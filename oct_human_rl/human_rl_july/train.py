"""
train.py — training entry point for DTS-ZSC.

Usage
    python train.py                                  # defaults
    python train.py --total_steps 500000
    python train.py --num_orders 3 --num_humans 2    # larger warehouse
    python train.py --device cuda

Training loop
─────────────
  1. Reset the env — a human profile is sampled from TRAIN_PROFILES only,
     so held-out partners stay unseen.
  2. Collect a fixed-length rollout.
  3. PPO update, with the learning rate annealed towards zero.
  4. Log rolling coordination metrics to console and CSV.
  5. Periodically measure the zero-shot gap on held-out partners.
  6. Checkpoint.

Correctness notes
─────────────────
  * Task completion is read from `info["tasks_done"]`, which the environment
    actually maintains.
  * Synchronisation is read from `info["num_busy"]`, i.e. how many workers are
    busy right now.
  * Time-limit truncation is bootstrapped (`r += gamma * V(s')`) instead of
    being treated as a true terminal, so the value function is not biased
    against long episodes.
  * Rollouts are always filled completely, so no stale slots are ever trained on.
"""

import os
import csv
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

from env.collaborative_env import (CollaborativeTaskEnv, obs_dim_for,
                                   action_dim_for, MAX_EPISODE_STEPS,
                                   DEFAULT_NUM_ORDERS, DEFAULT_NUM_HUMANS,
                                   DEFAULT_NUM_AI, ORDER_ARRIVAL_GAP)
from agent.ppo_agent import PPOAgent
from agent.human_observer import HumanBehaviorObserver, extra_obs_dim
from agent.curriculum import PartnerCurriculum
from utils.metrics import CoordinationMetrics
from utils.episode import EpisodeTracker
from utils.runner import profile_label, evaluate_policy
from human.simulated_human import TRAIN_PROFILES, TEST_PROFILES, HUMAN_PROFILES


# ── CLI ───────────────────────────────────────────────────────────────── #
def parse_args():
    p = argparse.ArgumentParser(description="Train the DTS-ZSC PPO agent")
    # scenario
    p.add_argument("--num_orders",  type=int, default=DEFAULT_NUM_ORDERS)
    p.add_argument("--num_humans",  type=int, default=DEFAULT_NUM_HUMANS)
    p.add_argument("--num_ai",      type=int, default=DEFAULT_NUM_AI)
    p.add_argument("--arrival_gap", type=int, default=ORDER_ARRIVAL_GAP)
    p.add_argument("--max_steps",   type=int, default=MAX_EPISODE_STEPS)
    p.add_argument("--train_profiles", type=str, default=",".join(TRAIN_PROFILES),
                   help="comma-separated partner pool to train on. Narrow it to a "
                        "single name to train a specialist, e.g. --train_profiles average")
    p.add_argument("--no_reassignment", action="store_true",
                   help="ablation: forbid taking an in-progress task off a worker")
    p.add_argument("--no_observer", action="store_true",
                   help="ablation: blank the human interaction abstraction layer. "
                        "The observation dimension is unchanged, so this isolates "
                        "the information rather than the network size")
    p.add_argument("--blind_partner", action="store_true",
                   help="ablation: also zero the per-worker capability signals "
                        "(observed speed, error rate, fatigue). Implies "
                        "--no_observer, leaving the agent able to see who is "
                        "free but nothing about how well they work")
    p.add_argument("--curriculum", action="store_true",
                   help="over-sample the partners the agent is currently worst "
                        "at, instead of sampling the pool uniformly")
    p.add_argument("--curriculum_temperature", type=float, default=0.35,
                   help="lower = sharper preference for weak partners; below "
                        "~0.2 the schedule collapses onto the hardest one")
    p.add_argument("--curriculum_min_share", type=float, default=0.4,
                   help="floor on each partner's sampling probability, as a "
                        "fraction of uniform; stops the schedule collapsing "
                        "onto the hardest partner")
    # optimisation
    p.add_argument("--total_steps",   type=int,   default=400_000)
    p.add_argument("--rollout_len",   type=int,   default=2048)
    p.add_argument("--hidden",        type=int,   default=256)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--lam",           type=float, default=0.95)
    p.add_argument("--clip_eps",      type=float, default=0.2)
    p.add_argument("--ent_coef",      type=float, default=0.01)
    p.add_argument("--vf_coef",       type=float, default=0.5)
    p.add_argument("--update_epochs", type=int,   default=4)
    p.add_argument("--minibatch",     type=int,   default=256)
    p.add_argument("--reward_scale",  type=float, default=0.1,
                   help="rewards are scaled by this for LEARNING only; the "
                        "rewards reported and logged remain unscaled")
    p.add_argument("--anneal_lr",     action="store_true", default=True)
    p.add_argument("--no_anneal_lr",  dest="anneal_lr", action="store_false")
    # bookkeeping
    p.add_argument("--device",        type=str, default="cpu")
    p.add_argument("--save_dir",      type=str, default="checkpoints")
    p.add_argument("--save_interval", type=int, default=100_000)
    p.add_argument("--eval_interval", type=int, default=40_000,
                   help="steps between held-out zero-shot probes (0 disables)")
    p.add_argument("--eval_episodes", type=int, default=3,
                   help="episodes per profile in the periodic probe")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--quiet",         action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


LOG_COLUMNS = [
    "step", "episodes", "mean_reward", "mean_ces", "completion_pct",
    "solve_rate", "on_time_pct", "sync_rate", "role_switch_rate",
    "conflict_rate", "human_share", "mean_steps",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "lr",
    "seen_ces", "held_out_ces", "zsc_gap",
]


# ── main ──────────────────────────────────────────────────────────────── #
def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    pool = [p.strip() for p in args.train_profiles.split(",") if p.strip()]
    unknown = [p for p in pool if p not in HUMAN_PROFILES]
    if unknown:
        raise SystemExit(f"unknown profile(s) {unknown}; "
                         f"choose from {sorted(HUMAN_PROFILES)}")
    leaked = [p for p in pool if p in TEST_PROFILES]
    if leaked:
        raise SystemExit(
            f"refusing to train on held-out profile(s) {leaked} — that would "
            "invalidate the zero-shot claim. Held out: " + ", ".join(TEST_PROFILES))

    if args.blind_partner:
        args.no_observer = True     # hiding the partner but keeping the
                                    # smoothed summary of it makes no sense

    # Reused for the periodic probe, so it sees the same capabilities.
    env_kwargs = dict(num_orders=args.num_orders,
                      num_humans=args.num_humans,
                      num_ai=args.num_ai,
                      arrival_gap=args.arrival_gap,
                      max_steps=args.max_steps,
                      allow_reassignment=not args.no_reassignment,
                      blind_partner=args.blind_partner)

    env = CollaborativeTaskEnv(train_profiles=pool, **env_kwargs)
    hbo = HumanBehaviorObserver(num_humans=env.num_humans,
                                total_tasks=env.num_global_tasks,
                                enabled=not args.no_observer)
    metrics = CoordinationMetrics(window=100)
    curriculum = PartnerCurriculum(pool,
                                   temperature=args.curriculum_temperature,
                                   min_share=args.curriculum_min_share)
    curriculum_rng = np.random.default_rng(args.seed + 7)

    def next_reset_options():
        """Partner override for the next episode; None lets the env sample."""
        if not args.curriculum:
            return None
        return {"profile": curriculum.sample(curriculum_rng)}

    obs_dim    = obs_dim_for(args.num_orders, env.num_workers) + extra_obs_dim(env.num_humans)
    action_dim = action_dim_for(args.num_orders, env.num_workers)

    agent = PPOAgent(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden=args.hidden, lr=args.lr, gamma=args.gamma, lam=args.lam,
        clip_eps=args.clip_eps, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        update_epochs=args.update_epochs, minibatch=args.minibatch,
        rollout_len=args.rollout_len, device=args.device)

    log_path = os.path.join(args.save_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(LOG_COLUMNS)

    print("=" * 78)
    print("DTS-ZSC training")
    print(f"  scenario   : {args.num_orders} orders x {env.num_global_tasks // args.num_orders} tasks "
          f"= {env.num_global_tasks} tasks | {args.num_ai} AI + {args.num_humans} human")
    print(f"  spaces     : obs_dim={obs_dim}  action_dim={action_dim} "
          f"(incl. WAIT)  max_steps={args.max_steps}")
    print(f"  train pool : {pool}"
          + ("   <-- SPECIALIST (single partner)" if len(pool) == 1 else ""))
    print(f"  held out   : {TEST_PROFILES}")
    print(f"  reassign   : {'enabled' if not args.no_reassignment else 'DISABLED (ablation)'}")
    print(f"  observer   : {'enabled' if not args.no_observer else 'BLANKED (ablation)'}")
    print(f"  partner    : {'BLIND (no capability signals)' if args.blind_partner else 'observable'}")
    print(f"  sampling   : {'curriculum (weighted to weak partners)' if args.curriculum else 'uniform'}")
    print(f"  budget     : {args.total_steps:,} steps, rollout={args.rollout_len}, "
          f"device={args.device}")
    print("=" * 78)

    # Recorded into every checkpoint so evaluation can reproduce the setup.
    run_config = {
        "train_profiles":   pool,
        "no_observer":      bool(args.no_observer),
        "no_reassignment":  bool(args.no_reassignment),
        "blind_partner":    bool(args.blind_partner),
        "curriculum":       bool(args.curriculum),
        "num_orders":       args.num_orders,
        "num_humans":       args.num_humans,
        "num_ai":           args.num_ai,
        "max_steps":        args.max_steps,
        "total_steps":      args.total_steps,
        "seed":             args.seed,
    }

    obs, info = env.reset(seed=args.seed, options=next_reset_options())
    hbo.reset()
    obs = hbo.enhance(obs)
    tracker = EpisodeTracker(env)

    global_step = 0
    episode = 0
    next_eval = args.eval_interval if args.eval_interval > 0 else None
    next_save = args.save_interval
    last_zsc = None            # set only on rows where a probe actually ran
    start_time = time.time()

    pbar = tqdm(total=args.total_steps, disable=args.quiet,
                unit="step", dynamic_ncols=True, desc="training")

    while global_step < args.total_steps:

        # ── collect one full rollout ──────────────────────────────────── #
        for _ in range(args.rollout_len):
            action, logp, value = agent.select_action(obs, info)
            hbo.observe_action(action, env)

            raw_next, reward, terminated, truncated, next_info = env.step(action)
            tracker.record(action, reward, next_info)
            hbo.update(next_info)
            next_obs = hbo.enhance(raw_next)

            done = terminated or truncated
            # The critic learns on scaled rewards (well-conditioned targets);
            # everything reported to the user stays in the original units.
            # Bootstrap through a time-limit truncation rather than pretending
            # the episode genuinely ended there — the bootstrap value is
            # already in scaled units, so the two terms are consistent.
            stored_reward = reward * args.reward_scale
            if truncated and not terminated:
                stored_reward += args.gamma * agent.value_of(next_obs, next_info)

            agent.store(obs, action, stored_reward, value, logp, done, info)

            obs, info = next_obs, next_info
            global_step += 1
            pbar.update(1)

            if done:
                stats = tracker.finish(profile_label(info), terminated)
                metrics.record(stats)
                curriculum.update(stats.profile, CoordinationMetrics.ces(stats))
                episode += 1
                tracker.reset()
                raw_obs, info = env.reset(options=next_reset_options())
                hbo.reset()
                obs = hbo.enhance(raw_obs)

        # ── PPO update ────────────────────────────────────────────────── #
        if args.anneal_lr:
            agent.set_lr_fraction(max(0.0, 1.0 - global_step / args.total_steps))
        loss_info = agent.update(obs, info, last_done=False)
        current_lr = agent.optimizer.param_groups[0]["lr"]

        # ── periodic zero-shot probe on held-out partners ─────────────── #
        # These columns are left blank on rows where no probe ran, so the
        # analytics charts plot one point per real measurement instead of
        # forward-filling a value across dozens of rows.
        last_zsc = None
        if next_eval is not None and global_step >= next_eval:
            next_eval += args.eval_interval
            agent.net.eval()
            # "Seen" is whatever this run actually trained on, so a specialist
            # is measured against its own partner, not the full training pool.
            unseen = [p for p in HUMAN_PROFILES if p not in pool]
            probe = evaluate_policy(agent, pool + unseen,
                                    episodes=args.eval_episodes,
                                    seed=10_000, env_kwargs=env_kwargs,
                                    observer_enabled=not args.no_observer)
            agent.net.train()
            report = probe.zsc_report(pool, unseen)
            last_zsc = {k: report[k] for k in ("seen_ces", "held_out_ces", "zsc_gap")}

        # ── logging ───────────────────────────────────────────────────── #
        row = [
            global_step, episode,
            round(metrics.mean_reward(), 2), round(metrics.mean_ces(), 4),
            round(metrics.mean_completion() * 100, 1),
            round(metrics.solve_rate() * 100, 1),
            round(metrics.on_time_rate() * 100, 1),
            round(metrics.sync_rate(), 4),
            round(metrics.role_switch_rate(), 4),
            round(metrics.conflict_rate(), 4),
            round(metrics.human_share(), 4),
            round(metrics.mean_steps(), 2),
            round(loss_info["policy_loss"], 5), round(loss_info["value_loss"], 5),
            round(loss_info["entropy"], 5), round(loss_info["approx_kl"], 5),
            round(loss_info["clip_frac"], 4), round(current_lr, 7),
        ] + ([round(last_zsc["seen_ces"], 4),
              round(last_zsc["held_out_ces"], 4),
              round(last_zsc["zsc_gap"], 4)] if last_zsc else ["", "", ""])
        writer.writerow(row)
        log_file.flush()

        pbar.set_postfix_str(
            f"R={metrics.mean_reward():+.1f} CES={metrics.mean_ces():.3f} "
            f"solved={metrics.solve_rate()*100:.0f}% H={metrics.human_share()*100:.0f}% "
            f"ent={loss_info['entropy']:.2f}")

        if not args.quiet:
            tqdm.write(
                f"step {global_step:>8,} | ep {episode:>6} | "
                f"R={metrics.mean_reward():+8.1f} | CES={metrics.mean_ces():.3f} | "
                f"compl={metrics.mean_completion()*100:5.1f}% | "
                f"solved={metrics.solve_rate()*100:5.1f}% | "
                f"onTime={metrics.on_time_rate()*100:5.1f}% | "
                f"sync={metrics.sync_rate():.2f} | "
                f"conf={metrics.conflict_rate():.3f} | "
                f"ent={loss_info['entropy']:.3f}")

        # ── checkpoint ────────────────────────────────────────────────── #
        if global_step >= next_save:
            next_save += args.save_interval
            agent.save(os.path.join(args.save_dir, f"agent_{global_step:08d}.pt"),
                       config=run_config)

    pbar.close()
    agent.save(os.path.join(args.save_dir, "agent_final.pt"), config=run_config)
    log_file.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 78)
    print(f"Training complete in {elapsed/60:.1f} min  "
          f"({global_step/max(elapsed,1e-9):.0f} steps/s)")
    print(metrics.summary())
    print("-" * 78)
    print(metrics.profile_summary())
    if args.curriculum:
        print("-" * 78)
        print("Final partner sampling weights: " + curriculum.summary())
    print("=" * 78)
    print(f"Log written to {log_path}")
    print("Next: python evaluate.py --checkpoint checkpoints/agent_final.pt --baselines")


if __name__ == "__main__":
    train(parse_args())

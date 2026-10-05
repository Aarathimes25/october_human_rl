"""
evaluate.py — evaluate a trained DTS-ZSC agent.

Usage
    python evaluate.py --checkpoint checkpoints/agent_final.pt
    python evaluate.py --checkpoint checkpoints/agent_final.pt --baselines
    python evaluate.py --checkpoint checkpoints/agent_final.pt --episodes 100
    python evaluate.py --checkpoint checkpoints/agent_final.pt --render --episodes 1

What it reports
───────────────
  * a per-profile table (CES, reward, steps, completion, on-time, sync, human share)
  * the zero-shot summary, split into partners the agent trained with ("seen")
    and partners it has never met ("held-out"), plus the gap between them
  * with --baselines, the same numbers for a random policy, a static
    predefined-sequence scheduler and a strong greedy heuristic

Every episode is seeded, and the environment is reset exactly once per episode,
so runs are reproducible.
"""

import os
import json
import argparse
import numpy as np
import torch

from env.collaborative_env import (CollaborativeTaskEnv, obs_dim_for,
                                   action_dim_for, MAX_EPISODE_STEPS,
                                   DEFAULT_NUM_ORDERS, DEFAULT_NUM_HUMANS,
                                   DEFAULT_NUM_AI, ORDER_ARRIVAL_GAP)
from agent.ppo_agent import PPOAgent
from agent.human_observer import extra_obs_dim
from utils.metrics import CoordinationMetrics
from utils.runner import evaluate_policy
from utils.stats import bootstrap_ci, paired_diff, episode_ces
from human.simulated_human import TRAIN_PROFILES, TEST_PROFILES
from baselines import BASELINES, make_baseline


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained DTS-ZSC agent")
    p.add_argument("--checkpoint", type=str, default="checkpoints/agent_final.pt")
    p.add_argument("--episodes",   type=int, default=50,
                   help="episodes per human profile")
    p.add_argument("--baselines",  action="store_true",
                   help="also evaluate random / static / greedy policies")
    p.add_argument("--compare", action="append", default=[], metavar="NAME=PATH",
                   help="also evaluate another checkpoint, e.g. "
                        "--compare specialist=checkpoints/specialist_average/agent_final.pt "
                        "(repeatable). Use this to contrast the zero-shot agent with a "
                        "partner-specific one trained on historical interaction data.")
    p.add_argument("--no_reassignment", action="store_true",
                   help="ablation: forbid taking an in-progress task off a worker")
    p.add_argument("--render",     action="store_true")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of taking the argmax")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=1000)
    p.add_argument("--results_dir", type=str, default="results")
    # scenario — must match the configuration the checkpoint was trained on
    p.add_argument("--num_orders",  type=int, default=DEFAULT_NUM_ORDERS)
    p.add_argument("--num_humans",  type=int, default=DEFAULT_NUM_HUMANS)
    p.add_argument("--num_ai",      type=int, default=DEFAULT_NUM_AI)
    p.add_argument("--arrival_gap", type=int, default=ORDER_ARRIVAL_GAP)
    p.add_argument("--max_steps",   type=int, default=MAX_EPISODE_STEPS)
    return p.parse_args()


# ── printing helpers ──────────────────────────────────────────────────── #
def print_profile_table(metrics: CoordinationMetrics, title: str):
    print(f"\n{title}")
    print(metrics.profile_summary())


def print_zsc(report: dict, label: str = ""):
    head = f"Zero-shot generalisation{(' — ' + label) if label else ''}"
    print("\n" + "─" * 78)
    print(head)
    print("─" * 78)
    print(f"  seen partners     {', '.join(report['seen'].keys()) or '—'}")
    print(f"    mean CES        {report['seen_ces']:.3f}")
    print(f"    mean reward     {report['seen_reward']:+.1f}")
    print(f"    solve rate      {report['seen_solve']*100:.1f}%")
    print(f"  held-out partners {', '.join(report['held_out'].keys()) or '—'}")
    print(f"    mean CES        {report['held_out_ces']:.3f}")
    print(f"    mean reward     {report['held_out_reward']:+.1f}")
    print(f"    solve rate      {report['held_out_solve']*100:.1f}%")
    print(f"  ZSC gap           {report['zsc_gap']:+.3f} "
          f"({report['relative_gap']*100:+.1f}% relative)")
    print("  A gap near zero means the agent coordinates with an unfamiliar")
    print("  partner about as well as with a familiar one.")


def ces_bars(metrics: CoordinationMetrics, seen, held):
    table = metrics.profile_table()
    print("\nPer-partner CES")
    width = 46
    for group, names in (("seen", seen), ("held-out", held)):
        for name in names:
            if name not in table:
                continue
            v = table[name]["ces"]
            bar = "#" * max(int(v * width), 0)
            print(f"  {name:<12} [{group:<8}] {bar:<{width}} {v:.3f}")


# ── main ──────────────────────────────────────────────────────────────── #
def evaluate(args):
    if args.render and args.episodes > 2:
        print(f"[note] --render with {args.episodes} episodes per profile would print "
              f"thousands of frames; capping at 1 episode per profile.")
        args.episodes = 1

    env_kwargs = dict(num_orders=args.num_orders,
                      num_humans=args.num_humans,
                      num_ai=args.num_ai,
                      arrival_gap=args.arrival_gap,
                      max_steps=args.max_steps,
                      allow_reassignment=not args.no_reassignment)

    probe = CollaborativeTaskEnv(**env_kwargs)
    obs_dim    = obs_dim_for(args.num_orders, probe.num_workers) + extra_obs_dim(args.num_humans)
    action_dim = action_dim_for(args.num_orders, probe.num_workers)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}\n"
            "Train one first:  python train.py")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=args.device)
    agent.load(args.checkpoint)
    agent.net.eval()

    all_profiles = list(TRAIN_PROFILES) + list(TEST_PROFILES)

    print("=" * 78)
    print(f"Evaluating   {args.checkpoint}")
    print(f"Scenario     {args.num_orders} orders, {args.num_ai} AI + "
          f"{args.num_humans} human, max {args.max_steps} steps")
    print(f"Episodes     {args.episodes} per profile x {len(all_profiles)} profiles")
    print(f"Policy       {'stochastic' if args.stochastic else 'deterministic (argmax)'}")
    print("=" * 78)

    # An ablated agent is handed a feature block it never learned to read
    # unless we evaluate it the way it was trained.
    agent_observer = not agent.config.get("no_observer", False)
    if not agent_observer:
        print("[note] trained with the abstraction layer blanked; "
              "evaluating it the same way.")

    rl_metrics = evaluate_policy(
        agent, all_profiles, episodes=args.episodes, seed=args.seed,
        env_kwargs=env_kwargs, deterministic=not args.stochastic,
        render=args.render, observer_enabled=agent_observer)

    print_profile_table(rl_metrics, "PPO agent — per-partner results")
    report = rl_metrics.zsc_report(TRAIN_PROFILES, TEST_PROFILES)
    print_zsc(report, "PPO agent")
    ces_bars(rl_metrics, TRAIN_PROFILES, TEST_PROFILES)
    print("\nOverall:", rl_metrics.summary(n=10 ** 9))

    results = {
        "checkpoint":  args.checkpoint,
        "episodes":    args.episodes,
        "scenario":    env_kwargs,
        "ppo": {
            "profiles":   rl_metrics.profile_table(),
            "zsc":        {k: v for k, v in report.items()
                           if k not in ("seen", "held_out")},
            "overall_ces":    rl_metrics.mean_ces(n=10 ** 9),
            "overall_reward": rl_metrics.mean_reward(n=10 ** 9),
            "solve_rate":     rl_metrics.solve_rate(n=10 ** 9),
        },
    }

    # ── other checkpoints (e.g. a partner-specific specialist) ────────── #
    extra: dict = {}
    for spec in args.compare:
        if "=" not in spec:
            raise SystemExit(f"--compare expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"--compare checkpoint not found: {path}")
        other = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=args.device)
        other.load(path)
        other.net.eval()
        m = evaluate_policy(other, all_profiles, episodes=args.episodes,
                            seed=args.seed, env_kwargs=env_kwargs,
                            deterministic=not args.stochastic,
                            observer_enabled=not other.config.get("no_observer", False))
        extra[name] = m
        rep = m.zsc_report(TRAIN_PROFILES, TEST_PROFILES)
        results.setdefault("compare", {})[name] = {
            "checkpoint": path,
            "profiles": m.profile_table(),
            "zsc": {k: v for k, v in rep.items() if k not in ("seen", "held_out")},
            "overall_ces":    m.mean_ces(10 ** 9),
            "overall_reward": m.mean_reward(10 ** 9),
            "solve_rate":     m.solve_rate(10 ** 9),
        }
        print_profile_table(m, f"{name} — per-partner results")

    if extra:
        print("\n" + "─" * 78)
        print("Generalist vs. partner-specific agent")
        print("─" * 78)
        print("A specialist is the 'behavioural model trained on historical "
              "interaction data'\nthe abstract contrasts with: one partner "
              "during training, nothing else.\n")
        head = f"{'partner':<12}{'zero-shot':>11}" + "".join(
            f"{n:>13}" for n in extra) + f"{'winner':>12}"
        print(head)
        print("-" * len(head))
        base = rl_metrics.profile_table()
        for p in all_profiles:
            row = f"{p:<12}{base[p]['ces']:>11.3f}"
            scores = {"zero-shot": base[p]["ces"]}
            for n, m in extra.items():
                v = m.profile_table()[p]["ces"]
                scores[n] = v
                row += f"{v:>13.3f}"
            best = max(scores, key=scores.get)
            print(row + f"{best:>12}")

    # ── scripted baselines ────────────────────────────────────────────── #
    baseline_metrics: dict = {}
    if args.baselines:
        results["baselines"] = {}
        for name in ("random", "static", "greedy"):
            m = evaluate_policy(
                lambda env, n=name: make_baseline(n, env, seed=args.seed),
                all_profiles, episodes=args.episodes, seed=args.seed,
                env_kwargs=env_kwargs)
            baseline_metrics[name] = m
            rep = m.zsc_report(TRAIN_PROFILES, TEST_PROFILES)
            results["baselines"][name] = {
                "profiles": m.profile_table(),
                "zsc": {k: v for k, v in rep.items()
                        if k not in ("seen", "held_out")},
                "overall_ces":    m.mean_ces(n=10 ** 9),
                "overall_reward": m.mean_reward(n=10 ** 9),
                "solve_rate":     m.solve_rate(n=10 ** 9),
            }

    # ── combined scoreboard ───────────────────────────────────────────── #
    rows = [("ppo", rl_metrics)] + list(extra.items()) + list(baseline_metrics.items())
    if len(rows) > 1:
        print("\n" + "=" * 78)
        print("Policy comparison")
        print("=" * 78)
        head = (f"{'policy':<12}{'CES':>8}{'95% CI':>18}{'reward':>9}{'steps':>7}"
                f"{'solved':>8}{'onTime':>8}{'sync':>7}{'human%':>8}{'reasgn':>8}"
                f"{'ZSCgap':>9}")
        print(head)
        print("-" * len(head))
        for name, m in rows:
            rep = m.zsc_report(TRAIN_PROFILES, TEST_PROFILES)
            est = bootstrap_ci(episode_ces(m), seed=args.seed)
            ci = f"[{est.lo:.3f}, {est.hi:.3f}]"
            print(f"{name:<12}{m.mean_ces(10**9):>8.3f}{ci:>18}"
                  f"{m.mean_reward(10**9):>9.1f}{m.mean_steps(10**9):>7.1f}"
                  f"{m.solve_rate(10**9)*100:>7.1f}%{m.on_time_rate(10**9)*100:>7.1f}%"
                  f"{m.sync_rate(10**9):>7.3f}{m.human_share(10**9)*100:>7.1f}%"
                  f"{m.mean_reassignments(10**9):>8.2f}{rep['zsc_gap']:>+9.3f}")
            results.setdefault("intervals", {})[name] = est.as_dict()

        best = max(rows, key=lambda kv: kv[1].mean_ces(10 ** 9))
        print(f"\nBest coordination efficiency: {best[0]} "
              f"(CES {best[1].mean_ces(10**9):.3f})")

    # ── significance ──────────────────────────────────────────────────── #
    if len(rows) > 1:
        print("\n" + "─" * 78)
        print("Paired bootstrap on per-episode CES")
        print("─" * 78)
        print(f"{args.episodes * len(all_profiles)} paired episodes per "
              f"comparison; identical seeds, so episode i is the same "
              f"warehouse and partner for every policy.\n")

        ppo_ces = episode_ces(rl_metrics)
        results["significance"] = {}
        for name, m in rows[1:]:
            cmp = paired_diff(ppo_ces, episode_ces(m), "ppo", name, seed=args.seed)
            results["significance"][name] = cmp.as_dict()
            print(f"  vs {name:<12} {cmp}")
            print(f"  {'':<15}-> PPO is {cmp.verdict()}\n")

        print("  *** p<0.001   ** p<0.01   * p<0.05 — an interval excluding "
              "zero means the gap is not sampling noise.")

    # ── persist for the analytics module ──────────────────────────────── #
    os.makedirs(args.results_dir, exist_ok=True)
    json_path = os.path.join(args.results_dir, "evaluation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    csv_path = os.path.join(args.results_dir, "episodes.csv")
    try:
        import pandas as pd
        frames = []
        for name, m in rows:
            df = pd.DataFrame(m.to_records())
            df.insert(0, "policy", name)
            frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
    except ImportError:
        csv_path = None

    print("\n" + "=" * 78)
    print(f"Results written to {json_path}")
    if csv_path:
        print(f"Per-episode records written to {csv_path}")
    print("Next: python plot_training.py   (training curves + comparison charts)")
    print("=" * 78)


if __name__ == "__main__":
    evaluate(parse_args())

"""
ablation.py — what is each component actually worth?

Evaluates agents trained on identical budgets and seeds with one component
changed each. Train the arms first (they are independent, so run them in
parallel), then run this:

    python train.py --total_steps 250000 --seed 42 --save_dir ablations/full
    python train.py --total_steps 250000 --seed 42 --no_reassignment \
                    --save_dir ablations/no_reassignment
    ...
    python ablation.py

Every arm sees the same partners on the same episode seeds, so the comparison
against `full` is paired. Each arm is also evaluated the way it was trained —
observer blanked, reassignment forbidden, partner hidden — read from the config
its checkpoint carries. Evaluating an ablated agent under the full setup would
hand it inputs or actions it never learned.
"""

import argparse
import json
import os

from agent.human_observer import extra_obs_dim
from agent.ppo_agent import PPOAgent
from env.collaborative_env import (CollaborativeTaskEnv, obs_dim_for,
                                   action_dim_for, MAX_EPISODE_STEPS,
                                   DEFAULT_NUM_ORDERS, DEFAULT_NUM_HUMANS,
                                   DEFAULT_NUM_AI, ORDER_ARRIVAL_GAP)
from human.simulated_human import TRAIN_PROFILES, TEST_PROFILES
from utils.runner import evaluate_policy
from utils.stats import bootstrap_ci, paired_diff, episode_ces


# name -> (directory, human-readable description)
DEFAULT_ARMS = [
    ("full",            "ablations/full",            "every component enabled"),
    ("no reassignment", "ablations/no_reassignment", "cannot take a task off a worker"),
    ("no observer",     "ablations/no_observer",
     "abstraction layer blanked, raw partner signals still visible"),
    ("blind partner",   "ablations/blind_partner",
     "no partner signals at all: sees who is free, not how capable"),
    ("curriculum",      "ablations/curriculum",      "weak partners over-sampled"),
]

BASELINE_ARM = "full"


def parse_args():
    p = argparse.ArgumentParser(description="DTS-ZSC component ablation study")
    p.add_argument("--episodes", type=int, default=30,
                   help="episodes per partner per arm")
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--dir", type=str, default="ablations",
                   help="directory holding one sub-directory per arm")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num_orders", type=int, default=DEFAULT_NUM_ORDERS)
    p.add_argument("--num_humans", type=int, default=DEFAULT_NUM_HUMANS)
    p.add_argument("--num_ai",     type=int, default=DEFAULT_NUM_AI)
    p.add_argument("--arrival_gap", type=int, default=ORDER_ARRIVAL_GAP)
    p.add_argument("--max_steps",  type=int, default=MAX_EPISODE_STEPS)
    return p.parse_args()


def run(args):
    profiles = list(TRAIN_PROFILES) + list(TEST_PROFILES)
    probe = CollaborativeTaskEnv(num_orders=args.num_orders,
                                 num_humans=args.num_humans,
                                 num_ai=args.num_ai)
    obs_dim = obs_dim_for(args.num_orders, probe.num_workers) + extra_obs_dim(args.num_humans)
    action_dim = action_dim_for(args.num_orders, probe.num_workers)

    arms = [(name, path, desc) for name, path, desc in DEFAULT_ARMS]
    missing = [(n, p) for n, p, _ in arms
               if not os.path.exists(os.path.join(p, "agent_final.pt"))]
    if missing:
        print("Missing trained arms — train them first:\n")
        for n, p in missing:
            print(f"  {n:<18} expected {os.path.join(p, 'agent_final.pt')}")
        print("\nSee the usage block at the top of ablation.py.")
        return 1

    print("=" * 82)
    print("DTS-ZSC component ablation")
    print(f"  {args.episodes} episodes x {len(profiles)} partners per arm, "
          f"identical seeds across arms")
    print("=" * 82)

    metrics, configs = {}, {}
    for name, path, desc in arms:
        ckpt = os.path.join(path, "agent_final.pt")
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=args.device)
        agent.load(ckpt)
        agent.net.eval()
        cfg = agent.config
        configs[name] = cfg

        env_kwargs = dict(num_orders=args.num_orders,
                          num_humans=args.num_humans,
                          num_ai=args.num_ai,
                          arrival_gap=args.arrival_gap,
                          max_steps=args.max_steps,
                          allow_reassignment=not cfg.get("no_reassignment", False),
                          blind_partner=cfg.get("blind_partner", False))

        metrics[name] = evaluate_policy(
            agent, profiles, episodes=args.episodes, seed=args.seed,
            env_kwargs=env_kwargs,
            observer_enabled=not cfg.get("no_observer", False))

    # ── table ─────────────────────────────────────────────────────────── #
    head = (f"{'arm':<18}{'CES':>8}{'95% CI':>18}{'reward':>9}{'steps':>7}"
            f"{'solved':>8}{'onTime':>8}{'reasgn':>8}{'ZSCgap':>9}")
    print("\n" + head)
    print("-" * len(head))
    for name, _, _ in arms:
        m = metrics[name]
        est = bootstrap_ci(episode_ces(m), seed=args.seed)
        rep = m.zsc_report(TRAIN_PROFILES, TEST_PROFILES)
        ci = f"[{est.lo:.3f}, {est.hi:.3f}]"
        print(f"{name:<18}{m.mean_ces(10**9):>8.3f}{ci:>18}"
              f"{m.mean_reward(10**9):>9.1f}{m.mean_steps(10**9):>7.1f}"
              f"{m.solve_rate(10**9)*100:>7.1f}%{m.on_time_rate(10**9)*100:>7.1f}%"
              f"{m.mean_reassignments(10**9):>8.2f}{rep['zsc_gap']:>+9.3f}")

    # ── what each component is worth ──────────────────────────────────── #
    print("\n" + "─" * 82)
    print(f"Paired comparison against '{BASELINE_ARM}' (same episodes, same seeds)")
    print("─" * 82)
    base_ces = episode_ces(metrics[BASELINE_ARM])
    comparisons = {}
    for name, _, desc in arms:
        if name == BASELINE_ARM:
            continue
        cmp = paired_diff(episode_ces(metrics[name]), base_ces,
                          name, BASELINE_ARM, seed=args.seed)
        comparisons[name] = cmp.as_dict()
        if not cmp.significant:
            direction = "no measurable effect"
        else:
            direction = "arm is better" if cmp.diff > 0 else "arm is worse"
        print(f"  {name:<18} {cmp.diff:+.4f} CES "
              f"[{cmp.lo:+.4f}, {cmp.hi:+.4f}]  p={cmp.p_value:.4f}   {direction}")
        print(f"  {'':<18} {desc}")

    print("\n  A negative number means removing that component made things "
          "worse, so it was earning its place.")

    # An overall average can hide a component that only matters on the hard
    # partners, so break it out per partner as well.
    print("\n" + "─" * 82)
    print("Per-partner CES by arm")
    print("─" * 82)
    head2 = f"{'partner':<12}" + "".join(f"{n:>18}" for n, _, _ in arms)
    print(head2)
    print("-" * len(head2))
    for p in profiles:
        row = f"{p:<12}"
        for name, _, _ in arms:
            row += f"{metrics[name].profile_table()[p]['ces']:>18.3f}"
        print(row)

    # ── persist ───────────────────────────────────────────────────────── #
    os.makedirs(args.results_dir, exist_ok=True)
    out = {
        "episodes": args.episodes,
        "seed": args.seed,
        "baseline_arm": BASELINE_ARM,
        "arms": {
            name: {
                "description": desc,
                "config": configs[name],
                "overall_ces":    metrics[name].mean_ces(10 ** 9),
                "overall_reward": metrics[name].mean_reward(10 ** 9),
                "solve_rate":     metrics[name].solve_rate(10 ** 9),
                "on_time_rate":   metrics[name].on_time_rate(10 ** 9),
                "mean_steps":     metrics[name].mean_steps(10 ** 9),
                "reassignments":  metrics[name].mean_reassignments(10 ** 9),
                "ci": bootstrap_ci(episode_ces(metrics[name]),
                                   seed=args.seed).as_dict(),
                "zsc": {k: v for k, v in
                        metrics[name].zsc_report(TRAIN_PROFILES, TEST_PROFILES).items()
                        if k not in ("seen", "held_out")},
                "profiles": metrics[name].profile_table(),
            }
            for name, _, desc in arms
        },
        "comparisons": comparisons,
    }
    path = os.path.join(args.results_dir, "ablation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("\n" + "=" * 82)
    print(f"Written to {path}")
    print("Run  python plot_training.py  to render the ablation chart.")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

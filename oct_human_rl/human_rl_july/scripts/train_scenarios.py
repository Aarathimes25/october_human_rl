"""
Train one PPO model for every dashboard scenario that does not have one yet.

A PPO network has fixed input and output sizes, so the default model
(2 orders, 1 AI + 1 human; checkpoints/agent_final.pt) cannot drive any other
workload or team. This script trains a separate model per scenario into

    checkpoints/scenarios/<N>o_<A>ai_<H>h/agent_final.pt

which the dashboard picks up automatically. The default model and the reported
research results are never touched.

After each run the new model is compared with the Greedy coordinator on the
same seeds, and the numbers are written to summary.json next to the model.

    python3 scripts/train_scenarios.py                 # every missing scenario
    python3 scripts/train_scenarios.py --jobs 2        # fewer runs at once
    python3 scripts/train_scenarios.py --only 3o_2ai_2h 1o_1ai_1h
    python3 scripts/train_scenarios.py --list          # show what exists
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import (MAX_UI_ORDERS, SCENARIO_CHECKPOINT_DIR, CHECKPOINT,  # noqa: E402
                 scenario_name, scenario_checkpoint, ui_max_steps)
from agent.human_observer import extra_obs_dim                         # noqa: E402
from agent.ppo_agent import PPOAgent                                   # noqa: E402
from baselines import make_baseline                                    # noqa: E402
from env.collaborative_env import obs_dim_for, action_dim_for          # noqa: E402
from human.simulated_human import TRAIN_PROFILES, TEST_PROFILES        # noqa: E402
from utils.runner import evaluate_policy                               # noqa: E402

# The dashboard's team selector: (num_ai, num_humans).
TEAMS = [(1, 1), (1, 2), (2, 2)]


def all_scenarios():
    for num_orders in range(1, MAX_UI_ORDERS + 1):
        for num_ai, num_humans in TEAMS:
            yield num_orders, num_ai, num_humans


def compare_with_greedy(scenario, episodes):
    num_orders, num_ai, num_humans = scenario
    env_kwargs = dict(num_orders=num_orders, num_ai=num_ai, num_humans=num_humans,
                      max_steps=ui_max_steps(num_orders))
    agent = PPOAgent(obs_dim=obs_dim_for(num_orders, num_ai + num_humans)
                     + extra_obs_dim(num_humans),
                     action_dim=action_dim_for(num_orders, num_ai + num_humans))
    agent.load(scenario_checkpoint(*scenario))
    agent.net.eval()
    profiles = TRAIN_PROFILES + TEST_PROFILES
    out = {}
    for name, policy in (("ppo", agent),
                         ("greedy", lambda env: make_baseline("greedy", env, seed=0))):
        m = evaluate_policy(policy, profiles, episodes=episodes, seed=5000,
                            env_kwargs=env_kwargs)
        out[name] = {"ces": m.mean_ces(10 ** 9),
                     "solve_rate": m.solve_rate(10 ** 9),
                     "steps": m.mean_steps(10 ** 9)}
    return out


def train_one(scenario, args):
    name = scenario_name(*scenario)
    save_dir = os.path.join(SCENARIO_CHECKPOINT_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    num_orders, num_ai, num_humans = scenario
    cmd = [sys.executable, os.path.join(ROOT, "train.py"),
           "--num_orders", str(num_orders),
           "--num_ai", str(num_ai),
           "--num_humans", str(num_humans),
           "--max_steps", str(ui_max_steps(num_orders)),
           "--total_steps", str(args.steps),
           "--save_interval", str(args.steps * 10),   # final model only
           "--seed", str(args.seed),
           "--save_dir", save_dir,
           "--quiet"]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    started = time.time()
    print(f"[start] {name}", flush=True)
    with open(os.path.join(save_dir, "train.log"), "w", encoding="utf-8") as log:
        code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT)
    minutes = (time.time() - started) / 60
    if code != 0 or not os.path.exists(scenario_checkpoint(*scenario)):
        print(f"[FAIL]  {name} after {minutes:.0f} min; see {save_dir}/train.log",
              flush=True)
        return name, False

    summary = {"scenario": name, "total_steps": args.steps, "seed": args.seed,
               "minutes": round(minutes, 1),
               "comparison": compare_with_greedy(scenario, args.eval_episodes)}
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    ppo, greedy = summary["comparison"]["ppo"], summary["comparison"]["greedy"]
    print(f"[done]  {name} in {minutes:.0f} min · CES ppo {ppo['ces']:.3f} "
          f"vs greedy {greedy['ces']:.3f} · solved {ppo['solve_rate']*100:.0f}% "
          f"vs {greedy['solve_rate']*100:.0f}%", flush=True)
    return name, True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None,
                   help="scenario names such as 3o_2ai_2h (default: all missing)")
    p.add_argument("--steps", type=int, default=400_000,
                   help="training steps per scenario (default matches train.py)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="scenarios trained at the same time")
    p.add_argument("--threads", type=int, default=2, help="CPU threads per run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="episodes per partner profile in the Greedy comparison")
    p.add_argument("--force", action="store_true", help="retrain existing models")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = p.parse_args()

    wanted = []
    for scenario in all_scenarios():
        name = scenario_name(*scenario)
        path = scenario_checkpoint(*scenario)
        exists = os.path.exists(path)
        if args.list:
            tag = "default" if path == CHECKPOINT else ("trained" if exists else "missing")
            print(f"  {name:<12} {tag}")
            continue
        if path == CHECKPOINT:
            continue                      # never retrain the reported model
        if args.only is not None and name not in args.only:
            continue
        if exists and not args.force:
            continue
        wanted.append(scenario)
    if args.list:
        return 0
    if not wanted:
        print("Nothing to train: every requested scenario already has a model.")
        return 0

    print(f"Training {len(wanted)} scenario(s), {args.jobs} at a time, "
          f"{args.steps:,} steps each.", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda s: train_one(s, args), wanted))
    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} scenario(s) trained."
          + (f" Failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

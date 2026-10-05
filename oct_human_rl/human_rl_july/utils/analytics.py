"""
Performance Analytics Module.

Turns the artefacts produced by training and evaluation into the charts the
project reports on:

  from checkpoints/training_log.csv
    1. training_progress.png    reward, CES, completion / solve rate, on-time rate
    2. coordination.png         synchronisation, role switching, conflicts,
                                share of work given to the human
    3. optimisation.png         policy loss, value loss, entropy, KL, clip
                                fraction, learning rate
    4. zero_shot_gap.png        seen vs held-out CES over the course of training

  from results/evaluation.json
    5. profile_ces.png          per-partner CES, seen vs held-out
    6. policy_comparison.png    PPO against the random / static / greedy baselines
    7. adaptive_allocation.png  how much work each partner type was given

Every chart is written to the results directory; nothing is displayed, so this
runs fine headless.
"""

import os
import json
from typing import Optional, Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.stats import significance_stars


DEFAULT_LOG      = os.path.join("checkpoints", "training_log.csv")
DEFAULT_EVAL     = os.path.join("results", "evaluation.json")
DEFAULT_OUT      = "results"

SEEN_COLOR = "#3B6D11"
HELD_COLOR = "#BA7517"
PPO_COLOR  = "#534AB7"
GRID_KW    = dict(alpha=0.3, linestyle="--", linewidth=0.6)


# ── helpers ───────────────────────────────────────────────────────────── #
def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, **GRID_KW)
    ax.tick_params(labelsize=8)


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _smooth(series: pd.Series, window: int = 5) -> pd.Series:
    if len(series) < window * 2:
        return series
    return series.rolling(window, min_periods=1, center=True).mean()


# ── training-log charts ───────────────────────────────────────────────── #
def plot_training_progress(df: pd.DataFrame, out_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0][0]
    ax.plot(df["step"], df["mean_reward"], color=PPO_COLOR, alpha=0.35, linewidth=1)
    ax.plot(df["step"], _smooth(df["mean_reward"]), color=PPO_COLOR, linewidth=2)
    _style(ax, "Episode reward", "training step", "mean reward")

    ax = axes[0][1]
    ax.plot(df["step"], df["mean_ces"], color=SEEN_COLOR, alpha=0.35, linewidth=1)
    ax.plot(df["step"], _smooth(df["mean_ces"]), color=SEEN_COLOR, linewidth=2)
    _style(ax, "Coordination Efficiency Score", "training step", "CES")

    ax = axes[1][0]
    ax.plot(df["step"], _smooth(df["completion_pct"]), label="tasks completed",
            color=PPO_COLOR, linewidth=2)
    ax.plot(df["step"], _smooth(df["solve_rate"]), label="episodes fully solved",
            color=HELD_COLOR, linewidth=2)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    _style(ax, "Task completion", "training step", "percent")

    ax = axes[1][1]
    ax.plot(df["step"], _smooth(df["on_time_pct"]), color="#A32D2D", linewidth=2)
    ax.set_ylim(0, 105)
    _style(ax, "Orders delivered before deadline", "training step", "percent on time")

    fig.suptitle("DTS-ZSC — training progress", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "training_progress.png")


def plot_coordination(df: pd.DataFrame, out_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0][0]
    ax.plot(df["step"], _smooth(df["sync_rate"]), color=PPO_COLOR, linewidth=2)
    _style(ax, "Human-AI synchronisation\n(fraction of steps with 2+ workers busy)",
           "training step", "sync rate")

    ax = axes[0][1]
    ax.plot(df["step"], _smooth(df["role_switch_rate"]), color=HELD_COLOR, linewidth=2)
    _style(ax, "Role switch rate", "training step", "switches per step")

    ax = axes[1][0]
    ax.plot(df["step"], df["conflict_rate"], color="#A32D2D", linewidth=2)
    ax.set_ylim(bottom=-0.005)
    _style(ax, "Illegal-action (conflict) rate", "training step", "conflicts per step")

    ax = axes[1][1]
    ax.plot(df["step"], _smooth(df["human_share"] * 100), color=SEEN_COLOR, linewidth=2)
    ax.axhline(50, color="grey", linestyle=":", linewidth=1, label="even split")
    ax.legend(fontsize=8)
    _style(ax, "Share of assignments given to the human",
           "training step", "percent of assignments")

    fig.suptitle("DTS-ZSC — coordination behaviour", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "coordination.png")


def plot_optimisation(df: pd.DataFrame, out_dir: str) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    panels = [
        ("policy_loss", "Policy loss",   PPO_COLOR),
        ("value_loss",  "Value loss",    "#A32D2D"),
        ("entropy",     "Policy entropy", SEEN_COLOR),
        ("approx_kl",   "Approx. KL",    HELD_COLOR),
        ("clip_frac",   "Clip fraction", "#6b6b66"),
        ("lr",          "Learning rate", "#1a1a18"),
    ]
    for ax, (col, title, colour) in zip(axes.flat, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df["step"], df[col], color=colour, linewidth=1.6)
        _style(ax, title, "training step", col)

    fig.suptitle("DTS-ZSC — PPO optimisation diagnostics",
                 fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "optimisation.png")


def plot_zero_shot_gap(df: pd.DataFrame, out_dir: str) -> Optional[str]:
    if "seen_ces" not in df.columns:
        return None
    probes = df.dropna(subset=["seen_ces", "held_out_ces"])
    probes = probes[probes["seen_ces"] != 0]
    # One marker per distinct measurement. Older logs carried a probe's value
    # forward on every row until the next probe, which drew a step function
    # implying far more measurements than were actually taken.
    if not probes.empty:
        changed = (probes["seen_ces"].ne(probes["seen_ces"].shift())
                   | probes["held_out_ces"].ne(probes["held_out_ces"].shift()))
        probes = probes[changed]
    if probes.empty:
        print("  (no zero-shot probes recorded — skipping zero_shot_gap.png)")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(probes["step"], probes["seen_ces"], "o-", color=SEEN_COLOR,
            linewidth=2, markersize=4, label="seen partners")
    ax.plot(probes["step"], probes["held_out_ces"], "s-", color=HELD_COLOR,
            linewidth=2, markersize=4, label="held-out partners (zero-shot)")
    ax.fill_between(probes["step"], probes["held_out_ces"], probes["seen_ces"],
                    color="grey", alpha=0.18, label="generalisation gap")
    ax.legend(fontsize=8)
    _style(ax, "Coordination efficiency: seen vs unseen partners",
           "training step", "CES")

    ax = axes[1]
    ax.plot(probes["step"], probes["zsc_gap"], "o-", color="#A32D2D",
            linewidth=2, markersize=4)
    ax.axhline(0, color="grey", linewidth=1)
    _style(ax, "Zero-shot generalisation gap\n(lower is better)",
           "training step", "seen CES - held-out CES")

    fig.suptitle("DTS-ZSC — zero-shot generalisation during training",
                 fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "zero_shot_gap.png")


# ── evaluation-json charts ────────────────────────────────────────────── #
def plot_profile_ces(results: Dict[str, Any], seen: List[str],
                     held: List[str], out_dir: str) -> str:
    table = results["ppo"]["profiles"]
    names  = [p for p in list(seen) + list(held) if p in table]
    values = [table[p]["ces"] for p in names]
    colors = [SEEN_COLOR if p in seen else HELD_COLOR for p in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=8)

    zsc = results["ppo"]["zsc"]
    ax.axhline(zsc["seen_ces"], color=SEEN_COLOR, linestyle="--", linewidth=1.2,
               label=f"seen mean {zsc['seen_ces']:.3f}")
    ax.axhline(zsc["held_out_ces"], color=HELD_COLOR, linestyle="--", linewidth=1.2,
               label=f"held-out mean {zsc['held_out_ces']:.3f}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(values + [0.1]) * 1.2)
    _style(ax, f"Per-partner coordination efficiency  "
               f"(zero-shot gap {zsc['zsc_gap']:+.3f})",
           "human partner profile", "CES")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, out_dir, "profile_ces.png")


def plot_policy_comparison(results: Dict[str, Any], out_dir: str) -> Optional[str]:
    if "baselines" not in results:
        print("  (no baseline results — run evaluate.py --baselines)")
        return None

    # Weakest-to-strongest baseline ordering, with the learned policy first.
    order = ["static", "random", "greedy"]
    available = results["baselines"]
    policies = ["ppo"] + [p for p in order if p in available] \
                       + [p for p in available if p not in order]

    def get(policy, key):
        node = results["ppo"] if policy == "ppo" else results["baselines"][policy]
        return node[key]

    metrics = [
        ("overall_ces",    "Coordination Efficiency Score", 1.0),
        ("overall_reward", "Mean episode reward",           1.0),
        ("solve_rate",     "Episodes fully solved (%)",     100.0),
    ]

    intervals = results.get("intervals", {})
    significance = results.get("significance", {})

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, title, scale) in zip(axes, metrics):
        vals = [get(p, key) * scale for p in policies]
        colors = [PPO_COLOR if p == "ppo" else "#9a9a93" for p in policies]
        bars = ax.bar(policies, vals, color=colors)

        # Confidence intervals are only stored for CES, the headline metric.
        tops = list(vals)
        if key == "overall_ces" and intervals:
            lo = [intervals.get(p, {}).get("ci_lo", v) for p, v in zip(policies, vals)]
            hi = [intervals.get(p, {}).get("ci_hi", v) for p, v in zip(policies, vals)]
            ax.errorbar(range(len(policies)), vals,
                        yerr=[[v - l for v, l in zip(vals, lo)],
                              [h - v for v, h in zip(vals, hi)]],
                        fmt="none", ecolor="#3d454e", elinewidth=1.2, capsize=4)
            title += "\n(bars show the 95% interval)"
            tops = hi                      # keep labels clear of the error caps

        pad = (max(tops) - min(min(vals), 0)) * 0.045
        for policy, bar, v, top in zip(policies, bars, vals, tops):
            label = f"{v:.2f}" if scale == 1.0 else f"{v:.0f}"
            if key == "overall_ces" and policy in significance:
                p_value = significance[policy].get("p_value", 1.0)
                label += significance_stars(p_value) or " n.s."
            ax.text(bar.get_x() + bar.get_width() / 2, top + pad,
                    label, ha="center", fontsize=8)
        ax.set_ylim(top=max(tops) + pad * 3.2)
        _style(ax, title, "policy", "")

    subtitle = "DTS-ZSC — learned policy vs. scripted baselines"
    if significance:
        subtitle += "   (significance vs PPO: *** p<0.001  ** p<0.01  * p<0.05)"
    fig.suptitle(subtitle, fontsize=12, fontweight="bold")
    return _save(fig, out_dir, "policy_comparison.png")


def plot_specialist_vs_general(results: Dict[str, Any], seen: List[str],
                               held: List[str], out_dir: str) -> Optional[str]:
    """
    The comparison the abstract implies: a zero-shot agent against a
    'behavioral model trained using historical interaction data' — i.e. an
    agent that saw exactly one partner during training.
    """
    if "compare" not in results or not results["compare"]:
        print("  (no comparison checkpoints — run evaluate.py --compare NAME=PATH)")
        return None

    general = results["ppo"]["profiles"]
    names = [p for p in list(seen) + list(held) if p in general]
    x = np.arange(len(names))
    others = results["compare"]
    width = 0.8 / (1 + len(others))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.4 + width / 2, [general[p]["ces"] for p in names], width,
           label="zero-shot agent (all training partners)", color=PPO_COLOR)
    palette = [HELD_COLOR, "#6b6b66", SEEN_COLOR]
    for i, (name, node) in enumerate(others.items()):
        tbl = node["profiles"]
        ax.bar(x - 0.4 + width * (1.5 + i), [tbl[p]["ces"] for p in names], width,
               label=f"{name} (one partner only)", color=palette[i % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    for i, p in enumerate(names):
        if p in held:
            ax.axvspan(i - 0.5, i + 0.5, color="grey", alpha=0.08, zorder=0)
    ax.legend(fontsize=8, loc="lower left")
    _style(ax, "Zero-shot agent vs. an agent trained on a single partner\n"
               "shaded = held out from the zero-shot agent's training",
           "human partner profile", "CES")
    ax.title.set_fontsize(10.5)
    return _save(fig, out_dir, "specialist_vs_general.png")


def plot_adaptive_allocation(results: Dict[str, Any], seen: List[str],
                             held: List[str], out_dir: str) -> str:
    """
    Evidence for adaptive task allocation: the share of work handed to the
    human should track how capable that partner actually is.
    """
    table = results["ppo"]["profiles"]
    names = [p for p in list(seen) + list(held) if p in table]
    share = [table[p]["human_share"] * 100 for p in names]
    steps = [table[p]["steps"] for p in names]
    colors = [SEEN_COLOR if p in seen else HELD_COLOR for p in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    bars = ax.bar(names, share, color=colors)
    for bar, v in zip(bars, share):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.0f}%",
                ha="center", fontsize=8)
    _style(ax, "Work delegated to each partner type",
           "human partner profile", "percent of assignments")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    ax = axes[1]
    ax.scatter(share, steps, c=colors, s=70, zorder=3)
    for n, x, y in zip(names, share, steps):
        ax.annotate(n, (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    _style(ax, "Delegation vs. episode length",
           "percent delegated to the human", "mean steps to finish")

    handles = [plt.Rectangle((0, 0), 1, 1, color=SEEN_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=HELD_COLOR)]
    fig.legend(handles, ["seen during training", "held-out (zero-shot)"],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("DTS-ZSC — adaptive task allocation", fontsize=13, fontweight="bold")
    fig.subplots_adjust(bottom=0.18)
    return _save(fig, out_dir, "adaptive_allocation.png")


def _arm_order(ablation: Dict[str, Any]) -> tuple:
    """Arm names with the reference arm first, and the arm data."""
    baseline = ablation.get("baseline_arm", "full")
    arms = ablation["arms"]
    return baseline, arms, [baseline] + [n for n in arms if n != baseline]


def _error_bars(values, los, his) -> list:
    """matplotlib wants distances from the point, not absolute bounds."""
    return [[v - lo for v, lo in zip(values, los)],
            [hi - v for v, hi in zip(values, his)]]


def plot_ablation(ablation: Dict[str, Any], out_dir: str) -> str:
    """Score per arm with intervals, and the paired difference against `full`."""
    baseline, arms, names = _arm_order(ablation)
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    means = [arms[n]["overall_ces"] for n in names]
    his = [arms[n]["ci"]["ci_hi"] for n in names]
    ax.barh(y, means, height=0.6,
            color=[PPO_COLOR if n == baseline else "#9a9a93" for n in names])
    ax.errorbar(means, y, fmt="none", ecolor="#3d454e", elinewidth=1.2, capsize=4,
                xerr=_error_bars(means, [arms[n]["ci"]["ci_lo"] for n in names], his))
    for row, mean in enumerate(means):
        ax.text(mean + 0.012, row, f"{mean:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(his) * 1.18)
    _style(ax, "Coordination efficiency by arm\n(bars show the 95% interval)",
           "CES", "")

    ax = axes[1]
    comparisons = ablation.get("comparisons", {})
    compared = [n for n in names if n in comparisons]
    if compared:
        diffs = [comparisons[n]["diff"] for n in compared]
        lo = [comparisons[n]["ci_lo"] for n in compared]
        hi = [comparisons[n]["ci_hi"] for n in compared]
        # Grey where the interval crosses zero — nothing can be claimed there.
        colours = ["#A32D2D" if comparisons[n]["significant"] and d < 0 else
                   SEEN_COLOR if comparisons[n]["significant"] else "#c9c8c2"
                   for n, d in zip(compared, diffs)]
        rows = np.arange(len(compared))
        ax.barh(rows, diffs, color=colours, height=0.6)
        ax.errorbar(diffs, rows, xerr=_error_bars(diffs, lo, hi), fmt="none",
                    ecolor="#3d454e", elinewidth=1.2, capsize=4)
        ax.axvline(0, color="#3d454e", linewidth=1.2)
        span = max(max(abs(min(lo)), abs(max(hi))) * 1.9, 0.02)
        pad = span * 0.04
        for row, (diff, name) in enumerate(zip(diffs, compared)):
            star = significance_stars(comparisons[name]["p_value"]) or "n.s."
            # Anchor past the interval so the text clears the error cap.
            edge = hi[row] + pad if diff >= 0 else lo[row] - pad
            ax.text(edge, row, f"{diff:+.3f} {star}", va="center", fontsize=8.5,
                    ha="left" if diff >= 0 else "right")
        ax.set_yticks(rows)
        ax.set_yticklabels(compared, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-span, span)
        _style(ax, f"Change vs '{baseline}'\n"
                   "negative = the component was earning its place",
               "difference in CES", "")

    fig.suptitle("DTS-ZSC — component ablation", fontsize=13, fontweight="bold")
    return _save(fig, out_dir, "ablation.png")


def plot_ablation_by_partner(ablation: Dict[str, Any], out_dir: str) -> str:
    """An overall mean can hide a component that only matters on hard partners."""
    baseline, arms, names = _arm_order(ablation)
    partners = list(arms[baseline]["profiles"])
    palette = [PPO_COLOR, "#A32D2D", HELD_COLOR, SEEN_COLOR, "#6b6b66"]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    width = 0.8 / len(names)
    x = np.arange(len(partners))
    for i, name in enumerate(names):
        scores = [arms[name]["profiles"][p]["ces"] for p in partners]
        ax.bar(x + i * width - 0.4 + width / 2, scores, width * 0.92,
               label=name, color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(partners, rotation=20, ha="right", fontsize=8.5)
    ax.legend(fontsize=8, ncol=len(names))
    _style(ax, "Per-partner coordination efficiency by ablation arm",
           "human partner profile", "CES")
    return _save(fig, out_dir, "ablation_by_partner.png")


# ── entry point ───────────────────────────────────────────────────────── #
def generate_all(log_path: str = DEFAULT_LOG,
                 eval_path: str = DEFAULT_EVAL,
                 out_dir: str = DEFAULT_OUT,
                 seen: Optional[List[str]] = None,
                 held: Optional[List[str]] = None) -> List[str]:
    from human.simulated_human import TRAIN_PROFILES, TEST_PROFILES
    seen = list(seen or TRAIN_PROFILES)
    held = list(held or TEST_PROFILES)

    written: List[str] = []

    if os.path.exists(log_path):
        print(f"Reading training log {log_path}")
        df = pd.read_csv(log_path)
        print(f"  {len(df)} log rows, {df['step'].iloc[-1]:,} training steps")
        written += [p for p in (plot_training_progress(df, out_dir),
                                plot_coordination(df, out_dir),
                                plot_optimisation(df, out_dir),
                                plot_zero_shot_gap(df, out_dir)) if p]
    else:
        print(f"No training log at {log_path} — run train.py first.")

    if os.path.exists(eval_path):
        print(f"Reading evaluation results {eval_path}")
        with open(eval_path) as f:
            results = json.load(f)
        written += [p for p in (plot_profile_ces(results, seen, held, out_dir),
                                plot_policy_comparison(results, out_dir),
                                plot_adaptive_allocation(results, seen, held, out_dir),
                                plot_specialist_vs_general(results, seen, held, out_dir))
                    if p]
    else:
        print(f"No evaluation results at {eval_path} — "
              "run evaluate.py --baselines first.")

    ablation_path = os.path.join(os.path.dirname(eval_path) or ".", "ablation.json")
    if os.path.exists(ablation_path):
        print(f"Reading ablation results {ablation_path}")
        with open(ablation_path) as f:
            ablation = json.load(f)
        written += [p for p in (plot_ablation(ablation, out_dir),
                                plot_ablation_by_partner(ablation, out_dir)) if p]
    else:
        print(f"No ablation results at {ablation_path} — "
              "train the arms and run ablation.py to add the component study.")

    return written

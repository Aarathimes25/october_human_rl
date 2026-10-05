"""
Render the documentation diagrams into docs/images.

    python scripts/make_diagrams.py                 # all three
    python scripts/make_diagrams.py --only usecase

Everything is laid out by hand on a 0-100 grid. Matplotlib is already a
dependency for the analytics charts, so this needs nothing extra.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diagram_style import (ENV, STATE, HUMAN, AGENT, MODULE, AMBER, OUTCOME,
                           ARROW, DASHED, PANEL_FS, TITLE_FS, SUB_FS,
                           actor, arrow, box, canvas, diamond, ellipse, line,
                           panel, route, save, stereotype, use_best_font)


# ── architecture ──────────────────────────────────────────────────────── #
def build_architecture():
    fig, ax = canvas(15, 13.2)

    panel(ax, (2.5, 2.5, 97.5, 97.5), ENV, radius=1.8, lw=2.4)
    ax.text(50, 93.8, "Collaborative task environment", ha="center",
            fontsize=PANEL_FS, fontweight="bold", color="#eafaf3", zorder=5)
    ax.text(50, 90.6, "Custom Gymnasium   ·   warehouse order fulfilment",
            ha="center", fontsize=SUB_FS + 1, color="#6fd6ac", zorder=5)

    box(ax, (6.5, 77, 33, 88), "Simulated human",
        "speed · fatigue · errors\nlatency · jitter · strategy", HUMAN)
    box(ax, (36.75, 77, 63.25, 88), "Order queue",
        "2 orders × 6 tasks · prerequisites\nstaggered arrival · priority · deadline")
    box(ax, (67, 77, 93.5, 88), "AI worker",
        "deterministic robot\nfixed time per task")

    # One shared bus: all three describe the same state.
    for x in (19.75, 50, 80.25):
        line(ax, (x, 77), (x, 74.8), lw=2.0, z=8)
    line(ax, (19.75, 74.8), (80.25, 74.8), lw=2.0, z=8)

    box(ax, (6.5, 62.5, 48, 72.8), "Observation   92-d",
        "12 tasks × 6  ·  2 workers × 6  ·  global 3  ·  observer 5")
    box(ax, (52, 62.5, 93.5, 72.8), "Action mask   25 actions",
        "the environment decides legality — the policy only reads it")

    arrow(ax, (27.25, 74.8), (27.25, 73.2))
    arrow(ax, (72.75, 74.8), (72.75, 73.2))
    arrow(ax, (27.25, 62.5), (27.25, 59.9))
    arrow(ax, (72.75, 62.5), (72.75, 59.9))

    panel(ax, (6, 22.5, 94, 59.5), AGENT, radius=1.6, lw=2.4, z=2)
    ax.text(50, 56.4, "RL agent   ·   PPO", ha="center", fontsize=TITLE_FS + 2,
            fontweight="bold", color="#f0eeff", zorder=6)
    ax.text(50, 53.4, "dynamic task sequencing policy", ha="center",
            fontsize=SUB_FS + 1, color="#c3bbff", zorder=6)

    box(ax, (10.5, 38.5, 34, 49.5), "Human observer",
        "smoothed speed · error rate\ncompletion ratio · trend", AMBER)
    box(ax, (38.25, 38.5, 61.75, 49.5), "Policy network",
        "shared trunk\nactor  ·  critic", MODULE)
    box(ax, (66, 38.5, 89.5, 49.5), "Masked softmax",
        "illegal moves removed\nbefore sampling", MODULE)
    box(ax, (28, 26, 72, 36), "Action",
        "assign (worker, task)   ·   reassign   ·   WAIT", AMBER)

    arrow(ax, (34, 44), (37.7, 44))
    arrow(ax, (61.75, 44), (65.45, 44))
    route(ax, [(77.75, 38.5), (77.75, 31), (72.55, 31)])

    box(ax, (6.5, 7, 33, 18), "Joint reward",
        "tasks + on-time orders\n− idle · late · reassign", OUTCOME)
    box(ax, (36.75, 7, 63.25, 18), "Coordination score",
        "completion · speed · sync\non-time − conflicts", OUTCOME)
    box(ax, (67, 7, 93.5, 18), "Zero-shot evaluation",
        "5 held-out partner types\nnever seen in training", OUTCOME)

    route(ax, [(50, 26), (50, 21), (19.75, 21), (19.75, 18.6)], style=DASHED)
    route(ax, [(6.5, 12.5), (4.6, 12.5), (4.6, 44), (9.9, 44)], style=DASHED)
    ax.text(3.5, 28, "feedback loop", rotation=90, ha="center", va="center",
            fontsize=SUB_FS - 0.5, color="#9ad3ba", zorder=8)

    ax.text(50, 4.7,
            "Partner sampled fresh each episode from the training pool only   ·   "
            "held-out types appear at evaluation",
            ha="center", fontsize=SUB_FS, color="#6fd6ac", zorder=5)
    return fig


# ── use case ──────────────────────────────────────────────────────────── #
OBSERVE, SEQUENCE, LEGAL, ASSIGN = 87.0, 76.5, 66.0, 55.5
REASSIGN, EXECUTE, SCORE, ZERO_SHOT = 45.0, 34.5, 24.0, 13.5

USE_CASES = [
    (OBSERVE,   "Observe partner behaviour",  "speed · errors · fatigue",  AMBER),
    (SEQUENCE,  "Sequence next task",         "re-decided every step",     MODULE),
    (LEGAL,     "Enforce legal actions",      "mask built by the environment", STATE),
    (ASSIGN,    "Assign task to worker",      "human or robot",            MODULE),
    (REASSIGN,  "Reassign stalled task",      "forfeits progress",         MODULE),
    (EXECUTE,   "Execute assigned task",      "at the partner's own pace", HUMAN),
    (SCORE,     "Score coordination",         "CES · on-time · sync",      OUTCOME),
    (ZERO_SHOT, "Evaluate on unseen partner", "5 held-out types",          OUTCOME),
]

ELLIPSE_W, ELLIPSE_H = 40, 8
LEFT_EDGE, RIGHT_EDGE = 31, 71


def build_usecase():
    fig, ax = canvas(15, 14.5)

    panel(ax, (26, 5, 76, 95), ENV, radius=1.6, lw=2.2)
    ax.text(51, 92, "DTS-ZSC System", ha="center", fontsize=TITLE_FS + 1,
            fontweight="bold", color="#eafaf3", zorder=5)

    for cy, title, subtitle, colours in USE_CASES:
        ellipse(ax, 51, cy, ELLIPSE_W, ELLIPSE_H, title, subtitle, colours)

    actor(ax, 10, 58, "Human partner", "#f0b9a5")
    actor(ax, 10, 13, "Evaluator", "#a9dc8a")
    actor(ax, 90, 58, "RL coordinator", "#c3bbff")

    # Ordered so each actor's associations fan out without crossing.
    for cy in (OBSERVE, ASSIGN, EXECUTE):
        line(ax, (10, 61), (LEFT_EDGE, cy), colour="#d8a48f", lw=1.6)
    for cy in (SCORE, ZERO_SHOT):
        line(ax, (10, 16), (LEFT_EDGE, cy), colour="#8fc972", lw=1.6)
    for cy in (OBSERVE, SEQUENCE, ASSIGN, REASSIGN):
        line(ax, (90, 61), (RIGHT_EDGE, cy), colour="#a79cff", lw=1.6)

    half = ELLIPSE_H / 2
    stereotype(ax, (51, SEQUENCE + half), (51, OBSERVE - half))
    stereotype(ax, (51, SEQUENCE - half), (51, LEGAL + half))
    stereotype(ax, (51, REASSIGN + half), (51, ASSIGN - half), "<<extend>>")

    ax.text(51, 2,
            "Sequencing always consults the partner signals and the legality "
            "mask; reassignment only\nhappens when an assignment is already "
            "running on another worker.",
            ha="center", fontsize=SUB_FS, color="#4a5560", linespacing=1.6)
    return fig


# ── activity ──────────────────────────────────────────────────────────── #
def build_activity():
    """Mirrors run_episode() in utils/runner.py and step() in the environment,
    including their order: the observer updates *after* the step, and its
    output only reaches the policy through the next observation."""
    from matplotlib.patches import Circle

    fig, ax = canvas(11.5, 17)
    ax.text(50, 98.5, "One coordination episode", ha="center",
            fontsize=TITLE_FS + 1, fontweight="bold", color="#1d2b33")

    ax.add_patch(Circle((50, 96), 1.5, facecolor="#2b3440",
                        edgecolor="#69748a", lw=1.8, zorder=6))

    steps = [
        ((89, 93.5), "Reset environment",
         "sample a partner from the training pool · build the orders", STATE),
        ((82, 86.5), "Build observation and legal action mask", "", STATE),
        ((75, 79.5), "Policy samples a legal action", "", MODULE),
        ((68, 72.5), "Observer records the assignment", "", AMBER),
    ]
    for (y0, y1), title, subtitle, colours in steps:
        box(ax, (26, y0, 74, y1), title, subtitle, colours,
            head_fs=12, sub_fs=8.5)

    diamond(ax, 50, 61.5, 32, 8, "which action?")

    box(ax, (10, 48.5, 32, 53), "Wait", "let work continue",
        STATE, head_fs=11.5, sub_fs=8.5)
    box(ax, (39, 48.5, 61, 53), "Assign", "start it on a free worker",
        MODULE, head_fs=11.5, sub_fs=8.5)
    box(ax, (68, 48.5, 90, 53), "Reassign", "release holder, lose progress",
        AMBER, head_fs=11.5, sub_fs=8.5)

    box(ax, (26, 40.5, 74, 45), "Advance every worker by one step",
        "each finished task pays out", STATE, head_fs=12, sub_fs=8.5)
    box(ax, (26, 33.5, 74, 38), "Score the step",
        "orders on time or late · sync bonus · idle penalty",
        OUTCOME, head_fs=12, sub_fs=8.5)
    box(ax, (26, 26.5, 74, 31), "Observer updates · rebuild observation",
        "these reach the policy on the next pass", AMBER, head_fs=12, sub_fs=8.5)

    diamond(ax, 50, 19.5, 36, 8, "all orders done,\nor step limit?")

    box(ax, (26, 8.5, 74, 13), "Record episode metrics",
        "CES · completion · sync · conflicts", OUTCOME, head_fs=12, sub_fs=8.5)
    ax.add_patch(Circle((50, 5), 1.9, facecolor="none",
                        edgecolor="#2b3440", lw=1.8, zorder=6))
    ax.add_patch(Circle((50, 5), 1.1, facecolor="#2b3440",
                        edgecolor="none", zorder=6))

    for start, end in [(94.5, 93.5), (89, 86.5), (82, 79.5), (75, 72.5),
                       (68, 65.5), (57.5, 55), (46, 45),
                       (40.5, 38), (33.5, 31), (26.5, 23.5), (15.5, 13),
                       (8.5, 6.9)]:
        arrow(ax, (50, start), (50, end))

    branches = (21, 50, 79)
    line(ax, (branches[0], 55), (branches[-1], 55), lw=2.0, z=8)
    for x in branches:
        arrow(ax, (x, 55), (x, 53.1))
        line(ax, (x, 48.5), (x, 46), lw=2.0, z=8)
    line(ax, (branches[0], 46), (branches[-1], 46), lw=2.0, z=8)

    for x, ha, label in ((19, "right", "wait"), (53, "left", "assign"),
                         (81, "left", "reassign")):
        ax.text(x, 53.9, label, ha=ha, va="center", fontsize=SUB_FS - 1,
                color="#5b6675", style="italic")

    # Loop lane sits left of every box, so it never crosses one.
    route(ax, [(32, 19.5), (4.5, 19.5), (4.5, 84.25), (25.6, 84.25)])
    ax.text(2.9, 52, "no — keep going", rotation=90, ha="center", va="center",
            fontsize=SUB_FS - 1, color="#5b6675", style="italic")
    ax.text(52.5, 14.2, "yes", ha="left", va="center", fontsize=SUB_FS - 1,
            color="#5b6675", style="italic")

    ax.text(50, -1.6,
            "Terminated when every order is complete, truncated at the step "
            "limit — the difference is what\n'solved' reports. Illegal actions "
            "are masked out, so that branch never fires for a masked policy.",
            ha="center", va="top", fontsize=SUB_FS - 1, color="#4a5560",
            linespacing=1.6)
    return fig


BUILDERS = {
    "architecture": (build_architecture, "architecture.png"),
    "usecase":      (build_usecase,      "usecase.png"),
    "activity":     (build_activity,     "activity.png"),
}


def main():
    parser = argparse.ArgumentParser(description="Render the docs diagrams")
    parser.add_argument("--only", choices=sorted(BUILDERS),
                        help="render a single diagram instead of all three")
    args = parser.parse_args()

    print(f"font: {use_best_font()}")
    images = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "images")
    os.makedirs(images, exist_ok=True)

    wanted = [args.only] if args.only else list(BUILDERS)
    for name in wanted:
        build, filename = BUILDERS[name]
        save(build(), os.path.join(images, filename))


if __name__ == "__main__":
    main()

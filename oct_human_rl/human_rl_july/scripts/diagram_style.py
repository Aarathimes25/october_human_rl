"""
Shared palette and drawing primitives for the diagrams in docs/images.

Cool for machinery, warm for the human, indigo for the policy, green for what
comes back out — the same mapping across every figure so the three read as one
set.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Ellipse,
                                Circle, Polygon)

PAGE    = "#eef1f4"
INK     = "#ffffff"
SUB_INK = "#dfe5ea"
ARROW   = "#b6c0ca"

ENV     = ("#0d4a3b", "#25876b")
STATE   = ("#39414d", "#69748a")
HUMAN   = ("#93331d", "#cd5533")
AGENT   = ("#362da6", "#8074f2")
MODULE  = ("#4a3fc4", "#a79cff")
AMBER   = ("#9e6100", "#d99418")
OUTCOME = ("#44761a", "#7fbb3a")

PANEL_FS, TITLE_FS, HEAD_FS, SUB_FS = 20, 15, 14, 10.5

SHADOW = ("#0a0f14", "#0a0f14")
DASHED = (0, (6, 4))


def use_best_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Corbel", "Calibri", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return str(plt.rcParams["font.family"])


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(PAGE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def panel(ax, rect, colours, radius=1.2, lw=1.8, z=1, alpha=1.0):
    x0, y0, x1, y1 = rect
    fill, edge = colours
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, alpha=alpha, zorder=z))


def _label(ax, cx, cy, title, subtitle, z, head_fs=HEAD_FS, sub_fs=SUB_FS,
           height=11.0):
    if subtitle:
        # Scale the split to the shape, or short boxes push their two lines
        # out against the edges.
        gap = min(1.7, height * 0.22)
        ax.text(cx, cy + gap, title, ha="center", va="center", zorder=z + 1,
                fontsize=head_fs, fontweight="bold", color=INK)
        ax.text(cx, cy - gap * 1.24, subtitle, ha="center", va="center",
                zorder=z + 1, fontsize=sub_fs, color=SUB_INK, linespacing=1.5)
    else:
        ax.text(cx, cy, title, ha="center", va="center", zorder=z + 1,
                fontsize=head_fs, fontweight="bold", color=INK)


def box(ax, rect, title, subtitle="", colours=STATE, z=4,
        head_fs=HEAD_FS, sub_fs=SUB_FS):
    x0, y0, x1, y1 = rect
    panel(ax, (x0 + 0.4, y0 - 0.55, x1 + 0.4, y1 - 0.55), SHADOW,
          radius=1.0, lw=0, z=z - 1, alpha=0.22)
    panel(ax, rect, colours, radius=1.0, lw=1.8, z=z)
    _label(ax, (x0 + x1) / 2, (y0 + y1) / 2, title, subtitle, z, head_fs,
           sub_fs, height=y1 - y0)


def ellipse(ax, cx, cy, width, height, title, subtitle="", colours=STATE, z=4,
            head_fs=HEAD_FS - 1.5, sub_fs=SUB_FS - 1.5):
    fill, edge = colours
    ax.add_patch(Ellipse((cx + 0.4, cy - 0.55), width, height,
                         facecolor=SHADOW[0], edgecolor="none",
                         alpha=0.22, zorder=z - 1))
    ax.add_patch(Ellipse((cx, cy), width, height, facecolor=fill,
                         edgecolor=edge, linewidth=1.8, zorder=z))
    _label(ax, cx, cy, title, subtitle, z, head_fs, sub_fs, height=height)


def diamond(ax, cx, cy, width, height, text, colours=AMBER, z=4,
            fs=SUB_FS + 0.5):
    fill, edge = colours
    pts = [(cx, cy + height / 2), (cx + width / 2, cy),
           (cx, cy - height / 2), (cx - width / 2, cy)]
    ax.add_patch(Polygon([(x + 0.4, y - 0.55) for x, y in pts],
                         facecolor=SHADOW[0], edgecolor="none",
                         alpha=0.22, zorder=z - 1))
    ax.add_patch(Polygon(pts, facecolor=fill, edgecolor=edge,
                         linewidth=1.8, zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", zorder=z + 1,
            fontsize=fs, fontweight="bold", color=INK, linespacing=1.4)


def actor(ax, cx, cy, name, colour="#f0f4f7", z=6, scale=1.0):
    """UML stick figure, drawn from cy upwards."""
    head_r = 1.5 * scale
    lw = 2.0
    ax.add_patch(Circle((cx, cy + 6.2 * scale), head_r, facecolor="none",
                        edgecolor=colour, linewidth=lw, zorder=z))
    ax.plot([cx, cx], [cy + 4.7 * scale, cy + 1.6 * scale],
            color=colour, lw=lw, zorder=z)
    ax.plot([cx - 2.4 * scale, cx + 2.4 * scale],
            [cy + 3.7 * scale, cy + 3.7 * scale], color=colour, lw=lw, zorder=z)
    ax.plot([cx - 2.0 * scale, cx], [cy - 1.4 * scale, cy + 1.6 * scale],
            color=colour, lw=lw, zorder=z)
    ax.plot([cx + 2.0 * scale, cx], [cy - 1.4 * scale, cy + 1.6 * scale],
            color=colour, lw=lw, zorder=z)
    ax.text(cx, cy - 3.4 * scale, name, ha="center", va="center", zorder=z,
            fontsize=SUB_FS + 1, fontweight="bold", color=colour)


def line(ax, start, end, colour=ARROW, lw=1.6, style="-", z=5):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=colour, lw=lw,
            linestyle=style, zorder=z, solid_capstyle="round")


def arrow(ax, start, end, style="-", lw=2.0, z=8, colour=ARROW, scale=17):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
        color=colour, linestyle=style, zorder=z, shrinkA=0, shrinkB=0))


def route(ax, points, style="-", lw=2.0, z=8, colour=ARROW):
    """Elbow connector: plain segments, arrowhead only on the last."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=colour, lw=lw, linestyle=style, zorder=z,
            solid_capstyle="round")
    arrow(ax, points[-2], points[-1], style=style, lw=lw, z=z, colour=colour)


def stereotype(ax, start, end, text="<<include>>", z=7, label_dx=2.4):
    """Dashed UML relationship. The label sits beside the arrow, not on it —
    the gap between two stacked ellipses is too shallow to stack text in."""
    arrow(ax, start, end, style=DASHED, lw=1.5, z=z, scale=13)
    ax.text((start[0] + end[0]) / 2 + label_dx, (start[1] + end[1]) / 2, text,
            ha="left", va="center", fontsize=SUB_FS - 1.5,
            color="#cfd6dd", style="italic", zorder=z + 1)


def save(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, facecolor=PAGE, bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path}")

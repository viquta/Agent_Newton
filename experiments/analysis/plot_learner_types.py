"""Trajectory and outcome panels for each learner-type category (Framing A).

Reads grid_learner_types/summary.json and produces:

  Per-category figures (default)
    One file per category (basic, linear, forgetful, slipper), each showing:
      Top panel    — remediation-ratio and accuracy trajectories over practice
                     items for both arms.
      Four panels  — one per outcome as mean paired difference ± 95 % CI.

  Combined figure (--combined)
    Two panels side by side, all four categories on the same axes:
      Left  — remediation ratio over items (shows sawtooth and flat forgetful)
      Right — accuracy over items (shows noise-floor plateau for slipper)
    Each category has its own colour; solid = coupled, dashed = decoupled.

Usage:
    uv run python experiments/analysis/plot_learner_types.py \\
        --summary results/grid_learner_types/summary.json \\
        [--out results/grid_learner_types/plots] \\
        [--format png|pdf] \\
        [--combined]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

# ---------------------------------------------------------------------------
# Style — match figures.py exactly
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

LOWER_IS_BETTER = frozenset({"distance_to_goal"})

OUTCOME_LABELS = {
    "remediation": "remediation ratio",
    "gain": "gain",
    "goals_mastered": "goals mastered",
    "distance_to_goal": "distance to goal",
}

ERROR_SHAPE = {
    "basic": "exponential",
    "linear": "linear",
    "forgetful": "sawtooth",
    "slipper": "noise floor",
}

#: One colour per category. The four are validated as a set on the light
#: surface alongside the existing three-slot palette.
CATEGORY_COLOR = {
    "basic":     BLUE,
    "linear":    AQUA,
    "forgetful": ORANGE,
    "slipper":   "#9b59b6",   # muted purple — 4th CVD-safe slot
}


def house_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SOFT,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "text.color": INK,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def _bare(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Trajectory panel
# ---------------------------------------------------------------------------

def _trajectory_panel(ax: Axes, cat_data: dict) -> None:
    """Remediation ratio and accuracy over practice items for both arms."""
    for arm, color in [("coupled", BLUE), ("decoupled", ORANGE)]:
        pts = cat_data["paths"][arm]
        if not pts:
            continue
        items = [p["item"] for p in pts]
        rem = [p["remediation"] for p in pts]
        acc = [p["accuracy"] for p in pts]

        ax.plot(items, rem, color=color, linewidth=1.5)
        ax.plot(items, acc, color=color, linewidth=1.0, linestyle="--", alpha=0.6)

        # Direct labels at the last point
        ax.text(items[-1] + 0.4, rem[-1], arm, color=color,
                fontsize=8, va="center", ha="left")

    ax.set_xlabel("practice item")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("learning trajectory  (blue = coupled, orange = decoupled)")

    # Style guide only — arms are direct-labelled on the lines
    ax.plot([], [], color=INK_MUTED, linewidth=1.5, label="remediation ratio")
    ax.plot([], [], color=INK_MUTED, linewidth=1.0, linestyle="--",
            alpha=0.6, label="accuracy (observed)")
    ax.legend(loc="lower right", fontsize=8)
    _bare(ax)


# ---------------------------------------------------------------------------
# Outcome panel
# ---------------------------------------------------------------------------

def _outcome_panel(ax: Axes, outcome_data: dict) -> None:
    """Mean paired difference ± 95 % CI for one outcome (Framing A)."""
    name = outcome_data["outcome"]
    diff = outcome_data["mean_difference"]
    lo, hi = outcome_data["ci95"]
    holm_p = outcome_data["holm_p"]
    significant = outcome_data["significant"]

    lower_better = name in LOWER_IS_BETTER
    # For lower-is-better outcomes, a negative diff means coupled is better.
    # Flip so positive always means "coupled is better" for consistent colouring.
    display_diff = -diff if lower_better else diff
    display_lo = -hi if lower_better else lo
    display_hi = -lo if lower_better else hi

    color = BLUE if display_diff >= 0 else ORANGE
    alpha = 1.0 if significant else 0.45

    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.errorbar(
        display_diff, 0,
        xerr=[[display_diff - display_lo], [display_hi - display_diff]],
        fmt="o", color=color, markersize=6, capsize=4, linewidth=1.5,
        alpha=alpha, zorder=2,
    )

    # p-value annotation
    p_str = f"p={holm_p:.2e}" if holm_p >= 0.0001 else f"p<0.0001"
    sig_mark = " *" if significant else ""
    ax.text(0.98, 0.88, p_str + sig_mark, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=INK_SOFT)

    label = OUTCOME_LABELS[name]
    if lower_better:
        label += "\n(lower is better;\nflipped: + = coupled wins)"
    ax.set_title(label)
    ax.set_xlabel("coupled − decoupled")
    ax.set_yticks([])
    _bare(ax)


# ---------------------------------------------------------------------------
# Per-category figure
# ---------------------------------------------------------------------------

def _category_figure(category: str, cat_data: dict, out: Path, suffix: str) -> None:
    shape = ERROR_SHAPE.get(category, category)
    dials = cat_data.get("dials", {})
    dials_str = ", ".join(f"{k}={v}" for k, v in dials.items()) if dials else "defaults"

    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        height_ratios=[2.2, 1],
        hspace=0.45, wspace=0.35,
    )

    # Trajectory panel spans all 4 columns
    ax_traj = fig.add_subplot(gs[0, :])
    _trajectory_panel(ax_traj, cat_data)

    # Four outcome panels
    outcomes_by_name = {o["outcome"]: o for o in cat_data["outcomes"]}
    outcome_order = ["remediation", "gain", "goals_mastered", "distance_to_goal"]
    for col, name in enumerate(outcome_order):
        if name not in outcomes_by_name:
            continue
        ax = fig.add_subplot(gs[1, col])
        _outcome_panel(ax, outcomes_by_name[name])

    fig.suptitle(
        f"learner type: {category}  ({shape})\n"
        f"dials: {dials_str}",
        fontsize=11, fontweight="bold", color=INK,
    )

    path = out / f"{category}.{suffix}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Combined figure (all categories on shared axes)
# ---------------------------------------------------------------------------

def _combined_figure(categories: dict[str, dict], out: Path, suffix: str) -> None:
    """Remediation ratio trajectories, all categories on one figure.

    Colour encodes population; solid = coupled arm, dashed = decoupled arm.

    Accuracy (observed) ranged 0.98–1.00 across all populations and arms and
    is omitted — the variation is within sampling noise.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Vertical nudges for end-of-line labels that would otherwise overlap.
    LABEL_NUDGE: dict[str, float] = {
        "basic":     0.018,
        "slipper":  -0.025,
    }

    for category, cat_data in categories.items():
        color = CATEGORY_COLOR.get(category, INK_MUTED)
        label: str = ERROR_SHAPE.get(category, category)
        for arm, linestyle, lw, alpha in [
            ("coupled",   "-",  1.8, 1.0),
            ("decoupled", "--", 1.1, 0.60),
        ]:
            pts = cat_data["paths"][arm]
            if not pts:
                continue
            items = [p["item"] for p in pts]
            vals  = [p["remediation"] for p in pts]
            ax.plot(items, vals, color=color, linestyle=linestyle,
                    linewidth=lw, alpha=alpha)

        # Direct label at the end of the coupled (solid) line.
        coupled_pts = cat_data["paths"]["coupled"]
        if coupled_pts:
            last  = coupled_pts[-1]
            nudge = LABEL_NUDGE.get(category, 0.0)
            ax.text(last["item"] + 0.5, last["remediation"] + nudge,
                    label, color=color, fontsize=8, va="center", ha="left")

    ax.set_xlabel("practice item")
    ax.set_ylabel("fraction remediated")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("remediation ratio over practice items")
    _bare(ax)

    # Build one combined legend: category colours + arm line styles.
    legend_handles = []
    import matplotlib.lines as mlines
    for cat, col in CATEGORY_COLOR.items():
        legend_handles.append(
            mlines.Line2D([], [], color=col, linewidth=1.8,
                          label=ERROR_SHAPE.get(cat, cat))
        )
    legend_handles.append(
        mlines.Line2D([], [], color=INK_MUTED, linewidth=1.8,
                      linestyle="-", label="coupled")
    )
    legend_handles.append(
        mlines.Line2D([], [], color=INK_MUTED, linewidth=1.1,
                      linestyle="--", alpha=0.6, label="decoupled")
    )
    ax.legend(handles=legend_handles, fontsize=8, frameon=False,
              loc="upper left", ncol=1)

    fig.suptitle(
        "Learning trajectories across simulated populations  (Framing A, N = 160)",
        fontsize=10, fontweight="bold", color=INK,
    )
    fig.text(
        0.5, -0.04,
        "Note: accuracy (observed) ranged 0.98–1.00 across all populations "
        "and arms and is omitted.",
        ha="center", fontsize=7.5, color=INK_MUTED,
    )

    path = out / f"combined.{suffix}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, required=True,
        help="Path to grid_learner_types/summary.json",
    )
    parser.add_argument(
        "--out", type=Path,
        help="Output directory (default: summary's parent / plots)",
    )
    parser.add_argument(
        "--format", choices=["png", "pdf"], default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Produce one combined figure instead of one per category.",
    )
    args = parser.parse_args()

    summary_path: Path = args.summary
    if not summary_path.exists():
        parser.error(f"summary not found: {summary_path}")

    out_dir: Path = args.out or summary_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open() as f:
        summary = json.load(f)

    house_style()

    categories: dict = summary["categories"]

    if args.combined:
        print(f"plotting combined figure → {out_dir}")
        _combined_figure(categories, out_dir, args.format)
    else:
        print(f"plotting {len(categories)} categories → {out_dir}")
        for category, cat_data in categories.items():
            _category_figure(category, cat_data, out_dir, args.format)

    print("done.")


if __name__ == "__main__":
    main()

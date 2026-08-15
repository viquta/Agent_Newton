"""Figures, drawn from the stored summaries.

    uv run python experiments/analysis/figures.py

Reads what the experiments already wrote and draws it. Nothing here re-runs a
cohort, so a figure cannot disagree with the number it is drawn from — and a
figure that cannot be regenerated from a committed summary is not evidence of
anything.

⚠️ **Output goes to `research_private/figures/` by default**, not to
`results/figures/`. The repository's `.gitignore` un-ignores the latter, on the
same reasoning that keeps aggregated metrics tracked. A figure is not prose, but
it *is* the argument in visual form, and it appears in the submitted document —
so publishing it early is the thing the publishability rule exists to prevent.
Pass ``--out results/figures`` to use the tracked location deliberately.

Three figures, and each one is the encoding the *shape of its data* asks for:

``prerequisite_sweep``
    Two quantities against the strength of the mechanism, in the same units and
    on one axis. The point is the contrast — one moves, one does not — so both
    are drawn, both direct-labelled.

``arbitration_substitution``
    A stacked bar whose total height barely changes while its composition does.
    That is the finding: raising the threshold re-attributes replans rather than
    preventing them, and a line of totals would show nothing at all.

``paired_discordance``
    A diverging stacked bar centred on the ties. The paired analysis is a sign
    test over discordant pairs, and most learners are *exactly* tied — a bar of
    means would show a difference while hiding that it rests on twelve people
    out of a hundred and sixty.

These are print figures, so there is no hover layer and one surface rather than
a selected dark mode. Colour is the validated categorical palette; every series
carries a direct label as well as a hue, so identity is never colour alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: Set by --format. PDF for the document; PNG exists so a figure can be looked
#: at before it is believed, which is the step no validator covers.
SUFFIX = "pdf"

#: Categorical slots 1–3 of the reference palette, validated as a set on the
#: light surface: all-pairs CVD ΔE 9.2, normal-vision 24.0. Aqua sits under 3:1
#: against the surface, which obliges visible labels — every series has one.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
#: Diverging poles and the neutral midpoint. Blue↔red reads as opposite; the
#: midpoint is grey so "no difference" looks like nothing rather than like a
#: third category.
POLE_LOW, NEUTRAL, POLE_HIGH = "#e34948", "#d8d7d2", "#2a78d6"

#: Outcomes where a *smaller* number is the better one.
#:
#: ⚠️ `statistics.compare` counts `favouring_first` as "the first arm's value is
#: larger", with no notion of which direction is good — so for these the raw
#: counts mean the opposite of what they say. `run_paired` carries the same
#: correction as a footnote under its table; a figure has no footnote, and the
#: first version of this one drew 160 learners favouring the decoupled arm on
#: the outcome the coupled arm wins by the widest margin.
LOWER_IS_BETTER = frozenset({"distance_to_goal"})

SURFACE = "#fcfcfb"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def house_style() -> None:
    """Recessive chrome, text in ink tokens, no chartjunk."""
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


def _bare(ax) -> None:  # noqa: ANN001
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def prerequisite_sweep(results: Path, out: Path) -> Path:
    """What the mechanism moved, and what it did not."""
    data = json.loads((results / "sweep_prerequisites" / "summary.json").read_text())
    ks = [point["k"] for point in data["curve"]]
    ordering = [point["ordering"]["mean_difference"] for point in data["curve"]]
    paired = [point["paired"]["remediation"]["mean_difference"] for point in data["curve"]]

    figure, ax = plt.subplots(figsize=(6.6, 3.4))
    _bare(ax)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.plot(ks, ordering, color=ORANGE, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ax.plot(ks, paired, color=BLUE, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)

    # Direct labels rather than a legend box: two series, and the reader should
    # not have to look away from the line to find out which is which. Placed in
    # a right-hand margin rather than over the plot — the first attempt put them
    # above their endpoints, where one collided with the title.
    ax.annotate(
        "prerequisite order against\narbitrary order",
        xy=(ks[-1], ordering[-1]), xytext=(8, 0), textcoords="offset points",
        ha="left", va="center", color=INK_SOFT, fontsize=8.5, linespacing=1.35,
    )
    ax.annotate(
        "coupled against\ndecoupled",
        xy=(ks[-1], paired[-1]), xytext=(8, 0), textcoords="offset points",
        ha="left", va="center", color=INK_SOFT, fontsize=8.5, linespacing=1.35,
    )
    ax.set_xlim(-0.04, 1.42)
    ax.set_xlabel("strength of prerequisite dependence (k)")
    ax.set_ylabel("difference in remediation")
    ax.set_title("Making sequencing matter does not separate the architectures",
                 loc="left")
    ax.set_xticks(ks)
    figure.savefig(out / f"prerequisite_sweep.{SUFFIX}")
    plt.close(figure)
    return out / f"prerequisite_sweep.{SUFFIX}"


def arbitration_substitution(results: Path, out: Path) -> Path:
    """Raising the threshold re-attributes replans rather than preventing them.

    Drawn as change from the lowest threshold, because the first attempt drew
    the counts themselves and the finding disappeared: the part that moves is
    two per cent of a bar whose other ninety-eight per cent is constant. What
    the data is actually saying is that one trigger loses exactly what the other
    gains, so the figure shows the two deltas mirrored about zero and lets the
    symmetry be the evidence.
    """
    data = json.loads(
        (results / "sweep_arbitration" / "summary_theta_k1.json").read_text()
    )
    points = data["points"]
    thetas = [point["theta"] for point in points]

    def counts(key: str) -> list[int]:
        return [int(point["replans_by_trigger"].get(key, 0)) for point in points]

    delta = counts("mastery_delta")
    repeat = counts("misconception_repeat")
    total = [
        sum(point["replans_by_trigger"].values()) for point in points
    ]
    base_delta, base_repeat = delta[0], repeat[0]

    figure, ax = plt.subplots(figsize=(6.6, 3.4))
    _bare(ax)
    ax.grid(axis="x", visible=False)
    positions = [float(i) for i in range(len(thetas))]
    width = 0.34
    ax.bar([p - width / 2 for p in positions], [d - base_delta for d in delta],
           width=width, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3,
           label="mastery moved")
    ax.bar([p + width / 2 for p in positions], [r - base_repeat for r in repeat],
           width=width, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
           label="misconception repeated")
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{t:g}" for t in thetas])
    ax.set_xlabel("replanning threshold (theta)")
    ax.set_ylabel("replans, against the lowest threshold")
    ax.set_title("What one trigger loses, the other gains", loc="left")
    ax.legend(loc="lower left", fontsize=8.5, labelcolor=INK_SOFT,
              handlelength=1.2, columnspacing=1.4, ncols=2)
    # The claim the mirror is evidence for, stated once rather than drawn as a
    # bar height nobody can compare across five bars.
    ax.annotate(
        f"total replans: {total[0]:,} at every threshold",
        xy=(1.0, 1.02), xycoords="axes fraction", ha="right", color=INK_SOFT,
        fontsize=8.5,
    )
    # The first column is empty because it is the point of comparison, which
    # reads as missing data unless it is said.
    ax.annotate(
        "baseline", xy=(positions[0], 0), xytext=(0, 8),
        textcoords="offset points", ha="center", color=INK_MUTED, fontsize=8,
    )
    figure.savefig(out / f"arbitration_substitution.{SUFFIX}")
    plt.close(figure)
    return out / f"arbitration_substitution.{SUFFIX}"


def paired_discordance(results: Path, out: Path) -> Path:
    """Where the paired comparison actually lives: a handful of learners."""
    data = json.loads((results / "paired_calculus" / "summary.json").read_text())
    rows = list(reversed(data["results"]))
    labels = [row["outcome"].replace("_", " ") for row in rows]

    figure, ax = plt.subplots(figsize=(5.6, 3.2))
    _bare(ax)
    ax.grid(axis="y", visible=False)
    positions = range(len(rows))
    for y, row in zip(positions, rows):
        ties = row["ties"]
        coupled, decoupled = row["favouring_coupled"], row["favouring_decoupled"]
        if row["outcome"] in LOWER_IS_BETTER:
            coupled, decoupled = decoupled, coupled
        # Centred on the ties, so "no difference" is the middle of the bar and
        # the two poles read as opposite directions rather than as magnitudes.
        ax.barh(y, -decoupled, left=-ties / 2, height=0.55, color=POLE_LOW,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.barh(y, ties, left=-ties / 2, height=0.55, color=NEUTRAL,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.barh(y, coupled, left=ties / 2, height=0.55, color=POLE_HIGH,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        if ties:
            # Only where there are any. A "0 tied" label lands inside a
            # full-width coloured bar, where it is both untrue-looking and
            # unreadable.
            ax.annotate(f"{ties} tied", xy=(0, y), ha="center", va="center",
                        color=INK_SOFT, fontsize=8)
        if decoupled:
            ax.annotate(f"{decoupled}", xy=(-ties / 2 - decoupled, y), xytext=(-5, 0),
                        textcoords="offset points", ha="right", va="center",
                        color=INK_SOFT, fontsize=8)
        if coupled:
            ax.annotate(f"{coupled}", xy=(ties / 2 + coupled, y), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center",
                        color=INK_SOFT, fontsize=8)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_xlabel("learners, by which architecture their pair favoured")
    ax.set_title("Every outcome rests on the few learners the two arms\n"
                 "treated differently")
    # Written rather than drawn with arrow glyphs: the house sans has no
    # arrows, and a missing glyph renders as a box in the submitted PDF.
    ax.annotate("favours decoupled", xy=(0.02, -0.16), xycoords="axes fraction",
                color=INK_SOFT, fontsize=8.5)
    ax.annotate("favours coupled", xy=(0.98, -0.16), xycoords="axes fraction",
                ha="right", color=INK_SOFT, fontsize=8.5)
    # Explicit room rather than margins: the bar lengths differ by two orders
    # of magnitude, so a count beside a two-learner bar lands next to the axis
    # label unless the space is reserved.
    reach = max(
        row["ties"] / 2 + max(row["favouring_coupled"], row["favouring_decoupled"])
        for row in rows
    )
    ax.set_xlim(-reach * 1.45, reach * 1.30)
    figure.savefig(out / f"paired_discordance.{SUFFIX}")
    plt.close(figure)
    return out / f"paired_discordance.{SUFFIX}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--format", default="pdf", choices=("pdf", "png"),
        help="PDF for the document; PNG to look at one quickly.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research_private" / "figures",
        help="Where the PDFs go. Private by default — see the module docstring.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    global SUFFIX
    SUFFIX = args.format

    house_style()
    for draw in (prerequisite_sweep, arbitration_substitution, paired_discordance):
        try:
            written = draw(args.results, args.out)
        except FileNotFoundError as missing:
            print(f"skipped {draw.__name__}: {missing.filename} has not been run")
            continue
        print(f"wrote {written}")


if __name__ == "__main__":
    main()

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

Five figures, and each one is the encoding the *shape of its data* asks for:

``prerequisite_sweep``
    Two quantities against the strength of the mechanism, in the same units and
    on one axis. The point is the contrast in *slope*: both rise, and one rises
    about six times more steeply. Same units and one axis is what makes that
    readable, so both are drawn and both direct-labelled.

``arbitration_substitution``
    A stacked bar whose total height barely changes while its composition does.
    That is the finding: raising the threshold re-attributes replans rather than
    preventing them, and a line of totals would show nothing at all.

``paired_discordance``
    A diverging stacked bar centred on the ties. The paired analysis is a sign
    test over discordant pairs, and most learners are *exactly* tied — a bar of
    means would show a difference while hiding that it rests on twelve people
    out of a hundred and sixty.

``learner_type_reversal``
    The same diverging bar, once per simulated population, beside the level each
    arm reached. Two panels rather than one because the bars give a direction,
    and a direction is only interpretable against how much either arm taught at
    that setting — and a figure travels without its caption, so the qualifier
    has to be a panel rather than a note.

``power_curve``
    Two decision rules over the same sizes, so the vertical gap between a line
    and its dashed partner is the quantity of interest. The outcomes that are
    powered at the smallest size simulated are stated rather than drawn: as
    curves they are flat lines at 1.0 that compress everything else.

These are print figures, so there is no hover layer and one surface rather than
a selected dark mode. Colour is the validated categorical palette; every series
carries a direct label as well as a hue, so identity is never colour alone.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
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

#: Where `learner_type_reversal` checks its categories against, relative to ROOT.
GRID_SOURCE = "experiments/grid_learner_types.py"

#: Category name -> the shape of the error proportion that category produces.
#: The summary keys name the mechanism a dial implements; these name what it
#: looks like plotted against items, which is the axis the four rows are
#: compared on and the one the mechanism names do not reveal. Checked against
#: the declared categories, so a new dial cannot draw without a shape.
ERROR_SHAPE: dict[str, str] = {
    "basic": "exponential",
    "linear": "linear",
    "forgetful": "sawtooth",
    "slipper": "noise floor",
}

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
    """How much the mechanism moved each quantity.

    ⚠️ Rewritten 2026-08-17. It previously read "what the mechanism moved, and
    what it did not", and was titled *"Making sequencing matter does not separate
    the architectures"* — because the paired difference was flat at +0.0136
    across the whole sweep while the ordering probe climbed. After a sixteenth
    catalogue entry redrew every learner's profile, the paired difference rises
    too: +0.0259 at k = 0 to +0.0350 at k = 1. **The figure was correct and the
    argument written on it was not**, which is the failure mode a figure is most
    prone to, since it travels without its caption.

    What the data now says, and it is still a contrast: the ordering probe rises
    by 0.0590 over the sweep and the paired difference by 0.0091 — about six
    times less. Making sequencing matter moves *sequencing sensitivity* sharply
    and the *architectural* difference only slightly.

    ⚠️ The paired series carries a caveat the figure states on its face rather
    than in a caption: the pilot gives N = 160 only 0.26 power for that effect,
    so every point on the blue line is a significant result obtained under low
    power. Drawn without that, the line reads as firmer than it is.
    """
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
    ax.set_title(
        "Prerequisite dependence moves sequencing sharply, the architectures little",
        loc="left",
    )
    ax.set_xticks(ks)
    # On the figure rather than in a caption. The direction convention learned
    # from `paired_discordance` applies to strength too: a figure has no
    # footnote, so a qualifier it needs has to be inside it.
    ax.annotate(
        "coupled–decoupled significant at every k, but at 0.26 power (N = 160)",
        xy=(0, 0), xytext=(0, -38), textcoords="offset points",
        xycoords="axes fraction", ha="left", va="top",
        color=INK_SOFT, fontsize=7.5,
    )
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


def _declared_categories() -> dict[str, dict]:
    """`CATEGORIES` as `grid_learner_types.py` declares it, parsed not imported.

    Parsed so drawing a stored summary does not import an experiment that would
    pull in the whole session stack to read one dict.
    """
    source = ROOT / GRID_SOURCE
    match = re.search(
        r"^CATEGORIES: dict\[str, dict\] = (\{.*?^\})", source.read_text(), re.M | re.S
    )
    if match is None:
        raise SystemExit(f"{GRID_SOURCE} no longer declares CATEGORIES as a literal")
    # Comments inside the literal are dropped by `literal_eval`'s parser.
    return ast.literal_eval(match.group(1))


def _dial_text(dials: dict) -> str:
    """One category's dials, short enough for a tick label.

    The field names carry the mechanism they belong to, which is redundant once
    the row is labelled with it and long enough to widen the panel by an inch.
    """
    if not dials:
        return "dials off"
    #: An unmapped key falls back to its own name rather than refusing to draw:
    #: the dial *values* are already checked against the source, so a gap here
    #: is presentational and costs a wider panel, not a wrong figure.
    short = {
        "forgetting_rate": "forgets",
        "forgetting_period": "every",
        "remediation_curve": "curve",
        "slip_rate": "slips",
    }
    return ", ".join(
        f"{short.get(key, key)} {value}" for key, value in dials.items()
    )


def _grid_complaints(summary: dict) -> list[str]:
    """What the summary and the experiment disagree about, in three directions.

    A figure drawn from a summary its source disowns is worse than no figure,
    because it looks current. Returning complaints is only half of it — the
    caller has to refuse to draw, or the check cannot fail and proves nothing.
    """
    declared = _declared_categories()
    drawn = set(summary["categories"])
    complaints = [
        f"{kind}: {', '.join(sorted(names))}"
        for kind, names in (
            (f"declared in {GRID_SOURCE} but not in the summary", set(declared) - drawn),
            (f"in the summary but not declared in {GRID_SOURCE}", drawn - set(declared)),
            ("drawn with no entry in ERROR_SHAPE", drawn - set(ERROR_SHAPE)),
        )
        if names
    ]
    # The dials matter as much as the names: a category re-tuned in place would
    # keep its key and mean something else.
    complaints += [
        f"category {name} was run with {summary['categories'][name].get('dials')} "
        f"but {GRID_SOURCE} now declares {declared[name]}"
        for name in sorted(drawn & set(declared))
        if summary["categories"][name].get("dials") != declared[name]
    ]
    if summary.get("standing") != "exploratory":
        complaints.append(
            f"the summary's standing is {summary.get('standing')!r}; this figure "
            f"states on its face that these populations are exploratory"
        )
    if "basic" not in drawn:
        complaints.append(
            "no `basic` row — it is the population the stored results were "
            "produced on, and without it the others have no baseline"
        )
    return complaints


def learner_type_reversal(results: Path, out: Path) -> Path:
    """The paired difference per population, and the level each arm reached.

    Two panels, because the direction and the level cannot be read apart.

    The left panel is direction, in `paired_discordance`'s encoding: bars
    centred on the ties, one row per population. It is the encoding that makes
    the finding visible at all — three rows are almost entirely tied, and the
    fourth both leans the other way and has lost nearly all its ties, so the
    outcome only has discriminating power in the row where it changes sign.

    The right panel is the level, and it is here rather than in a caption
    because a figure travels without one — the lesson `prerequisite_sweep`
    records from the other direction. At the strength the grid used, both arms
    clear a small fraction of what they clear with the dial off, so the row that
    changes sign compares two arms that have nearly stopped teaching. The
    weakest non-zero setting is marked too: the sign has already changed there,
    which is what separates a floor from the whole explanation.

    Both marked settings and every label are read from the stored summaries, and
    the categories are checked three ways against the experiment's own source —
    see `_grid_complaints`.
    """
    grid = json.loads((results / "grid_learner_types" / "summary.json").read_text())
    sweep = json.loads((results / "sweep_forgetting" / "summary.json").read_text())
    complaints = _grid_complaints(grid)
    if complaints:
        raise SystemExit(
            "learner_type_reversal refuses to draw:\n  " + "\n  ".join(complaints)
        )

    primary = grid["primary_outcome"]
    measure = primary.replace("_", " ")

    def row_for(name: str) -> dict:
        outcomes = grid["categories"][name]["outcomes"]
        return next(row for row in outcomes if row["outcome"] == primary)

    # Declared order, reversed: `barh` puts index 0 at the bottom, and `basic`
    # is the baseline the rest are read against, so it belongs at the top.
    names = list(reversed(list(grid["categories"])))

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    for ax in (left, right):
        _bare(ax)

    # ---- left: which architecture each learner's pair favoured -------------
    left.grid(axis="y", visible=False)
    reach = max(
        row_for(name)["ties"] / 2
        + max(row_for(name)["favouring_coupled"], row_for(name)["favouring_decoupled"])
        for name in names
    )
    for y, name in enumerate(names):
        row = row_for(name)
        ties = row["ties"]
        coupled, decoupled = row["favouring_coupled"], row["favouring_decoupled"]
        if primary in LOWER_IS_BETTER:
            coupled, decoupled = decoupled, coupled
        left.barh(y, -decoupled, left=-ties / 2, height=0.54, color=POLE_LOW,
                  edgecolor=SURFACE, linewidth=2, zorder=3)
        left.barh(y, ties, left=-ties / 2, height=0.54, color=NEUTRAL,
                  edgecolor=SURFACE, linewidth=2, zorder=3)
        left.barh(y, coupled, left=ties / 2, height=0.54, color=POLE_HIGH,
                  edgecolor=SURFACE, linewidth=2, zorder=3)
        if ties:
            inside = ties > 0.09 * reach * 3.5
            left.annotate(
                f"{ties} tied", xy=(0, y if inside else y + 0.40), ha="center",
                va="center", color=INK_SOFT, fontsize=8,
            )
        if decoupled:
            left.annotate(f"{decoupled}", xy=(-ties / 2 - decoupled, y), xytext=(-5, 0),
                          textcoords="offset points", ha="right", va="center",
                          color=INK_SOFT, fontsize=8)
        if coupled:
            left.annotate(f"{coupled}", xy=(ties / 2 + coupled, y), xytext=(5, 0),
                          textcoords="offset points", ha="left", va="center",
                          color=INK_SOFT, fontsize=8)
        # The difference in its own right-hand column, on a blended transform so
        # the four land in a line whatever the bars underneath them do.
        left.annotate(
            f"{row['mean_difference']:+.4f}{' *' if row['significant'] else ''}",
            xy=(0.995, y), xycoords=left.get_yaxis_transform(), ha="right",
            va="center", fontsize=8.5, family="monospace",
            color=POLE_HIGH if row["mean_difference"] > 0 else POLE_LOW,
        )

    left.set_yticks(list(range(len(names))))
    left.set_yticklabels(
        [
            "\n".join(
                part for part in (
                    ERROR_SHAPE[name],
                    # Dropped when the shape and the mechanism share a name, as
                    # they do for `linear`, where printing both reads as a
                    # stutter and neither word is doing work the other is not.
                    None if name == ERROR_SHAPE[name] else name,
                    _dial_text(grid["categories"][name]["dials"]),
                ) if part
            )
            for name in names
        ],
        fontsize=8.5, linespacing=1.5,
    )
    left.set_xticks([])
    left.set_xlim(-reach * 1.35, reach * 2.15)
    left.set_ylim(-0.6, len(names) - 0.4)
    left.set_xlabel(f"learners, by which architecture their pair favoured"
                    f"      (* Holm-adjusted p < {grid['alpha']:g})")
    left.set_title(f"{measure} changes sign against one population,\n"
                   f"and only there does it stop tying", loc="left")
    # Written rather than drawn with arrow glyphs: the house sans has no arrows,
    # and a missing glyph renders as a box in the PDF. Placed by where the bars
    # actually reach, not at the panel edges, which the difference column owns.
    left.annotate("favours decoupled", xy=(0.02, -0.155), xycoords="axes fraction",
                  color=INK_SOFT, fontsize=8)
    left.annotate("favours coupled", xy=(0.50, -0.155), xycoords="axes fraction",
                  color=INK_SOFT, fontsize=8)

    # ---- right: what either arm actually achieved -------------------------
    points = sweep["points"]
    rates = [point["forgetting_rate"] for point in points]
    achieved = {
        arm: [point["mean_remediation"][arm] for point in points]
        for arm in ("coupled", "decoupled")
    }
    for arm, colour in (("coupled", POLE_HIGH), ("decoupled", POLE_LOW)):
        right.plot(rates, achieved[arm], color=colour, linewidth=2, marker="o",
                   markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                   zorder=3, label=arm)

    # Headroom for the two marked settings, whose labels sit in a band above the
    # data rather than beside it: the curve descends across the whole width, so
    # there is no interior region wide enough that does not touch it.
    ceiling = max(max(series) for series in achieved.values())
    right.set_ylim(0, ceiling * 1.30)

    # Both settings are read from the data. The grid's is where the left panel's
    # sign change was measured; the flip point is the weakest setting at which it
    # had already happened, and the two being different is why this panel exists.
    grid_rate = grid["categories"]["forgetful"]["dials"]["forgetting_rate"]
    flipped = [
        point["forgetting_rate"] for point in points
        if point["forgetting_rate"] > 0
        and point["framing_a"]["mean_difference"] < 0
        and point["framing_a"]["significant"]
    ]
    marks = [(grid_rate, "the left panel's setting")]
    if flipped and flipped[0] != grid_rate:
        marks.insert(0, (flipped[0], "sign already changed"))
    for rate, label in marks:
        right.axvline(rate, color=AXIS, linewidth=0.8, linestyle=(0, (2, 2)), zorder=2)
        right.annotate(label, xy=(rate, 0.985), xycoords=right.get_xaxis_transform(),
                       xytext=(3, 0), textcoords="offset points", ha="left",
                       va="top", color=INK_MUTED, fontsize=7.5)

    right.set_xticks(rates)
    right.set_xticklabels([f"{rate:g}" for rate in rates])
    right.set_xlabel(f"forgetting rate, every {sweep['period']} items")
    right.set_ylabel(f"{measure} achieved")
    right.set_title("Where it changes sign, both arms have nearly\nstopped teaching",
                    loc="left")
    right.legend(loc="upper right", fontsize=8.5, labelcolor="linecolor",
                 handlelength=1.2, bbox_to_anchor=(1.0, 0.90))
    at_grid = achieved["coupled"][rates.index(grid_rate)]
    right.annotate(
        f"at that setting the coupled arm clears {at_grid:.2f},\n"
        f"against {achieved['coupled'][0]:.2f} with the dial off — "
        f"{at_grid / achieved['coupled'][0]:.0%} of it",
        xy=(0.98, 0.58), xycoords="axes fraction", ha="right", va="top",
        color=INK_SOFT, fontsize=7.5, linespacing=1.45,
    )

    # As a figure-level label rather than placed text: constrained layout
    # accounts for this one and reserves the strip, where free text in figure
    # coordinates lands on top of the axis labels below it.
    figure.supxlabel(
        f"Exploratory. N = {grid['n_learners']} per arm, paired by learner; grid "
        f"seed {grid['seed']}, forgetting sweep seed {sweep['seed']}. "
        f"`basic` is the population the stored results were produced on.",
        fontsize=7.5, color=INK_MUTED,
    )
    figure.savefig(out / f"learner_type_reversal.{SUFFIX}")
    plt.close(figure)
    return out / f"learner_type_reversal.{SUFFIX}"


def power_curve(results: Path, out: Path) -> Path:
    """Power against sample size, under both decision rules.

    Two outcomes and two rules, so colour carries the outcome and dash carries
    the rule. **The gap between a solid line and its dashed partner is the
    point**: the correction is applied to the analysis but was not applied when
    the size was chosen, so every solid line overstates the power of the study
    as run. Drawing the corrected series alone would hide that there are two
    numbers; drawing only the uncorrected one is what happened.

    The two goal outcomes are left off the axes deliberately and stated instead.
    They sit at 1.0 across the whole range and reach the target at the smallest
    size simulated, so as curves they are two flat lines that compress the
    region where anything happens — and the fact worth carrying is a sample
    size, not a shape.

    The target rule and the marked size are read from the file, and so is the
    claim that the target is never reached: `required_n` is null exactly when no
    simulated size cleared it.
    """
    data = json.loads((results / "power_calculus" / "power.json").read_text())
    outcomes = data["outcomes"]
    target = data["target_power"]
    primary = data["primary_outcome"]

    #: The outcomes whose power actually varies over the range. Selected by
    #: whether the target was ever reached, so a re-run that changes which
    #: outcomes are underpowered redraws rather than mislabels.
    climbing = [
        name for name, series in outcomes.items() if series["required_n_holm"] is None
    ]
    reached = {
        name: series["required_n_holm"]
        for name, series in outcomes.items()
        if series["required_n_holm"] is not None
    }

    figure, ax = plt.subplots(figsize=(6.6, 3.6))
    _bare(ax)
    ax.axhline(target, color=POLE_LOW, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"target {target:g}", xy=(0.005, target), xycoords=ax.get_yaxis_transform(),
        xytext=(0, 5), textcoords="offset points", color=POLE_LOW, fontsize=8,
    )

    colours = {name: colour for name, colour in zip(climbing, (BLUE, ORANGE, AQUA))}
    for name in climbing:
        points = outcomes[name]["curve"]
        ns = [point["n_learners"] for point in points]
        for key, dash, rule in (
            ("power_sign_test", None, "sign test"),
            ("power_sign_test_holm", (0, (3, 2)), "under Holm"),
        ):
            ax.plot(
                ns, [point[key] for point in points], color=colours[name],
                linewidth=2 if dash is None else 1.6, linestyle=dash or "solid",
                marker="o", markersize=4, markeredgecolor=SURFACE,
                markeredgewidth=1.1, zorder=3,
            )
            ax.annotate(
                f"{name.replace('_', ' ')}, {rule}"
                if key == "power_sign_test" else rule,
                xy=(ns[-1], points[-1][key]), xytext=(7, 0),
                textcoords="offset points", ha="left", va="center",
                color=colours[name], fontsize=8,
            )

    # The size actually used, which is the number a reader came for.
    configured = 160
    at = {
        name: (
            next(p for p in outcomes[name]["curve"] if p["n_learners"] == configured)
        )
        for name in climbing
        if any(p["n_learners"] == configured for p in outcomes[name]["curve"])
    }
    if configured in [p["n_learners"] for p in outcomes[primary]["curve"]]:
        ax.axvline(configured, color=AXIS, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
        row = at[primary]
        ax.annotate(
            f"N = {configured}: {row['power_sign_test']:.2f} uncorrected,\n"
            f"{row['power_sign_test_holm']:.2f} under Holm",
            xy=(configured, 0.98), xycoords=ax.get_xaxis_transform(),
            xytext=(-6, 0), textcoords="offset points", ha="right", va="top",
            color=INK_SOFT, fontsize=8, linespacing=1.4,
        )

    ax.set_xlim(0, max(p["n_learners"] for p in outcomes[primary]["curve"]) * 1.34)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("learners per arm")
    ax.set_ylabel("power")
    ax.set_title(
        f"Neither fine-grained outcome reaches {target:g} at any size simulated",
        loc="left",
    )
    if reached:
        ax.annotate(
            "  ·  ".join(
                f"{name.replace('_', ' ')} reaches it at N = {n}"
                for name, n in reached.items()
            )
            + " (both rules), so they are not drawn",
            xy=(0, 0), xytext=(0, -38), textcoords="offset points",
            xycoords="axes fraction", ha="left", va="top",
            color=INK_SOFT, fontsize=7.5,
        )
    figure.supxlabel(
        f"One {data['pilot_learners']}-learner pilot pool, "
        f"{data['replicates']:,} replicates, alpha {data['alpha']:g} — so this is "
        f"conditional on that pool's own draw.",
        fontsize=7.5, color=INK_MUTED,
    )
    figure.savefig(out / f"power_curve.{SUFFIX}")
    plt.close(figure)
    return out / f"power_curve.{SUFFIX}"


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
    for draw in (
        prerequisite_sweep,
        arbitration_substitution,
        paired_discordance,
        learner_type_reversal,
        power_curve,
    ):
        try:
            written = draw(args.results, args.out)
        except FileNotFoundError as missing:
            print(f"skipped {draw.__name__}: {missing.filename} has not been run")
            continue
        print(f"wrote {written}")


if __name__ == "__main__":
    main()

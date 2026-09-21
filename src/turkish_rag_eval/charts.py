"""Render the two charts the README leads with.

A benchmark repository with no picture in it asks every reader to reconstruct
the finding from a twelve-row table. These two carry the findings that a table
hides:

``ndcg-intervals``  every configuration with its 95% bootstrap interval, with
                    the ones the data cannot separate from the best picked out.
                    The interval is the point: the ranking looks decisive and
                    mostly is not.

``abstention``      coverage against selective accuracy for both confidence
                    signals. This is the repository's least expected result -
                    the intuitive signal (top-1 margin) is useless on fused
                    rankings, because RRF scores are 1/(60+rank) and the
                    top-two gap is about 2% for every query, confident or not.

Both are written in light and dark variants, because a light-background PNG in
a dark README is a glare rectangle; the README pairs them with <picture>.

Colours come from the reference palette and were validated for contrast and
colour-vision separation on both surfaces before use. The emphasis chart is
one hue plus grey rather than a colour per bar: the reader's question is
"which of these are actually different", and that is a two-group question.
"""

import argparse
import json
from pathlib import Path

from .paths import ROOT
from .report import DEFAULT_METRIC, load_configurations, recommend

#: Points resting on a handful of answered queries swing between 0 and 1 on
#: one question. abstain.pick_threshold ignores them for the same reason.
MIN_ANSWERED = 5

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "accent": "#2a78d6",
        "second": "#eb6834",
        "recessive": "#898781",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "accent": "#3987e5",
        "second": "#d95926",
        "recessive": "#898781",
    },
}


def _figure(theme: dict, size: tuple[float, float]):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=size)
    figure.patch.set_facecolor(theme["surface"])
    axes.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme["axis"])
        axes.spines[side].set_linewidth(1)
    axes.tick_params(colors=theme["muted"], labelsize=9, length=0)
    return figure, axes


def _titles(axes, theme: dict, title: str, subtitle: str) -> None:
    """Title and subtitle stacked above the axes.

    Offset in *points* from the axes corner rather than in axes fractions: the
    interval chart's height grows with the number of configurations, so a
    fractional offset would drift away from the title as rows are added, and a
    fixed title pad would collide with it.
    """
    subtitle_lines = subtitle.count("\n") + 1
    axes.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                  xytext=(0, 10), textcoords="offset points",
                  ha="left", va="bottom", fontsize=9.5,
                  color=theme["secondary"], linespacing=1.45)
    axes.annotate(title, xy=(0, 1), xycoords="axes fraction",
                  xytext=(0, 10 + 15 * subtitle_lines + 8),
                  textcoords="offset points", ha="left", va="bottom",
                  fontsize=13.5, fontweight="bold", color=theme["ink"])


def intervals_chart(results_dir: Path, out: Path, theme_name: str,
                    metric: str = DEFAULT_METRIC) -> Path:
    """Every configuration, sorted, with its interval; the tied ones picked out."""
    theme = THEMES[theme_name]
    configurations = load_configurations(results_dir)
    recommendation = recommend(configurations, metric)
    tied = {id(c) for c in recommendation.tied_with_best}

    rows = sorted(configurations, key=lambda c: c.mean(metric))
    figure, axes = _figure(theme, (8.4, 0.44 * len(rows) + 2.3))

    for position, configuration in enumerate(rows):
        low, high = configuration.interval(metric)
        mean = configuration.mean(metric)
        emphasised = id(configuration) in tied
        colour = theme["accent"] if emphasised else theme["recessive"]

        axes.plot([low, high], [position, position], color=colour, linewidth=2,
                  solid_capstyle="round", zorder=2,
                  alpha=1.0 if emphasised else 0.55)
        axes.plot([mean], [position], marker="o", markersize=8, color=colour,
                  markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3,
                  alpha=1.0 if emphasised else 0.55)
        axes.text(high + 0.012, position, f"{mean:.3f}", va="center",
                  fontsize=9, color=theme["ink"] if emphasised
                  else theme["muted"])

    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([c.name for c in rows], fontsize=9.5,
                         color=theme["secondary"])
    axes.set_xlabel(metric, color=theme["secondary"], fontsize=9.5, labelpad=8)
    axes.set_xlim(0.22, 0.82)
    axes.set_ylim(-0.7, len(rows) - 0.3)
    axes.xaxis.grid(True, color=theme["grid"], linewidth=1)
    axes.set_axisbelow(True)

    _titles(axes, theme,
            "Most of this ranking is not a ranking",
            f"{metric} with 95% bootstrap intervals over "
            f"{len(recommendation.best.scored)} queries. Highlighted: the "
            f"configurations a paired\nbootstrap cannot separate from the best.")

    # A two-group emphasis chart still needs its groups named.
    from matplotlib.lines import Line2D
    axes.legend(
        handles=[
            Line2D([], [], color=theme["accent"], linewidth=2, marker="o",
                   markersize=7, label="within noise of the best"),
            Line2D([], [], color=theme["recessive"], alpha=0.55, linewidth=2,
                   marker="o", markersize=7, label="measurably worse"),
        ],
        loc="lower right", frameon=False, fontsize=9,
        labelcolor=theme["secondary"])

    return _save(figure, out)


def abstention_chart(results_dir: Path, out: Path, theme_name: str) -> Path:
    """Coverage against selective accuracy, one line per confidence signal."""
    theme = THEMES[theme_name]
    figure, axes = _figure(theme, (7.6, 5.0))

    series = [
        ("score", "dense top-1 cosine", theme["accent"]),
        ("margin", "top-1 margin", theme["second"]),
    ]
    plotted = 0
    for signal, label, colour in series:
        path = results_dir / f"abstain_curve_{signal}.json"
        if not path.exists():
            continue
        curve = [p for p in json.loads(path.read_text(encoding="utf-8"))
                 if p["answered"] >= MIN_ANSWERED
                 and p["selective_accuracy"] is not None]
        if not curve:
            continue
        coverage = [p["coverage"] for p in curve]
        accuracy = [p["selective_accuracy"] for p in curve]

        axes.plot(coverage, accuracy, color=colour, linewidth=2, marker="o",
                  markersize=6, markeredgecolor=theme["surface"],
                  markeredgewidth=1.5, label=label, zorder=3)
        # Direct label at the low-coverage end, where the two lines separate.
        leftmost = min(range(len(coverage)), key=lambda i: coverage[i])
        axes.annotate(label, (coverage[leftmost], accuracy[leftmost]),
                      textcoords="offset points", xytext=(8, 8),
                      fontsize=9.5, color=theme["ink"], fontweight="bold")
        plotted += 1

    axes.set_xlabel("coverage - share of queries answered automatically",
                    color=theme["secondary"], fontsize=9.5, labelpad=8)
    axes.set_ylabel("selective accuracy - correct among those answered",
                    color=theme["secondary"], fontsize=9.5, labelpad=8)
    axes.set_xlim(0, 1.03)
    axes.set_ylim(0, 1.03)
    axes.grid(True, color=theme["grid"], linewidth=1)
    axes.set_axisbelow(True)
    if plotted > 1:
        axes.legend(loc="lower left", frameon=False, fontsize=9,
                    labelcolor=theme["secondary"])

    _titles(axes, theme,
            "The intuitive confidence signal is the useless one",
            f"Thresholds keeping at least {MIN_ANSWERED} answered queries. RRF "
            f"fuses ranks as 1/(60+rank), so the top-two\ngap is ~2% on every "
            f"query and the margin buys almost no coverage to trade.")
    return _save(figure, out)


def _save(figure, out: Path) -> Path:
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    plt.close(figure)
    return out


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "charts")
    parser.add_argument("--metric", default=DEFAULT_METRIC)


def run(args) -> int:
    if not (args.results / "summary.json").exists():
        print(f"{args.results} holds no results - run 'turkish-rag-eval run' first")
        return 1

    written = []
    for theme_name in THEMES:
        suffix = "" if theme_name == "light" else "-dark"
        written.append(intervals_chart(
            args.results, args.out / f"ndcg-intervals{suffix}.png",
            theme_name, args.metric))
        if (args.results / "abstain_curve_score.json").exists():
            written.append(abstention_chart(
                args.results, args.out / f"abstention{suffix}.png", theme_name))

    for path in written:
        print(f"  {path.relative_to(ROOT) if ROOT in path.parents else path}")
    print(f"{len(written)} chart(s) written")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

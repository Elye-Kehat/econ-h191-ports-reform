#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Figure based on the current preferred Table 3 values for Model 1B:
# "Haifa reforms and ln(K/L): binned event-study estimates"
# This version hard-codes the reported coefficients and standard errors from the
# table the user supplied, so the figure exactly matches the table currently in
# the thesis draft.


COMPETITION_WINDOWS = [
    ("avg_pre", "Average pre\nm ∈ [-12,-2]"),
    ("post_1_6", "Average post\nm ∈ [1,6]"),
    ("post_7_12", "Average post\nm ∈ [7,12]"),
    ("post_13_24", "Average post\nm ∈ [13,24]"),
    ("post_full", "Average full post\nm ∈ [1,24]"),
]

PRIVATIZATION_WINDOWS = [
    ("avg_pre", "Average pre\nm ∈ [-12,-2]"),
    ("post_1_6", "Average post\nm ∈ [1,6]"),
    ("post_7_23", "Average post\nm ∈ [7,23]"),
    ("post_full", "Average full post\nm ∈ [1,23]"),
]

STYLE = {
    "baseline": {"label": "Baseline", "color": "#1f77b4", "marker": "o", "offset": -0.12},
    "ctrls_tr": {"label": "Ctrls+Tr", "color": "#ff7f0e", "marker": "s", "offset": 0.12},
}

DATA = {
    "competition": {
        "title": "Panel A: Competition clock (Bayport entry, Sep. 2021)",
        "windows": COMPETITION_WINDOWS,
        "legend_y": 0.947,
        "heading_y": 0.965,
        "series": {
            "legacy": {
                "title": "Haifa legacy (ln(K/L))",
                "ylim": (-0.8, 5.25),
                "pretrend": {"baseline": "<0.001", "ctrls_tr": "0.199"},
                "baseline": {
                    "avg_pre": (0.022, 0.017),
                    "post_1_6": (-0.092, 0.041),
                    "post_7_12": (0.310, 0.119),
                    "post_13_24": (1.659, 0.147),
                    "post_full": (0.884, 0.079),
                },
                "ctrls_tr": {
                    "avg_pre": (-0.354, 0.572),
                    "post_1_6": (0.198, 0.285),
                    "post_7_12": (0.984, 0.712),
                    "post_13_24": (2.615, 1.211),
                    "post_full": (1.603, 0.849),
                },
            },
            "aggregate": {
                "title": "Aggregate port (ln(K/L))",
                "ylim": (-0.45, 1.10),
                "pretrend": {"baseline": "<0.001", "ctrls_tr": "0.002"},
                "baseline": {
                    "avg_pre": (0.035, 0.016),
                    "post_1_6": (-0.066, 0.033),
                    "post_7_12": (0.039, 0.046),
                    "post_13_24": (0.341, 0.015),
                    "post_full": (0.164, 0.018),
                },
                "ctrls_tr": {
                    "avg_pre": (-0.060, 0.133),
                    "post_1_6": (0.062, 0.067),
                    "post_7_12": (0.319, 0.173),
                    "post_13_24": (0.692, 0.272),
                    "post_full": (0.441, 0.193),
                },
            },
        },
    },
    "privatization": {
        "title": "Panel B: Privatization clock (Jan. 2023)",
        "windows": PRIVATIZATION_WINDOWS,
        "legend_y": 0.485,
        "heading_y": 0.503,
        "series": {
            "legacy": {
                "title": "Haifa legacy (ln(K/L))",
                "ylim": (-2.1, 0.45),
                "pretrend": {"baseline": "<0.001", "ctrls_tr": "<0.001"},
                "baseline": {
                    "avg_pre": (-0.581, 0.142),
                    "post_1_6": (-0.351, 0.454),
                    "post_7_23": (-1.392, 0.109),
                    "post_full": (-1.120, 0.157),
                },
                "ctrls_tr": {
                    "avg_pre": (-0.592, 0.140),
                    "post_1_6": (-0.319, 0.509),
                    "post_7_23": (-1.273, 0.249),
                    "post_full": (-1.024, 0.277),
                },
            },
            "aggregate": {
                "title": "Aggregate port (ln(K/L))",
                "ylim": (-0.70, 0.15),
                "pretrend": {"baseline": "0.138", "ctrls_tr": "0.423"},
                "baseline": {
                    "avg_pre": (-0.029, 0.019),
                    "post_1_6": (-0.219, 0.082),
                    "post_7_23": (-0.186, 0.068),
                    "post_full": (-0.195, 0.062),
                },
                "ctrls_tr": {
                    "avg_pre": (-0.019, 0.024),
                    "post_1_6": (-0.248, 0.120),
                    "post_7_23": (-0.406, 0.105),
                    "post_full": (-0.365, 0.102),
                },
            },
        },
    },
}


def build_legend_handles():
    handles = []
    for _, style in STYLE.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                linewidth=0,
                markersize=6,
                label=style["label"],
            )
        )
    return handles


def plot_panel(ax, panel_key, series_key, show_ylabel):
    panel = DATA[panel_key]
    spec = panel["series"][series_key]
    windows = panel["windows"]

    for model_key, style in STYLE.items():
        xs, ys, yerrs = [], [], []
        for i, (win_key, _) in enumerate(windows):
            beta, se = spec[model_key][win_key]
            xs.append(i + style["offset"])
            ys.append(beta)
            yerrs.append(1.96 * se)

        ax.errorbar(
            xs,
            ys,
            yerr=yerrs,
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.8,
            capsize=3,
            markersize=4.8,
            linestyle="none",
            zorder=3,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(-0.45, len(windows) - 0.55)
    ax.set_ylim(*spec["ylim"])
    ax.set_title(spec["title"], loc="left", fontsize=11, weight="bold")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([label for _, label in windows], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", pad=3)
    ax.grid(True, axis="y", alpha=0.20)
    ax.grid(True, axis="x", alpha=0.08)

    if show_ylabel:
        ax.set_ylabel("Estimated effect on ln(K/L)", fontsize=10)

    ax.text(
        0.00,
        -0.22,
        f"Pretrend p-values: Baseline = {spec['pretrend']['baseline']}   |   Ctrls+Tr = {spec['pretrend']['ctrls_tr']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="Directory to write the figure and manifest")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.6, 10.3))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.32)

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])

    plot_panel(ax11, "competition", "legacy", show_ylabel=True)
    plot_panel(ax12, "competition", "aggregate", show_ylabel=False)
    plot_panel(ax21, "privatization", "legacy", show_ylabel=True)
    plot_panel(ax22, "privatization", "aggregate", show_ylabel=False)

    ax11.set_xlabel("Event-study window", fontsize=10)
    ax12.set_xlabel("Event-study window", fontsize=10)
    ax21.set_xlabel("Event-study window", fontsize=10)
    ax22.set_xlabel("Event-study window", fontsize=10)

    fig.text(
        0.02,
        DATA["competition"]["heading_y"],
        DATA["competition"]["title"],
        fontsize=12.5,
        weight="bold",
        ha="left",
    )
    fig.text(
        0.02,
        DATA["privatization"]["heading_y"],
        DATA["privatization"]["title"],
        fontsize=12.5,
        weight="bold",
        ha="left",
    )

    fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, DATA["competition"]["legend_y"]),
        ncol=2,
        frameon=False,
        fontsize=10,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, DATA["privatization"]["legend_y"]),
        ncol=2,
        frameon=False,
        fontsize=10,
        handletextpad=0.4,
        columnspacing=1.1,
    )

    fig.subplots_adjust(top=0.91, bottom=0.14, left=0.08, right=0.98)

    outpath = outdir / "figure_model1b_kl_binned_eventstudy.png"
    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "description": "Binned event-study figure for Model 1B Table 3 values",
        "output_file": str(outpath),
        "based_on": "User-supplied Table 3 LaTeX values for Model 1B preferred specification",
        "panels": {
            "competition": DATA["competition"]["title"],
            "privatization": DATA["privatization"]["title"],
        },
    }

    manifest_path = outdir / "figure_model1b_kl_binned_eventstudy_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Wrote: {outpath}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()

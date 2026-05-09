#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


STATIC_FILE = "model1a_q_static_betas_twfe.tsv"
NYT_WINDOW_FILE = "model1a_q_window_betas_nyt.tsv"
NYT_PRETREND_FILE = "model1a_q_pretrend_tests_nyt.tsv"

CONVENTIONAL_PANELS = [
    {
        "reform": "haifa_comp",
        "target": "Haifa-Legacy",
        "title": "Haifa legacy terminal",
        "rows": ["full_post", "post_y1", "post_y2", "post_y1_2", "post_y3"],
    },
    {
        "reform": "haifa_comp",
        "target": "Haifa-Aggregate",
        "title": "Haifa aggregate port",
        "rows": ["full_post", "post_y1", "post_y2", "post_y1_2", "post_y3"],
    },
    {
        "reform": "ashdod_comp",
        "target": "Ashdod-Legacy",
        "title": "Ashdod legacy terminal",
        "rows": ["full_post", "post_y1", "post_y2", "post_y1_2"],
    },
    {
        "reform": "ashdod_comp",
        "target": "Ashdod-Aggregate",
        "title": "Ashdod aggregate port",
        "rows": ["full_post", "post_y1", "post_y2", "post_y1_2"],
    },
]

NYT_PANELS = [
    {
        "reform": "haifa_comp",
        "target": "Haifa-Legacy",
        "title": "Haifa legacy terminal",
        "rows": ["avg_pre", "post_y1"],
    },
    {
        "reform": "haifa_comp",
        "target": "Haifa-Aggregate",
        "title": "Haifa aggregate port",
        "rows": ["avg_pre", "post_y1"],
    },
]

ROW_LABELS_CONVENTIONAL = {
    "full_post": "q ≥ 1",
    "post_y1": "q ∈ [1,4]",
    "post_y2": "q ∈ [5,8]",
    "post_y1_2": "q ∈ [1,8]",
    "post_y3": "q ∈ [9,12]",
}

ROW_LABELS_NYT = {
    "avg_pre": "Average pre\nq ∈ [-4,-2]",
    "post_y1": "Average post\nq ∈ [1,4]",
    "full_post": "Average post\nq ∈ [1,4]",
}

STYLE_CONVENTIONAL = {
    "baseline": {"label": "TWFE", "color": "#1f77b4", "marker": "o", "offset": -0.12},
    "porttr": {"label": "TWFE+Tr", "color": "#ff7f0e", "marker": "s", "offset": 0.12},
}

STYLE_NYT = {
    "baseline": {"label": "NYT", "color": "#2ca02c", "marker": "D", "offset": -0.12},
    "porttr": {"label": "NYT+Tr", "color": "#d62728", "marker": "^", "offset": 0.12},
}


def infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent] + list(here.parents):
        if (candidate / "Data").exists() and (candidate / "Design").exists():
            return candidate
    return Path.cwd().resolve()


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_file(repo_root: Path, filename: str, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (repo_root / p)

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / filename,
        script_dir.parent / filename,
        repo_root / filename,
        repo_root / "Design" / "Output (new)" / "Model_1A_v8_2" / filename,
        repo_root / "Model_1A_v8_2" / filename,
        Path("/mnt/data") / filename,
    ]

    direct = first_existing(candidates)
    if direct is not None:
        return direct

    matches = sorted(repo_root.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find {filename}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def conventional_ylim(target: str) -> tuple[float, float]:
    if target == "Haifa-Legacy":
        return -1.5, 1.5
    return -1.0, 1.0


def nyt_ylim() -> tuple[float, float]:
    return -0.5, 0.5


def set_y_ticks(ax, ylim):
    if ylim == (-1.5, 1.5):
        ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    elif ylim == (-1.0, 1.0):
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    elif ylim == (-0.5, 0.5):
        ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])


def get_nyt_row(sub: pd.DataFrame, row_key: str, spec_name: str):
    candidates = [row_key]
    if row_key == "post_y1":
        candidates = ["post_y1", "full_post"]

    tmp = sub[(sub["window"].isin(candidates)) & (sub["spec_name"] == spec_name)].copy()
    if tmp.empty:
        return None

    preferred = tmp[tmp["window"] == row_key]
    if not preferred.empty:
        return preferred.iloc[0]

    return tmp.iloc[0]


def style_axis(ax, ylim, ylabel=None, xlabel=None):
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(*ylim)
    set_y_ticks(ax, ylim)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    else:
        ax.set_ylabel("")

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    else:
        ax.set_xlabel("")

    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.10)
    ax.tick_params(axis="x", labelsize=9, pad=5)
    ax.tick_params(axis="y", labelsize=9)


def plot_conventional_panel(ax, df, spec, show_ylabel, show_xlabel):
    sub = df[
        (df["table_group"] == "competition")
        & (df["reform"] == spec["reform"])
        & (df["target"] == spec["target"])
        & (df["horizon"].isin(spec["rows"]))
    ].copy()

    x_map = {row: i for i, row in enumerate(spec["rows"])}

    for spec_name, style in STYLE_CONVENTIONAL.items():
        tmp = sub[sub["spec_name"] == spec_name].copy()
        if tmp.empty:
            continue

        xs = []
        ys = []
        yerrs = []

        for row_key in spec["rows"]:
            one = tmp[tmp["horizon"] == row_key]
            if one.empty:
                continue
            row = one.iloc[0]
            xs.append(x_map[row_key] + style["offset"])
            ys.append(float(row["beta"]))
            yerrs.append(1.96 * float(row["se"]))

        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.7,
                capsize=3,
                markersize=5.5,
                linestyle="none",
                zorder=3,
            )

    ax.set_title(spec["title"], loc="left", fontsize=11, weight="bold")
    ax.set_xticks(range(len(spec["rows"])))
    ax.set_xticklabels([ROW_LABELS_CONVENTIONAL[r] for r in spec["rows"]])
    ax.set_xlim(-0.4, len(spec["rows"]) - 0.6)

    style_axis(
        ax,
        conventional_ylim(spec["target"]),
        "Estimated effect on ln(LP)" if show_ylabel else None,
        "Post-reform window" if show_xlabel else None,
    )


def plot_nyt_panel(ax, df, spec, show_ylabel):
    sub = df[
        (df["table_group"] == "competition")
        & (df["reform"] == spec["reform"])
        & (df["target"] == spec["target"])
    ].copy()

    x_map = {row: i for i, row in enumerate(spec["rows"])}

    for spec_name, style in STYLE_NYT.items():
        xs = []
        ys = []
        yerrs = []

        for row_key in spec["rows"]:
            row = get_nyt_row(sub, row_key=row_key, spec_name=spec_name)
            if row is None:
                continue

            xs.append(x_map[row_key] + style["offset"])
            ys.append(float(row["beta"]))
            yerrs.append(1.96 * float(row["se"]))

        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.7,
                capsize=3,
                markersize=6.0,
                linestyle="none",
                zorder=3,
            )

    ax.set_title(spec["title"], loc="left", fontsize=11, weight="bold")
    ax.set_xticks(range(len(spec["rows"])))
    ax.set_xticklabels([ROW_LABELS_NYT[r] for r in spec["rows"]])
    ax.set_xlim(-0.4, len(spec["rows"]) - 0.6)

    style_axis(
        ax,
        nyt_ylim(),
        "Estimated effect on ln(LP)" if show_ylabel else None,
        "NYT summary window",
    )


def build_legend_handles(style_map):
    handles = []
    for _, style in style_map.items():
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                linewidth=0,
                markersize=7,
                label=style["label"],
            )
        )
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", default=None, help=f"Path to {STATIC_FILE}")
    ap.add_argument("--nyt", default=None, help=f"Path to {NYT_WINDOW_FILE}")
    ap.add_argument("--pretrend", default=None, help=f"Path to {NYT_PRETREND_FILE}")
    ap.add_argument("--outdir", default=None, help="Output directory")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    repo_root = infer_repo_root()

    static_path = resolve_file(repo_root, STATIC_FILE, args.static)
    nyt_path = resolve_file(repo_root, NYT_WINDOW_FILE, args.nyt)
    pretrend_path = resolve_file(repo_root, NYT_PRETREND_FILE, args.pretrend)

    static_df = load_tsv(static_path)
    nyt_df = load_tsv(nyt_path)

    if args.outdir is not None:
        outdir = Path(args.outdir)
        if not outdir.is_absolute():
            outdir = repo_root / outdir
    else:
        outdir = static_path.parent / "Figures"

    ensure_dir(outdir)

    fig = plt.figure(figsize=(15.5, 11.6))

    outer = fig.add_gridspec(
        nrows=5,
        ncols=4,
        height_ratios=[0.10, 1.12, 1.12, 0.10, 0.90],
        hspace=0.28,
        wspace=0.30,
    )

    ax_leg_a = fig.add_subplot(outer[0, :])
    ax_leg_a.axis("off")

    ax11 = fig.add_subplot(outer[1, 0:2])
    ax12 = fig.add_subplot(outer[1, 2:4])
    ax21 = fig.add_subplot(outer[2, 0:2])
    ax22 = fig.add_subplot(outer[2, 2:4])

    ax_leg_b = fig.add_subplot(outer[3, :])
    ax_leg_b.axis("off")

    nyt_gs = outer[4, :].subgridspec(1, 8, wspace=0.70)
    ax31 = fig.add_subplot(nyt_gs[0, 1:3])
    ax32 = fig.add_subplot(nyt_gs[0, 5:7])

    plot_conventional_panel(ax11, static_df, CONVENTIONAL_PANELS[0], show_ylabel=True, show_xlabel=False)
    plot_conventional_panel(ax12, static_df, CONVENTIONAL_PANELS[1], show_ylabel=False, show_xlabel=False)
    plot_conventional_panel(ax21, static_df, CONVENTIONAL_PANELS[2], show_ylabel=True, show_xlabel=True)
    plot_conventional_panel(ax22, static_df, CONVENTIONAL_PANELS[3], show_ylabel=False, show_xlabel=True)

    plot_nyt_panel(ax31, nyt_df, NYT_PANELS[0], show_ylabel=True)
    plot_nyt_panel(ax32, nyt_df, NYT_PANELS[1], show_ylabel=False)

    ax_leg_a.legend(
        handles=build_legend_handles(STYLE_CONVENTIONAL),
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    ax_leg_b.legend(
        handles=build_legend_handles(STYLE_NYT),
        loc="center",
        bbox_to_anchor=(0.5, 0.30),
        ncol=2,
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    fig.subplots_adjust(top=1, bottom=0.07, left=0.07, right=0.98)


    pos_leg_a = ax_leg_a.get_position()
    pos_leg_b = ax_leg_b.get_position()

    fig.text(
        0.03,
        pos_leg_a.y1 + 0.002,
        "Panel A: Conventional DiD estimates by post-reform window",
        fontsize=12,
        weight="bold",
        ha="left",
    )

    fig.text(
        0.03,
        pos_leg_b.y1 - 0.006,
        "Panel B: Haifa NYT average pre- and post-reform estimates",
        fontsize=12,
        weight="bold",
        ha="left",
    )

    outpath = outdir / "figure_model1a_competition_entry_lnLP_coefficients_v3.png"
    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "description": "Competition entry and labor productivity: conventional DiD window estimates and Haifa NYT summaries",
        "inputs": {
            "static_twfe": str(static_path),
            "window_nyt": str(nyt_path),
            "pretrend_nyt": str(pretrend_path),
        },
        "output_file": str(outpath),
        "title": "Competition entry and labor productivity: conventional DiD window estimates and Haifa NYT summaries",
        "panel_a_heading": "Conventional DiD estimates by post-reform window",
        "panel_b_heading": "Haifa NYT average pre- and post-reform estimates",
        "conventional_y_limits": {
            "Haifa-Legacy": [-1.5, 1.5],
            "Other conventional panels": [-1.0, 1.0],
        },
        "nyt_y_limits": [-0.5, 0.5],
        "spacing_controls": {
            "outer_gridspec_height_ratios": [0.10, 1.12, 1.12, 0.10, 0.90],
            "outer_gridspec_hspace": 0.28,
            "outer_gridspec_wspace": 0.30,
            "subplots_adjust_top": 0.935,
            "subplots_adjust_bottom": 0.07,
            "subplots_adjust_left": 0.07,
            "subplots_adjust_right": 0.98,
        },
    }

    with open(outdir / "figure_model1a_competition_entry_lnLP_coefficients_v3_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Static TWFE input: {static_path}")
    print(f"NYT window input:  {nyt_path}")
    print(f"NYT pretrend:      {pretrend_path}")
    print(f"Wrote:             {outpath}")


if __name__ == "__main__":
    main()
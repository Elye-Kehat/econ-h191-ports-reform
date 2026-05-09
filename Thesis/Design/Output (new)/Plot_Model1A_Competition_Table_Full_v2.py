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
        "ylim": (-1.5, 1.5),
    },
    {
        "reform": "haifa_comp",
        "target": "Haifa-Aggregate",
        "title": "Haifa aggregate port",
        "rows": ["full_post", "post_y1", "post_y2", "post_y1_2", "post_y3"],
        "ylim": (-1.0, 1.0),
    },
    {
        "reform": "ashdod_comp",
        "target": "Ashdod-Legacy",
        "title": "Ashdod legacy terminal",
        "rows": ["full_post", "post_y1", "post_y2", "post_y1_2"],
        "ylim": (-1.0, 1.0),
    },
    {
        "reform": "ashdod_comp",
        "target": "Ashdod-Aggregate",
        "title": "Ashdod aggregate port",
        "rows": ["full_post", "post_y1", "post_y2", "post_y1_2"],
        "ylim": (-1.0, 1.0),
    },
]

NYT_PANELS = [
    {
        "reform": "haifa_comp",
        "target": "Haifa-Legacy",
        "title": "Haifa legacy terminal",
        "rows": ["avg_pre", "post_y1"],
        "ylim": (-0.5, 0.5),
    },
    {
        "reform": "haifa_comp",
        "target": "Haifa-Aggregate",
        "title": "Haifa aggregate port",
        "rows": ["avg_pre", "post_y1"],
        "ylim": (-0.5, 0.5),
    },
]

ROW_LABELS_CONVENTIONAL = {
    "full_post": "Full post\nq ≥ 1",
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


def first_existing(paths: list[Path]) -> Path | None:
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


def get_nyt_row(sub: pd.DataFrame, row_key: str, spec_name: str) -> pd.Series | None:
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


def build_legend_handles(style_map: dict) -> list[Line2D]:
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
                markersize=6,
                label=style["label"],
            )
        )
    return handles


def plot_conventional_panel(ax: plt.Axes, df: pd.DataFrame, spec: dict) -> None:
    sub = df[
        (df["table_group"] == "competition")
        & (df["reform"] == spec["reform"])
        & (df["target"] == spec["target"])
        & (df["horizon"].isin(spec["rows"]))
    ].copy()

    x_positions = list(range(len(spec["rows"])))
    x_map = {row: i for i, row in enumerate(spec["rows"])}

    for spec_name, style in STYLE_CONVENTIONAL.items():
        tmp = sub[sub["spec_name"] == spec_name].copy()
        if tmp.empty:
            continue

        for row_key in spec["rows"]:
            one = tmp[tmp["horizon"] == row_key]
            if one.empty:
                continue
            row = one.iloc[0]
            x = x_map[row_key] + style["offset"]
            y = float(row["beta"])
            err = 1.96 * float(row["se"])

            ax.errorbar(
                x,
                y,
                yerr=err,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.6,
                capsize=3,
                markersize=4.5,
                linestyle="none",
                zorder=3,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(*spec["ylim"])
    ax.set_xlim(-0.45, len(spec["rows"]) - 0.55)
    ax.set_title(spec["title"], loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([ROW_LABELS_CONVENTIONAL[r] for r in spec["rows"]], fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.20)
    ax.grid(True, axis="x", alpha=0.10)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", pad=2)


def plot_nyt_panel(ax: plt.Axes, df: pd.DataFrame, pretrend_df: pd.DataFrame | None, spec: dict) -> None:
    sub = df[
        (df["table_group"] == "competition")
        & (df["reform"] == spec["reform"])
        & (df["target"] == spec["target"])
    ].copy()

    x_positions = list(range(len(spec["rows"])))
    x_map = {row: i for i, row in enumerate(spec["rows"])}

    for spec_name, style in STYLE_NYT.items():
        for row_key in spec["rows"]:
            row = get_nyt_row(sub, row_key=row_key, spec_name=spec_name)
            if row is None:
                continue

            x = x_map[row_key] + style["offset"]
            y = float(row["beta"])
            err = 1.96 * float(row["se"])

            ax.errorbar(
                x,
                y,
                yerr=err,
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.4,
                capsize=3,
                markersize=4.5,
                linestyle="none",
                zorder=3,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(*spec["ylim"])
    ax.set_xlim(-0.45, len(spec["rows"]) - 0.55)
    ax.set_title(spec["title"], loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([ROW_LABELS_NYT[r] for r in spec["rows"]], fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.20)
    ax.grid(True, axis="x", alpha=0.10)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", pad=2)

    if pretrend_df is not None:
        psub = pretrend_df[
            (pretrend_df["table_group"] == "competition")
            & (pretrend_df["reform"] == spec["reform"])
            & (pretrend_df["target"] == spec["target"])
        ].copy()

        p_base = psub.loc[psub["spec_name"] == "baseline", "pvalue"]
        p_tr = psub.loc[psub["spec_name"] == "porttr", "pvalue"]

        if not p_base.empty and not p_tr.empty:
            txt = f"Pretrend p-values: NYT = {p_base.iloc[0]:.3f}   |   NYT+Tr = {p_tr.iloc[0]:.3f}"
            ax.text(
                0.0,
                -0.22,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
            )


def main() -> None:
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
    pretrend_df = load_tsv(pretrend_path)

    if args.outdir is not None:
        outdir = Path(args.outdir)
        if not outdir.is_absolute():
            outdir = repo_root / outdir
    else:
        outdir = static_path.parent / "Figures"

    ensure_dir(outdir)

    fig = plt.figure(figsize=(14.5, 10.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.68], hspace=0.28, wspace=0.30)

    ax_tl = fig.add_subplot(gs[0, 0:2])
    ax_tr = fig.add_subplot(gs[0, 2:4])
    gs_mid = gs[1, :].subgridspec(1, 2, wspace=0.28)
    ax_bl = fig.add_subplot(gs_mid[0, 0])
    ax_br = fig.add_subplot(gs_mid[0, 1])

    top_axes = [ax_tl, ax_tr]
    top_pairs = [
        (CONVENTIONAL_PANELS[0], CONVENTIONAL_PANELS[1]),
        (CONVENTIONAL_PANELS[2], CONVENTIONAL_PANELS[3]),
    ]

    for ax, (left_spec, right_spec) in zip(top_axes, top_pairs):
        ax.remove()

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tm = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[0, 2])
    ax_t4 = fig.add_subplot(gs[0, 3])
    top_axes = [ax_tl, ax_tm, ax_tr, ax_t4]
    bottom_axes = [ax_bl, ax_br]

    for ax, spec in zip(top_axes, CONVENTIONAL_PANELS):
        plot_conventional_panel(ax, static_df, spec)

    for ax, spec in zip(bottom_axes, NYT_PANELS):
        plot_nyt_panel(ax, nyt_df, pretrend_df, spec)

    for ax in top_axes:
        ax.set_xlabel("Post-reform window", fontsize=9.5)
        ax.set_ylabel("Estimated effect on ln(LP)", fontsize=9.5)

    for ax in bottom_axes:
        ax.set_xlabel("NYT summary window", fontsize=9.5)
        ax.set_ylabel("Estimated effect on ln(LP)", fontsize=9.5)

    fig.text(
        0.015,
        0.962,
        "Panel A: Conventional DiD estimates by post-reform window",
        fontsize=12,
        weight="bold",
        ha="left",
    )

    fig.text(
        0.50,
        0.945,
        "TWFE      TWFE+Tr",
        fontsize=10,
        ha="center",
        va="center",
        alpha=0.0,
    )

    fig.legend(
        handles=build_legend_handles(STYLE_CONVENTIONAL),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handletextpad=0.3,
        columnspacing=1.0,
    )

    fig.text(
        0.015,
        0.365,
        "Panel B: Haifa NYT average pre- and post-reform estimates",
        fontsize=12,
        weight="bold",
        ha="left",
    )

    fig.legend(
        handles=build_legend_handles(STYLE_NYT),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.368),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handletextpad=0.3,
        columnspacing=1.0,
    )

    outpath = outdir / "figure_model1a_competition_full_coefficients_v2.png"
    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "description": "Competition LP figure with quarter-based window labels and no overall title",
        "inputs": {
            "static_twfe": str(static_path),
            "window_nyt": str(nyt_path),
            "pretrend_nyt": str(pretrend_path),
        },
        "output_file": str(outpath),
    }

    with open(outdir / "figure_model1a_competition_full_coefficients_v2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Static TWFE input: {static_path}")
    print(f"NYT window input:  {nyt_path}")
    print(f"NYT pretrend:      {pretrend_path}")
    print(f"Wrote:             {outpath}")


if __name__ == "__main__":
    main()

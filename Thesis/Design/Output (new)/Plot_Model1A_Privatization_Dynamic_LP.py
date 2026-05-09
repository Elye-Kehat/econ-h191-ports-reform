#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd


DYNAMIC_TWFE_FILE = "model1a_q_dynamic_betas_twfe.tsv"
DYNAMIC_NYT_FILE = "model1a_q_dynamic_betas_nyt.tsv"


TARGET_STYLE = {
    "Haifa-Aggregate": {
        "label": "Aggregate port (main object)",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.6,
        "markersize": 5.2,
        "zorder": 4,
    },
    "Haifa-Legacy": {
        "label": "Haifa-Legacy (diagnostic)",
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.8,
        "markersize": 4.6,
        "zorder": 3,
    },
    "Haifa-Bayport": {
        "label": "Haifa-Bayport placebo (diagnostic)",
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 1.8,
        "markersize": 4.8,
        "zorder": 3,
    },
}

TARGET_ORDER = ["Haifa-Aggregate", "Haifa-Legacy", "Haifa-Bayport"]


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


def resolve_input(repo_root: Path, filename: str, explicit: str | None = None) -> Path:
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


def pretty_bin_label(a: float, b: float) -> str:
    a = int(a)
    b = int(b)
    if a == b:
        if a == 0:
            return "q = 0"
        return f"q = {a}"
    return f"q ∈ [{a},{b}]"


def keep_supported_bins(df: pd.DataFrame) -> pd.DataFrame:
    # For the privatization figure, keep the bins that contain the supported
    # dynamic shell that is informative in the current design:
    # pre bins, q0, q1..q4, and the first later post bin q[5,8].
    allowed = {"pre_8_5", "pre_4_2", "q0", "q1", "q2", "q3", "q4", "y2_q5_8"}
    out = df[df["bin_label"].isin(allowed)].copy()

    # Drop rows that are structurally empty placeholders.
    out = out[~((out["beta"].abs() < 1e-12) & (out["se"].abs() < 1e-12) & (out["a"] >= 9))].copy()

    # Midpoint x-position for interval bins.
    out["x"] = (out["a"] + out["b"]) / 2.0
    out["tick_label"] = out.apply(lambda r: pretty_bin_label(r["a"], r["b"]), axis=1)
    out = out.sort_values(["target", "x"]).reset_index(drop=True)
    return out


def plot_series(ax: plt.Axes, sub: pd.DataFrame) -> None:
    for target in TARGET_ORDER:
        tdf = sub[sub["target"] == target].copy()
        if tdf.empty:
            continue

        style = TARGET_STYLE[target]
        ax.errorbar(
            tdf["x"],
            tdf["beta"],
            yerr=1.96 * tdf["se"],
            fmt=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.5,
            capsize=3,
            label=style["label"],
            zorder=style["zorder"],
        )

    # Reform marker
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(
        0.06,
        0.98,
        "Haifa privatization",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="top",
        ha="left",
        fontsize=11,
    )

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.10)

    # Use the actual plotted x-positions as ticks.
    ticks_df = sub[["x", "tick_label"]].drop_duplicates().sort_values("x")
    ax.set_xticks(ticks_df["x"])
    ax.set_xticklabels(ticks_df["tick_label"], fontsize=9)

    ax.set_xlabel("Event time (quarters relative to privatization)", fontsize=11)
    ax.set_ylabel("Estimated effect on ln(LP)", fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=["nyt", "twfe"],
        default="nyt",
        help="Which dynamic coefficient file family to use",
    )
    ap.add_argument(
        "--spec",
        choices=["baseline", "porttr"],
        default="porttr",
        help="Which specification inside the chosen dataset to plot",
    )
    ap.add_argument("--nyt-file", default=None, help=f"Path to {DYNAMIC_NYT_FILE}")
    ap.add_argument("--twfe-file", default=None, help=f"Path to {DYNAMIC_TWFE_FILE}")
    ap.add_argument("--outdir", default=".", help="Directory to write the figure and manifest")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    repo_root = infer_repo_root()

    nyt_path = resolve_input(repo_root, DYNAMIC_NYT_FILE, args.nyt_file)
    twfe_path = resolve_input(repo_root, DYNAMIC_TWFE_FILE, args.twfe_file)

    data_path = nyt_path if args.dataset == "nyt" else twfe_path
    df = pd.read_csv(data_path, sep="\t")

    sub = df[
        (df["reform"] == "haifa_priv")
        & (df["target"].isin(TARGET_ORDER))
        & (df["spec_name"] == args.spec)
        & (df["table_group"].isin(["privatization", "privatization_diag"]))
    ].copy()

    sub = keep_supported_bins(sub)

    if sub.empty:
        raise RuntimeError("No rows found for the requested privatization dynamic plot selection.")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    plot_series(ax, sub)

    dataset_label = "NYT" if args.dataset == "nyt" else "TWFE"
    spec_label = "Ctrls+Tr" if args.spec == "porttr" else "Baseline"

    ax.legend(frameon=False, fontsize=10.5, loc="upper left")

    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.10, right=0.98)

    outfile = outdir / f"figure_model1a_privatization_dynamic_lp_{args.dataset}_{args.spec}.png"
    fig.savefig(outfile, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "description": "Combined Haifa privatization dynamic ln(LP) plot",
        "input_file_used": str(data_path),
        "dataset": args.dataset,
        "spec_name": args.spec,
        "reform": "haifa_priv",
        "targets": TARGET_ORDER,
        "output_file": str(outfile),
        "kept_bin_labels": sorted(sub["bin_label"].unique().tolist()),
    }

    manifest_path = outdir / f"figure_model1a_privatization_dynamic_lp_{args.dataset}_{args.spec}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Input used: {data_path}")
    print(f"Wrote:      {outfile}")
    print(f"Wrote:      {manifest_path}")


if __name__ == "__main__":
    main()

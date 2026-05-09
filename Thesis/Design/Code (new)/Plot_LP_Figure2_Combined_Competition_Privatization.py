#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot_LP_Figure2_Combined_Competition_Privatization.py

Combined dynamic LP figure:
Haifa competition vs. Haifa privatization

Recommended use:
- same target for both lines, ideally Haifa aggregate
- same spec for both lines
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import matplotlib.pyplot as plt
import pandas as pd


PLOT_BINS = [
    "pre_8_5",
    "pre_4_2",
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
    "y2_q5_8",
    "y3_q9_12",
]

BIN_LABELS = {
    "pre_8_5": "-8 to -5",
    "pre_4_2": "-4 to -2",
    "q0": "0",
    "q1": "1",
    "q2": "2",
    "q3": "3",
    "q4": "4",
    "y2_q5_8": "5 to 8",
    "y3_q9_12": "9 to 12",
}


def infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent] + list(here.parents):
        if (candidate / "Data").exists() and (candidate / "Design").exists():
            return candidate
    return Path.cwd().resolve()


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_input_path(repo_root: Path, explicit_path: str | None) -> Path:
    if explicit_path is not None:
        p = Path(explicit_path)
        return p if p.is_absolute() else (repo_root / p)

    candidates = [
        repo_root / "Design" / "Output (new)" / "Model_1A_v8_2" / "model1a_q_dynamic_betas_twfe.tsv",
        repo_root / "Design" / "Output (new)" / "Model_1A_v8_1" / "model1a_q_dynamic_betas_twfe.tsv",
        repo_root / "model1a_q_dynamic_betas_twfe.tsv",
        Path("/mnt/data/model1a_q_dynamic_betas_twfe.tsv"),
    ]

    direct = find_first_existing(candidates)
    if direct is not None:
        return direct

    matches = sorted(repo_root.rglob("model1a_q_dynamic_betas_twfe.tsv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not find model1a_q_dynamic_betas_twfe.tsv. Pass it with --dynamic."
    )


def resolve_outdir(repo_root: Path, explicit_outdir: str | None) -> Path:
    if explicit_outdir is not None:
        p = Path(explicit_outdir)
        return p if p.is_absolute() else (repo_root / p)
    return repo_root / "Design" / "Output (new)" / "LP" / "Visuals"


def load_dynamic_betas(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = {
        "table_group",
        "target",
        "spec_name",
        "bin_label",
        "beta",
        "se",
        "n_event_obs",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dynamic beta file missing columns: {sorted(missing)}")

    df = df.copy()
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["se"] = pd.to_numeric(df["se"], errors="coerce")
    df["n_event_obs"] = pd.to_numeric(df["n_event_obs"], errors="coerce").fillna(0).astype(int)

    return df


def prep_subset(
    df: pd.DataFrame,
    table_group: str,
    target: str,
    spec_name: str,
) -> pd.DataFrame:
    sub = df[
        (df["table_group"] == table_group)
        & (df["target"] == target)
        & (df["spec_name"] == spec_name)
        & (df["bin_label"].isin(PLOT_BINS))
    ].copy()

    sub["bin_order"] = sub["bin_label"].map({b: i for i, b in enumerate(PLOT_BINS)})
    sub = sub.sort_values("bin_order").reset_index(drop=True)

    sub = sub[
        ~(
            (sub["n_event_obs"] == 0)
            & (sub["beta"].fillna(0) == 0)
            & (sub["se"].fillna(0) == 0)
        )
    ].copy()

    sub["x"] = sub["bin_label"].map({b: i for i, b in enumerate(PLOT_BINS)})
    sub["ci_low"] = sub["beta"] - 1.96 * sub["se"]
    sub["ci_high"] = sub["beta"] + 1.96 * sub["se"]

    return sub


def set_y_limits(ax: plt.Axes, plotted: list[pd.DataFrame]) -> None:
    vals = []
    for sub in plotted:
        vals.extend(sub["ci_low"].tolist())
        vals.extend(sub["ci_high"].tolist())

    if not vals:
        return

    ymin = min(vals)
    ymax = max(vals)
    span = ymax - ymin
    pad = 0.10 * span if span > 0 else 0.10

    lower = math.floor((ymin - pad) * 100) / 100
    upper = math.ceil((ymax + pad) * 100) / 100
    ax.set_ylim(lower, upper)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamic", default=None, help="Path to model1a_q_dynamic_betas_twfe.tsv")
    ap.add_argument("--outdir", default=None, help="Output directory")
    ap.add_argument("--spec", default="porttr", choices=["baseline", "porttr"])
    ap.add_argument(
        "--target",
        default="Haifa-Aggregate",
        help="Common target used for both competition and privatization lines. Recommended: Haifa-Aggregate",
    )
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    repo_root = infer_repo_root()
    dynamic_path = resolve_input_path(repo_root, args.dynamic)
    outdir = resolve_outdir(repo_root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dynamic_betas(dynamic_path)

    comp = prep_subset(
        df=df,
        table_group="competition",
        target=args.target,
        spec_name=args.spec,
    )

    priv = prep_subset(
        df=df,
        table_group="privatization",
        target=args.target,
        spec_name=args.spec,
    )

    if comp.empty or priv.empty:
        print("\nNo rows found for one or both lines with the current settings.")
        print("Try checking the available target/spec combinations in the TSV.")
        print(f"Requested target: {args.target}")
        print(f"Requested spec:   {args.spec}\n")

        combos = (
            df[["table_group", "target", "spec_name"]]
            .drop_duplicates()
            .sort_values(["table_group", "target", "spec_name"])
        )
        print(combos.to_string(index=False))
        raise ValueError("Could not build combined figure with current filters.")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(
        comp["x"],
        comp["beta"],
        yerr=1.96 * comp["se"],
        fmt="o-",
        linewidth=2.2,
        markersize=5.5,
        capsize=3,
        label="Competition",
    )

    ax.errorbar(
        priv["x"],
        priv["beta"],
        yerr=1.96 * priv["se"],
        fmt="o-",
        linewidth=2.2,
        markersize=5.5,
        capsize=3,
        label="Privatization",
    )

    ax.set_title("Haifa dynamic LP estimates: competition vs. privatization", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("Event time (quarters relative to reform)")
    ax.set_ylabel("Estimated effect on ln(LP)")

    ax.set_xticks(range(len(PLOT_BINS)))
    ax.set_xticklabels([BIN_LABELS[b] for b in PLOT_BINS])

    ax.axhline(0, color="k", linewidth=1.0)
    ax.axvline(1.5, linestyle="--", linewidth=1.0, color="0.35", alpha=0.8)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    set_y_limits(ax, [comp, priv])

    ymin, ymax = ax.get_ylim()
    ax.text(
        1.5,
        ymax * 0.98 if ymax > 0 else ymax,
        "reform",
        rotation=90,
        va="top",
        ha="right",
        fontsize=11,
    )

    fig.tight_layout()

    outpath = outdir / "figure2_lp_combined_comp_priv_dynamic.png"
    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figure": 2,
        "description": "Combined dynamic LP figure: Haifa competition vs. privatization",
        "input_dynamic_betas": str(dynamic_path),
        "spec_name": args.spec,
        "target": args.target,
        "output_file": str(outpath),
    }

    with open(outdir / "figure2_lp_combined_comp_priv_dynamic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Read:  {dynamic_path}")
    print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()
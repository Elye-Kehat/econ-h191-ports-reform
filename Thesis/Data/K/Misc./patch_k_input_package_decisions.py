#!/usr/bin/env python3
"""
patch_k_input_package_decisions.py

Apply user-approved decision rules to an existing K input package.

Expected existing files in --indir:
- k_monthly_redesign_master_v4_repaired.tsv
- k_entity_rules.tsv
- k_entity_year_anchors.tsv
- k_dep_lookup.tsv
- k_dated_events.tsv
- k_annual_class_pools.tsv
- k_month_shares.tsv
- k_pool_identity_diagnostics.tsv
- k_package_qa_summary.tsv

What this patch does
--------------------
1) Leaves missing anchors as missing. They are treated as acceptable for now.
2) Treats annual totals as authoritative and proportionally rescales
   HPC class pools within affected years.
3) Excludes two problematic HPC dated-event rows:
     - HPC_Sea_Department_Transfer_2020
     - HPC_Waterfront_Relocation_2024
4) Keeps right-of-use rows untouched.
5) Keeps IPC explicit service-entry information in the anchor table only.
   It is intentionally NOT duplicated into class pools.
6) Keeps SIPG simple: annual anchors + annual pools + monthly shares.
7) Rebuilds diagnostics and QA summary.

Usage
-----
python patch_k_input_package_decisions.py \
  --indir "/path/to/k_input_package" \
  --outdir "/path/to/k_input_package_patched"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


EXCLUDE_PROJECT_IDS = {
    "HPC_Sea_Department_Transfer_2020",
    "HPC_Waterfront_Relocation_2024",
}

SHARE_TOL = 1e-8
IDENTITY_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True, help="Existing package directory")
    p.add_argument("--outdir", required=True, help="Patched package output directory")
    return p.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ensure_columns(df: pd.DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def patch_dated_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ensure_columns(df, ["project_id"], "k_dated_events.tsv")
    out = df.loc[~df["project_id"].isin(EXCLUDE_PROJECT_IDS)].copy()
    return out.reset_index(drop=True)


def patch_class_pools(df_pools: pd.DataFrame, df_anchors: pd.DataFrame) -> pd.DataFrame:
    """
    Treat annual totals as authoritative.
    For HPC years with known annual totals, proportionally rescale all
    non-background class-pool rows so that:

        sum(non-background pools) + background_sum = annual_total

    This leaves years with missing totals untouched.
    """
    pools = df_pools.copy()
    anchors = df_anchors.copy()

    ensure_columns(
        pools,
        ["entity", "year", "pool_type", "annual_amount_kNIS"],
        "k_annual_class_pools.tsv",
    )
    ensure_columns(
        anchors,
        ["entity", "year", "I_annual_total_kNIS"],
        "k_entity_year_anchors.tsv",
    )

    pools["year"] = to_num(pools["year"]).astype("Int64")
    pools["annual_amount_kNIS"] = to_num(pools["annual_amount_kNIS"])
    anchors["year"] = to_num(anchors["year"]).astype("Int64")
    anchors["I_annual_total_kNIS"] = to_num(anchors["I_annual_total_kNIS"])

    # Only patch HPC, only where annual total is available
    anchors_hpc = anchors.loc[anchors["entity"] == "HPC", ["year", "I_annual_total_kNIS"]].dropna()

    for _, ar in anchors_hpc.iterrows():
        year = int(ar["year"])
        annual_total = float(ar["I_annual_total_kNIS"])

        mask_year = (pools["entity"] == "HPC") & (pools["year"] == year)
        if not mask_year.any():
            continue

        mask_bg = mask_year & (pools["pool_type"] == "background")
        mask_nonbg = mask_year & (pools["pool_type"] != "background")

        bg_sum = pools.loc[mask_bg, "annual_amount_kNIS"].fillna(0.0).sum()
        nonbg_sum = pools.loc[mask_nonbg, "annual_amount_kNIS"].fillna(0.0).sum()

        target_nonbg = annual_total - bg_sum

        if pd.isna(nonbg_sum) or nonbg_sum <= 0:
            continue

        if abs(nonbg_sum - target_nonbg) <= IDENTITY_TOL:
            continue

        scale = target_nonbg / nonbg_sum
        pools.loc[mask_nonbg, "annual_amount_kNIS"] = (
            pools.loc[mask_nonbg, "annual_amount_kNIS"].fillna(0.0) * scale
        )

        # Pretty formatting consistency
        pools.loc[mask_nonbg, "annual_amount_kNIS"] = pools.loc[mask_nonbg, "annual_amount_kNIS"].round(6)

        # Annotate notes
        note_mask = mask_nonbg
        pools.loc[note_mask, "notes"] = (
            pools.loc[note_mask, "notes"].fillna("").astype(str)
            + " | proportionally rescaled to annual authoritative total"
        ).str.strip(" |")

    return pools


def rebuild_pool_identity_diagnostics(df_pools: pd.DataFrame, df_anchors: pd.DataFrame) -> pd.DataFrame:
    pools = df_pools.copy()
    anchors = df_anchors.copy()

    pools["year"] = to_num(pools["year"]).astype("Int64")
    pools["annual_amount_kNIS"] = to_num(pools["annual_amount_kNIS"])
    anchors["year"] = to_num(anchors["year"]).astype("Int64")
    anchors["I_annual_total_kNIS"] = to_num(anchors["I_annual_total_kNIS"])

    pool_sum = (
        pools.groupby(["entity", "year", "pool_type"], as_index=False)["annual_amount_kNIS"]
        .sum()
        .pivot_table(index=["entity", "year"], columns="pool_type", values="annual_amount_kNIS", aggfunc="first")
        .reset_index()
    )
    pool_sum.columns.name = None

    out = anchors[["entity", "year", "I_annual_total_kNIS"]].merge(pool_sum, on=["entity", "year"], how="left")

    for c in ["mapped_undated", "background", "lease_addition", "service_entry_undated"]:
        if c not in out.columns:
            out[c] = np.nan

    out["mapped_undated_sum_kNIS"] = out["mapped_undated"].fillna(0.0)
    out["background_kNIS"] = out["background"].fillna(0.0)
    out["lease_addition_kNIS"] = out["lease_addition"].fillna(0.0)
    out["service_entry_undated_kNIS"] = out["service_entry_undated"].fillna(0.0)

    # Identity used here intentionally excludes any informational anchor-only service-entry field.
    out["pool_total_kNIS"] = (
        out["mapped_undated_sum_kNIS"]
        + out["background_kNIS"]
        + out["lease_addition_kNIS"]
        + out["service_entry_undated_kNIS"]
    )

    out["residual_kNIS"] = out["I_annual_total_kNIS"] - out["pool_total_kNIS"]

    out = out.rename(columns={"I_annual_total_kNIS": "annual_total_kNIS"})
    out = out[
        [
            "entity",
            "year",
            "annual_total_kNIS",
            "mapped_undated_sum_kNIS",
            "lease_addition_kNIS",
            "service_entry_undated_kNIS",
            "background_kNIS",
            "pool_total_kNIS",
            "residual_kNIS",
        ]
    ].sort_values(["entity", "year"]).reset_index(drop=True)

    return out


def rebuild_qa_summary(
    df_master: pd.DataFrame,
    df_rules: pd.DataFrame,
    df_anchors: pd.DataFrame,
    df_dep: pd.DataFrame,
    df_events: pd.DataFrame,
    df_pools: pd.DataFrame,
    df_shares: pd.DataFrame,
    df_diag: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(check_name, entity_scope, year_scope, value, status, notes):
        rows.append(
            {
                "check_name": check_name,
                "entity_scope": entity_scope,
                "year_scope": year_scope,
                "value": value,
                "status": status,
                "notes": notes,
            }
        )

    # counts
    add("master_v4_repaired_rows", "global", "", len(df_master), "info", "Repaired v4 raw master row count")
    dup_count = int(df_master["record_id"].duplicated().sum()) if "record_id" in df_master.columns else np.nan
    add("master_v4_duplicate_record_ids", "global", "", dup_count, "pass" if dup_count == 0 else "fail", "Duplicate record IDs should be zero")
    add("entity_rules_rows", "global", "", len(df_rules), "info", "Entity rules row count")
    add("anchors_rows", "global", "", len(df_anchors), "info", "Entity-year anchors row count")
    add("dep_rows", "global", "", len(df_dep), "info", "Depreciation lookup row count")
    add("dated_event_rows", "global", "", len(df_events), "info", "Dated event row count after exclusions")
    add("class_pool_rows", "global", "", len(df_pools), "info", "Annual class pools row count")
    add("month_shares_rows", "global", "", len(df_shares), "info", "Month shares row count")

    # month shares
    shares = df_shares.copy()
    ensure_columns(shares, ["entity", "year", "pool_type", "share"], "k_month_shares.tsv")
    shares["year"] = to_num(shares["year"]).astype("Int64")
    shares["share"] = to_num(shares["share"])
    share_sum = shares.groupby(["entity", "year", "pool_type"], as_index=False)["share"].sum()
    share_fails = int((share_sum["share"] - 1.0).abs().gt(SHARE_TOL).sum())
    add("month_share_sum_failures", "global", "", share_fails, "pass" if share_fails == 0 else "fail", "Entity-year-pool share sums not equal to 1")

    # missing anchors
    anchors = df_anchors.copy()
    anchors["I_annual_total_kNIS"] = to_num(anchors["I_annual_total_kNIS"])
    missing_I = int(anchors["I_annual_total_kNIS"].isna().sum())
    add("anchors_missing_I_total", "global", "", missing_I, "info", "Missing annual total values intentionally left for interpolation-stage handling")

    anchors["K_anchor_dec_kNIS"] = to_num(anchors["K_anchor_dec_kNIS"])
    missing_K = int(anchors["K_anchor_dec_kNIS"].isna().sum())
    add("anchors_missing_K_anchor", "global", "", missing_K, "info", "Missing annual anchor values intentionally left for interpolation-stage handling")

    # dep coverage for used classes
    dep = df_dep.copy()
    pools = df_pools.copy()
    ensure_columns(dep, ["entity", "asset_class_std"], "k_dep_lookup.tsv")
    ensure_columns(pools, ["entity", "asset_class_std"], "k_annual_class_pools.tsv")
    used_pairs = set(
        pools.loc[pools["annual_amount_kNIS"].astype(str).ne("0") & pools["asset_class_std"].notna(), ["entity", "asset_class_std"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    dep_pairs = set(dep[["entity", "asset_class_std"]].drop_duplicates().itertuples(index=False, name=None))
    missing_pairs = sorted(used_pairs - dep_pairs)
    add("missing_dep_classes_count", "global", "", len(missing_pairs), "pass" if len(missing_pairs) == 0 else "fail", f"Missing dep rows for used classes: {missing_pairs}")

    # pool identity residuals, but only for rows with observed annual totals
    diag = df_diag.copy()
    diag["annual_total_kNIS"] = to_num(diag["annual_total_kNIS"])
    diag["residual_kNIS"] = to_num(diag["residual_kNIS"])
    diag_known = diag.loc[diag["annual_total_kNIS"].notna()].copy()
    bad_resid = int(diag_known["residual_kNIS"].abs().gt(IDENTITY_TOL).sum())
    add("pool_identity_residual_failures", "global", "", bad_resid, "pass" if bad_resid == 0 else "warn", "Rows with known annual totals where pool identity residual is non-zero")

    # explicit record of the excluded events
    add(
        "excluded_problematic_events",
        "HPC",
        "",
        len(EXCLUDE_PROJECT_IDS),
        "info",
        "Excluded from dated events: " + ", ".join(sorted(EXCLUDE_PROJECT_IDS)),
    )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "master": indir / "k_monthly_redesign_master_v4_repaired.tsv",
        "rules": indir / "k_entity_rules.tsv",
        "anchors": indir / "k_entity_year_anchors.tsv",
        "dep": indir / "k_dep_lookup.tsv",
        "events": indir / "k_dated_events.tsv",
        "pools": indir / "k_annual_class_pools.tsv",
        "shares": indir / "k_month_shares.tsv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file in input package: {path}")

    master = read_tsv(paths["master"])
    rules = read_tsv(paths["rules"])
    anchors = read_tsv(paths["anchors"])
    dep = read_tsv(paths["dep"])
    events = read_tsv(paths["events"])
    pools = read_tsv(paths["pools"])
    shares = read_tsv(paths["shares"])

    # Patch tables
    events_p = patch_dated_events(events)
    pools_p = patch_class_pools(pools, anchors)
    diag_p = rebuild_pool_identity_diagnostics(pools_p, anchors)
    qa_p = rebuild_qa_summary(master, rules, anchors, dep, events_p, pools_p, shares, diag_p)

    # Write everything through. Master, rules, anchors, dep, shares unchanged.
    write_tsv(master, outdir / "k_monthly_redesign_master_v4_repaired.tsv")
    write_tsv(rules, outdir / "k_entity_rules.tsv")
    write_tsv(anchors, outdir / "k_entity_year_anchors.tsv")
    write_tsv(dep, outdir / "k_dep_lookup.tsv")
    write_tsv(events_p, outdir / "k_dated_events.tsv")
    write_tsv(pools_p, outdir / "k_annual_class_pools.tsv")
    write_tsv(shares, outdir / "k_month_shares.tsv")
    write_tsv(diag_p, outdir / "k_pool_identity_diagnostics.tsv")
    write_tsv(qa_p, outdir / "k_package_qa_summary.tsv")

    manifest = {
        "input_dir": str(indir),
        "output_dir": str(outdir),
        "decisions_applied": {
            "missing_anchors_left_for_interpolation_stage": True,
            "annual_totals_authoritative_for_hpc_pool_scaling": True,
            "excluded_problematic_hpc_dated_events": sorted(EXCLUDE_PROJECT_IDS),
            "right_of_use_assets_left_in_raw_inputs": True,
            "ipc_explicit_service_entry_kept_anchor_only": True,
            "sipg_left_simple_for_interpolation_stage": True,
            "fallback_choice_deferred_to_interpolation_stage": True,
        },
    }
    with open(outdir / "_patch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Wrote patched package to: {outdir}")


if __name__ == "__main__":
    main()

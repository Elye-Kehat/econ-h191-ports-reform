#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY_TOL = 1e-6
SHARE_TOL = 1e-8

EXCLUDE_PROJECT_IDS = {
    "HPC_Sea_Department_Transfer_2020",
    "HPC_Waterfront_Relocation_2024",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True, help="Input package directory")
    p.add_argument("--outdir", required=True, help="Output package directory")
    return p.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def annual_to_monthly_dep(delta_a: float) -> float:
    return 1.0 - (1.0 - float(delta_a)) ** (1.0 / 12.0)


def exclude_problematic_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    if "project_id" in events.columns:
        events = events.loc[~events["project_id"].isin(EXCLUDE_PROJECT_IDS)].copy()
    return events.reset_index(drop=True)


def add_ipc_mixed_dep_row(dep: pd.DataFrame) -> pd.DataFrame:
    dep = dep.copy()

    if ((dep["entity"] == "IPC") & (dep["asset_class_std"] == "mixed")).any():
        return dep

    ipc = dep.loc[dep["entity"] == "IPC"].copy()
    ipc["dep_rate_annual"] = to_num(ipc["dep_rate_annual"])

    if ipc["dep_rate_annual"].notna().any():
        fallback_annual = float(ipc["dep_rate_annual"].dropna().mean())
    else:
        fallback_annual = 0.06

    row = {
        "entity": "IPC",
        "asset_class_std": "mixed",
        "dep_rate_annual": f"{fallback_annual:.6f}",
        "dep_rate_monthly": f"{annual_to_monthly_dep(fallback_annual):.6f}",
        "dep_method": "fallback_average",
        "is_fallback": "1",
        "fallback_group": "mixed",
        "source_id": "IPC_MIXED_FALLBACK_GENERATED_V2",
        "notes": "Generated fallback row for IPC background residual class using average across IPC categories",
    }

    dep = pd.concat([dep, pd.DataFrame([row])], ignore_index=True)
    return dep.sort_values(["entity", "asset_class_std", "source_id"]).reset_index(drop=True)


def scale_hpc_pools_to_annual(pools: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    pools = pools.copy()
    anchors = anchors.copy()

    pools["year"] = to_num(pools["year"]).astype("Int64")
    pools["annual_amount_kNIS"] = to_num(pools["annual_amount_kNIS"])
    anchors["year"] = to_num(anchors["year"]).astype("Int64")
    anchors["I_annual_total_kNIS"] = to_num(anchors["I_annual_total_kNIS"])

    hpc_anchors = anchors.loc[
        (anchors["entity"] == "HPC") & anchors["I_annual_total_kNIS"].notna(),
        ["year", "I_annual_total_kNIS"],
    ]

    for _, ar in hpc_anchors.iterrows():
        year = int(ar["year"])
        annual_total = float(ar["I_annual_total_kNIS"])

        mask = (pools["entity"] == "HPC") & (pools["year"] == year)
        if not mask.any():
            continue

        bg_mask = mask & (pools["pool_type"] == "background")
        nonbg_mask = mask & (pools["pool_type"] != "background")

        bg_sum = pools.loc[bg_mask, "annual_amount_kNIS"].fillna(0.0).sum()
        nonbg_sum = pools.loc[nonbg_mask, "annual_amount_kNIS"].fillna(0.0).sum()

        target_nonbg = annual_total - bg_sum

        if nonbg_sum <= 0:
            continue
        if abs(nonbg_sum - target_nonbg) <= IDENTITY_TOL:
            continue

        scale = target_nonbg / nonbg_sum

        pools.loc[nonbg_mask, "annual_amount_kNIS"] = (
            pools.loc[nonbg_mask, "annual_amount_kNIS"].fillna(0.0) * scale
        ).round(6)

        pools.loc[nonbg_mask, "notes"] = (
            pools.loc[nonbg_mask, "notes"].fillna("").astype(str)
            + " | proportionally rescaled to annual authoritative total"
        ).str.strip(" |")

    return pools


def add_missing_background_rows(pools: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    pools = pools.copy()
    anchors = anchors.copy()

    pools["year"] = to_num(pools["year"]).astype("Int64")
    pools["annual_amount_kNIS"] = to_num(pools["annual_amount_kNIS"])
    anchors["year"] = to_num(anchors["year"]).astype("Int64")
    anchors["I_annual_total_kNIS"] = to_num(anchors["I_annual_total_kNIS"])

    current_totals = (
        pools.groupby(["entity", "year"], as_index=False)["annual_amount_kNIS"]
        .sum()
        .rename(columns={"annual_amount_kNIS": "pool_total_kNIS"})
    )

    merged = anchors[["entity", "year", "I_annual_total_kNIS"]].merge(
        current_totals, on=["entity", "year"], how="left"
    )
    merged["pool_total_kNIS"] = merged["pool_total_kNIS"].fillna(0.0)
    merged["residual_kNIS"] = merged["I_annual_total_kNIS"] - merged["pool_total_kNIS"]

    rows_to_add = []

    for _, r in merged.iterrows():
        entity = r["entity"]
        year = r["year"]
        annual_total = r["I_annual_total_kNIS"]
        residual = r["residual_kNIS"]

        if pd.isna(annual_total) or pd.isna(residual):
            continue
        if abs(float(residual)) <= IDENTITY_TOL:
            continue

        if entity == "IPC":
            asset_class_std = "mixed"
            source_id = "IPC_BACKGROUND_GENERATED_V2"
        elif entity == "SIPG":
            asset_class_std = "unknown_bayport_class"
            source_id = "SIPG_BACKGROUND_GENERATED_V2"
        else:
            continue

        mask_existing = (
            (pools["entity"] == entity)
            & (pools["year"] == year)
            & (pools["pool_type"] == "background")
            & (pools["asset_class_std"] == asset_class_std)
        )

        if mask_existing.any():
            pools.loc[mask_existing, "annual_amount_kNIS"] = (
                pools.loc[mask_existing, "annual_amount_kNIS"].fillna(0.0) + float(residual)
            ).round(6)
            pools.loc[mask_existing, "notes"] = (
                pools.loc[mask_existing, "notes"].fillna("").astype(str)
                + " | topped up by generated background residual"
            ).str.strip(" |")
        else:
            rows_to_add.append(
                {
                    "entity": entity,
                    "year": str(int(year)),
                    "asset_class_std": asset_class_std,
                    "pool_type": "background",
                    "annual_amount_kNIS": f"{float(residual):.6f}",
                    "include_in_productive_K": "1",
                    "allocation_priority": "3",
                    "source_id": source_id,
                    "notes": "Generated background residual row to reconcile known annual total",
                }
            )

    if rows_to_add:
        pools = pd.concat([pools, pd.DataFrame(rows_to_add)], ignore_index=True)

    return pools.sort_values(
        ["entity", "year", "pool_type", "asset_class_std", "source_id"]
    ).reset_index(drop=True)


def rebuild_pool_identity_diagnostics(pools: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    pools = pools.copy()
    anchors = anchors.copy()

    pools["year"] = to_num(pools["year"]).astype("Int64")
    pools["annual_amount_kNIS"] = to_num(pools["annual_amount_kNIS"])
    anchors["year"] = to_num(anchors["year"]).astype("Int64")
    anchors["I_annual_total_kNIS"] = to_num(anchors["I_annual_total_kNIS"])

    pivot = (
        pools.groupby(["entity", "year", "pool_type"], as_index=False)["annual_amount_kNIS"]
        .sum()
        .pivot_table(
            index=["entity", "year"],
            columns="pool_type",
            values="annual_amount_kNIS",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns.name = None

    out = anchors[["entity", "year", "I_annual_total_kNIS"]].merge(
        pivot, on=["entity", "year"], how="left"
    )

    for c in ["mapped_undated", "background", "lease_addition", "service_entry_undated"]:
        if c not in out.columns:
            out[c] = np.nan

    out["mapped_undated_sum_kNIS"] = out["mapped_undated"].fillna(0.0)
    out["lease_addition_kNIS"] = out["lease_addition"].fillna(0.0)
    out["service_entry_undated_kNIS"] = out["service_entry_undated"].fillna(0.0)
    out["background_kNIS"] = out["background"].fillna(0.0)

    out["pool_total_kNIS"] = (
        out["mapped_undated_sum_kNIS"]
        + out["lease_addition_kNIS"]
        + out["service_entry_undated_kNIS"]
        + out["background_kNIS"]
    )

    out["residual_kNIS"] = out["I_annual_total_kNIS"] - out["pool_total_kNIS"]

    out = out.rename(columns={"I_annual_total_kNIS": "annual_total_kNIS"})

    return out[
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


def rebuild_qa(master, rules, anchors, dep, events, pools, shares, diag) -> pd.DataFrame:
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

    add("master_v4_repaired_rows", "global", "", len(master), "info", "Repaired v4 raw master row count")
    dup = int(master["record_id"].duplicated().sum()) if "record_id" in master.columns else np.nan
    add("master_v4_duplicate_record_ids", "global", "", dup, "pass" if dup == 0 else "fail", "Duplicate record IDs should be zero")
    add("entity_rules_rows", "global", "", len(rules), "info", "Entity rules row count")
    add("anchors_rows", "global", "", len(anchors), "info", "Entity-year anchors row count")
    add("dep_rows", "global", "", len(dep), "info", "Depreciation lookup row count")
    add("dated_event_rows", "global", "", len(events), "info", "Dated event row count")
    add("class_pool_rows", "global", "", len(pools), "info", "Annual class pools row count")
    add("month_shares_rows", "global", "", len(shares), "info", "Month shares row count")

    shares = shares.copy()
    shares["year"] = to_num(shares["year"]).astype("Int64")
    shares["share"] = to_num(shares["share"])
    ss = shares.groupby(["entity", "year", "pool_type"], as_index=False)["share"].sum()
    sf = int((ss["share"] - 1.0).abs().gt(SHARE_TOL).sum())
    add("month_share_sum_failures", "global", "", sf, "pass" if sf == 0 else "fail", "Entity-year-pool share sums not equal to 1")

    anchors = anchors.copy()
    anchors["I_annual_total_kNIS"] = to_num(anchors["I_annual_total_kNIS"])
    anchors["K_anchor_dec_kNIS"] = to_num(anchors["K_anchor_dec_kNIS"])
    add("anchors_missing_I_total", "global", "", int(anchors["I_annual_total_kNIS"].isna().sum()), "info", "Missing annual totals intentionally deferred to interpolation stage")
    add("anchors_missing_K_anchor", "global", "", int(anchors["K_anchor_dec_kNIS"].isna().sum()), "info", "Missing annual anchors intentionally deferred to interpolation stage")

    dep_pairs = set(dep[["entity", "asset_class_std"]].drop_duplicates().itertuples(index=False, name=None))

    pools_n = pools.copy()
    pools_n["annual_amount_kNIS"] = to_num(pools_n["annual_amount_kNIS"])
    used_pairs = set(
        pools_n.loc[
            pools_n["annual_amount_kNIS"].fillna(0.0).abs().gt(IDENTITY_TOL),
            ["entity", "asset_class_std"],
        ].drop_duplicates().itertuples(index=False, name=None)
    )
    missing_pairs = sorted(used_pairs - dep_pairs)
    add("missing_dep_classes_count", "global", "", len(missing_pairs), "pass" if len(missing_pairs) == 0 else "fail", f"Missing dep rows for used classes: {missing_pairs}")

    diag = diag.copy()
    diag["annual_total_kNIS"] = to_num(diag["annual_total_kNIS"])
    diag["residual_kNIS"] = to_num(diag["residual_kNIS"])
    known = diag.loc[diag["annual_total_kNIS"].notna()].copy()
    bad = int(known["residual_kNIS"].abs().gt(IDENTITY_TOL).sum())
    add("pool_identity_residual_failures", "global", "", bad, "pass" if bad == 0 else "warn", "Rows with known annual totals where pool identity residual is non-zero")

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = {
        "master": indir / "k_monthly_redesign_master_v4_repaired.tsv",
        "rules": indir / "k_entity_rules.tsv",
        "anchors": indir / "k_entity_year_anchors.tsv",
        "dep": indir / "k_dep_lookup.tsv",
        "events": indir / "k_dated_events.tsv",
        "pools": indir / "k_annual_class_pools.tsv",
        "shares": indir / "k_month_shares.tsv",
    }

    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    master = read_tsv(files["master"])
    rules = read_tsv(files["rules"])
    anchors = read_tsv(files["anchors"])
    dep = read_tsv(files["dep"])
    events = read_tsv(files["events"])
    pools = read_tsv(files["pools"])
    shares = read_tsv(files["shares"])

    events2 = exclude_problematic_events(events)
    dep2 = add_ipc_mixed_dep_row(dep)
    pools2 = scale_hpc_pools_to_annual(pools, anchors)
    pools2 = add_missing_background_rows(pools2, anchors)
    diag2 = rebuild_pool_identity_diagnostics(pools2, anchors)
    qa2 = rebuild_qa(master, rules, anchors, dep2, events2, pools2, shares, diag2)

    write_tsv(master, outdir / "k_monthly_redesign_master_v4_repaired.tsv")
    write_tsv(rules, outdir / "k_entity_rules.tsv")
    write_tsv(anchors, outdir / "k_entity_year_anchors.tsv")
    write_tsv(dep2, outdir / "k_dep_lookup.tsv")
    write_tsv(events2, outdir / "k_dated_events.tsv")
    write_tsv(pools2, outdir / "k_annual_class_pools.tsv")
    write_tsv(shares, outdir / "k_month_shares.tsv")
    write_tsv(diag2, outdir / "k_pool_identity_diagnostics.tsv")
    write_tsv(qa2, outdir / "k_package_qa_summary.tsv")

    manifest = {
        "input_dir": str(indir),
        "output_dir": str(outdir),
        "fixes_applied": [
            "exclude_problematic_hpc_events",
            "scale_hpc_pools_to_authoritative_annual_totals",
            "add_ipc_mixed_fallback_dep_row",
            "add_missing_ipc_and_sipg_background_rows",
        ],
    }
    with open(outdir / "_finalize_manifest_v2.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Wrote finalized package to: {outdir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
prepare_k_inputs_v2.py

Purpose
-------
1) Safely append hpc_reports_complement_extract.csv into k_monthly_redesign_master_v2.tsv
   and write k_monthly_redesign_master_v3.tsv.
2) Rebuild a stronger first-pass set of processed K-input tables from the updated raw archive.
3) Handle the metric-name heterogeneity that exists across HPC, IPC, and SIPG rows.

Key upgrades vs. prior version
------------------------------
- Depreciation aliases:
    annual_depr_rate
    annual_depr_rate_pct_min
    annual_depr_rate_pct_max
  are now mapped into the standardized depreciation fields.
- Annual flow aliases:
    purchases
    additons
    additions
    new_leases_additions
    realizations_derecognitions_transfers
    depreciation_expense
    nbv_close
    transfer_to_ppe
    commissioned_assets_or_transfers
  are now recognized.
- Service-entry pool logic now prefers explicit service-entry signals where they exist:
    transfer_to_ppe
    commissioned_assets_or_transfers
  and otherwise falls back to observed spending net of allocated WIP.
- Event extraction now also includes row_type = event.

Usage
-----
python prepare_k_inputs_v2.py \
  --master "/path/to/k_monthly_redesign_master_v2.tsv" \
  --hpc "/path/to/hpc_reports_complement_extract.csv" \
  --outdir "/path/to/k_prepared"

Notes
-----
This script is conservative and append-only on the raw side.
It writes a new raw archive and new processed files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_MASTER_COLUMNS = [
    "record_id",
    "entity",
    "year",
    "source_group",
    "source_file",
    "source_doc",
    "source_page",
    "source_section",
    "row_type",
    "asset_class_raw",
    "asset_class_std",
    "project_id",
    "project_name",
    "metric",
    "value_num",
    "value_text",
    "unit",
    "currency",
    "date_precision",
    "event_date",
    "event_year",
    "event_month",
    "confidence",
    "include_in_k",
    "include_in_productive_k",
    "needs_monthly_timing",
    "notes",
]

YES_VALUES = {"yes", "y", "true", "1"}
NO_VALUES = {"no", "n", "false", "0"}

# Depreciation aliases
DEP_METRIC_ALIASES = {
    "annual_depreciation_rate_main_pct": [
        "annual_depreciation_rate_main_pct",
        "annual_depr_rate",
    ],
    "annual_depreciation_rate_low_pct": [
        "annual_depreciation_rate_low_pct",
        "annual_depr_rate_pct_min",
    ],
    "annual_depreciation_rate_high_pct": [
        "annual_depreciation_rate_high_pct",
        "annual_depr_rate_pct_max",
    ],
    "depreciation_rule_text": [
        "depreciation_rule_text",
    ],
}

# PPE rollforward / annual-flow aliases
FLOW_ALIASES = {
    "gross_additions_spend": [
        "cost_additions",
        "additions",
        "additons",
        "purchases",
    ],
    "gross_additions_leases": [
        "cost_new_leases",
        "new_leases_additions",
    ],
    "explicit_service_entry": [
        "transfer_to_ppe",
        "commissioned_assets_or_transfers",
    ],
    "gross_reductions": [
        "cost_disposals",
        "disposals",
        "cost_derecognitions_terminated_leases",
        "cost_transfer_discontinued_ops",
        "cost_transfer_held_for_sale",
        "realizations_derecognitions_transfers",
    ],
    "annual_depreciation": [
        "accdep_depreciation",
        "depreciation_expense",
    ],
    "net_ppe_close_std": [
        "net_ppe_close",
        "nbv_close",
    ],
    "net_ppe_open_std": [
        "net_ppe_open",
        "nbv_open",
    ],
    "cost_open_std": [
        "cost_open",
    ],
    "cost_close_std": [
        "cost_close",
    ],
    "accdep_open_std": [
        "accdep_open",
    ],
    "accdep_close_std": [
        "accdep_close",
    ],
}

WIP_METRICS = [
    "payments_on_account_purchase_fixed_assets",
]

EVENT_ROW_TYPES = {
    "event",
    "project_event",
    "project_characteristic",
    "planned_capex",
    "corporate_event",
}

NEGATIVE_FLOW_OUTPUTS = {
    "gross_reductions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True, help="Path to k_monthly_redesign_master_v2.tsv")
    parser.add_argument("--hpc", required=True, help="Path to hpc_reports_complement_extract.csv")
    parser.add_argument("--outdir", required=True, help="Directory to write outputs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    return pd.read_csv(path, sep="\t", dtype=str)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def normalize_yes_no(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip().str.lower()
    out = np.where(x.isin(YES_VALUES), "yes", np.where(x.isin(NO_VALUES), "no", ""))
    return pd.Series(out, index=s.index)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def harmonize_master_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in REQUIRED_MASTER_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[REQUIRED_MASTER_COLUMNS].copy()

    out["include_in_k"] = normalize_yes_no(out["include_in_k"])
    out["include_in_productive_k"] = normalize_yes_no(out["include_in_productive_k"])
    out["needs_monthly_timing"] = normalize_yes_no(out["needs_monthly_timing"])

    out = coerce_numeric(out, ["year", "source_page", "value_num", "event_year", "event_month"])

    for c in ["record_id", "entity", "source_group", "row_type", "asset_class_std", "metric", "currency", "unit"]:
        out[c] = out[c].astype("string").str.strip()

    return out


def append_master(master: pd.DataFrame, hpc: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([master, hpc], ignore_index=True)
    dup_mask = both["record_id"].duplicated(keep=False)
    if dup_mask.any():
        dups = both.loc[dup_mask, ["record_id", "source_group", "source_file"]].sort_values("record_id")
        raise ValueError(
            "Duplicate record_id values found after append.\n"
            f"Sample duplicates:\n{dups.head(20).to_string(index=False)}"
        )
    return both


def _filter_by_metrics(df: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    metric_set = {m.strip().lower() for m in metric_names}
    metric_col = df["metric"].astype("string").str.strip().str.lower()
    return df.loc[metric_col.isin(metric_set)].copy()


def _wide_first(
    df: pd.DataFrame,
    index_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=index_cols)
    out = (
        df.pivot_table(
            index=index_cols,
            columns="metric_std",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
    )
    out.columns.name = None
    return out


def _wide_sum(
    df: pd.DataFrame,
    index_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=index_cols)
    out = (
        df.pivot_table(
            index=index_cols,
            columns="metric_std",
            values=value_col,
            aggfunc="sum",
        )
        .reset_index()
    )
    out.columns.name = None
    return out


def build_dep_table(master: pd.DataFrame) -> pd.DataFrame:
    dep = master.loc[master["row_type"].eq("dep_policy")].copy()
    if dep.empty:
        return pd.DataFrame()

    dep_numeric_parts = []
    dep_text_parts = []

    index_cols = ["entity", "year", "source_group", "source_file", "asset_class_std", "asset_class_raw"]

    for metric_std, aliases in DEP_METRIC_ALIASES.items():
        sub = _filter_by_metrics(dep, aliases)
        if sub.empty:
            continue
        sub["metric_std"] = metric_std

        if metric_std == "depreciation_rule_text":
            dep_text_parts.append(sub)
        else:
            dep_numeric_parts.append(sub)

    dep_num = pd.concat(dep_numeric_parts, ignore_index=True) if dep_numeric_parts else pd.DataFrame(columns=dep.columns.tolist() + ["metric_std"])
    dep_txt = pd.concat(dep_text_parts, ignore_index=True) if dep_text_parts else pd.DataFrame(columns=dep.columns.tolist() + ["metric_std"])

    dep_wide_num = _wide_first(dep_num, index_cols=index_cols, value_col="value_num")
    dep_wide_txt = _wide_first(dep_txt, index_cols=index_cols, value_col="value_text")

    dep_out = dep_wide_num.merge(dep_wide_txt, on=index_cols, how="outer")

    if "annual_depreciation_rate_main_pct" in dep_out.columns:
        dep_out["annual_dep_rate_main_decimal"] = dep_out["annual_depreciation_rate_main_pct"] / 100.0
        dep_out["monthly_dep_rate_main_decimal"] = 1 - (1 - dep_out["annual_dep_rate_main_decimal"]).pow(1 / 12)

    if "annual_depreciation_rate_low_pct" in dep_out.columns:
        dep_out["annual_dep_rate_low_decimal"] = dep_out["annual_depreciation_rate_low_pct"] / 100.0
        dep_out["monthly_dep_rate_low_decimal"] = 1 - (1 - dep_out["annual_dep_rate_low_decimal"]).pow(1 / 12)

    if "annual_depreciation_rate_high_pct" in dep_out.columns:
        dep_out["annual_dep_rate_high_decimal"] = dep_out["annual_depreciation_rate_high_pct"] / 100.0
        dep_out["monthly_dep_rate_high_decimal"] = 1 - (1 - dep_out["annual_dep_rate_high_decimal"]).pow(1 / 12)

    dep_out = dep_out.sort_values(["entity", "year", "asset_class_std", "source_group"]).reset_index(drop=True)
    return dep_out


def build_annual_total_components(master: pd.DataFrame) -> pd.DataFrame:
    ppe = master.loc[master["row_type"].eq("ppe_rollforward")].copy()
    if ppe.empty:
        return pd.DataFrame()

    index_cols = ["entity", "year", "source_group", "source_file", "asset_class_std", "asset_class_raw"]

    ppe_parts = []
    for metric_std, aliases in FLOW_ALIASES.items():
        sub = _filter_by_metrics(ppe, aliases)
        if sub.empty:
            continue
        sub["metric_std"] = metric_std
        ppe_parts.append(sub)

    ppe_long = pd.concat(ppe_parts, ignore_index=True) if ppe_parts else pd.DataFrame(columns=ppe.columns.tolist() + ["metric_std"])
    annual = _wide_sum(ppe_long, index_cols=index_cols, value_col="value_num")

    for c in FLOW_ALIASES.keys():
        if c not in annual.columns:
            annual[c] = 0.0

    annual["observed_pool_spend_component"] = annual["gross_additions_spend"].fillna(0.0)
    annual["observed_pool_leases_component"] = annual["gross_additions_leases"].fillna(0.0)
    annual["observed_pool_component"] = annual["observed_pool_spend_component"] + annual["observed_pool_leases_component"]

    annual["gross_reductions"] = annual["gross_reductions"].abs().fillna(0.0)
    annual["net_observed_flow_before_wip"] = annual["observed_pool_component"] - annual["gross_reductions"]

    annual = annual.sort_values(["entity", "year", "asset_class_std", "source_group"]).reset_index(drop=True)
    return annual


def build_wip_by_entity_year(master: pd.DataFrame) -> pd.DataFrame:
    wip = _filter_by_metrics(master, WIP_METRICS).copy()
    if wip.empty:
        return pd.DataFrame(columns=["entity", "year", "wip_total"])
    out = (
        wip.groupby(["entity", "year"], as_index=False)["value_num"]
        .sum()
        .rename(columns={"value_num": "wip_total"})
    )
    return out


def build_annual_observed_pool_components(annual_total: pd.DataFrame) -> pd.DataFrame:
    if annual_total.empty:
        return pd.DataFrame()
    obs = annual_total[
        [
            "entity",
            "year",
            "source_group",
            "source_file",
            "asset_class_std",
            "asset_class_raw",
            "observed_pool_spend_component",
            "observed_pool_leases_component",
            "observed_pool_component",
            "gross_reductions",
            "net_observed_flow_before_wip",
            "annual_depreciation",
            "net_ppe_close_std",
            "explicit_service_entry",
        ]
    ].copy()
    obs = obs.sort_values(["entity", "year", "asset_class_std", "source_group"]).reset_index(drop=True)
    return obs


def build_annual_service_entry_pool_components(
    annual_observed: pd.DataFrame,
    wip_by_entity_year: pd.DataFrame,
) -> pd.DataFrame:
    if annual_observed.empty:
        return pd.DataFrame()

    svc = annual_observed.merge(wip_by_entity_year, on=["entity", "year"], how="left")
    svc["wip_total"] = svc["wip_total"].fillna(0.0)

    # Only allocate WIP across spend-based rows that do not already have explicit service-entry.
    svc["needs_wip_fallback"] = np.where(
        svc["explicit_service_entry"].fillna(0.0) <= 0.0,
        1,
        0,
    )

    svc["fallback_spend_base"] = np.where(
        svc["needs_wip_fallback"] == 1,
        svc["observed_pool_spend_component"].fillna(0.0),
        0.0,
    )

    grp_cols = ["entity", "year"]
    svc["entity_year_fallback_spend_total"] = svc.groupby(grp_cols)["fallback_spend_base"].transform("sum")

    svc["allocated_wip_component"] = np.where(
        svc["entity_year_fallback_spend_total"] > 0,
        svc["wip_total"] * (svc["fallback_spend_base"] / svc["entity_year_fallback_spend_total"]),
        0.0,
    )

    svc["fallback_service_entry_component"] = (
        svc["fallback_spend_base"] - svc["allocated_wip_component"]
    ).clip(lower=0.0)

    svc["service_entry_pool_component"] = np.where(
        svc["explicit_service_entry"].fillna(0.0) > 0.0,
        svc["explicit_service_entry"].fillna(0.0) + svc["observed_pool_leases_component"].fillna(0.0),
        svc["fallback_service_entry_component"] + svc["observed_pool_leases_component"].fillna(0.0),
    )

    svc["service_entry_logic"] = np.where(
        svc["explicit_service_entry"].fillna(0.0) > 0.0,
        "explicit_service_entry_plus_leases",
        "fallback_spend_minus_allocated_wip_plus_leases",
    )

    svc["service_entry_pool_ratio"] = np.where(
        svc["observed_pool_component"].fillna(0.0) > 0.0,
        svc["service_entry_pool_component"] / svc["observed_pool_component"],
        np.nan,
    )

    svc = svc.sort_values(["entity", "year", "asset_class_std", "source_group"]).reset_index(drop=True)
    return svc


def build_observed_event_components(master: pd.DataFrame) -> pd.DataFrame:
    events = master.copy()

    row_type_event = events["row_type"].astype("string").str.strip().str.lower().isin(EVENT_ROW_TYPES)
    has_month = events["event_month"].notna()
    has_year = events["event_year"].notna()
    has_explicit_event_date = events["event_date"].notna() & events["event_date"].astype("string").str.strip().ne("")
    needs_month = events["needs_monthly_timing"].eq("yes")
    include_k = events["include_in_k"].eq("yes") | events["include_in_productive_k"].eq("yes")

    keep = include_k & (
        row_type_event
        | has_month
        | has_explicit_event_date
        | (needs_month & has_year)
    )

    out = events.loc[keep].copy()

    out["event_has_month"] = out["event_month"].notna().map({True: "yes", False: "no"})
    out["event_has_exact_date"] = out["event_date"].astype("string").str.strip().ne("").map({True: "yes", False: "no"})

    keep_cols = [
        "record_id",
        "entity",
        "year",
        "source_group",
        "source_file",
        "row_type",
        "asset_class_std",
        "asset_class_raw",
        "project_id",
        "project_name",
        "metric",
        "value_num",
        "value_text",
        "unit",
        "date_precision",
        "event_date",
        "event_year",
        "event_month",
        "event_has_month",
        "event_has_exact_date",
        "confidence",
        "include_in_k",
        "include_in_productive_k",
        "needs_monthly_timing",
        "notes",
    ]
    out = out[keep_cols].sort_values(["entity", "event_year", "event_month", "source_group", "record_id"]).reset_index(drop=True)
    return out


def build_summary(
    master_v3: pd.DataFrame,
    dep: pd.DataFrame,
    annual_observed: pd.DataFrame,
    annual_service: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    total_rows = (
        master_v3.groupby(["entity", "year"], as_index=False)
        .size()
        .rename(columns={"size": "n_master_rows"})
    )

    if dep.empty:
        dep_cov = pd.DataFrame(columns=["entity", "year", "n_dep_asset_classes", "n_dep_rows"])
    else:
        dep_cov = (
            dep.groupby(["entity", "year"], as_index=False)
            .agg(
                n_dep_asset_classes=("asset_class_std", "nunique"),
                n_dep_rows=("asset_class_std", "size"),
            )
        )

    if annual_observed.empty:
        obs_sum = pd.DataFrame(columns=[
            "entity", "year", "observed_pool_total", "observed_spend_total",
            "observed_leases_total", "explicit_service_entry_total",
            "annual_dep_expense_total", "annual_net_ppe_close_total"
        ])
    else:
        obs_sum = (
            annual_observed.groupby(["entity", "year"], as_index=False)
            .agg(
                observed_pool_total=("observed_pool_component", "sum"),
                observed_spend_total=("observed_pool_spend_component", "sum"),
                observed_leases_total=("observed_pool_leases_component", "sum"),
                explicit_service_entry_total=("explicit_service_entry", "sum"),
                annual_dep_expense_total=("annual_depreciation", "sum"),
                annual_net_ppe_close_total=("net_ppe_close_std", "sum"),
            )
        )

    if annual_service.empty:
        svc_sum = pd.DataFrame(columns=[
            "entity", "year", "wip_total",
            "allocated_wip_total", "service_entry_pool_total"
        ])
    else:
        svc_sum = (
            annual_service.groupby(["entity", "year"], as_index=False)
            .agg(
                wip_total=("wip_total", "first"),
                allocated_wip_total=("allocated_wip_component", "sum"),
                service_entry_pool_total=("service_entry_pool_component", "sum"),
            )
        )

    if events.empty:
        evt_sum = pd.DataFrame(columns=["entity", "year", "n_event_rows", "n_event_rows_with_month"])
    else:
        evt_sum = (
            events.groupby(["entity", "event_year"], as_index=False)
            .agg(
                n_event_rows=("record_id", "size"),
                n_event_rows_with_month=("event_has_month", lambda s: int((s == "yes").sum())),
            )
            .rename(columns={"event_year": "year"})
        )

    out = total_rows.merge(dep_cov, on=["entity", "year"], how="left")
    out = out.merge(obs_sum, on=["entity", "year"], how="left")
    out = out.merge(svc_sum, on=["entity", "year"], how="left")
    out = out.merge(evt_sum, on=["entity", "year"], how="left")

    out["service_entry_share_of_observed"] = np.where(
        out["observed_pool_total"].fillna(0) > 0,
        out["service_entry_pool_total"] / out["observed_pool_total"],
        np.nan,
    )

    out = out.sort_values(["entity", "year"]).reset_index(drop=True)
    return out


def build_diagnostics(
    master_v3: pd.DataFrame,
    dep: pd.DataFrame,
    annual_total: pd.DataFrame,
    annual_service: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    dup_count = int(master_v3["record_id"].duplicated().sum())
    rows.append({
        "check_name": "duplicate_record_id_count",
        "scope": "master_v3",
        "value_num": dup_count,
        "value_text": "",
        "status": "pass" if dup_count == 0 else "fail",
        "notes": "Must be zero.",
    })

    dep_missing_class = int(dep["asset_class_std"].isna().sum()) if not dep.empty else 0
    rows.append({
        "check_name": "dep_rows_missing_asset_class_std",
        "scope": "depreciation_params_clean_v2",
        "value_num": dep_missing_class,
        "value_text": "",
        "status": "pass" if dep_missing_class == 0 else "warn",
        "notes": "Depreciation rows should ideally have mapped asset classes.",
    })

    svc_neg = int((annual_service["service_entry_pool_component"] < 0).sum()) if not annual_service.empty else 0
    rows.append({
        "check_name": "negative_service_entry_pool_count",
        "scope": "annual_service_entry_pool_components_v2",
        "value_num": svc_neg,
        "value_text": "",
        "status": "pass" if svc_neg == 0 else "fail",
        "notes": "Service-entry pool is clipped at zero, so negatives should be impossible.",
    })

    svc_ratio_gt1 = int((annual_service["service_entry_pool_ratio"] > 1.000001).sum()) if not annual_service.empty else 0
    rows.append({
        "check_name": "service_entry_ratio_gt_one_count",
        "scope": "annual_service_entry_pool_components_v2",
        "value_num": svc_ratio_gt1,
        "value_text": "",
        "status": "pass" if svc_ratio_gt1 == 0 else "warn",
        "notes": "Explicit service-entry can exceed observed spending in some archival setups; investigate if non-zero.",
    })

    events_with_month = int((events["event_has_month"] == "yes").sum()) if not events.empty else 0
    rows.append({
        "check_name": "event_rows_with_month",
        "scope": "observed_event_components_v2",
        "value_num": events_with_month,
        "value_text": "",
        "status": "info",
        "notes": "Higher is better for true monthly timing.",
    })

    total_annual_rows = len(annual_total) if not annual_total.empty else 0
    rows.append({
        "check_name": "annual_total_component_rows",
        "scope": "annual_total_components_v2",
        "value_num": total_annual_rows,
        "value_text": "",
        "status": "info",
        "notes": "Wide annual class-year table row count.",
    })

    # Coverage diagnostics by entity for observed and service-entry pools
    if not annual_total.empty:
        by_entity = (
            annual_total.groupby("entity", as_index=False)
            .agg(
                observed_pool_total=("observed_pool_component", "sum"),
                explicit_service_entry_total=("explicit_service_entry", "sum"),
                annual_dep_total=("annual_depreciation", "sum"),
            )
        )
        for _, r in by_entity.iterrows():
            rows.append({
                "check_name": "entity_pool_coverage",
                "scope": str(r["entity"]),
                "value_num": float(r["observed_pool_total"]),
                "value_text": f'explicit_service_entry_total={r["explicit_service_entry_total"]}; annual_dep_total={r["annual_dep_total"]}',
                "status": "info",
                "notes": "Entity-level annual pool totals after alias mapping.",
            })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    master_path = Path(args.master)
    hpc_path = Path(args.hpc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("Reading raw inputs...", args.verbose)
    master_raw = read_table(master_path)
    hpc_raw = read_table(hpc_path)

    validate_columns(master_raw, REQUIRED_MASTER_COLUMNS, "Master TSV")
    validate_columns(hpc_raw, REQUIRED_MASTER_COLUMNS, "HPC supplement CSV")

    master = harmonize_master_schema(master_raw)
    hpc = harmonize_master_schema(hpc_raw)

    log("Appending HPC supplement into master...", args.verbose)
    master_v3 = append_master(master, hpc)

    master_v3_path = outdir / "k_monthly_redesign_master_v3.tsv"
    write_tsv(master_v3, master_v3_path)

    log("Building processed tables...", args.verbose)
    dep = build_dep_table(master_v3)
    annual_total = build_annual_total_components(master_v3)
    annual_observed = build_annual_observed_pool_components(annual_total)
    wip_by_entity_year = build_wip_by_entity_year(master_v3)
    annual_service = build_annual_service_entry_pool_components(annual_observed, wip_by_entity_year)
    events = build_observed_event_components(master_v3)
    summary = build_summary(master_v3, dep, annual_observed, annual_service, events)
    diagnostics = build_diagnostics(master_v3, dep, annual_total, annual_service, events)

    write_tsv(dep, outdir / "depreciation_params_clean_v2.tsv")
    write_tsv(annual_total, outdir / "annual_total_components_v2.tsv")
    write_tsv(annual_observed, outdir / "annual_observed_pool_components_v2.tsv")
    write_tsv(annual_service, outdir / "annual_service_entry_pool_components_v2.tsv")
    write_tsv(events, outdir / "observed_event_components_v2.tsv")
    write_tsv(summary, outdir / "emi_k_feedback_summary_v2.tsv")
    write_tsv(diagnostics, outdir / "emi_k_feedback_diagnostics_v2.tsv")

    manifest = {
        "inputs": {
            "master": str(master_path),
            "hpc": str(hpc_path),
        },
        "outputs": [
            "k_monthly_redesign_master_v3.tsv",
            "depreciation_params_clean_v2.tsv",
            "annual_total_components_v2.tsv",
            "annual_observed_pool_components_v2.tsv",
            "annual_service_entry_pool_components_v2.tsv",
            "observed_event_components_v2.tsv",
            "emi_k_feedback_summary_v2.tsv",
            "emi_k_feedback_diagnostics_v2.tsv",
        ],
        "notes": [
            "The raw archive is append-only and lossless.",
            "Processed inputs now map heterogenous metric names across HPC, IPC, and SIPG.",
            "Service-entry pool prefers explicit service-entry signals where available and otherwise falls back to observed spending net of allocated WIP.",
            "Monthly K code should read the processed inputs, not the raw archive.",
        ],
    }

    with open(outdir / "_prepare_k_inputs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Wrote: {master_v3_path}")
    print(f"Wrote processed inputs to: {outdir}")


if __name__ == "__main__":
    main()

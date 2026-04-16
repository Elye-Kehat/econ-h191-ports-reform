from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "k_monthly_redesign_master_v2.tsv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "k_feedback_output"

MIN_YEAR = 2017
MAX_YEAR = 2024

DEBUG = True


# =============================================================================
# DEBUG HELPERS
# =============================================================================

def debug_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def debug_show(
    df: pd.DataFrame,
    title: str,
    cols: Optional[list[str]] = None,
    n: int = 200,
) -> None:
    debug_header(title)
    if df is None or df.empty:
        print("[empty]")
        return
    out = df.copy()
    if cols is not None:
        keep = [c for c in cols if c in out.columns]
        out = out[keep]
    print(out.head(n).to_string(index=False))


def debug_warn(msg: str) -> None:
    print(f"\n[DEBUG WARNING] {msg}")


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def normalize_yes_no(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    if s in {"maybe", "unclear"}:
        return "maybe"
    return s


def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def to_num(x: object) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def first_non_null(df: pd.DataFrame, by: list[str], sort_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.sort_values(sort_cols).drop_duplicates(by, keep="first").copy()
    return out


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def load_master(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input TSV at:\n{input_path}\n\n"
            f"Script location:\n{SCRIPT_DIR}"
        )

    master = pd.read_csv(input_path, sep="\t", dtype=str)

    needed = [
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
    master = ensure_columns(master, needed)

    text_cols = [
        "record_id",
        "entity",
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
    for c in text_cols:
        master[c] = master[c].map(clean_text)

    master["include_in_k"] = master["include_in_k"].map(normalize_yes_no)
    master["include_in_productive_k"] = master["include_in_productive_k"].map(normalize_yes_no)
    master["needs_monthly_timing"] = master["needs_monthly_timing"].map(normalize_yes_no)

    master["year"] = pd.to_numeric(master["year"], errors="coerce")
    master["event_year"] = pd.to_numeric(master["event_year"], errors="coerce")
    master["value_num"] = master["value_num"].map(to_num)

    return master


def in_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)].copy()


def choose_best_metric_row(
    df: pd.DataFrame,
    entity: str,
    year: int,
    metric: str,
    preferred_source_groups: list[str],
) -> Optional[pd.Series]:
    sub = df[
        (df["entity"] == entity)
        & (df["year"] == year)
        & (df["metric"] == metric)
        & df["value_num"].notna()
    ].copy()

    if sub.empty:
        return None

    source_rank = {sg: i for i, sg in enumerate(preferred_source_groups)}
    sub["source_rank"] = sub["source_group"].map(lambda x: source_rank.get(x, 999))
    sub = sub.sort_values(["source_rank", "record_id"])
    return sub.iloc[0]


def add_component_row(
    rows: list[dict],
    entity: str,
    year: int,
    component_type: str,
    value: float,
    source_rule: str,
    row: Optional[pd.Series] = None,
    extra: Optional[dict] = None,
) -> None:
    out = {
        "entity": entity,
        "year": year,
        "component_type": component_type,
        "value": float(value) if pd.notna(value) else np.nan,
        "source_rule": source_rule,
        "record_id": row["record_id"] if row is not None and "record_id" in row else "",
        "source_group": row["source_group"] if row is not None and "source_group" in row else "",
        "metric": row["metric"] if row is not None and "metric" in row else "",
        "asset_class_std": row["asset_class_std"] if row is not None and "asset_class_std" in row else "",
        "project_name": row["project_name"] if row is not None and "project_name" in row else "",
        "event_month": row["event_month"] if row is not None and "event_month" in row else "",
    }
    if extra:
        out.update(extra)
    rows.append(out)


def metric_contains_any(metric: str, needles: list[str]) -> bool:
    s = clean_text(metric).lower()
    return any(n in s for n in needles)


# =============================================================================
# DEPRECIATION TABLE
# =============================================================================

def build_depreciation_params_clean(master: pd.DataFrame) -> pd.DataFrame:
    dep = master[master["row_type"] == "dep_policy"].copy()
    dep = in_window(dep)

    if dep.empty:
        return dep

    dep["metric_l"] = dep["metric"].str.lower()
    dep["entity"] = dep["entity"].str.strip()
    dep["asset_class_std"] = dep["asset_class_std"].str.strip()

    keep_metrics = {
        "annual_depr_rate",
        "annual_depr_rate_pct_min",
        "annual_depr_rate_pct_max",
        "annual_depr_rate_pct",
        "useful_life_years",
        "useful_life_years_min",
        "useful_life_years_max",
        "residual_pct_min",
        "residual_pct_max",
        "capitalization_rate",
        "depr_method",
        "depreciation_start_rule",
        "cip_to_ppe_and_depr_start_rule",
    }
    dep = dep[dep["metric"].isin(keep_metrics) | dep["metric_l"].isin(keep_metrics)].copy()

    if dep.empty:
        return dep

    wide = (
        dep.pivot_table(
            index=["entity", "year", "asset_class_std", "source_group"],
            columns="metric",
            values="value_num",
            aggfunc="first",
        )
        .reset_index()
    )

    for c in [
        "annual_depr_rate",
        "annual_depr_rate_pct_min",
        "annual_depr_rate_pct_max",
        "useful_life_years",
        "useful_life_years_min",
        "useful_life_years_max",
    ]:
        if c not in wide.columns:
            wide[c] = np.nan

    # Normalize pct metrics into decimal annual rates if possible
    wide["annual_rate_low"] = np.nan
    wide["annual_rate_high"] = np.nan

    mask_direct = wide["annual_depr_rate"].notna()
    wide.loc[mask_direct, "annual_rate_low"] = wide.loc[mask_direct, "annual_depr_rate"]
    wide.loc[mask_direct, "annual_rate_high"] = wide.loc[mask_direct, "annual_depr_rate"]

    mask_pct_range = wide["annual_depr_rate_pct_min"].notna() | wide["annual_depr_rate_pct_max"].notna()
    wide.loc[mask_pct_range, "annual_rate_low"] = wide.loc[mask_pct_range, "annual_depr_rate_pct_min"] / 100.0
    wide.loc[mask_pct_range, "annual_rate_high"] = wide.loc[mask_pct_range, "annual_depr_rate_pct_max"] / 100.0

    # If no direct annual rate, infer from useful lives
    mask_life = wide["annual_rate_low"].isna() & wide["useful_life_years_max"].notna()
    wide.loc[mask_life, "annual_rate_low"] = 1.0 / wide.loc[mask_life, "useful_life_years_max"]

    mask_life2 = wide["annual_rate_high"].isna() & wide["useful_life_years_min"].notna()
    wide.loc[mask_life2, "annual_rate_high"] = 1.0 / wide.loc[mask_life2, "useful_life_years_min"]

    mask_life3 = wide["annual_rate_low"].isna() & wide["useful_life_years"].notna()
    wide.loc[mask_life3, "annual_rate_low"] = 1.0 / wide.loc[mask_life3, "useful_life_years"]

    mask_life4 = wide["annual_rate_high"].isna() & wide["useful_life_years"].notna()
    wide.loc[mask_life4, "annual_rate_high"] = 1.0 / wide.loc[mask_life4, "useful_life_years"]

    wide["annual_rate_mid"] = np.where(
        wide["annual_rate_low"].notna() & wide["annual_rate_high"].notna(),
        0.5 * (wide["annual_rate_low"] + wide["annual_rate_high"]),
        np.where(wide["annual_rate_low"].notna(), wide["annual_rate_low"], wide["annual_rate_high"]),
    )

    wide["monthly_rate_mid_geometric"] = np.where(
        wide["annual_rate_mid"].notna(),
        1.0 - np.power(1.0 - wide["annual_rate_mid"], 1.0 / 12.0),
        np.nan,
    )

    wide = wide.sort_values(["entity", "year", "asset_class_std", "source_group"]).reset_index(drop=True)
    return wide


# =============================================================================
# ANNUAL TOTALS
# =============================================================================

def build_annual_total_components(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    years = list(range(MIN_YEAR, MAX_YEAR + 1))

    for year in years:
        # HPC
        r = choose_best_metric_row(
            master,
            "HPC",
            year,
            "approx_gross_ppe_investment",
            ["THESIS_2026_SUMMARY", "HPC_FINANCIALS_RAW"],
        )
        if r is not None:
            add_component_row(
                rows,
                "HPC",
                year,
                "annual_total_primary",
                r["value_num"],
                "HPC primary total = explicit annual gross investment summary",
                r,
            )
        else:
            r2 = choose_best_metric_row(
                master,
                "HPC",
                year,
                "purchase_fixed_assets_cashflow",
                ["HPC_FINANCIALS_RAW"],
            )
            if r2 is not None:
                add_component_row(
                    rows,
                    "HPC",
                    year,
                    "annual_total_primary",
                    abs(r2["value_num"]),
                    "HPC fallback primary total = abs(purchase_fixed_assets_cashflow)",
                    r2,
                )

        # IPC
        r = choose_best_metric_row(
            master,
            "IPC",
            year,
            "approx_gross_ppe_investment",
            ["THESIS_2026_SUMMARY", "IPC_RAW_TABLE2", "IPC_2024_AR"],
        )
        if r is not None:
            add_component_row(
                rows,
                "IPC",
                year,
                "annual_total_primary",
                r["value_num"],
                "IPC primary total = explicit annual gross investment summary",
                r,
            )
        else:
            r2 = choose_best_metric_row(
                master,
                "IPC",
                year,
                "purchase_ppe_cashflow",
                ["IPC_RAW_TABLE2"],
            )
            if r2 is not None:
                add_component_row(
                    rows,
                    "IPC",
                    year,
                    "annual_total_primary",
                    abs(r2["value_num"]),
                    "IPC fallback primary total = abs(purchase_ppe_cashflow)",
                    r2,
                )

        # SIPG
        r = choose_best_metric_row(
            master,
            "SIPG",
            year,
            "transfer_to_ppe_knis",
            ["THESIS_2026_SUMMARY"],
        )
        if r is not None:
            add_component_row(
                rows,
                "SIPG",
                year,
                "annual_total_primary",
                r["value_num"],
                "SIPG primary total = explicit annual Bayport transfer_to_ppe summary",
                r,
            )
        else:
            r2 = choose_best_metric_row(
                master,
                "SIPG",
                year,
                "transfer_to_ppe",
                ["SIPG_BAYPORT_CIP", "SIPG_K_REDESIGN_CLEAN"],
            )
            if r2 is not None:
                add_component_row(
                    rows,
                    "SIPG",
                    year,
                    "annual_total_primary",
                    r2["value_num"],
                    "SIPG fallback primary total = raw transfer_to_ppe",
                    r2,
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["entity", "year", "record_id"]).reset_index(drop=True)
    return out


# =============================================================================
# OBSERVED ANNUAL POOL
#
# IMPORTANT:
# This is the annual amount we can explicitly point to in the current data as
# observed annual capital flow for the same broad concept used in annual_total.
#
# For HPC / IPC:
#   if the primary total itself comes from an explicit annual financial summary,
#   we count that as observed annual-undated, because the annual amount is known.
#
# For SIPG:
#   transfer_to_ppe is the primary annual concept for monthly-K entry.
#
# We ALSO build a separate "service-entry alt pool" below for narrower
# recognized-in-service amounts, so we can inspect concept mismatch explicitly.
# =============================================================================

def build_annual_observed_pool_components(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    years = list(range(MIN_YEAR, MAX_YEAR + 1))

    for year in years:
        # HPC observed annual pool
        r = choose_best_metric_row(
            master,
            "HPC",
            year,
            "approx_gross_ppe_investment",
            ["THESIS_2026_SUMMARY", "HPC_FINANCIALS_RAW"],
        )
        if r is not None:
            add_component_row(
                rows,
                "HPC",
                year,
                "annual_observed_pool",
                r["value_num"],
                "HPC observed annual pool = explicit annual gross investment summary",
                r,
            )
        else:
            r2 = choose_best_metric_row(
                master,
                "HPC",
                year,
                "purchase_fixed_assets_cashflow",
                ["HPC_FINANCIALS_RAW"],
            )
            if r2 is not None:
                add_component_row(
                    rows,
                    "HPC",
                    year,
                    "annual_observed_pool",
                    abs(r2["value_num"]),
                    "HPC observed annual pool = abs(purchase_fixed_assets_cashflow)",
                    r2,
                )

        # IPC observed annual pool
        r = choose_best_metric_row(
            master,
            "IPC",
            year,
            "approx_gross_ppe_investment",
            ["THESIS_2026_SUMMARY", "IPC_RAW_TABLE2", "IPC_2024_AR"],
        )
        if r is not None:
            add_component_row(
                rows,
                "IPC",
                year,
                "annual_observed_pool",
                r["value_num"],
                "IPC observed annual pool = explicit annual gross investment summary",
                r,
            )
        else:
            r2 = choose_best_metric_row(
                master,
                "IPC",
                year,
                "purchase_ppe_cashflow",
                ["IPC_RAW_TABLE2"],
            )
            if r2 is not None:
                add_component_row(
                    rows,
                    "IPC",
                    year,
                    "annual_observed_pool",
                    abs(r2["value_num"]),
                    "IPC observed annual pool = abs(purchase_ppe_cashflow)",
                    r2,
                )

        # SIPG observed annual pool
        r = choose_best_metric_row(
            master,
            "SIPG",
            year,
            "transfer_to_ppe_knis",
            ["THESIS_2026_SUMMARY"],
        )
        if r is not None:
            add_component_row(
                rows,
                "SIPG",
                year,
                "annual_observed_pool",
                r["value_num"],
                "SIPG observed annual pool = explicit annual Bayport transfer_to_ppe summary",
                r,
            )
        else:
            r2 = choose_best_metric_row(
                master,
                "SIPG",
                year,
                "transfer_to_ppe",
                ["SIPG_BAYPORT_CIP", "SIPG_K_REDESIGN_CLEAN"],
            )
            if r2 is not None:
                add_component_row(
                    rows,
                    "SIPG",
                    year,
                    "annual_observed_pool",
                    r2["value_num"],
                    "SIPG observed annual pool = raw transfer_to_ppe",
                    r2,
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["entity", "year", "record_id"]).reset_index(drop=True)
    return out


# =============================================================================
# ALTERNATIVE NARROWER SERVICE-ENTRY POOL
#
# This is a DIFFERENT concept from the broad annual total above.
# It is included for debugging / evaluation because it helps show where
# "recognized in service" amounts differ from broad annual capex / annual flow.
# =============================================================================

def build_annual_service_entry_pool_components(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    m = in_window(master)

    # HPC: sum Note 8 purchases + new_leases_additions across productive classes
    hpc = m[
        (m["entity"] == "HPC")
        & (m["row_type"] == "ppe_rollforward")
        & (m["metric"].isin(["purchases", "new_leases_additions"]))
        & (m["include_in_productive_k"] == "yes")
        & m["value_num"].notna()
    ].copy()

    if not hpc.empty:
        grp = (
            hpc.groupby("year", as_index=False)["value_num"]
            .sum()
            .rename(columns={"value_num": "value"})
        )
        for _, r in grp.iterrows():
            add_component_row(
                rows,
                "HPC",
                int(r["year"]),
                "annual_service_entry_pool",
                r["value"],
                "HPC alternative service-entry pool = Note 8 purchases + new_leases_additions",
                None,
            )

    # IPC: sum positive commissioned_assets_or_transfers in productive classes,
    # excluding total / wip / land
    ipc = m[
        (m["entity"] == "IPC")
        & (m["row_type"] == "ppe_rollforward")
        & (m["metric"] == "commissioned_assets_or_transfers")
        & (m["include_in_productive_k"] == "yes")
        & (~m["asset_class_std"].isin(["total", "wip", "land"]))
        & m["value_num"].notna()
        & (m["value_num"] > 0)
    ].copy()

    if not ipc.empty:
        grp = (
            ipc.groupby("year", as_index=False)["value_num"]
            .sum()
            .rename(columns={"value_num": "value"})
        )
        for _, r in grp.iterrows():
            add_component_row(
                rows,
                "IPC",
                int(r["year"]),
                "annual_service_entry_pool",
                r["value"],
                "IPC alternative service-entry pool = positive commissioned_assets_or_transfers in productive classes",
                None,
            )

    # SIPG: transfer_to_ppe is already the narrow service-entry concept
    years = list(range(MIN_YEAR, MAX_YEAR + 1))
    for year in years:
        r = choose_best_metric_row(
            master,
            "SIPG",
            year,
            "transfer_to_ppe_knis",
            ["THESIS_2026_SUMMARY"],
        )
        if r is not None:
            add_component_row(
                rows,
                "SIPG",
                year,
                "annual_service_entry_pool",
                r["value_num"],
                "SIPG alternative service-entry pool = transfer_to_ppe summary",
                r,
            )
        else:
            r2 = choose_best_metric_row(
                master,
                "SIPG",
                year,
                "transfer_to_ppe",
                ["SIPG_BAYPORT_CIP", "SIPG_K_REDESIGN_CLEAN"],
            )
            if r2 is not None:
                add_component_row(
                    rows,
                    "SIPG",
                    year,
                    "annual_service_entry_pool",
                    r2["value_num"],
                    "SIPG alternative service-entry pool = raw transfer_to_ppe",
                    r2,
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["entity", "year"]).reset_index(drop=True)
    return out


# =============================================================================
# EVENTS
# =============================================================================

def event_direction(row: pd.Series) -> str:
    notes = clean_text(row.get("notes", "")).lower()
    name = clean_text(row.get("project_name", "")).lower()
    metric = clean_text(row.get("metric", "")).lower()

    disposal_words = [
        "disposal",
        "derecognition",
        "realization",
        "redemption",
        "sale of",
        "sale price",
        "transfer of marine dept",
        "removed from",
        "book value of disposed assets",
    ]
    if any(w in notes for w in disposal_words) or any(w in name for w in disposal_words):
        return "disposal"

    if row.get("include_in_k", "") != "yes":
        return "excluded"

    return "addition"


def event_is_proxy_only(row: pd.Series) -> bool:
    notes = clean_text(row.get("notes", "")).lower()
    value = row.get("value_num", np.nan)

    # If no numeric amount, do not use in dated amount sums
    if pd.isna(value):
        return True

    strong_proxy_words = [
        "not capitalized",
        "not yet capitalized",
        "commission upon completion",
        "upon completion",
        "max compensation agreed",
        "sale price",
        "book value of disposed assets tbd",
        "book value of disposed assets (not the sale price)",
        "estimated total",
        "estimated hpc share",
        "approx hpc share",
        "approx share",
        "approx. hpc share",
        "actual use in assets",
        "deliveries 2023–2026",
        "deliveries 2023-2026",
        "phased",
        "not disclosed separately",
        "could be >",
    ]
    return any(w in notes for w in strong_proxy_words)


def build_event_components(master: pd.DataFrame) -> pd.DataFrame:
    m = in_window(master)
    events = m[(m["row_type"] == "event")].copy()

    rows: list[dict] = []

    if events.empty:
        return pd.DataFrame(rows)

    for _, r in events.iterrows():
        direction = event_direction(r)
        proxy_only = event_is_proxy_only(r)

        if r["include_in_k"] != "yes":
            add_component_row(
                rows,
                r["entity"],
                int(r["year"]),
                "excluded_event",
                r["value_num"] if pd.notna(r["value_num"]) else np.nan,
                "include_in_k != yes",
                r,
                extra={"component_subtype": direction},
            )
            continue

        if proxy_only:
            add_component_row(
                rows,
                r["entity"],
                int(r["year"]),
                "ignored_proxy_event",
                r["value_num"] if pd.notna(r["value_num"]) else np.nan,
                "proxy-only or not-yet-in-service event excluded",
                r,
                extra={"component_subtype": direction},
            )
            continue

        date_precision = clean_text(r.get("date_precision", "")).lower()
        if direction == "addition" and date_precision == "year_month":
            add_component_row(
                rows,
                r["entity"],
                int(r["year"]),
                "observed_event_dated",
                r["value_num"],
                "event rows classified heuristically; date_precision=year_month",
                r,
                extra={"component_subtype": "addition"},
            )
        elif direction == "disposal" and date_precision == "year_month":
            add_component_row(
                rows,
                r["entity"],
                int(r["year"]),
                "observed_event_dated",
                r["value_num"],
                "event rows classified heuristically; date_precision=year_month",
                r,
                extra={"component_subtype": "disposal"},
            )
        else:
            add_component_row(
                rows,
                r["entity"],
                int(r["year"]),
                "observed_event_annual_undated",
                r["value_num"],
                "event rows classified heuristically; no month-level date",
                r,
                extra={"component_subtype": direction},
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["entity", "year", "component_type", "record_id"]).reset_index(drop=True)
    return out


# =============================================================================
# SUMMARY + DIAGNOSTICS
# =============================================================================

def aggregate_component(df: pd.DataFrame, component_type: str, subtype: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["entity", "year", "value"])
    sub = df[df["component_type"] == component_type].copy()
    if subtype is not None and "component_subtype" in sub.columns:
        sub = sub[sub["component_subtype"] == subtype].copy()
    if sub.empty:
        return pd.DataFrame(columns=["entity", "year", "value"])
    out = sub.groupby(["entity", "year"], as_index=False)["value"].sum()
    return out


def build_emi_summary(
    annual_total: pd.DataFrame,
    annual_pool: pd.DataFrame,
    annual_service_pool: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    total_sum = (
        aggregate_component(annual_total, "annual_total_primary")
        .rename(columns={"value": "annual_total_primary"})
    )
    pool_sum = (
        aggregate_component(annual_pool, "annual_observed_pool")
        .rename(columns={"value": "explicit_annual_observed_pool"})
    )
    service_sum = (
        aggregate_component(annual_service_pool, "annual_service_entry_pool")
        .rename(columns={"value": "alt_service_entry_pool"})
    )
    dated_add = (
        aggregate_component(events, "observed_event_dated", "addition")
        .rename(columns={"value": "dated_additions"})
    )
    dated_disp = (
        aggregate_component(events, "observed_event_dated", "disposal")
        .rename(columns={"value": "dated_disposals"})
    )
    undated_add_events = (
        aggregate_component(events, "observed_event_annual_undated", "addition")
        .rename(columns={"value": "annual_undated_event_additions"})
    )

    universe = pd.concat(
        [
            total_sum[["entity", "year"]],
            pool_sum[["entity", "year"]],
            service_sum[["entity", "year"]],
            dated_add[["entity", "year"]],
            dated_disp[["entity", "year"]],
            undated_add_events[["entity", "year"]],
        ],
        axis=0,
        ignore_index=True,
    ).drop_duplicates()

    if universe.empty:
        return universe

    s = universe.merge(total_sum, on=["entity", "year"], how="left")
    s = s.merge(pool_sum, on=["entity", "year"], how="left")
    s = s.merge(service_sum, on=["entity", "year"], how="left")
    s = s.merge(dated_add, on=["entity", "year"], how="left")
    s = s.merge(dated_disp, on=["entity", "year"], how="left")
    s = s.merge(undated_add_events, on=["entity", "year"], how="left")

    for c in [
        "annual_total_primary",
        "explicit_annual_observed_pool",
        "alt_service_entry_pool",
        "dated_additions",
        "dated_disposals",
        "annual_undated_event_additions",
    ]:
        if c not in s.columns:
            s[c] = 0.0
        s[c] = s[c].fillna(0.0)

    # Broad annual-undated amount for the same concept as annual_total_primary.
    # Since explicit_annual_observed_pool is already an observed annual amount,
    # it is acceptable for it to equal annual_total_primary in many rows.
    s["annual_observed_undated"] = np.maximum(
        s["explicit_annual_observed_pool"] - s["dated_additions"],
        0.0,
    ) + s["annual_undated_event_additions"]

    s["residual_unexplained"] = np.maximum(
        s["annual_total_primary"] - s["dated_additions"] - s["annual_observed_undated"],
        0.0,
    )

    # Shares
    with np.errstate(divide="ignore", invalid="ignore"):
        s["dated_share"] = np.where(
            s["annual_total_primary"] > 0,
            s["dated_additions"] / s["annual_total_primary"],
            np.nan,
        )
        s["annual_observed_undated_share"] = np.where(
            s["annual_total_primary"] > 0,
            s["annual_observed_undated"] / s["annual_total_primary"],
            np.nan,
        )
        s["residual_share"] = np.where(
            s["annual_total_primary"] > 0,
            s["residual_unexplained"] / s["annual_total_primary"],
            np.nan,
        )
        s["alt_service_entry_pool_share"] = np.where(
            s["annual_total_primary"] > 0,
            s["alt_service_entry_pool"] / s["annual_total_primary"],
            np.nan,
        )

    s = s.sort_values(["entity", "year"]).reset_index(drop=True)
    return s


def build_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()

    d = summary.copy()
    d["flag_missing_primary_total"] = d["annual_total_primary"] <= 0
    d["flag_missing_observed_pool"] = d["explicit_annual_observed_pool"] <= 0
    d["flag_total_equals_pool"] = np.isclose(
        d["annual_total_primary"],
        d["explicit_annual_observed_pool"],
        equal_nan=False,
    )
    d["flag_no_dated_additions"] = d["dated_additions"] <= 0
    d["flag_alt_service_entry_exceeds_total"] = d["alt_service_entry_pool"] > d["annual_total_primary"]
    d["flag_zero_residual"] = np.isclose(d["residual_unexplained"], 0.0)
    d["diagnostic_note"] = ""

    notes = []
    for _, r in d.iterrows():
        parts: list[str] = []

        if r["flag_missing_primary_total"]:
            parts.append("missing_primary_total")
        if r["flag_missing_observed_pool"]:
            parts.append("missing_observed_pool")
        if r["flag_total_equals_pool"]:
            parts.append("primary_total_equals_observed_pool")
        if r["flag_no_dated_additions"]:
            parts.append("no_dated_additions_survive_filter")
        if r["flag_alt_service_entry_exceeds_total"]:
            parts.append("alt_service_entry_pool_exceeds_primary_total_concept_mismatch")
        if r["flag_zero_residual"] and r["annual_total_primary"] > 0:
            parts.append("zero_residual_given_positive_primary_total")

        notes.append("; ".join(parts))

    d["diagnostic_note"] = notes
    return d


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master = load_master(input_path)

    if DEBUG:
        debug_header("DEBUG 1: INPUT FILE AND MASTER COVERAGE")
        print(f"Input path actually used: {input_path}")
        print(f"Master rows loaded: {len(master):,}")

        coverage = (
            master.groupby(["entity", "source_group", "row_type"], dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values(["entity", "source_group", "row_type"])
        )
        print(coverage.to_string(index=False))

        thesis_rows = master[master["source_group"] == "THESIS_2026_SUMMARY"].copy()
        debug_show(
            thesis_rows.sort_values(["entity", "year", "metric"]),
            "DEBUG 2: THESIS_2026_SUMMARY ROWS FOUND IN MASTER",
            cols=["entity", "year", "metric", "value_num", "record_id"],
        )

        hpc_annual = master[
            (master["entity"] == "HPC") & (master["row_type"] == "annual_summary")
        ].copy()
        debug_show(
            hpc_annual.sort_values(["year", "metric"]),
            "DEBUG 3: HPC annual_summary rows in master",
            cols=["year", "metric", "value_num", "source_group", "record_id"],
        )

    dep = build_depreciation_params_clean(master)
    annual_total = build_annual_total_components(master)
    annual_pool = build_annual_observed_pool_components(master)
    annual_service_pool = build_annual_service_entry_pool_components(master)
    events = build_event_components(master)
    summary = build_emi_summary(annual_total, annual_pool, annual_service_pool, events)
    diagnostics = build_diagnostics(summary)

    if DEBUG:
        debug_show(
            annual_total.sort_values(["entity", "year"]),
            "DEBUG 4: annual_total_components",
            cols=["entity", "year", "component_type", "value", "source_rule", "record_id"],
        )

        debug_show(
            annual_pool.sort_values(["entity", "year"]),
            "DEBUG 5: annual_observed_pool_components",
            cols=["entity", "year", "component_type", "value", "source_rule", "record_id"],
        )

        total_sum = (
            annual_total.groupby(["entity", "year"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "annual_total_sum"})
        )
        pool_sum = (
            annual_pool.groupby(["entity", "year"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "annual_pool_sum"})
        )
        service_sum = (
            annual_service_pool.groupby(["entity", "year"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "annual_service_entry_pool_sum"})
        )

        compare = total_sum.merge(pool_sum, on=["entity", "year"], how="outer").fillna(0.0)
        compare = compare.merge(service_sum, on=["entity", "year"], how="outer").fillna(0.0)
        compare["diff_total_minus_pool"] = compare["annual_total_sum"] - compare["annual_pool_sum"]
        compare["diff_total_minus_service_entry_pool"] = (
            compare["annual_total_sum"] - compare["annual_service_entry_pool_sum"]
        )

        debug_show(
            compare.sort_values(["entity", "year"]),
            "DEBUG 6: annual_total vs annual_observed_pool vs annual_service_entry_pool",
            cols=[
                "entity",
                "year",
                "annual_total_sum",
                "annual_pool_sum",
                "annual_service_entry_pool_sum",
                "diff_total_minus_pool",
                "diff_total_minus_service_entry_pool",
            ],
        )

        raw_events = master[
            (master["row_type"] == "event")
            & (master["year"] >= MIN_YEAR)
            & (master["year"] <= MAX_YEAR)
        ].copy()
        if not raw_events.empty:
            raw_events["direction_guess"] = raw_events.apply(event_direction, axis=1)
            raw_events["proxy_only_flag"] = raw_events.apply(event_is_proxy_only, axis=1)

            debug_show(
                raw_events.sort_values(["entity", "year", "record_id"]),
                "DEBUG 7: RAW EVENT ROWS BEFORE FILTERING",
                cols=[
                    "entity",
                    "year",
                    "record_id",
                    "project_name",
                    "value_num",
                    "date_precision",
                    "include_in_k",
                    "direction_guess",
                    "proxy_only_flag",
                    "notes",
                ],
            )

        debug_show(
            events.sort_values(["entity", "year", "component_type"]),
            "DEBUG 8: EVENT COMPONENTS AFTER FILTERING / CLASSIFICATION",
            cols=[
                "entity",
                "year",
                "component_type",
                "component_subtype",
                "value",
                "record_id",
                "project_name",
                "event_month",
                "source_rule",
            ],
        )

        debug_show(
            summary.sort_values(["entity", "year"]),
            "DEBUG 9: FINAL EMI SUMMARY",
            cols=[
                "entity",
                "year",
                "annual_total_primary",
                "explicit_annual_observed_pool",
                "alt_service_entry_pool",
                "dated_additions",
                "annual_observed_undated",
                "residual_unexplained",
                "dated_share",
                "annual_observed_undated_share",
                "residual_share",
                "alt_service_entry_pool_share",
            ],
        )

        same_total_pool = summary[
            (summary["annual_total_primary"] > 0)
            & np.isclose(summary["annual_total_primary"], summary["explicit_annual_observed_pool"])
        ].copy()
        if not same_total_pool.empty:
            debug_warn(
                "primary total equals observed annual pool in some rows. "
                "This is NOT automatically a bug here: it means the broad annual amount itself is observed "
                "at annual frequency. Use alt_service_entry_pool to inspect narrower in-service amounts."
            )
            print(
                same_total_pool[
                    [
                        "entity",
                        "year",
                        "annual_total_primary",
                        "explicit_annual_observed_pool",
                        "alt_service_entry_pool",
                        "dated_additions",
                        "residual_unexplained",
                    ]
                ].to_string(index=False)
            )

        alt_gt_total = summary[
            (summary["alt_service_entry_pool"] > summary["annual_total_primary"])
            & (summary["annual_total_primary"] > 0)
        ].copy()
        if not alt_gt_total.empty:
            debug_warn(
                "alternative service-entry pool exceeds primary annual total in some rows. "
                "This indicates concept mismatch (e.g. transfers from prior WIP can exceed current-year cash/addition totals)."
            )
            print(
                alt_gt_total[
                    [
                        "entity",
                        "year",
                        "annual_total_primary",
                        "alt_service_entry_pool",
                        "alt_service_entry_pool_share",
                    ]
                ].to_string(index=False)
            )

        no_dated = summary[
            (summary["annual_total_primary"] > 0)
            & (summary["dated_additions"] <= 0)
        ].copy()
        if not no_dated.empty:
            debug_warn(
                "No dated additions survive filtering in some positive-total rows. "
                "This usually means the event file either lacks numeric amounts or the surviving event amounts are too proxy-like to use."
            )
            print(
                no_dated[
                    [
                        "entity",
                        "year",
                        "annual_total_primary",
                        "dated_additions",
                        "annual_observed_undated",
                        "residual_unexplained",
                    ]
                ].to_string(index=False)
            )

    # Write outputs
    dep_path = output_dir / "depreciation_params_clean.tsv"
    annual_total_path = output_dir / "annual_total_components.tsv"
    annual_pool_path = output_dir / "annual_observed_pool_components.tsv"
    annual_service_pool_path = output_dir / "annual_service_entry_pool_components.tsv"
    events_path = output_dir / "observed_event_components.tsv"
    summary_path = output_dir / "emi_k_feedback_summary.tsv"
    diagnostics_path = output_dir / "emi_k_feedback_diagnostics.tsv"

    dep.to_csv(dep_path, sep="\t", index=False)
    annual_total.to_csv(annual_total_path, sep="\t", index=False)
    annual_pool.to_csv(annual_pool_path, sep="\t", index=False)
    annual_service_pool.to_csv(annual_service_pool_path, sep="\t", index=False)
    events.to_csv(events_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    diagnostics.to_csv(diagnostics_path, sep="\t", index=False)

    print("Wrote outputs to:")
    print(f"  {dep_path}")
    print(f"  {annual_total_path}")
    print(f"  {annual_pool_path}")
    print(f"  {annual_service_pool_path}")
    print(f"  {events_path}")
    print(f"  {summary_path}")
    print(f"  {diagnostics_path}")

    if not summary.empty:
        print("\nPreview of emi_k_feedback_summary.tsv:")
        preview_cols = [
            "entity",
            "year",
            "annual_total_primary",
            "dated_additions",
            "annual_observed_undated",
            "residual_unexplained",
            "dated_share",
            "annual_observed_undated_share",
            "residual_share",
            "alt_service_entry_pool",
            "alt_service_entry_pool_share",
        ]
        print(summary[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
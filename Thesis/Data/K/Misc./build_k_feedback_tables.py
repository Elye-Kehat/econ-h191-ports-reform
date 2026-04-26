#!/usr/bin/env python3
"""
Build diagnostic tables for the monthly-K redesign from the compiled master TSV.

Primary goal:
- produce the Emi-facing summary of what share of annual capital movement is:
    1) explicitly dated to month,
    2) observed annually but not month-dated,
    3) still residual / unexplained.

This script does NOT yet build final monthly K.
It is a diagnostic builder that turns the master TSV into clean accounting objects.

Usage:
    python build_k_feedback_tables.py \
        --input k_monthly_redesign_master.tsv \
        --output-dir output_k_feedback

Main outputs:
- depreciation_params_clean.tsv
- annual_total_components.tsv
- observed_event_components.tsv
- emi_k_feedback_summary.tsv
- emi_k_feedback_diagnostics.tsv

Assumptions used here are explicit and intentionally conservative.
They are written into the output columns so you can revise them later.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "k_monthly_redesign_master.tsv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "k_feedback_output"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _as_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()



def normalize_flag(x: object, default: Optional[bool] = None) -> Optional[bool]:
    s = _as_str(x).lower()
    if s in {"yes", "y", "true", "1"}:
        return True
    if s in {"no", "n", "false", "0"}:
        return False
    if s in {"maybe", "partial", "unknown", ""}:
        return default
    return default



def normalize_text(s: object) -> str:
    return _as_str(s).lower().replace("-", "_")



def annual_to_monthly_geometric(annual_rate: float) -> float:
    if pd.isna(annual_rate):
        return np.nan
    if annual_rate < 0 or annual_rate >= 1:
        return np.nan
    return 1.0 - (1.0 - annual_rate) ** (1.0 / 12.0)



def to_rate_decimal(value: float, unit: str, metric: str) -> float:
    """
    Convert a stored rate into decimal form when possible.
    Examples:
    - 0.05 with unit pct_decimal -> 0.05
    - 5 with unit pct -> 0.05
    """
    if pd.isna(value):
        return np.nan
    u = normalize_text(unit)
    m = normalize_text(metric)
    v = float(value)

    if u == "pct_decimal":
        return v
    if u == "pct":
        return v / 100.0
    if "rate" in m and v > 1.0:
        return v / 100.0
    return v



def safe_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.fillna(0).sum())



def choose_first_nonempty(values: List[object]) -> str:
    for v in values:
        s = _as_str(v)
        if s:
            return s
    return ""



def row_included(row: pd.Series) -> bool:
    include_in_k = normalize_flag(row.get("include_in_k"), default=True)
    return bool(include_in_k)



def row_in_productive_k(row: pd.Series) -> bool:
    include_in_k = normalize_flag(row.get("include_in_k"), default=True)
    include_prod = normalize_flag(row.get("include_in_productive_k"), default=True)
    return bool(include_in_k and include_prod)



def event_direction(row: pd.Series) -> str:
    """
    Heuristic parse of the HPC event notes.
    Returns one of: addition, disposal, excluded, ambiguous.
    """
    if not row_included(row):
        return "excluded"

    text = " ".join([
        _as_str(row.get("project_name")),
        _as_str(row.get("notes")),
        _as_str(row.get("value_text")),
    ]).lower()

    if "event_type=disposal" in text or "removal of" in text or "book value of disposed" in text:
        return "disposal"
    if "not capitalized" in text or "operating_project_not_ppe" in text:
        return "excluded"
    if "event_type=addition" in text or "event_type=relocation" in text or "event_type=addition_and_upgrade" in text:
        return "addition"
    if "purchase" in text or "new cranes" in text or "reactivation" in text or "rehab" in text:
        return "addition"
    return "ambiguous"



def productive_mask(df: pd.DataFrame) -> pd.Series:
    return df.apply(row_in_productive_k, axis=1)


# -----------------------------------------------------------------------------
# Load / basic cleanup
# -----------------------------------------------------------------------------


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    expected_cols = {
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
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing expected columns: {sorted(missing)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce")
    df["value_num"] = pd.to_numeric(df["value_num"], errors="coerce")

    # Standardize blank strings for convenience.
    text_cols = [
        "entity", "source_group", "source_file", "source_doc", "source_page", "source_section",
        "row_type", "asset_class_raw", "asset_class_std", "project_id", "project_name", "metric",
        "value_text", "unit", "currency", "date_precision", "event_date", "event_month", "confidence",
        "include_in_k", "include_in_productive_k", "needs_monthly_timing", "notes",
    ]
    for c in text_cols:
        df[c] = df[c].fillna("")

    return df


# -----------------------------------------------------------------------------
# Depreciation table
# -----------------------------------------------------------------------------


def build_depreciation_params(master: pd.DataFrame) -> pd.DataFrame:
    dep = master[master["row_type"] == "dep_policy"].copy()
    if dep.empty:
        return pd.DataFrame()

    out_rows = []
    group_cols = ["entity", "asset_class_std"]

    for (entity, asset_class_std), g in dep.groupby(group_cols, dropna=False):
        g = g.copy()

        asset_class_raw = choose_first_nonempty(g["asset_class_raw"].tolist())
        source_groups = sorted(set(x for x in g["source_group"].tolist() if x))
        source_docs = sorted(set(x for x in g["source_doc"].tolist() if x))
        source_sections = sorted(set(x for x in g["source_section"].tolist() if x))
        method = choose_first_nonempty(g.loc[g["metric"] == "depr_method", "value_text"].tolist())

        # Useful lives.
        life_point = np.nan
        life_min = np.nan
        life_max = np.nan

        s = g.loc[g["metric"] == "useful_life_years", "value_num"]
        if not s.empty:
            life_point = float(s.iloc[0])
            life_min = float(s.iloc[0])
            life_max = float(s.iloc[0])
        else:
            smin = g.loc[g["metric"] == "useful_life_years_min", "value_num"]
            smax = g.loc[g["metric"] == "useful_life_years_max", "value_num"]
            if not smin.empty:
                life_min = float(smin.iloc[0])
            if not smax.empty:
                life_max = float(smax.iloc[0])
            if not pd.isna(life_min) and not pd.isna(life_max):
                life_point = (life_min + life_max) / 2.0

        # Residuals.
        residual_min = np.nan
        residual_max = np.nan
        rmin = g.loc[g["metric"] == "residual_pct_min", ["value_num", "unit", "metric"]]
        rmax = g.loc[g["metric"] == "residual_pct_max", ["value_num", "unit", "metric"]]
        if not rmin.empty:
            residual_min = to_rate_decimal(rmin.iloc[0]["value_num"], rmin.iloc[0]["unit"], rmin.iloc[0]["metric"])
        if not rmax.empty:
            residual_max = to_rate_decimal(rmax.iloc[0]["value_num"], rmax.iloc[0]["unit"], rmax.iloc[0]["metric"])

        # Direct annual rates when present.
        direct_low = np.nan
        direct_high = np.nan
        direct_point = np.nan

        sr = g.loc[g["metric"] == "annual_depr_rate", ["value_num", "unit", "metric"]]
        if not sr.empty:
            direct_point = to_rate_decimal(sr.iloc[0]["value_num"], sr.iloc[0]["unit"], sr.iloc[0]["metric"])
            direct_low = direct_point
            direct_high = direct_point
        else:
            smin = g.loc[g["metric"] == "annual_depr_rate_pct_min", ["value_num", "unit", "metric"]]
            smax = g.loc[g["metric"] == "annual_depr_rate_pct_max", ["value_num", "unit", "metric"]]
            if not smin.empty:
                direct_low = to_rate_decimal(smin.iloc[0]["value_num"], smin.iloc[0]["unit"], smin.iloc[0]["metric"])
            if not smax.empty:
                direct_high = to_rate_decimal(smax.iloc[0]["value_num"], smax.iloc[0]["unit"], smax.iloc[0]["metric"])
            if not pd.isna(direct_low) and not pd.isna(direct_high):
                direct_point = (direct_low + direct_high) / 2.0

        # If no direct rate exists, derive from useful life as a transparent fallback.
        # This is intentionally simple: annual delta ~= 1 / useful life.
        # It is a diagnostic fallback, not a claim of exact economic depreciation.
        rate_source = "direct_policy_rate"
        annual_low = direct_low
        annual_high = direct_high
        annual_central = direct_point

        if pd.isna(annual_central):
            rate_source = "derived_from_life_inverse"
            if not pd.isna(life_min) and not pd.isna(life_max):
                annual_low = 1.0 / life_max if life_max > 0 else np.nan
                annual_high = 1.0 / life_min if life_min > 0 else np.nan
                annual_central = 1.0 / life_point if life_point > 0 else np.nan
            elif not pd.isna(life_point):
                annual_low = annual_high = annual_central = 1.0 / life_point if life_point > 0 else np.nan

        monthly_low = annual_to_monthly_geometric(annual_low)
        monthly_central = annual_to_monthly_geometric(annual_central)
        monthly_high = annual_to_monthly_geometric(annual_high)

        include_prod = True
        if not g.empty:
            include_prod = any(row_in_productive_k(r) for _, r in g.iterrows())

        out_rows.append(
            {
                "entity": entity,
                "asset_class_std": asset_class_std,
                "asset_class_raw": asset_class_raw,
                "depr_method": method,
                "useful_life_years_min": life_min,
                "useful_life_years_central": life_point,
                "useful_life_years_max": life_max,
                "residual_pct_min": residual_min,
                "residual_pct_max": residual_max,
                "annual_delta_low": annual_low,
                "annual_delta_central": annual_central,
                "annual_delta_high": annual_high,
                "monthly_delta_low": monthly_low,
                "monthly_delta_central": monthly_central,
                "monthly_delta_high": monthly_high,
                "annual_delta_source": rate_source,
                "include_in_productive_k": "yes" if include_prod else "no",
                "source_groups": " | ".join(source_groups),
                "source_docs": " | ".join(source_docs),
                "source_sections": " | ".join(source_sections),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["entity", "asset_class_std"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Annual totals and event buckets
# -----------------------------------------------------------------------------


def build_annual_total_components(master: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # HPC: use class-level purchases plus new lease additions as annual capital flow proxy.
    hpc = master[(master["entity"] == "HPC") & (master["row_type"] == "ppe_rollforward")].copy()
    hpc = hpc[productive_mask(hpc)]
    hpc = hpc[hpc["metric"].isin(["purchases", "new_leases_additions"])]
    hpc = hpc[hpc["value_num"] > 0]
    for _, r in hpc.iterrows():
        rows.append(
            {
                "entity": "HPC",
                "year": int(r["year"]),
                "asset_class_std": r["asset_class_std"],
                "component_type": "annual_total_primary",
                "component_subtype": r["metric"],
                "value": float(r["value_num"]),
                "source_rule": "HPC primary total = class purchases + new_leases_additions from PPE rollforward",
                "record_id": r["record_id"],
            }
        )

    # IPC: use positive commissioned_assets_or_transfers into productive classes, excluding total and wip.
    ipc = master[(master["entity"] == "IPC") & (master["row_type"] == "ppe_rollforward")].copy()
    ipc = ipc[productive_mask(ipc)]
    ipc = ipc[ipc["metric"] == "commissioned_assets_or_transfers"]
    ipc = ipc[ipc["asset_class_std"].isin(["total", "wip"]) == False]
    ipc = ipc[ipc["value_num"] > 0]
    for _, r in ipc.iterrows():
        rows.append(
            {
                "entity": "IPC",
                "year": int(r["year"]),
                "asset_class_std": r["asset_class_std"],
                "component_type": "annual_total_primary",
                "component_subtype": r["metric"],
                "value": float(r["value_num"]),
                "source_rule": "IPC primary total = commissioned_assets_or_transfers into productive PPE classes (exclude wip and total)",
                "record_id": r["record_id"],
            }
        )

    # SIPG / Bayport: use CIP transfer_to_ppe as the annual movement into service.
    sipg = master[(master["entity"] == "SIPG") & (master["row_type"] == "cip_flow")].copy()
    sipg = sipg[sipg["metric"] == "transfer_to_ppe"]
    sipg = sipg[sipg["value_num"] > 0]
    sipg = sipg[sipg.apply(row_included, axis=1)]
    for _, r in sipg.iterrows():
        rows.append(
            {
                "entity": "SIPG",
                "year": int(r["year"]),
                "asset_class_std": r["asset_class_std"],
                "component_type": "annual_total_primary",
                "component_subtype": r["metric"],
                "value": float(r["value_num"]),
                "source_rule": "SIPG primary total = Bayport CIP transfer_to_ppe",
                "record_id": r["record_id"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["entity", "year", "asset_class_std", "component_subtype"]).reset_index(drop=True)
    return out



def build_event_components(master: pd.DataFrame) -> pd.DataFrame:
    events = master[master["row_type"] == "event"].copy()
    if events.empty:
        return pd.DataFrame()

    rows = []
    for _, r in events.iterrows():
        direction = event_direction(r)
        if direction == "excluded":
            continue

        precision = normalize_text(r["date_precision"]) or "unknown"
        value = float(r["value_num"]) if not pd.isna(r["value_num"]) else np.nan

        component_type = "observed_event_undated"
        if precision == "year_month":
            component_type = "observed_event_dated"
        elif precision in {"year", "annual_known_month_unknown"}:
            component_type = "observed_event_undated"
        else:
            component_type = "observed_event_unknown_precision"

        rows.append(
            {
                "entity": r["entity"],
                "year": int(r["year"]),
                "asset_class_std": r["asset_class_std"],
                "component_type": component_type,
                "component_subtype": direction,
                "value": value,
                "source_rule": f"event rows classified heuristically from notes; date_precision={precision}",
                "record_id": r["record_id"],
                "project_id": r["project_id"],
                "project_name": r["project_name"],
                "event_month": r["event_month"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["entity", "year", "component_type", "project_id"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Emi summary
# -----------------------------------------------------------------------------


def build_emi_summary(master: pd.DataFrame, annual_total_components: pd.DataFrame, event_components: pd.DataFrame) -> pd.DataFrame:
    entity_years = master[["entity", "year"]].dropna().drop_duplicates().copy()
    entity_years["year"] = entity_years["year"].astype(int)
    entity_years = entity_years.sort_values(["entity", "year"]).reset_index(drop=True)

    rows = []

    for _, ey in entity_years.iterrows():
        entity = ey["entity"]
        year = int(ey["year"])

        total_rows = annual_total_components[(annual_total_components["entity"] == entity) & (annual_total_components["year"] == year)]
        event_rows = event_components[(event_components["entity"] == entity) & (event_components["year"] == year)]

        annual_total = safe_sum(total_rows["value"])

        dated_additions = safe_sum(event_rows[(event_rows["component_type"] == "observed_event_dated") & (event_rows["component_subtype"] == "addition")]["value"])
        undated_event_additions = safe_sum(event_rows[(event_rows["component_type"] != "observed_event_dated") & (event_rows["component_subtype"] == "addition")]["value"])
        dated_disposal_proxy = safe_sum(event_rows[(event_rows["component_type"] == "observed_event_dated") & (event_rows["component_subtype"] == "disposal")]["value"])
        undated_disposal_proxy = safe_sum(event_rows[(event_rows["component_type"] != "observed_event_dated") & (event_rows["component_subtype"] == "disposal")]["value"])

        # Entity-specific logic for annual-but-undated observed amount.
        if entity in {"IPC", "SIPG"}:
            # For IPC and SIPG, the primary total already comes from annual observed transfer/commissioning pools.
            # So the annual-but-undated amount is whatever remains after subtracting explicitly month-dated additions.
            annual_observed_undated = max(annual_total - dated_additions, 0.0)
            annual_observed_rule = (
                f"{entity}: annual_observed_undated = annual_total_primary - dated_additions, because annual_total is already an observed annual transfer/commissioning pool"
            )
        else:
            # For HPC, only explicit year-only event additions are counted as observed annual-but-undated.
            annual_observed_undated = undated_event_additions
            annual_observed_rule = (
                "HPC: annual_observed_undated comes only from year-precision event additions; purchases not linked to explicit events remain residual"
            )

        residual_unexplained = max(annual_total - dated_additions - annual_observed_undated, 0.0)
        observed_excess_over_annual_total = max(dated_additions + annual_observed_undated - annual_total, 0.0)

        dated_share = dated_additions / annual_total if annual_total > 0 else np.nan
        annual_observed_undated_share = annual_observed_undated / annual_total if annual_total > 0 else np.nan
        observed_total_share = (dated_additions + annual_observed_undated) / annual_total if annual_total > 0 else np.nan
        residual_share = residual_unexplained / annual_total if annual_total > 0 else np.nan

        likely_issue = ""
        recommended_next_source = ""
        if entity == "HPC":
            if residual_unexplained > 0:
                likely_issue = "Purchases are only partly linked to dated projects; commissioning month and disposal book value still incomplete"
                recommended_next_source = "Targeted HPC annual reports / fixed-asset and held-for-sale notes / project-level materials"
            else:
                likely_issue = "Most annual movement linked to explicit events"
                recommended_next_source = "Optional cleanup only"
        elif entity == "IPC":
            likely_issue = "Annual commissioning/transfer totals are observed, but month-level timing is mostly missing"
            recommended_next_source = "Targeted IPC project timing sources, not broad annual extraction"
        elif entity == "SIPG":
            likely_issue = "Annual transfer-to-PPE is observed, but month-level timing and Bayport-only class mix remain weak"
            recommended_next_source = "Targeted Bayport project/subsidiary materials; timing sources if available"

        total_basis = ""
        if not total_rows.empty:
            total_basis = " | ".join(sorted(total_rows["source_rule"].dropna().unique().tolist()))

        rows.append(
            {
                "entity": entity,
                "year": year,
                "annual_total_primary": annual_total,
                "dated_additions": dated_additions,
                "annual_observed_undated": annual_observed_undated,
                "residual_unexplained": residual_unexplained,
                "dated_share": dated_share,
                "annual_observed_undated_share": annual_observed_undated_share,
                "observed_total_share": observed_total_share,
                "residual_share": residual_share,
                "dated_disposal_proxy": dated_disposal_proxy,
                "undated_disposal_proxy": undated_disposal_proxy,
                "observed_excess_over_annual_total": observed_excess_over_annual_total,
                "annual_total_basis": total_basis,
                "annual_observed_rule": annual_observed_rule,
                "likely_issue": likely_issue,
                "recommended_next_source": recommended_next_source,
            }
        )

    out = pd.DataFrame(rows).sort_values(["entity", "year"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Diagnostics / human-readable notes
# -----------------------------------------------------------------------------


def build_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        status = "ready_for_emi"
        if r["annual_total_primary"] <= 0:
            status = "no_annual_total_identified"
        elif r.get("observed_excess_over_annual_total", 0) > 0:
            status = "event_proxies_exceed_booked_total_refine_amounts"
        elif r["residual_share"] > 0.25:
            status = "needs_targeted_extraction_before_emi"
        elif r["annual_observed_undated_share"] > 0.5 and r["dated_share"] < 0.25:
            status = "good_annual_accounting_but_weak_monthly_timing"

        rows.append(
            {
                "entity": r["entity"],
                "year": int(r["year"]),
                "status": status,
                "headline": (
                    f"{r['entity']} {int(r['year'])}: dated={r['dated_share']:.1%} | annual-undated={r['annual_observed_undated_share']:.1%} | residual={r['residual_share']:.1%}"
                    if pd.notna(r["dated_share"]) else f"{r['entity']} {int(r['year'])}: no annual total identified"
                ),
                "likely_issue": r["likely_issue"],
                "recommended_next_source": r["recommended_next_source"],
            }
        )
    return pd.DataFrame(rows).sort_values(["entity", "year"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build monthly-K redesign diagnostic tables from the master TSV.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to k_monthly_redesign_master.tsv",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output tables",
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input TSV at:\n{input_path}\n\n"
            f"Script location:\n{SCRIPT_DIR}\n\n"
            "Make sure k_monthly_redesign_master.tsv is in the same folder as build_k_feedback_tables.py."
    )

    master = load_master(input_path)
    depreciation = build_depreciation_params(master)
    annual_total_components = build_annual_total_components(master)
    event_components = build_event_components(master)
    summary = build_emi_summary(master, annual_total_components, event_components)
    diagnostics = build_diagnostics(summary)

    depreciation.to_csv(output_dir / "depreciation_params_clean.tsv", sep="\t", index=False)
    annual_total_components.to_csv(output_dir / "annual_total_components.tsv", sep="\t", index=False)
    event_components.to_csv(output_dir / "observed_event_components.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "emi_k_feedback_summary.tsv", sep="\t", index=False)
    diagnostics.to_csv(output_dir / "emi_k_feedback_diagnostics.tsv", sep="\t", index=False)

    print("Wrote outputs to:")
    print(f"  {output_dir / 'depreciation_params_clean.tsv'}")
    print(f"  {output_dir / 'annual_total_components.tsv'}")
    print(f"  {output_dir / 'observed_event_components.tsv'}")
    print(f"  {output_dir / 'emi_k_feedback_summary.tsv'}")
    print(f"  {output_dir / 'emi_k_feedback_diagnostics.tsv'}")

    if not summary.empty:
        print("\nPreview of emi_k_feedback_summary.tsv:")
        cols = [
            "entity",
            "year",
            "annual_total_primary",
            "dated_additions",
            "annual_observed_undated",
            "residual_unexplained",
            "dated_share",
            "annual_observed_undated_share",
            "residual_share",
        ]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Build a clean K-interpolation input package from:
- k_monthly_redesign_master_v2.tsv
- k_monthly_redesign_master_v3.tsv
- hpc_reports_complement_extract.csv

Outputs
-------
1. k_monthly_redesign_master_v4_repaired.tsv
2. k_entity_rules.tsv
3. k_entity_year_anchors.tsv
4. k_dep_lookup.tsv
5. k_dated_events.tsv
6. k_annual_class_pools.tsv
7. k_month_shares.tsv
8. k_package_qa_summary.tsv

Design notes
------------
This script is intentionally "code-prep" rather than "final economics."
It tries to:
- repair v3 using v2 and the HPC supplement
- produce a narrow input package for a new interpolation codebase
- preserve SIPG as a legitimate missing-data case where appropriate

Units
-----
The package is written in kNIS-style amounts where the source archive already
reports those values in NIS_thousands / kNIS terms. The script does not do a
full FX + deflator conversion. That is a later step in the new pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


REQUIRED_COLS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", required=True, help="Path to k_monthly_redesign_master_v2.tsv")
    parser.add_argument("--v3", required=True, help="Path to k_monthly_redesign_master_v3.tsv")
    parser.add_argument("--hpc", required=True, help="Path to hpc_reports_complement_extract.csv")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path, dtype=str)
    return pd.read_csv(path, sep="\t", dtype=str)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def blank_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].replace(r"^\s*$", np.nan, regex=True)
    return out


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_yes_no(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return 1
    return 0


def canonical_asset_class(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    mapping = {
        "infrastructure_buildings_structures": "infrastructure_buildings",
        "infrastructure_buildings_infrastructure": "infrastructure_buildings",
        "spare_parts_under_15000": "spare_parts",
        "spare_parts_over_15000": "spare_parts",
        "spare_parts_inventory": "spare_parts",
        "right_of_use_real_estate": "right_of_use_assets",
        "right_of_use_vehicles": "right_of_use_assets",
        "mixed_terminal_infrastructure": "mixed",
        "marine_assets_disposal": "marine_assets",
        "mixed_relocation_assets": "mixed",
        "terminal_infrastructure": "mixed",
        "leased_operating_equipment": "other_operational_equipment",
        "mobile_cranes_heavy_equipment": "other_operational_equipment",
        "wip_bayport": "unknown_bayport_class",
    }
    return mapping.get(s, s)


def repair_v3(v2_path: Path, v3_path: Path, hpc_path: Path) -> pd.DataFrame:
    v2 = blank_to_nan(read_table(v2_path)[REQUIRED_COLS])
    v3 = blank_to_nan(read_table(v3_path)[REQUIRED_COLS])
    hpc = blank_to_nan(read_table(hpc_path)[REQUIRED_COLS])

    reference = pd.concat([v2, hpc], ignore_index=True)
    reference = reference.drop_duplicates("record_id", keep="last").set_index("record_id")

    out = v3.drop_duplicates("record_id", keep="last").set_index("record_id")

    for c in REQUIRED_COLS:
        if c == "record_id":
            continue
        if c not in out.columns:
            out[c] = np.nan
        if c in reference.columns:
            out[c] = out[c].combine_first(reference[c])

    missing_ids = sorted(set(reference.index) - set(out.index))
    if missing_ids:
        out = pd.concat([out, reference.loc[missing_ids]], axis=0)

    out = out.reset_index()[REQUIRED_COLS]
    return out


def build_entity_rules() -> pd.DataFrame:
    rows = [
        ["HPC", "Haifa Port Company Legacy", "Haifa", 1, 0, np.nan, 0, 0, "average_across_categories", "uniform", "Legacy terminal"],
        ["IPC", "Israel Ports Company Haifa Infrastructure", "Haifa", 0, 1, np.nan, 0, 0, "average_across_categories", "backloaded", "Landlord infrastructure"],
        ["SIPG", "Bayport SIPG", "Haifa", 1, 0, "2021-09", 1, 1, "average_across_categories", "post_opening_only", "Annual-only Bayport detail"],
    ]
    cols = [
        "entity",
        "entity_label",
        "port",
        "is_operator",
        "is_landlord",
        "opening_month",
        "pre_opening_zero_rule",
        "allow_empty_dated_events",
        "dep_fallback_method",
        "default_share_rule",
        "notes",
    ]
    return pd.DataFrame(rows, columns=cols)


def build_anchors(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["val"] = to_num(m["value_num"])

    def first_val(sub: pd.DataFrame, metrics: list[str], prefer_sources: list[str] | None = None, absval: bool = False):
        x = sub[sub["metric"].isin(metrics)].copy()
        x = x[x["val"].notna()]
        if x.empty:
            return np.nan, np.nan
        if prefer_sources:
            x["src_rank"] = x["source_group"].apply(lambda s: prefer_sources.index(s) if s in prefer_sources else 999)
            x = x.sort_values(["src_rank", "source_group"])
        row = x.iloc[0]
        val = float(row["val"])
        if absval:
            val = abs(val)
        return val, row["source_group"]

    rows = []

    # HPC
    hpc_years = sorted(pd.to_numeric(m[(m["entity"] == "HPC") & (m["metric"] == "ppe_net_eoy")]["year"], errors="coerce").dropna().astype(int).unique())
    for year in hpc_years:
        sub = m[(m["entity"] == "HPC") & (m["year"].astype(str) == str(year))]
        K_anchor, src_k = first_val(sub, ["ppe_net_eoy"], ["THESIS_2026_SUMMARY", "HPC_FINANCIALS_RAW"])
        I_total, src_i = first_val(sub, ["approx_gross_ppe_investment"], ["THESIS_2026_SUMMARY"])
        if pd.isna(I_total):
            I_total, src_i = first_val(sub, ["purchase_fixed_assets_cashflow"], ["HPC_FINANCIALS_RAW"], absval=True)
        disposals, src_d = first_val(sub, ["disposal_proceeds"], ["THESIS_2026_SUMMARY"])
        if pd.isna(disposals):
            disposals, src_d = first_val(sub, ["proceeds_realization_fixed_assets"], ["HPC_FINANCIALS_RAW"], absval=True)
        dep_exp, src_dep = first_val(sub, ["dep_amort_total"], ["THESIS_2026_SUMMARY"])
        if pd.isna(dep_exp):
            dep_exp, src_dep = first_val(sub, ["depr_amort_total_cf_adj"], ["HPC_FINANCIALS_RAW"])
        wip = sub[sub["row_type"] == "wip_indicator"]["val"].sum(min_count=1)

        rows.append([
            "HPC",
            year,
            K_anchor,
            I_total,
            np.nan,
            disposals,
            dep_exp,
            wip,
            "net_ppe",
            1,
            src_k if pd.notna(src_k) else (src_i if pd.notna(src_i) else "HPC"),
            "Operating PPE anchor",
        ])

    # IPC
    ipc = m[m["entity"] == "IPC"].copy()
    ipc_years = sorted(pd.to_numeric(ipc["year"], errors="coerce").dropna().astype(int).unique())
    for year in ipc_years:
        sub = ipc[ipc["year"].astype(str) == str(year)]

        x = sub[(sub["row_type"] == "ppe_rollforward") & (sub["asset_class_std"] == "total") & (sub["metric"] == "nbv_close") & (sub["val"].notna())]
        if not x.empty:
            K_anchor = float(x.iloc[0]["val"])
            src_k = x.iloc[0]["source_group"]
        else:
            K_anchor, src_k = first_val(sub, ["ppe_net_eoy"], ["THESIS_2026_SUMMARY"])

        x = sub[(sub["row_type"] == "ppe_rollforward") & (sub["asset_class_std"] == "total") & (sub["metric"].isin(["additions", "capitalized_interest"])) & (sub["val"].notna())]
        I_total = float(x["val"].sum()) if not x.empty else np.nan

        x = sub[
            (sub["row_type"] == "ppe_rollforward")
            & (~sub["asset_class_std"].isin(["total", "wip"]))
            & (sub["metric"] == "commissioned_assets_or_transfers")
            & (sub["val"] > 0)
        ]
        explicit_service = float(x["val"].sum()) if not x.empty else np.nan
        if pd.notna(I_total) and pd.notna(explicit_service) and explicit_service > 1.5 * I_total:
            explicit_service = np.nan

        x = sub[(sub["row_type"] == "ppe_rollforward") & (sub["asset_class_std"] == "total") & (sub["metric"] == "disposals") & (sub["val"].notna())]
        disposals = abs(float(x.iloc[0]["val"])) if not x.empty else np.nan

        x = sub[(sub["row_type"] == "ppe_rollforward") & (sub["asset_class_std"] == "total") & (sub["metric"] == "depreciation_expense") & (sub["val"].notna())]
        dep_exp = float(x.iloc[0]["val"]) if not x.empty else np.nan

        x = sub[(sub["row_type"] == "ppe_rollforward") & (sub["asset_class_std"] == "wip") & (sub["metric"] == "nbv_close") & (sub["val"].notna())]
        wip = float(x.iloc[0]["val"]) if not x.empty else np.nan

        rows.append([
            "IPC",
            year,
            K_anchor,
            I_total,
            explicit_service,
            disposals,
            dep_exp,
            wip,
            "net_ppe",
            1,
            src_k if pd.notna(src_k) else "IPC_RAW_TABLE1",
            "IPC total anchor from Note 8 / summary",
        ])

    # SIPG
    sipg = m[m["entity"] == "SIPG"].copy()
    sipg_years = sorted(pd.to_numeric(sipg["year"], errors="coerce").dropna().astype(int).unique())
    for year in sipg_years:
        sub = sipg[sipg["year"].astype(str) == str(year)]

        def thesis_metric(metric: str):
            x = sub[(sub["source_group"] == "THESIS_2026_SUMMARY") & (sub["metric"] == metric) & (sub["val"].notna())]
            if x.empty:
                return np.nan, np.nan
            return float(x.iloc[0]["val"]), x.iloc[0]["source_group"]

        K_anchor, src_k = thesis_metric("cum_transfer_to_ppe_knis")
        if year < 2021:
            K_anchor = 0.0

        I_total, src_i = thesis_metric("cip_additions_plus_capint_knis")
        explicit_service, src_e = thesis_metric("transfer_to_ppe_knis")
        wip, src_w = thesis_metric("cip_close_knis")

        rows.append([
            "SIPG",
            year,
            K_anchor,
            I_total,
            explicit_service,
            0.0,
            np.nan,
            wip,
            "cumulative_transfers",
            1,
            src_k if pd.notna(src_k) else (src_i if pd.notna(src_i) else "SIPG"),
            "Bayport productive K rough anchor",
        ])

    cols = [
        "entity",
        "year",
        "K_anchor_dec_kNIS",
        "I_annual_total_kNIS",
        "explicit_service_entry_annual_kNIS",
        "disposals_annual_kNIS",
        "dep_expense_annual_kNIS",
        "wip_eoy_kNIS",
        "anchor_basis",
        "include_in_preferred_spec",
        "source_id",
        "notes",
    ]
    return pd.DataFrame(rows, columns=cols).sort_values(["entity", "year"]).reset_index(drop=True)


def normalize_rate(val) -> float:
    if pd.isna(val):
        return np.nan
    v = float(val)
    return v / 100.0 if v > 1 else v


def annual_to_monthly_dep_rate(delta_annual: float) -> float:
    if pd.isna(delta_annual):
        return np.nan
    return 1.0 - (1.0 - delta_annual) ** (1.0 / 12.0)


def build_dep_lookup(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["val"] = to_num(m["value_num"])
    m["asset_class_std_canon"] = m["asset_class_std"].map(canonical_asset_class)

    rows = []

    for entity in ["HPC", "IPC", "SIPG"]:
        sub = m[(m["entity"] == entity) & (m["row_type"] == "dep_policy")].copy()
        if sub.empty:
            continue

        for cls in sorted(sub["asset_class_std_canon"].dropna().unique()):
            ss = sub[sub["asset_class_std_canon"] == cls].copy()

            annual = np.nan
            dep_method = np.nan
            source_id = np.nan

            # 1. Direct annual rate
            x = ss[(ss["metric"] == "annual_depr_rate") & (ss["val"].notna())]
            if not x.empty:
                annual = normalize_rate(x.iloc[0]["val"])
                dep_method = "reported_rate"
                source_id = "|".join(sorted(x["source_group"].dropna().unique()))

            # 2. Main percentage
            if pd.isna(annual):
                x = ss[(ss["metric"] == "annual_depreciation_rate_main_pct") & (ss["val"].notna())]
                if not x.empty:
                    annual = normalize_rate(x.iloc[0]["val"])
                    dep_method = "reported_rate"
                    source_id = "|".join(sorted(x["source_group"].dropna().unique()))

            # 3. Low/high midpoint
            if pd.isna(annual):
                low = ss[ss["metric"].isin(["annual_depreciation_rate_low_pct", "annual_depr_rate_pct_min"])]["val"].dropna().astype(float)
                high = ss[ss["metric"].isin(["annual_depreciation_rate_high_pct", "annual_depr_rate_pct_max"])]["val"].dropna().astype(float)
                vals = []
                if len(low):
                    vals.append(low.mean())
                if len(high):
                    vals.append(high.mean())
                if vals:
                    annual = np.mean(vals) / 100.0
                    dep_method = "reported_rate"
                    source_id = "|".join(sorted(ss[ss["metric"].isin([
                        "annual_depreciation_rate_low_pct",
                        "annual_depr_rate_pct_min",
                        "annual_depreciation_rate_high_pct",
                        "annual_depr_rate_pct_max",
                    ])]["source_group"].dropna().unique()))

            # 4. Useful-life implied
            if pd.isna(annual):
                life_vals = ss[ss["metric"].isin(["useful_life_years", "useful_life_years_min", "useful_life_years_max"])]["val"].dropna().astype(float).tolist()
                if life_vals:
                    annual = 1.0 / float(np.mean(life_vals))
                    dep_method = "implied_life"
                    source_id = "|".join(sorted(ss[ss["metric"].isin(["useful_life_years", "useful_life_years_min", "useful_life_years_max"])]["source_group"].dropna().unique()))

            if pd.notna(annual):
                rows.append([
                    entity,
                    cls,
                    annual,
                    annual_to_monthly_dep_rate(annual),
                    dep_method,
                    0,
                    np.nan,
                    source_id,
                    "Observed or implied class rate",
                ])

    dep = pd.DataFrame(
        rows,
        columns=[
            "entity",
            "asset_class_std",
            "dep_rate_annual",
            "dep_rate_monthly",
            "dep_method",
            "is_fallback",
            "fallback_group",
            "source_id",
            "notes",
        ],
    )

    # Add fallback rows for any class used in productive rows that still has no rate
    m["asset_class_std_canon"] = m["asset_class_std"].map(canonical_asset_class)
    productive = m[m["include_in_productive_k"].apply(parse_yes_no) == 1].copy()
    productive = productive[~productive["asset_class_std_canon"].isin(["total", "wip"])]
    used_classes = productive.groupby("entity")["asset_class_std_canon"].apply(lambda s: set(s.dropna())).to_dict()

    for entity, cls_set in used_classes.items():
        observed = dep[(dep["entity"] == entity) & (dep["is_fallback"] == 0)].copy()
        obs_rates = observed["dep_rate_annual"]
        if entity == "SIPG":
            obs_rates = obs_rates[(obs_rates > 0) & (obs_rates < 0.5)]
        fallback = float(obs_rates.mean()) if len(obs_rates) else 0.06

        for cls in sorted(cls_set):
            if dep[(dep["entity"] == entity) & (dep["asset_class_std"] == cls)].empty:
                dep.loc[len(dep)] = [
                    entity,
                    cls,
                    fallback,
                    annual_to_monthly_dep_rate(fallback),
                    "fallback_average",
                    1,
                    "mixed",
                    f"{entity}_FALLBACK",
                    "Fallback average across categories",
                ]

    return dep.sort_values(["entity", "asset_class_std", "is_fallback"]).reset_index(drop=True)


def parse_event_month(row: pd.Series):
    for field in ["event_month", "event_date"]:
        val = row.get(field)
        if pd.notna(val):
            s = str(val).strip()
            m = re.search(r"(\d{4})-(\d{1,2})", s)
            if m:
                year = int(m.group(1))
                month = int(m.group(2))
                if 1 <= month <= 12:
                    return f"{year:04d}-{month:02d}"
    return np.nan


def parse_event_type(row: pd.Series):
    txt = " ".join([
        str(row.get("source_section") or ""),
        str(row.get("notes") or ""),
        str(row.get("metric") or ""),
    ])
    m = re.search(r"event_type=([A-Za-z_]+)", txt)
    raw = m.group(1).lower() if m else ""

    mapping = {
        "addition": "service_entry",
        "addition_and_upgrade": "service_entry",
        "disposal": "disposal",
        "relocation": "transfer_in",
        "operating_project_not_ppe": "other",
    }
    if raw in mapping:
        return mapping[raw]
    return "other"


def build_dated_events(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["val"] = to_num(m["value_num"])
    m["asset_class_std_canon"] = m["asset_class_std"].map(canonical_asset_class)

    events = m[m["row_type"].isin(["event", "project_event"])].copy()
    events["event_month_std"] = events.apply(parse_event_month, axis=1)
    events["event_type_std"] = events.apply(parse_event_type, axis=1)
    events["include_prod"] = events["include_in_productive_k"].apply(parse_yes_no)

    # Conservative exclusions:
    # - no parsed month
    # - no numeric amount
    # - explicitly not productive
    # - "other" event type
    # - notes that clearly say the observed month is not a commissioning month
    bad_phrase = events["notes"].fillna("").str.contains("Commission upon completion", case=False, na=False)

    events = events[
        (events["include_prod"] == 1)
        & (events["event_month_std"].notna())
        & (events["val"].notna())
        & (events["event_type_std"].isin(["service_entry", "disposal", "transfer_out", "transfer_in"]))
        & (~bad_phrase)
    ].copy()

    events["amount_sign"] = np.where(events["event_type_std"].isin(["disposal", "transfer_out"]), -1, 1)
    events["include_in_productive_K"] = 1

    out = events[
        [
            "entity",
            "record_id",
            "project_id",
            "project_name",
            "asset_class_std_canon",
            "event_month_std",
            "event_type_std",
            "val",
            "amount_sign",
            "confidence",
            "source_group",
            "notes",
            "include_in_productive_K",
        ]
    ].copy()

    out.columns = [
        "entity",
        "source_record_id",
        "project_id",
        "project_name",
        "asset_class_std",
        "event_month",
        "event_type",
        "amount_kNIS",
        "amount_sign",
        "confidence",
        "source_id",
        "notes",
        "include_in_productive_K",
    ]

    return out.sort_values(["entity", "event_month", "project_id"]).reset_index(drop=True)


def build_class_pools(master: pd.DataFrame, anchors: pd.DataFrame):
    m = master.copy()
    m["val"] = to_num(m["value_num"])
    m["asset_class_std_canon"] = m["asset_class_std"].map(canonical_asset_class)

    rows = []
    diag_rows = []

    # HPC
    hpc = m[(m["entity"] == "HPC") & (m["row_type"] == "ppe_rollforward")].copy()
    hpc = hpc[~hpc["asset_class_std_canon"].isin(["total", "wip"])]
    hpc_years = sorted(pd.to_numeric(hpc["year"], errors="coerce").dropna().astype(int).unique())

    for year in hpc_years:
        annual_total_series = anchors[(anchors["entity"] == "HPC") & (anchors["year"] == year)]["I_annual_total_kNIS"]
        annual_total = float(annual_total_series.iloc[0]) if (not annual_total_series.empty and pd.notna(annual_total_series.iloc[0])) else np.nan

        sub = hpc[hpc["year"].astype(str) == str(year)]

        mapped_sum = 0.0
        for pool_type, metrics in {
            "mapped_undated": ["cost_additions", "purchases"],
            "lease_addition": ["cost_new_leases", "new_leases_additions"],
        }.items():
            grp = sub[sub["metric"].isin(metrics)].groupby("asset_class_std_canon", dropna=True)["val"].sum().reset_index()
            for _, r in grp.iterrows():
                if pd.isna(r["val"]) or abs(float(r["val"])) < 1e-9:
                    continue
                rows.append([
                    "HPC",
                    year,
                    r["asset_class_std_canon"],
                    pool_type,
                    float(r["val"]),
                    1,
                    1 if pool_type == "mapped_undated" else 2,
                    "|".join(sorted(sub[sub["metric"].isin(metrics)]["source_group"].dropna().unique())),
                    "From class-level ppe rollforward",
                ])
                if pool_type == "mapped_undated":
                    mapped_sum += float(r["val"])

        if pd.notna(annual_total):
            bg = max(annual_total - mapped_sum, 0.0)
            rows.append([
                "HPC",
                year,
                "mixed",
                "background",
                bg,
                1,
                3,
                "HPC_BG_RESIDUAL",
                "Residual after mapped undated pools",
            ])
            diag_rows.append(["HPC", year, annual_total, mapped_sum, bg])

    # IPC
    ipc = m[(m["entity"] == "IPC") & (m["row_type"] == "ppe_rollforward")].copy()
    ipc = ipc[~ipc["asset_class_std_canon"].isin(["total", "wip"])]
    ipc_years = sorted(pd.to_numeric(ipc["year"], errors="coerce").dropna().astype(int).unique())

    for year in ipc_years:
        annual_total_series = anchors[(anchors["entity"] == "IPC") & (anchors["year"] == year)]["I_annual_total_kNIS"]
        annual_total = float(annual_total_series.iloc[0]) if (not annual_total_series.empty and pd.notna(annual_total_series.iloc[0])) else np.nan

        sub = ipc[ipc["year"].astype(str) == str(year)]
        grp = sub[sub["metric"].isin(["additions", "capitalized_interest"])].groupby("asset_class_std_canon", dropna=True)["val"].sum().reset_index()

        mapped_sum = 0.0
        for _, r in grp.iterrows():
            if pd.isna(r["val"]) or abs(float(r["val"])) < 1e-9:
                continue
            rows.append([
                "IPC",
                year,
                r["asset_class_std_canon"],
                "mapped_undated",
                float(r["val"]),
                1,
                1,
                "IPC_RAW_TABLE1",
                "From class additions + capitalized interest",
            ])
            mapped_sum += float(r["val"])

        if pd.notna(annual_total):
            bg = max(annual_total - mapped_sum, 0.0)
            rows.append([
                "IPC",
                year,
                "mixed",
                "background",
                bg,
                1,
                3,
                "IPC_BG_RESIDUAL",
                "Residual after class-observed pools",
            ])
            diag_rows.append(["IPC", year, annual_total, mapped_sum, bg])

    # SIPG
    sipg = anchors[(anchors["entity"] == "SIPG") & (anchors["year"] >= 2021) & (anchors["I_annual_total_kNIS"].notna())].copy()
    for _, r in sipg.iterrows():
        rows.append([
            "SIPG",
            int(r["year"]),
            "unknown_bayport_class",
            "background",
            float(r["I_annual_total_kNIS"]),
            1,
            3,
            "SIPG_BG_ANNUAL",
            "Annual Bayport investment with missing class timing",
        ])
        diag_rows.append(["SIPG", int(r["year"]), float(r["I_annual_total_kNIS"]), 0.0, float(r["I_annual_total_kNIS"])])

    pools = pd.DataFrame(
        rows,
        columns=[
            "entity",
            "year",
            "asset_class_std",
            "pool_type",
            "annual_amount_kNIS",
            "include_in_productive_K",
            "allocation_priority",
            "source_id",
            "notes",
        ],
    ).sort_values(["entity", "year", "allocation_priority", "asset_class_std"]).reset_index(drop=True)

    diag = pd.DataFrame(
        diag_rows,
        columns=["entity", "year", "annual_total_kNIS", "mapped_undated_sum_kNIS", "background_kNIS"],
    ).sort_values(["entity", "year"]).reset_index(drop=True)

    return pools, diag


def generate_shares_for_year(rule_name: str, year: int, opening_month: str | float | None):
    months = [f"{year}-{m:02d}" for m in range(1, 13)]

    if rule_name == "uniform":
        shares = np.repeat(1.0 / 12.0, 12)

    elif rule_name == "backloaded":
        weights = np.arange(1, 13, dtype=float)
        shares = weights / weights.sum()

    elif rule_name == "post_opening_only":
        if pd.isna(opening_month):
            shares = np.repeat(1.0 / 12.0, 12)
        else:
            open_year, open_month_num = map(int, str(opening_month).split("-")[:2])
            if year < open_year:
                return None
            elif year == open_year:
                shares = np.array([0.0] * (open_month_num - 1) + [1.0 / (13 - open_month_num)] * (13 - open_month_num))
            else:
                shares = np.repeat(1.0 / 12.0, 12)

    else:
        shares = np.repeat(1.0 / 12.0, 12)

    return pd.DataFrame({"month": months, "share": shares})


def build_month_shares(entity_rules: pd.DataFrame, pools: pd.DataFrame) -> pd.DataFrame:
    rule_map = entity_rules.set_index("entity")[["default_share_rule", "opening_month"]].to_dict("index")
    keys = pools[["entity", "year", "pool_type"]].drop_duplicates()

    rows = []

    for _, r in keys.iterrows():
        entity = r["entity"]
        year = int(r["year"])
        pool_type = r["pool_type"]

        default_rule = rule_map[entity]["default_share_rule"]
        opening_month = rule_map[entity]["opening_month"]

        rule_name = "uniform" if pool_type == "lease_addition" else default_rule
        shares = generate_shares_for_year(rule_name, year, opening_month)

        if shares is None:
            continue

        for _, rr in shares.iterrows():
            rows.append([
                entity,
                year,
                pool_type,
                rr["month"],
                float(rr["share"]),
                rule_name,
                f"{entity}_RULE",
                "Generated default monthly shares",
            ])

    out = pd.DataFrame(
        rows,
        columns=["entity", "year", "pool_type", "month", "share", "rule_name", "source_id", "notes"],
    ).sort_values(["entity", "year", "pool_type", "month"]).reset_index(drop=True)

    return out


def build_qa(master: pd.DataFrame, anchors: pd.DataFrame, dep: pd.DataFrame, events: pd.DataFrame, pools: pd.DataFrame, shares: pd.DataFrame, pool_diag: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append(["master_v4_repaired_rows", "global", "", len(master), "info", "Repaired v4 raw master row count"])
    rows.append(["master_v4_duplicate_record_ids", "global", "", int(master["record_id"].duplicated().sum()), "pass" if int(master["record_id"].duplicated().sum()) == 0 else "fail", "Duplicate record IDs should be zero"])
    rows.append(["anchors_rows", "global", "", len(anchors), "info", "Entity-year anchors row count"])
    rows.append(["dep_rows", "global", "", len(dep), "info", "Depreciation lookup row count"])
    rows.append(["dated_event_rows", "global", "", len(events), "info", "Dated event row count"])
    rows.append(["class_pool_rows", "global", "", len(pools), "info", "Annual class pools row count"])
    rows.append(["month_shares_rows", "global", "", len(shares), "info", "Month shares row count"])

    # Share sums
    share_sums = shares.groupby(["entity", "year", "pool_type"], as_index=False)["share"].sum()
    bad_share_sums = share_sums[np.abs(share_sums["share"] - 1.0) > 1e-8]
    rows.append(["month_share_sum_failures", "global", "", len(bad_share_sums), "pass" if len(bad_share_sums) == 0 else "warn", "Entity-year-pool share sums not equal to 1"])

    # Missing dep rows for used classes
    used_classes = set(pools["asset_class_std"].dropna().unique()) | set(events["asset_class_std"].dropna().unique())
    dep_classes = set(dep["asset_class_std"].dropna().unique())
    missing_dep_classes = sorted(c for c in used_classes if c not in dep_classes and c not in {"mixed"})
    rows.append(["missing_dep_classes_count", "global", "", len(missing_dep_classes), "pass" if len(missing_dep_classes) == 0 else "warn", f"Missing dep rows for used classes: {missing_dep_classes}"])

    # Missing anchor annual totals
    missing_I = anchors["I_annual_total_kNIS"].isna().sum()
    rows.append(["anchors_missing_I_total", "global", "", int(missing_I), "info", "Missing annual total values in anchor file"])

    # Entity-level diagnostics
    for entity in sorted(anchors["entity"].unique()):
        rows.append(["entity_anchor_rows", entity, "", int(len(anchors[anchors["entity"] == entity])), "info", "Anchor rows for entity"])
        rows.append(["entity_event_rows", entity, "", int(len(events[events["entity"] == entity])), "info", "Dated event rows for entity"])
        rows.append(["entity_pool_rows", entity, "", int(len(pools[pools["entity"] == entity])), "info", "Class pool rows for entity"])

    # Pool residual diagnostics
    for _, r in pool_diag.iterrows():
        gap = float(r["annual_total_kNIS"]) - float(r["mapped_undated_sum_kNIS"]) - float(r["background_kNIS"])
        rows.append([
            "pool_identity_gap",
            r["entity"],
            str(int(r["year"])),
            gap,
            "info",
            "annual_total - mapped_undated_sum - background",
        ])

    return pd.DataFrame(rows, columns=["check_name", "entity_scope", "year_scope", "value", "status", "notes"])


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    master_v4 = repair_v3(Path(args.v2), Path(args.v3), Path(args.hpc))
    rules = build_entity_rules()
    anchors = build_anchors(master_v4)
    dep = build_dep_lookup(master_v4)
    events = build_dated_events(master_v4)
    pools, pool_diag = build_class_pools(master_v4, anchors)
    shares = build_month_shares(rules, pools)
    qa = build_qa(master_v4, anchors, dep, events, pools, shares, pool_diag)

    write_tsv(master_v4, outdir / "k_monthly_redesign_master_v4_repaired.tsv")
    write_tsv(rules, outdir / "k_entity_rules.tsv")
    write_tsv(anchors, outdir / "k_entity_year_anchors.tsv")
    write_tsv(dep, outdir / "k_dep_lookup.tsv")
    write_tsv(events, outdir / "k_dated_events.tsv")
    write_tsv(pools, outdir / "k_annual_class_pools.tsv")
    write_tsv(shares, outdir / "k_month_shares.tsv")
    write_tsv(pool_diag, outdir / "k_pool_identity_diagnostics.tsv")
    write_tsv(qa, outdir / "k_package_qa_summary.tsv")

    print("Done.")
    print(f"Wrote repaired master and package files to: {outdir}")


if __name__ == "__main__":
    main()

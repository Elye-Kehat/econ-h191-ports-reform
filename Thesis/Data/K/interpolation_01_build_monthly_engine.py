#!/usr/bin/env python3
"""
interpolation_01_build_monthly_engine_v6.py

Purpose
-------
Build the monthly class-level investment ledger and the monthly class-level capital ledger.

Fixes relative to v3
--------------------
1. Robust month-share parsing remains in place.
2. Keeps the schedule-fallback hierarchy for generated pool rows such as IPC 2021 background.
3. Nets positive dated events out of annual pools before monthly allocation so they are not double counted.
4. Checks annual identity against gross additions with a slightly looser tolerance for harmless rounding noise.

Schedule fallback hierarchy
---------------------------
For mapped_undated and background pools:
A. exact (entity, year, pool_type)
B. nearest available year for the same (entity, pool_type)
C. if still missing, use a uniform within-year schedule

This is needed because interpolation_00 can generate productive background rows
for entity-years that did not exist as separate rows in the finalized raw month-share
table. The annual totals remain authoritative; this logic only supplies within-year timing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SAMPLE_START = "2018-01"
SAMPLE_END = "2024-12"

DEFAULT_FINAL_INPUT_DIR = "Data/K/Final Input Files"
DEFAULT_PREPARED_DIR = "Data/K/Interpolation Output"
DEFAULT_OUTPUT_DIR = "Data/K/Interpolation Output"

TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS = {"0", "false", "f", "no", "n"}

ANCHOR_REQUIRED = "interpolation_00_prepared_anchors.tsv"
POOL_REQUIRED = "interpolation_00_working_annual_pools.tsv"
DEP_REQUIRED = "interpolation_00_working_dep_lookup.tsv"
SERVICE_REQUIRED = "interpolation_00_working_service_entry_pools.tsv"
EVENTS_REQUIRED = "interpolation_00_working_dated_events.tsv"
MILESTONES_OPTIONAL = "interpolation_00_project_milestones.tsv"

RULE_ALIASES = {
    "entity": ["entity", "owner", "firm", "operator"],
    "opening_month": ["opening_month", "open_month", "operational_start_month", "first_operating_month"],
    "pre_open_zero": ["pre_open_zero_rule", "zero_before_opening", "pre_opening_zero_rule"],
    "allow_empty_dated_events": ["allow_empty_dated_events", "dated_events_optional"],
    "default_share_rule": ["default_share_rule", "share_rule"],
    "notes": ["notes", "description"],
}

WORKING_EVENT_ALIASES = {
    "project_id": ["project_id", "event_id", "record_id"],
    "entity": ["entity", "owner", "firm", "operator"],
    "event_month": ["event_month", "month", "commission_month", "date_month"],
    "asset_class_std": ["asset_class_std", "asset_class", "class_std", "class"],
    "signed_amount": ["signed_amount_kNIS", "amount_kNIS", "event_amount_kNIS", "annual_amount_kNIS"],
    "event_type": ["event_type", "kind", "direction", "flow_type"],
    "include_prod": [
        "include_in_productive_K",
        "include_productive",
        "productive_flag",
        "productive_capital_flag",
    ],
    "notes": ["notes", "description"],
}

SHARE_ALIASES = {
    "entity": ["entity", "owner", "firm", "operator"],
    "year": ["year", "fiscal_year"],
    "pool_type": ["pool_type", "component_type", "component"],
    "month_num": ["month_num", "month", "calendar_month", "month_index"],
    "share": ["share", "month_share", "weight"],
    "notes": ["notes", "description"],
}

IDENTITY_TOL = 1e-6
FLOW_IDENTITY_TOL = 1e-5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    p.add_argument("--final-input-dir", default=DEFAULT_FINAL_INPUT_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).replace({np.nan: ""})


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def as_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def canonicalize(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def pick_column(columns: Sequence[str], aliases: Sequence[str], label: str, required: bool = True) -> Optional[str]:
    raw_to_canon = {c: canonicalize(c) for c in columns}
    alias_set = {canonicalize(a) for a in aliases}
    for raw, canon in raw_to_canon.items():
        if canon in alias_set:
            return raw
    for raw, canon in raw_to_canon.items():
        if any(a in canon or canon in a for a in alias_set):
            return raw
    if required:
        raise KeyError(f"Could not resolve required column '{label}'. Available columns: {list(columns)}")
    return None


def rename_by_aliases(df: pd.DataFrame, alias_map: Dict[str, Sequence[str]], required_fields: Sequence[str]) -> pd.DataFrame:
    rename_map = {}
    for field, aliases in alias_map.items():
        raw = pick_column(df.columns, aliases, field, required=(field in required_fields))
        if raw is not None:
            rename_map[raw] = field
    out = df.rename(columns=rename_map).copy()
    missing = [f for f in required_fields if f not in out.columns]
    if missing:
        raise KeyError(f"Missing required fields after alias resolution: {missing}. Available columns: {list(df.columns)}")
    return out


def normalize_entity(x: str) -> str:
    s = str(x).strip().upper()
    mapping = {
        "HAIFA PORT COMPANY": "HPC",
        "HPC": "HPC",
        "HAIFA LEGACY": "HPC",
        "ISRAEL PORTS COMPANY": "IPC",
        "IPC": "IPC",
        "SIPG": "SIPG",
        "BAYPORT": "SIPG",
        "SIPG BAYPORT": "SIPG",
        "HAIFA BAYPORT": "SIPG",
    }
    return mapping.get(s, s)


def normalize_month_str(x: str) -> str:
    s = str(x).strip()
    if s == "":
        return ""
    s = s.replace("/", "-")
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return s
    if re.fullmatch(r"\d{4}-\d{1}", s):
        y, m = s.split("-")
        return f"{y}-{int(m):02d}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s[:7]
    if re.fullmatch(r"\d{6}", s):
        return f"{s[:4]}-{s[4:]}"
    return s


def str_truthy(x: str) -> Optional[bool]:
    s = str(x).strip().lower()
    if s == "":
        return None
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    return None


def annual_to_monthly_dep(delta_a: float) -> float:
    return 1.0 - (1.0 - float(delta_a)) ** (1.0 / 12.0)


def month_range(start: str, end: str) -> List[str]:
    return [str(p) for p in pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")]


def month_num(month_str: str) -> int:
    return int(month_str[-2:])


def year_num(month_str: str) -> int:
    return int(month_str[:4])


def load_prepared(prepared_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = {
        "anchors": prepared_dir / ANCHOR_REQUIRED,
        "pools": prepared_dir / POOL_REQUIRED,
        "dep": prepared_dir / DEP_REQUIRED,
        "service": prepared_dir / SERVICE_REQUIRED,
    }
    missing = [str(v) for v in files.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing required interpolation_00 outputs:\n" + "\n".join(missing))

    anchors = read_tsv(files["anchors"])
    pools = read_tsv(files["pools"])
    dep = read_tsv(files["dep"])
    service = read_tsv(files["service"])

    anchors["entity"] = anchors["entity"].map(normalize_entity)
    anchors["year"] = as_num(anchors["year"]).astype("Int64")
    anchors["annual_total_kNIS"] = as_num(anchors["annual_total_kNIS"])
    anchors["stock_anchor_dec_kNIS"] = as_num(anchors["stock_anchor_dec_kNIS"])
    anchors["service_entry_annual_kNIS"] = as_num(anchors["service_entry_annual_kNIS"])
    if "stock_anchor_scope_note" not in anchors.columns:
        anchors["stock_anchor_scope_note"] = ""
    anchors["hard_anchor_flag"] = anchors.apply(is_hard_stock_anchor_row, axis=1)

    pools["entity"] = pools["entity"].map(normalize_entity)
    pools["year"] = as_num(pools["year"]).astype("Int64")
    pools["annual_amount_kNIS"] = as_num(pools["annual_amount_kNIS"]).fillna(0.0)
    pools["allocation_priority"] = as_num(pools["allocation_priority"]).fillna(0).astype(int)

    dep["entity"] = dep["entity"].map(normalize_entity)
    dep["dep_rate_annual"] = as_num(dep["dep_rate_annual"])
    dep["dep_rate_monthly"] = as_num(dep["dep_rate_monthly"])

    if not service.empty:
        service["entity"] = service["entity"].map(normalize_entity)
        service["year"] = as_num(service["year"]).astype("Int64")
        service["annual_amount_kNIS"] = as_num(service["annual_amount_kNIS"]).fillna(0.0)

    return anchors, pools, dep, service


def load_rules(final_input_dir: Path) -> pd.DataFrame:
    path = final_input_dir / "k_entity_rules.tsv"
    df = read_tsv(path)
    df = rename_by_aliases(df, RULE_ALIASES, required_fields=["entity"])
    for col in ["opening_month", "pre_open_zero", "allow_empty_dated_events", "default_share_rule", "notes"]:
        if col not in df.columns:
            df[col] = ""
    df["entity"] = df["entity"].map(normalize_entity)
    df["opening_month"] = df["opening_month"].apply(normalize_month_str)
    df["pre_open_zero_rule"] = df["pre_open_zero"].apply(str_truthy)
    df["allow_empty_dated_events"] = df["allow_empty_dated_events"].apply(str_truthy)
    return df.drop_duplicates(subset=["entity"]).reset_index(drop=True)



def load_working_events(prepared_dir: Path) -> pd.DataFrame:
    """
    Read the cleaned additive dated-event ledger written by interpolation_00.

    Why this file is authoritative for Step 01
    ------------------------------------------
    Step 00 now performs the economically sensitive classification work that decides
    whether a raw dated-event row should remain additive or instead become a
    non-additive project milestone. Step 01 must therefore read the cleaned Step 00
    working dated-events file, not the raw k_dated_events.tsv from Final Input Files.
    """
    path = prepared_dir / EVENTS_REQUIRED
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required Step 00 working dated-events file: {path}\n"
            "Run interpolation_00 first and make sure it wrote "
            "interpolation_00_working_dated_events.tsv."
        )

    df = read_tsv(path)
    if df.empty:
        return pd.DataFrame(columns=[
            "project_id",
            "entity",
            "event_month",
            "asset_class_std",
            "event_type",
            "signed_amount_kNIS",
            "include_in_productive_K",
            "source_id",
            "notes",
        ])

    df = rename_by_aliases(
        df,
        WORKING_EVENT_ALIASES,
        required_fields=["project_id", "entity", "event_month", "asset_class_std", "signed_amount"],
    )
    for col in ["event_type", "include_prod", "source_id", "notes"]:
        if col not in df.columns:
            df[col] = ""

    df["entity"] = df["entity"].map(normalize_entity)
    df["event_month"] = df["event_month"].apply(normalize_month_str)
    df["signed_amount_kNIS"] = as_num(df["signed_amount"]).fillna(0.0)
    if "include_prod" in df.columns:
        df["include_in_productive_K"] = df["include_prod"].apply(str_truthy)
        df = df.loc[df["include_in_productive_K"].ne(False)].copy()
    else:
        df["include_in_productive_K"] = True

    df = df.loc[df["event_month"] != ""].copy()
    return df.reset_index(drop=True)


def load_project_milestones(prepared_dir: Path) -> pd.DataFrame:
    """
    Read optional non-additive project milestones from Step 00.

    These rows are for diagnostics and narrative only. They do NOT enter the additive
    monthly investment ledger.
    """
    path = prepared_dir / MILESTONES_OPTIONAL
    if not path.exists():
        return pd.DataFrame()
    return read_tsv(path)



def is_hard_stock_anchor_row(row: pd.Series) -> bool:
    """
    Decide whether a prepared anchor should be enforced as a hard stock anchor.

    Important SIPG note
    -------------------
    Step 00 v6 rebuilds SIPG "anchors" from cumulative transfers to PPE. Those are useful
    diagnostics about service-entry scale, but they are NOT true net productive-PPE stock
    anchors. If Step 01 forces the monthly stock path to hit them exactly, it creates an
    artificial SIPG background stock via reconciliation.

    Therefore:
    - HPC and IPC anchors remain hard anchors.
    - SIPG cumulative-transfer-based anchors are treated as soft diagnostic anchors only.
    """
    entity = str(row.get("entity", ""))
    note = str(row.get("stock_anchor_scope_note", ""))
    val = row.get("stock_anchor_dec_kNIS", np.nan)

    if pd.isna(val):
        return False

    if entity == "SIPG":
        return False

    return True


def first_observed_hard_anchor_month(anchors: pd.DataFrame, entity: str) -> Optional[str]:
    a = anchors.loc[(anchors["entity"] == entity)].copy()
    if a.empty:
        return None
    if "hard_anchor_flag" not in a.columns:
        a["hard_anchor_flag"] = a.apply(is_hard_stock_anchor_row, axis=1)
    a = a.loc[a["hard_anchor_flag"]].copy()
    if a.empty:
        return None
    y = int(a["year"].min())
    return f"{y}-12"
def derive_month_num_from_string(x: str) -> Optional[int]:
    s = normalize_month_str(x)
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return int(s[-2:])
    if re.fullmatch(r"\d{1,2}", str(x).strip()):
        return int(str(x).strip())
    return None


def load_shares(final_input_dir: Path) -> pd.DataFrame:
    path = final_input_dir / "k_month_shares.tsv"
    raw = read_tsv(path)
    df = rename_by_aliases(raw, SHARE_ALIASES, required_fields=["entity", "year", "pool_type", "month_num", "share"])
    if "notes" not in df.columns:
        df["notes"] = ""

    df["entity"] = df["entity"].map(normalize_entity)
    df["year"] = as_num(df["year"]).astype("Int64")
    df["pool_type"] = df["pool_type"].astype(str).str.strip().str.lower()
    df["share"] = as_num(df["share"]).fillna(0.0)

    raw_month = df["month_num"].astype(str).str.strip()
    month_num_numeric = as_num(raw_month)
    month_num_from_string = raw_month.apply(derive_month_num_from_string)
    month_num_final = month_num_numeric.where(month_num_numeric.notna(), pd.Series(month_num_from_string, index=df.index))
    df["month_num"] = pd.to_numeric(month_num_final, errors="coerce").astype("Int64")

    bad = df.loc[df["month_num"].isna() | ~df["month_num"].between(1, 12)].copy()
    if not bad.empty:
        raise ValueError(
            "Could not resolve valid month_num values from k_month_shares.tsv. "
            "The first problematic rows are:\n"
            + bad.head(20).to_string(index=False)
        )

    return df[["entity", "year", "pool_type", "month_num", "share", "notes"]].sort_values(
        ["entity", "year", "pool_type", "month_num"]
    ).reset_index(drop=True)


def validate_share_table(shares: pd.DataFrame) -> None:
    g = shares.groupby(["entity", "year", "pool_type"], dropna=False)["share"].sum().reset_index()
    bad = g.loc[~np.isclose(g["share"], 1.0, atol=1e-8)].copy()
    if not bad.empty:
        raise ValueError("Month-share table has non-unit sums:\n" + bad.head(20).to_string(index=False))


def resolve_background_dep_rates(pools: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    dep_map = dep[["entity", "asset_class_std", "dep_rate_annual", "dep_rate_monthly", "is_fallback", "source_id"]].copy()

    mapped = pools.loc[pools["pool_type"].eq("mapped_undated")].copy()
    mapped = mapped.merge(
        dep_map[["entity", "asset_class_std", "dep_rate_annual", "dep_rate_monthly"]],
        on=["entity", "asset_class_std"],
        how="left",
    )

    entity_wide = (
        mapped.groupby(["entity", "asset_class_std"], as_index=False)["annual_amount_kNIS"].sum()
        .merge(dep_map[["entity", "asset_class_std", "dep_rate_annual", "dep_rate_monthly"]], on=["entity", "asset_class_std"], how="left")
    )

    output_rows = []
    bg_rows = pools.loc[pools["pool_type"].isin(["background", "service_entry_undated"])].copy()

    for _, row in bg_rows.iterrows():
        entity = row["entity"]
        year = int(row["year"])
        asset_class_std = row["asset_class_std"]

        mapped_same = mapped.loc[(mapped["entity"] == entity) & (mapped["year"] == year) & mapped["dep_rate_annual"].notna()].copy()
        source = None
        dep_a = np.nan
        dep_m = np.nan
        weight_basis = np.nan

        if not mapped_same.empty and mapped_same["annual_amount_kNIS"].sum() > IDENTITY_TOL:
            w = mapped_same["annual_amount_kNIS"] / mapped_same["annual_amount_kNIS"].sum()
            dep_a = float((w * mapped_same["dep_rate_annual"]).sum())
            dep_m = annual_to_monthly_dep(dep_a)
            source = "entity_year_weighted_average"
            weight_basis = float(mapped_same["annual_amount_kNIS"].sum())
        else:
            ew = entity_wide.loc[(entity_wide["entity"] == entity) & entity_wide["dep_rate_annual"].notna()].copy()
            if not ew.empty and ew["annual_amount_kNIS"].sum() > IDENTITY_TOL:
                w = ew["annual_amount_kNIS"] / ew["annual_amount_kNIS"].sum()
                dep_a = float((w * ew["dep_rate_annual"]).sum())
                dep_m = annual_to_monthly_dep(dep_a)
                source = "entity_wide_weighted_average"
                weight_basis = float(ew["annual_amount_kNIS"].sum())
            else:
                direct = dep_map.loc[(dep_map["entity"] == entity) & (dep_map["asset_class_std"] == asset_class_std)].copy()
                if direct.empty:
                    raise ValueError(
                        f"Could not resolve any depreciation rate for background/service-entry row: "
                        f"{entity}, {year}, {asset_class_std}"
                    )
                dep_a = float(direct["dep_rate_annual"].iloc[0])
                dep_m = float(direct["dep_rate_monthly"].iloc[0])
                source = "direct_lookup_fallback"
                weight_basis = np.nan

        output_rows.append({
            "entity": entity,
            "year": year,
            "pool_type": row["pool_type"],
            "asset_class_std": asset_class_std,
            "annual_amount_kNIS": float(row["annual_amount_kNIS"]),
            "dep_rate_annual_used": dep_a,
            "dep_rate_monthly_used": dep_m,
            "source_rule": source,
            "weight_basis_kNIS": weight_basis,
        })

    return pd.DataFrame(output_rows).sort_values(["entity", "year", "pool_type", "asset_class_std"]).reset_index(drop=True)


def attach_dep_rates_to_pools(pools: pd.DataFrame, dep: pd.DataFrame, bg_rates: pd.DataFrame) -> pd.DataFrame:
    pools = pools.copy()
    direct = dep[["entity", "asset_class_std", "dep_rate_annual", "dep_rate_monthly"]].drop_duplicates()

    out = pools.merge(direct, on=["entity", "asset_class_std"], how="left")

    bg = bg_rates.rename(columns={
        "dep_rate_annual_used": "bg_dep_rate_annual",
        "dep_rate_monthly_used": "bg_dep_rate_monthly",
    })

    out = out.merge(
        bg[["entity", "year", "pool_type", "asset_class_std", "bg_dep_rate_annual", "bg_dep_rate_monthly"]],
        on=["entity", "year", "pool_type", "asset_class_std"],
        how="left",
    )

    use_bg = out["pool_type"].isin(["background", "service_entry_undated"])
    out["dep_rate_annual_used"] = np.where(use_bg, out["bg_dep_rate_annual"], out["dep_rate_annual"])
    out["dep_rate_monthly_used"] = np.where(use_bg, out["bg_dep_rate_monthly"], out["dep_rate_monthly"])

    missing = out.loc[out["dep_rate_annual_used"].isna() | out["dep_rate_monthly_used"].isna()].copy()
    if not missing.empty:
        raise ValueError(
            "Some working annual pool rows are missing a resolved depreciation rate:\n"
            + missing.head(20).to_string(index=False)
        )

    return out




def residualize_pools_for_positive_dated_events(
    pools_with_dep: pd.DataFrame,
    events: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Net positive dated events out of the annual pool ledger before monthly allocation.

    Why this is needed
    ------------------
    The prepared annual pools coming from interpolation_00 already sum to the authoritative
    annual investment totals. If we now add positive dated events on top without subtracting
    them from the corresponding annual pools, we mechanically double count those additions.

    Strategy
    --------
    For each positive dated event inside the sample:
    1. subtract it first from same-entity, same-year, same-class mapped_undated pools
    2. then from same-entity, same-year, same-class service_entry/background pools
    3. then, only if still needed, from any remaining positive same-entity-year pools
       in priority order mapped_undated -> service_entry_undated -> background

    This keeps annual totals authoritative while preserving the dated monthly timing.
    Negative dated events (for example disposals) are NOT netted out of annual pools here,
    because the annual_total_kNIS object is treated as a gross additions object.
    """
    pools = pools_with_dep.copy()
    events = events.copy()

    if pools.empty or events.empty:
        empty = pd.DataFrame(columns=[
            "entity", "year", "project_id", "event_month", "asset_class_std",
            "dated_event_amount_kNIS", "pool_row_source_id", "pool_type",
            "pool_asset_class_std", "amount_subtracted_kNIS", "rule_used"
        ])
        return pools, empty

    events["year"] = events["event_month"].astype(str).str[:4].astype(int)
    events["signed_amount_kNIS"] = pd.to_numeric(events["signed_amount_kNIS"], errors="coerce").fillna(0.0)

    adjustments = []

    def subtract_from_candidates(cands: pd.DataFrame, remaining: float, project_id: str, entity: str, year: int,
                                 event_month: str, event_asset: str, event_amount: float, rule_used: str) -> float:
        nonlocal pools, adjustments
        if remaining <= IDENTITY_TOL or cands.empty:
            return remaining

        cands = cands.loc[cands["annual_amount_kNIS"] > IDENTITY_TOL].copy()
        if cands.empty:
            return remaining

        total_available = float(cands["annual_amount_kNIS"].sum())
        take_total = min(total_available, remaining)
        if take_total <= IDENTITY_TOL:
            return remaining

        weights = cands["annual_amount_kNIS"] / total_available
        planned = weights * take_total
        planned = planned.to_numpy(dtype=float)

        # preserve exact total by pushing tiny remainder to the last candidate
        rem = take_total - float(planned.sum())
        if len(planned) > 0:
            planned[-1] += rem

        for row_idx, amt in zip(cands.index.tolist(), planned.tolist()):
            if abs(amt) <= IDENTITY_TOL:
                continue
            current = float(pools.at[row_idx, "annual_amount_kNIS"])
            if amt > current:
                amt = current
            pools.at[row_idx, "annual_amount_kNIS"] = current - amt
            adjustments.append({
                "entity": entity,
                "year": year,
                "project_id": project_id,
                "event_month": event_month,
                "asset_class_std": event_asset,
                "dated_event_amount_kNIS": event_amount,
                "pool_row_source_id": pools.at[row_idx, "source_id"],
                "pool_type": pools.at[row_idx, "pool_type"],
                "pool_asset_class_std": pools.at[row_idx, "asset_class_std"],
                "amount_subtracted_kNIS": amt,
                "rule_used": rule_used,
            })
        return remaining - take_total

    pos_events = events.loc[(events["signed_amount_kNIS"] > IDENTITY_TOL) & (events["event_month"].between(SAMPLE_START, SAMPLE_END))].copy()

    for _, ev in pos_events.iterrows():
        entity = ev["entity"]
        year = int(ev["year"])
        event_asset = ev["asset_class_std"]
        project_id = ev["project_id"]
        event_month = ev["event_month"]
        event_amount = float(ev["signed_amount_kNIS"])
        remaining = event_amount

        base = pools.loc[(pools["entity"] == entity) & (pools["year"] == year)].copy()
        if base.empty:
            raise ValueError(f"No annual pool rows available to net out positive dated event {project_id} for {entity}, {year}")

        # 1. same class mapped_undated
        c1 = base.loc[(base["asset_class_std"] == event_asset) & (base["pool_type"] == "mapped_undated")].copy()
        remaining = subtract_from_candidates(c1, remaining, project_id, entity, year, event_month, event_asset, event_amount, "same_class_mapped_undated")

        # 2. same class service-entry/background
        c2 = base.loc[(base["asset_class_std"] == event_asset) & (base["pool_type"].isin(["service_entry_undated", "background"]))].copy()
        remaining = subtract_from_candidates(c2, remaining, project_id, entity, year, event_month, event_asset, event_amount, "same_class_nonmapped")

        # 3. any same-year positive pools by hierarchy
        if remaining > IDENTITY_TOL:
            for pool_type in ["mapped_undated", "service_entry_undated", "background"]:
                c3 = pools.loc[
                    (pools["entity"] == entity)
                    & (pools["year"] == year)
                    & (pools["pool_type"] == pool_type)
                    & (pools["annual_amount_kNIS"] > IDENTITY_TOL)
                ].copy()
                remaining = subtract_from_candidates(c3, remaining, project_id, entity, year, event_month, event_asset, event_amount, f"fallback_{pool_type}")
                if remaining <= IDENTITY_TOL:
                    break

        if remaining > 1e-4:
            raise ValueError(
                f"Could not fully net out positive dated event from annual pools: "
                f"{project_id} ({entity}, {year}) amount={event_amount}, unmatched remainder={remaining}"
            )

    # Clean tiny negative noise
    pools.loc[pools["annual_amount_kNIS"].abs() < FLOW_IDENTITY_TOL, "annual_amount_kNIS"] = 0.0

    adjustments_df = pd.DataFrame(adjustments)
    if adjustments_df.empty:
        adjustments_df = pd.DataFrame(columns=[
            "entity", "year", "project_id", "event_month", "asset_class_std",
            "dated_event_amount_kNIS", "pool_row_source_id", "pool_type",
            "pool_asset_class_std", "amount_subtracted_kNIS", "rule_used"
        ])

    return pools, adjustments_df

def uniform_schedule() -> pd.DataFrame:
    return pd.DataFrame({"month_num": list(range(1, 13)), "share": [1.0 / 12.0] * 12})


def build_pool_share_schedule(entity: str, year: int, pool_type: str, shares: pd.DataFrame, rules: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if pool_type in {"mapped_undated", "background"}:
        exact = shares.loc[
            (shares["entity"] == entity)
            & (shares["year"] == year)
            & (shares["pool_type"] == pool_type),
            ["month_num", "share"]
        ].copy()

        if not exact.empty:
            if exact["month_num"].isna().any():
                raise ValueError(f"Exact month-share schedule contains missing month_num for {entity}, {year}, {pool_type}")
            if not np.isclose(float(exact["share"].sum()), 1.0, atol=1e-8):
                raise ValueError(f"Month shares do not sum to 1 for {entity}, {year}, {pool_type}")
            return exact.sort_values("month_num").reset_index(drop=True), "exact"

        same_pool = shares.loc[
            (shares["entity"] == entity)
            & (shares["pool_type"] == pool_type),
            ["year", "month_num", "share"]
        ].copy()

        if not same_pool.empty:
            years_available = sorted(int(y) for y in same_pool["year"].dropna().unique())
            nearest_year = min(years_available, key=lambda yy: (abs(yy - year), yy))
            fallback = same_pool.loc[same_pool["year"] == nearest_year, ["month_num", "share"]].copy()
            if fallback["month_num"].isna().any():
                raise ValueError(
                    f"Nearest-year fallback schedule contains missing month_num for {entity}, requested {year}, fallback {nearest_year}, {pool_type}"
                )
            if not np.isclose(float(fallback["share"].sum()), 1.0, atol=1e-8):
                raise ValueError(f"Nearest-year fallback shares do not sum to 1 for {entity}, requested {year}, fallback {nearest_year}, {pool_type}")
            return fallback.sort_values("month_num").reset_index(drop=True), f"nearest_year_fallback:{nearest_year}"

        return uniform_schedule(), "uniform_fallback"

    if pool_type == "service_entry_undated":
        rr = rules.loc[rules["entity"] == entity].copy()
        opening_month = rr["opening_month"].iloc[0] if not rr.empty else ""
        if opening_month == "":
            allowed_months = list(range(1, 13))
        else:
            op_y = int(opening_month[:4])
            op_m = int(opening_month[-2:])
            if year < op_y:
                allowed_months = []
            elif year > op_y:
                allowed_months = list(range(1, 13))
            else:
                allowed_months = list(range(op_m, 13))

        if not allowed_months:
            return pd.DataFrame({"month_num": list(range(1, 13)), "share": [0.0] * 12}), "service_entry_post_opening_empty"

        ranks = np.arange(1, len(allowed_months) + 1, dtype=float)
        shares_vec = ranks / ranks.sum()
        full = pd.DataFrame({"month_num": list(range(1, 13)), "share": [0.0] * 12})
        full.loc[full["month_num"].isin(allowed_months), "share"] = shares_vec
        return full, "service_entry_post_opening_backloaded"

    raise ValueError(f"Unsupported pool_type for share schedule: {pool_type}")


def allocate_annual_amount_to_months(amount: float, sched: pd.DataFrame, year: int) -> pd.DataFrame:
    sched = sched.copy().sort_values("month_num").reset_index(drop=True)
    if sched["month_num"].isna().any():
        raise ValueError(
            "allocate_annual_amount_to_months received a schedule with missing month_num values:\n"
            + sched.to_string(index=False)
        )
    sched["amount_kNIS"] = amount * sched["share"]
    remainder = amount - float(sched["amount_kNIS"].sum())
    if abs(remainder) > 0:
        nonzero = sched.index[sched["share"].abs() > 0]
        target_idx = int(nonzero[-1]) if len(nonzero) else len(sched) - 1
        sched.loc[target_idx, "amount_kNIS"] += remainder
    sched["month"] = sched["month_num"].apply(lambda m: f"{year}-{int(m):02d}")
    return sched[["month", "month_num", "share", "amount_kNIS"]]


def build_monthly_investment_ledger(pools: pd.DataFrame, events: pd.DataFrame, shares: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if not events.empty:
        for _, row in events.iterrows():
            rows.append({
                "entity": row["entity"],
                "month": row["event_month"],
                "year": year_num(row["event_month"]),
                "pool_type": "dated",
                "asset_class_std": row["asset_class_std"],
                "engine_bucket": f"dated::{row['asset_class_std']}",
                "component_source": f"dated_event::{row['project_id']}",
                "amount_kNIS": float(row["signed_amount_kNIS"]),
                "share_used": np.nan,
                "share_rule_used": "dated_exact",
                "dep_rate_annual_used": np.nan,
                "dep_rate_monthly_used": np.nan,
                "notes": row.get("notes", ""),
            })

    for _, row in pools.iterrows():
        entity = row["entity"]
        year = int(row["year"])
        pool_type = row["pool_type"]
        asset_class_std = row["asset_class_std"]
        amount = float(row["annual_amount_kNIS"])
        sched, share_rule_used = build_pool_share_schedule(entity, year, pool_type, shares, rules)
        alloc = allocate_annual_amount_to_months(amount, sched, year)

        engine_bucket = f"{pool_type}::{asset_class_std}"

        for _, a in alloc.iterrows():
            rows.append({
                "entity": entity,
                "month": a["month"],
                "year": year,
                "pool_type": pool_type,
                "asset_class_std": asset_class_std,
                "engine_bucket": engine_bucket,
                "component_source": row.get("source_id", ""),
                "amount_kNIS": float(a["amount_kNIS"]),
                "share_used": float(a["share"]),
                "share_rule_used": share_rule_used,
                "dep_rate_annual_used": float(row["dep_rate_annual_used"]),
                "dep_rate_monthly_used": float(row["dep_rate_monthly_used"]),
                "notes": row.get("notes", ""),
            })

    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return pd.DataFrame(columns=[
            "entity", "month", "year", "pool_type", "asset_class_std", "engine_bucket",
            "component_source", "amount_kNIS", "share_used", "share_rule_used",
            "dep_rate_annual_used", "dep_rate_monthly_used", "notes"
        ])

    ledger["month"] = ledger["month"].astype(str)
    ledger = ledger.loc[ledger["month"].between(SAMPLE_START, SAMPLE_END)].copy()
    return ledger.sort_values(["entity", "month", "engine_bucket", "component_source"]).reset_index(drop=True)


def attach_dated_dep_rates(ledger: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    direct = dep[["entity", "asset_class_std", "dep_rate_annual", "dep_rate_monthly"]].drop_duplicates()

    mask = out["pool_type"].eq("dated")
    if mask.any():
        merged = out.loc[mask, ["entity", "asset_class_std"]].merge(direct, on=["entity", "asset_class_std"], how="left")
        out.loc[mask, "dep_rate_annual_used"] = merged["dep_rate_annual"].to_numpy()
        out.loc[mask, "dep_rate_monthly_used"] = merged["dep_rate_monthly"].to_numpy()

    missing = out.loc[
        out["dep_rate_annual_used"].isna() | out["dep_rate_monthly_used"].isna(),
        ["entity", "month", "pool_type", "asset_class_std", "component_source"]
    ]
    if not missing.empty:
        raise ValueError(
            "Some monthly investment rows are missing a resolved depreciation rate:\n"
            + missing.head(20).to_string(index=False)
        )

    return out


def choose_background_bucket(entity: str, active_buckets: List[str], pools: pd.DataFrame) -> Tuple[str, str]:
    bg = [b for b in active_buckets if b.startswith("background::")]
    if bg:
        bucket = sorted(bg)[0]
        asset_class = bucket.split("::", 1)[1]
        return bucket, asset_class

    pool_bg = pools.loc[(pools["entity"] == entity) & (pools["pool_type"] == "background"), "asset_class_std"].tolist()
    asset_class = sorted(pool_bg)[0] if pool_bg else "background_unclassified"
    bucket = f"background::{asset_class}"
    return bucket, asset_class



def backsolve_initial_background_stock(entity: str, first_anchor_month: str, anchors: pd.DataFrame, monthly_flows: pd.DataFrame, pools_with_dep: pd.DataFrame) -> Dict[str, object]:
    anchor_year = int(first_anchor_month[:4])
    anchor_val = float(
        anchors.loc[(anchors["entity"] == entity) & (anchors["year"] == anchor_year), "stock_anchor_dec_kNIS"].iloc[0]
    )

    bg_pool = pools_with_dep.loc[
        (pools_with_dep["entity"] == entity)
        & (pools_with_dep["year"] == anchor_year)
        & (pools_with_dep["pool_type"] == "background")
    ].copy()
    if bg_pool.empty:
        raise ValueError(f"No background pool found for {entity} in first anchor year {anchor_year}")

    bg_asset = str(bg_pool["asset_class_std"].iloc[0])
    bg_bucket = f"background::{bg_asset}"
    bg_delta = float(bg_pool["dep_rate_monthly_used"].iloc[0])

    months = month_range(SAMPLE_START, first_anchor_month)
    n = len(months)

    flows_entity = monthly_flows.loc[(monthly_flows["entity"] == entity) & (monthly_flows["month"].isin(months))].copy()

    stock_by_bucket = {}
    for m in months:
        fm = flows_entity.loc[flows_entity["month"] == m].copy()
        for bucket in set(list(stock_by_bucket.keys()) + fm["engine_bucket"].tolist()):
            if bucket not in stock_by_bucket:
                stock_by_bucket[bucket] = 0.0
        for bucket in list(stock_by_bucket.keys()):
            sub = fm.loc[fm["engine_bucket"] == bucket]
            if not sub.empty:
                bucket_delta = float(sub["dep_rate_monthly_used"].iloc[0])
            else:
                if bucket.startswith("background::"):
                    bucket_delta = bg_delta
                else:
                    asset = bucket.split("::", 1)[1]
                    ap = pools_with_dep.loc[
                        (pools_with_dep["entity"] == entity)
                        & (pools_with_dep["asset_class_std"] == asset)
                        & (~pools_with_dep["pool_type"].isin(["background"]))
                    ]
                    bucket_delta = float(ap["dep_rate_monthly_used"].iloc[0]) if not ap.empty else bg_delta
            stock_by_bucket[bucket] = (1.0 - bucket_delta) * stock_by_bucket[bucket] + float(sub["amount_kNIS"].sum())

    non_bg_stock = float(sum(v for k, v in stock_by_bucket.items() if k != bg_bucket))
    bg_flow_path = flows_entity.loc[flows_entity["engine_bucket"] == bg_bucket, ["month", "amount_kNIS"]].copy()
    bg_flow_map = {r.month: float(r.amount_kNIS) for r in bg_flow_path.itertuples(index=False)}

    discounted_bg_flows = 0.0
    for idx, m in enumerate(months, start=1):
        amt = bg_flow_map.get(m, 0.0)
        power = n - idx
        discounted_bg_flows += ((1.0 - bg_delta) ** power) * amt

    bg_target_at_anchor = anchor_val - non_bg_stock
    denom = (1.0 - bg_delta) ** n
    if abs(denom) < 1e-12:
        raise ValueError(f"Background depreciation implies near-zero denominator for {entity}")

    initial_bg = (bg_target_at_anchor - discounted_bg_flows) / denom

    return {
        "entity": entity,
        "first_anchor_month": first_anchor_month,
        "background_engine_bucket": bg_bucket,
        "background_asset_class_std": bg_asset,
        "background_dep_rate_monthly": bg_delta,
        "anchor_stock_kNIS": anchor_val,
        "non_background_stock_at_anchor_kNIS": non_bg_stock,
        "background_target_at_anchor_kNIS": bg_target_at_anchor,
        "implied_initial_background_stock_kNIS": initial_bg,
        "is_materially_negative": bool(initial_bg < -1e-4),
    }


def run_monthly_engine(
    anchors: pd.DataFrame,
    pools_with_dep: pd.DataFrame,
    monthly_flows: pd.DataFrame,
    rules: pd.DataFrame,
    initial_bg_rows: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    months_all = month_range(SAMPLE_START, SAMPLE_END)
    entities = sorted(set(pools_with_dep["entity"]).union(set(anchors["entity"])).union(set(monthly_flows["entity"])))

    class_rows = []
    recon_rows = []

    initial_map = {r["entity"]: r for _, r in initial_bg_rows.iterrows()}

    for entity in entities:
        rr = rules.loc[rules["entity"] == entity].copy()
        opening_month = rr["opening_month"].iloc[0] if not rr.empty else ""
        pre_open_zero = rr["pre_open_zero_rule"].iloc[0] if not rr.empty else None

        flows_e = monthly_flows.loc[monthly_flows["entity"] == entity].copy()
        anchor_e = anchors.loc[anchors["entity"] == entity].copy()

        state: Dict[str, float] = {}
        dep_m: Dict[str, float] = {}
        bucket_to_class: Dict[str, str] = {}
        bucket_to_pool: Dict[str, str] = {}

        init = initial_map.get(entity)
        if init is not None:
            bg_bucket = init["background_engine_bucket"]
            state[bg_bucket] = float(init["implied_initial_background_stock_kNIS"])
            dep_m[bg_bucket] = float(init["background_dep_rate_monthly"])
            bucket_to_class[bg_bucket] = str(init["background_asset_class_std"])
            bucket_to_pool[bg_bucket] = "background"

        for month in months_all:
            y = year_num(month)
            mnum = month_num(month)

            if opening_month and pre_open_zero is True and month < opening_month:
                state = {k: 0.0 for k in state.keys()}

            fm = flows_e.loc[flows_e["month"] == month].copy()
            active_buckets = sorted(set(list(state.keys()) + fm["engine_bucket"].tolist()))

            if not active_buckets and init is None:
                continue

            bg_bucket, bg_class = choose_background_bucket(entity, active_buckets, pools_with_dep)

            if bg_bucket not in state:
                state[bg_bucket] = 0.0
                bg_row = pools_with_dep.loc[
                    (pools_with_dep["entity"] == entity)
                    & (pools_with_dep["year"] == y)
                    & (pools_with_dep["pool_type"] == "background")
                ]
                if bg_row.empty:
                    bg_row = pools_with_dep.loc[
                        (pools_with_dep["entity"] == entity)
                        & (pools_with_dep["pool_type"] == "background")
                    ]
                dep_m[bg_bucket] = float(bg_row["dep_rate_monthly_used"].iloc[0]) if not bg_row.empty else 0.0
                bucket_to_class[bg_bucket] = bg_class
                bucket_to_pool[bg_bucket] = "background"

            for bucket in active_buckets:
                if bucket not in state:
                    state[bucket] = 0.0
                if bucket not in dep_m:
                    sub = fm.loc[fm["engine_bucket"] == bucket]
                    if sub.empty:
                        raise ValueError(f"New bucket without same-month flow or stored dep rate: {entity}, {month}, {bucket}")
                    dep_m[bucket] = float(sub["dep_rate_monthly_used"].iloc[0])
                    bucket_to_class[bucket] = str(sub["asset_class_std"].iloc[0])
                    bucket_to_pool[bucket] = str(sub["pool_type"].iloc[0])

            for bucket in active_buckets:
                begin_stock = float(state.get(bucket, 0.0))
                delta_m = float(dep_m[bucket])
                investment = float(fm.loc[fm["engine_bucket"] == bucket, "amount_kNIS"].sum())
                end_stock_pre = (1.0 - delta_m) * begin_stock + investment

                class_rows.append({
                    "entity": entity,
                    "month": month,
                    "year": y,
                    "pool_type": bucket_to_pool[bucket],
                    "asset_class_std": bucket_to_class[bucket],
                    "engine_bucket": bucket,
                    "begin_stock_kNIS": begin_stock,
                    "dep_rate_monthly_used": delta_m,
                    "investment_kNIS": investment,
                    "reconciliation_adjustment_kNIS": 0.0,
                    "end_stock_kNIS": end_stock_pre,
                })
                state[bucket] = end_stock_pre

            if mnum == 12:
                ar = anchor_e.loc[anchor_e["year"] == y]
                anchor_available = not ar.empty and pd.notna(ar["stock_anchor_dec_kNIS"].iloc[0])
                hard_anchor = bool(ar["hard_anchor_flag"].iloc[0]) if (anchor_available and "hard_anchor_flag" in ar.columns) else False
                observed_anchor = float(ar["stock_anchor_dec_kNIS"].iloc[0]) if anchor_available else np.nan
                anchor_policy = str(ar["stock_anchor_scope_note"].iloc[0]) if (anchor_available and "stock_anchor_scope_note" in ar.columns) else ""

                provisional_total = float(sum(state.values()))
                residual = float(observed_anchor - provisional_total) if hard_anchor else 0.0
                post_total = provisional_total

                if hard_anchor:
                    state[bg_bucket] = float(state.get(bg_bucket, 0.0) + residual)
                    post_total = float(sum(state.values()))

                    idx = None
                    for j in range(len(class_rows) - 1, -1, -1):
                        rrj = class_rows[j]
                        if rrj["entity"] == entity and rrj["month"] == month and rrj["engine_bucket"] == bg_bucket:
                            idx = j
                            break
                    if idx is None:
                        class_rows.append({
                            "entity": entity,
                            "month": month,
                            "year": y,
                            "pool_type": "background",
                            "asset_class_std": bg_class,
                            "engine_bucket": bg_bucket,
                            "begin_stock_kNIS": float(state[bg_bucket] - residual),
                            "dep_rate_monthly_used": float(dep_m[bg_bucket]),
                            "investment_kNIS": 0.0,
                            "reconciliation_adjustment_kNIS": residual,
                            "end_stock_kNIS": float(state[bg_bucket]),
                        })
                    else:
                        class_rows[idx]["reconciliation_adjustment_kNIS"] = residual
                        class_rows[idx]["end_stock_kNIS"] = float(class_rows[idx]["end_stock_kNIS"] + residual)

                if hard_anchor:
                    status = "reanchored"
                    residual_out = residual
                    post_out = post_total
                    observed_out = observed_anchor
                    hard_flag_out = True
                    available_flag_out = True
                elif anchor_available:
                    # soft diagnostic anchor only
                    status = "soft_anchor_diagnostic_only"
                    residual_out = np.nan
                    post_out = provisional_total
                    observed_out = observed_anchor
                    hard_flag_out = False
                    available_flag_out = True
                else:
                    status = "propagated_no_anchor"
                    residual_out = np.nan
                    post_out = provisional_total
                    observed_out = np.nan
                    hard_flag_out = False
                    available_flag_out = False

                recon_rows.append({
                    "entity": entity,
                    "year": y,
                    "december_month": month,
                    "anchor_observed_flag": bool(available_flag_out),
                    "hard_anchor_flag": bool(hard_flag_out),
                    "anchor_policy": anchor_policy if available_flag_out else "",
                    "observed_stock_anchor_kNIS": observed_out,
                    "provisional_dec_stock_kNIS": provisional_total,
                    "background_reconciliation_residual_kNIS": residual_out,
                    "post_reconciliation_dec_stock_kNIS": post_out,
                    "status": status,
                })

    class_df = pd.DataFrame(class_rows)
    recon_df = pd.DataFrame(recon_rows)

    if class_df.empty:
        class_df = pd.DataFrame(columns=[
            "entity", "month", "year", "pool_type", "asset_class_std", "engine_bucket",
            "begin_stock_kNIS", "dep_rate_monthly_used", "investment_kNIS",
            "reconciliation_adjustment_kNIS", "end_stock_kNIS"
        ])

    return class_df.sort_values(["entity", "month", "engine_bucket"]).reset_index(drop=True), \
        recon_df.sort_values(["entity", "year"]).reset_index(drop=True)


def build_engine_debug_summary(monthly_flows: pd.DataFrame, class_capital: pd.DataFrame, recon: pd.DataFrame, initial_bg: pd.DataFrame, project_milestones: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric, value, notes):
        rows.append({"metric": metric, "value": value, "notes": notes})

    add("monthly_flow_rows", int(len(monthly_flows)), "Rows in the monthly class investment ledger")
    add("monthly_capital_rows", int(len(class_capital)), "Rows in the monthly class capital ledger")
    add("reconciliation_rows", int(len(recon)), "Entity-year December reconciliation rows")
    add("initial_background_rows", int(len(initial_bg)), "Entities with a backsolved initial background stock")
    add("project_milestone_rows", int(len(project_milestones)), "Non-additive milestone rows passed through from Step 00")

    n_neg = int(initial_bg["is_materially_negative"].sum()) if not initial_bg.empty else 0
    add("materially_negative_initial_background_count", n_neg, "Count of entities with materially negative initial background stock")

    # Helpful diagnostic for schedule fallback use
    if not monthly_flows.empty and "share_rule_used" in monthly_flows.columns:
        fallback_rows = monthly_flows["share_rule_used"].astype(str).str.contains("fallback", case=False, na=False).sum()
    else:
        fallback_rows = 0
    add("rows_using_share_schedule_fallback", int(fallback_rows), "Monthly flow rows allocated with nearest-year or uniform fallback share rules")
    if not recon.empty and "status" in recon.columns:
        soft_rows = int((recon["status"] == "soft_anchor_diagnostic_only").sum())
    else:
        soft_rows = 0
    add("soft_anchor_diagnostic_rows", soft_rows, "Entity-year December rows where a diagnostic-only anchor was available but not enforced")

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    final_input_dir = Path(args.final_input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors, pools, dep, service = load_prepared(prepared_dir)
    rules = load_rules(final_input_dir)
    events = load_working_events(prepared_dir)
    project_milestones = load_project_milestones(prepared_dir)
    shares = load_shares(final_input_dir)

    validate_share_table(shares)

    bg_rates = resolve_background_dep_rates(pools, dep)
    pools_with_dep = attach_dep_rates_to_pools(pools, dep, bg_rates)

    # Step 01 now uses ONLY the cleaned additive dated-event ledger from Step 00.
    # We still net those positive dated events out of annual pools before monthly
    # allocation so annual totals remain authoritative and dated additions are not
    # double counted.
    pools_with_dep, dated_event_pool_netting = residualize_pools_for_positive_dated_events(pools_with_dep, events)

    monthly_flows = build_monthly_investment_ledger(pools_with_dep, events, shares, rules)
    monthly_flows = attach_dated_dep_rates(monthly_flows, dep)

    # Annual identity is checked against gross additions, not net flows.
    # Negative dated events such as disposals may still enter the monthly stock recursion,
    # but annual_total_kNIS is treated as an additions object.
    annual_from_months = (
        monthly_flows.assign(gross_addition_kNIS=monthly_flows["amount_kNIS"].clip(lower=0.0))
        .groupby(["entity", "year"], as_index=False)["gross_addition_kNIS"].sum()
        .rename(columns={"gross_addition_kNIS": "annual_from_months_kNIS"})
    )
    annual_compare = anchors[["entity", "year", "annual_total_kNIS"]].merge(
        annual_from_months, on=["entity", "year"], how="left"
    )
    annual_compare["annual_from_months_kNIS"] = annual_compare["annual_from_months_kNIS"].fillna(0.0)
    annual_compare["diff_kNIS"] = annual_compare["annual_total_kNIS"] - annual_compare["annual_from_months_kNIS"]
    bad = annual_compare.loc[annual_compare["annual_total_kNIS"].notna() & (annual_compare["diff_kNIS"].abs() > FLOW_IDENTITY_TOL)]
    if not bad.empty:
        raise ValueError(
            "Annual flow identity failed before the engine ran:\n"
            + bad.head(20).to_string(index=False)
        )

    initial_rows = []
    for entity in sorted(anchors["entity"].dropna().unique()):
        first_anchor = first_observed_hard_anchor_month(anchors, entity)
        if first_anchor is None:
            continue
        initial_rows.append(backsolve_initial_background_stock(entity, first_anchor, anchors, monthly_flows, pools_with_dep))
    initial_bg = pd.DataFrame(initial_rows)

    class_capital, recon = run_monthly_engine(anchors, pools_with_dep, monthly_flows, rules, initial_bg)
    debug_summary = build_engine_debug_summary(monthly_flows, class_capital, recon, initial_bg, project_milestones)

    write_tsv(monthly_flows, output_dir / "interpolation_01_monthly_class_investment.tsv")
    write_tsv(class_capital, output_dir / "interpolation_01_monthly_class_capital.tsv")
    write_tsv(bg_rates, output_dir / "interpolation_01_background_dep_rates.tsv")
    write_tsv(dated_event_pool_netting, output_dir / "interpolation_01_dated_event_pool_netting.tsv")
    write_tsv(initial_bg, output_dir / "interpolation_01_initial_stock_backsolve.tsv")
    write_tsv(recon, output_dir / "interpolation_01_year_end_reconciliation.tsv")
    write_tsv(debug_summary, output_dir / "interpolation_01_engine_debug_summary.tsv")

    manifest = {
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "prepared_dir": str(prepared_dir),
        "final_input_dir": str(final_input_dir),
        "output_dir": str(output_dir),
        "rows": {
            "working_pools": int(len(pools)),
            "working_additive_events": int(len(events)),
            "project_milestones": int(len(project_milestones)),
            "monthly_flows": int(len(monthly_flows)),
            "class_capital": int(len(class_capital)),
            "background_dep_rates": int(len(bg_rates)),
            "dated_event_pool_netting_rows": int(len(dated_event_pool_netting)),
            "initial_background_rows": int(len(initial_bg)),
            "year_end_reconciliation_rows": int(len(recon)),
        },
        "design_choices": {
            "annual_totals_authoritative": True,
            "end_of_month_investment_timing": True,
            "reconciliation_margin_is_background_stock": True,
            "missing_anchor_years_are_propagated": True,
            "dated_events_read_from_step00_working_file": True,
            "project_milestones_are_non_additive": True,
            "sipg_cumulative_transfer_pseudo_anchors_are_diagnostic_only": True,
            "share_schedule_fallback_hierarchy": [
                "exact entity-year-pool_type",
                "nearest available year within entity-pool_type",
                "uniform within-year fallback"
            ],
        },
    }
    with open(output_dir / "interpolation_01_engine_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Wrote: {output_dir / 'interpolation_01_monthly_class_investment.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_monthly_class_capital.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_background_dep_rates.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_dated_event_pool_netting.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_initial_stock_backsolve.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_year_end_reconciliation.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_engine_debug_summary.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_engine_manifest.json'}")


if __name__ == "__main__":
    main()

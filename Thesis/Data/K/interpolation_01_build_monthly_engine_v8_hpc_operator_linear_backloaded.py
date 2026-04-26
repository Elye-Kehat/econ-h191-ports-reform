#!/usr/bin/env python3
"""
interpolation_01_build_monthly_engine_v8_hpc_linear_backloaded.py

Purpose
-------
Build the monthly class-level investment ledger and the monthly class-level capital ledger,
while fitting hard annual stock anchors smoothly over each anchor block rather than with
a single December stock plug.

Changes in this version
-----------------------
1. Keeps the stable v6 baseline monthly interpolation logic for dated flows, mapped-undated pools,
   background pools, service-entry timing, depreciation lookup, and annual-flow identity checks.
2. Replaces the old December-only hard-anchor stock plug with a smooth background-flow bridge over
   each hard-anchor block.
3. HPC now uses the same simple operator-style linear backloaded timing logic as SIPG for:
   - mapped_undated annual pools, and
   - background / bridge-distribution timing.
   Concretely, within a full operating year:
       w_m = m / 78,   m = 1, ..., 12.
4. The bridge still stays close to the previous strategy overall: only the within-year timing of the
   HPC background margin is changed from uniform to linear backloaded.
5. Missing hard-anchor years are still propagated; if the next observed stock anchor is later, the bridge
   spans the full open block.
6. SIPG remains unchanged conceptually: its cumulative-transfer pseudo-anchors are still diagnostic only.

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


def linear_backloaded_schedule(allowed_months: List[int]) -> pd.DataFrame:
    """
    Simple operator-style linear backloaded schedule.

    If allowed_months = [1, ..., 12], this gives:
        w_m = m / 78,   m = 1, ..., 12

    More generally, over any allowed subset of months, weights are proportional to the
    rank within the allowed operating months. This matches the existing SIPG service-entry
    logic and is now also used for HPC undated operator investment timing.
    """
    full = pd.DataFrame({"month_num": list(range(1, 13)), "share": [0.0] * 12})
    if not allowed_months:
        return full
    ranks = np.arange(1, len(allowed_months) + 1, dtype=float)
    shares_vec = ranks / ranks.sum()
    full.loc[full["month_num"].isin(allowed_months), "share"] = shares_vec
    return full



def build_pool_share_schedule(entity: str, year: int, pool_type: str, shares: pd.DataFrame, rules: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    # HPC operator-style timing patch:
    # HPC is now aligned with SIPG's operator logic. For undated operator investment, we do not
    # use the old flat month-share schedules. Instead, we use a simple linear backloaded rule
    # within the operating year:
    #
    #     w_m = m / 78,   m = 1, ..., 12
    #
    # We apply this to both:
    # - HPC mapped_undated baseline annual pools
    # - HPC background timing, which also governs the smooth bridge distribution
    #
    # This preserves the smooth-bridge strategy while avoiding the visually straight HPC path
    # produced by constant within-year monthly flows.
    if entity == "HPC" and pool_type in {"mapped_undated", "background"}:
        return linear_backloaded_schedule(list(range(1, 13))), "hpc_operator_linear_backloaded"

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

        full = linear_backloaded_schedule(allowed_months)
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




def next_month(month_str: str) -> str:
    return str(pd.Period(month_str, freq="M") + 1)


def get_background_bucket_row(entity: str, year: int, pools_with_dep: pd.DataFrame) -> pd.Series:
    same_year = pools_with_dep.loc[
        (pools_with_dep["entity"] == entity)
        & (pools_with_dep["year"] == year)
        & (pools_with_dep["pool_type"] == "background")
    ].copy()
    if not same_year.empty:
        return same_year.iloc[0]

    all_bg = pools_with_dep.loc[
        (pools_with_dep["entity"] == entity)
        & (pools_with_dep["pool_type"] == "background")
    ].copy()
    if all_bg.empty:
        raise ValueError(f"No background pool rows available for entity {entity}")

    all_bg["year_distance"] = (all_bg["year"].astype(int) - int(year)).abs()
    all_bg = all_bg.sort_values(["year_distance", "year", "asset_class_std"]).reset_index(drop=True)
    return all_bg.iloc[0]


def build_background_bridge_schedule(entity: str, start_month: str, end_month: str, shares: pd.DataFrame, rules: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    months = month_range(start_month, end_month)
    rows = []
    source_rules = []

    for month in months:
        year = year_num(month)
        mnum = month_num(month)
        sched, rule_used = build_pool_share_schedule(entity, year, "background", shares, rules)
        raw_share = sched.loc[sched["month_num"] == mnum, "share"]
        raw_share = float(raw_share.iloc[0]) if not raw_share.empty else 0.0
        rows.append({
            "entity": entity,
            "month": month,
            "year": year,
            "month_num": mnum,
            "raw_share": raw_share,
            "share_rule_fragment": rule_used,
        })
        source_rules.append(rule_used)

    out = pd.DataFrame(rows)
    raw_total = float(out["raw_share"].sum())
    if raw_total <= IDENTITY_TOL:
        out["bridge_share"] = 1.0 / len(out)
        share_rule_used = "block_uniform_fallback"
    else:
        out["bridge_share"] = out["raw_share"] / raw_total
        share_rule_used = "block_background_schedule::" + "|".join(dict.fromkeys(source_rules))

    return out, share_rule_used


def build_bridge_rows_for_block(
    entity: str,
    start_month: str,
    end_month: str,
    residual_to_anchor_kNIS: float,
    shares: pd.DataFrame,
    rules: pd.DataFrame,
    pools_with_dep: pd.DataFrame,
    block_label: str,
) -> Tuple[pd.DataFrame, Dict[str, float | str]]:
    sched, share_rule_used = build_background_bridge_schedule(entity, start_month, end_month, shares, rules)
    if sched.empty:
        raise ValueError(f"Empty bridge schedule for {entity}, {start_month} -> {end_month}")

    dep_rows = []
    for month in sched["month"].tolist():
        bg_row = get_background_bucket_row(entity, year_num(month), pools_with_dep)
        dep_rows.append({
            "month": month,
            "asset_class_std": str(bg_row["asset_class_std"]),
            "dep_rate_annual_used": float(bg_row["dep_rate_annual_used"]),
            "dep_rate_monthly_used": float(bg_row["dep_rate_monthly_used"]),
        })
    dep_df = pd.DataFrame(dep_rows)
    sched = sched.merge(dep_df, on="month", how="left")

    months = sched["month"].tolist()
    deltas = sched["dep_rate_monthly_used"].tolist()
    survivals = []
    for i in range(len(months)):
        s = 1.0
        for j in range(i + 1, len(months)):
            s *= (1.0 - float(deltas[j]))
        survivals.append(s)
    sched["survival_to_block_end"] = survivals

    effective_weight = float((sched["bridge_share"] * sched["survival_to_block_end"]).sum())
    if abs(effective_weight) <= 1e-12:
        raise ValueError(
            f"Effective bridge weight is numerically zero for {entity}, {start_month} -> {end_month}. "
            "Cannot translate block-end stock residual into monthly background adjustments."
        )

    bridge_scale_lambda = float(residual_to_anchor_kNIS) / effective_weight
    sched["amount_kNIS"] = bridge_scale_lambda * sched["bridge_share"]

    rows = []
    for _, row in sched.iterrows():
        asset_class_std = str(row["asset_class_std"])
        rows.append({
            "entity": entity,
            "month": row["month"],
            "year": int(row["year"]),
            "pool_type": "background",
            "asset_class_std": asset_class_std,
            "engine_bucket": f"background::{asset_class_std}",
            "component_source": f"bridge_adjustment::{block_label}",
            "amount_kNIS": float(row["amount_kNIS"]),
            "share_used": float(row["bridge_share"]),
            "share_rule_used": share_rule_used,
            "dep_rate_annual_used": float(row["dep_rate_annual_used"]),
            "dep_rate_monthly_used": float(row["dep_rate_monthly_used"]),
            "notes": (
                "Smooth bridge adjustment added only to the background margin so the monthly path lands "
                "on the next hard annual stock anchor without a December-only stock plug."
            ),
        })

    meta = {
        "share_rule_used": share_rule_used,
        "effective_weight": effective_weight,
        "bridge_scale_lambda_kNIS": bridge_scale_lambda,
        "bridge_total_adjustment_kNIS": float(pd.DataFrame(rows)["amount_kNIS"].sum()) if rows else 0.0,
    }
    return pd.DataFrame(rows), meta


def simulate_entity_segment(
    entity: str,
    months_seq: List[str],
    flows_e: pd.DataFrame,
    pools_with_dep: pd.DataFrame,
    state: Dict[str, float],
    dep_m: Dict[str, float],
    bucket_to_class: Dict[str, str],
    bucket_to_pool: Dict[str, str],
    opening_month: str,
    pre_open_zero: Optional[bool],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, str], Dict[str, str]]:
    class_rows = []
    month_rows = []

    for month in months_seq:
        y = year_num(month)

        if opening_month and pre_open_zero is True and month < opening_month:
            state = {k: 0.0 for k in state.keys()}

        fm = flows_e.loc[flows_e["month"] == month].copy()
        active_buckets = sorted(set(list(state.keys()) + fm["engine_bucket"].tolist()))

        if not active_buckets and not state:
            month_rows.append({"entity": entity, "month": month, "year": y, "end_stock_total_kNIS": 0.0})
            continue

        bg_bucket, bg_class = choose_background_bucket(entity, active_buckets, pools_with_dep)

        if bg_bucket not in state:
            state[bg_bucket] = 0.0
            bg_row = get_background_bucket_row(entity, y, pools_with_dep)
            dep_m[bg_bucket] = float(bg_row["dep_rate_monthly_used"])
            bucket_to_class[bg_bucket] = str(bg_row["asset_class_std"])
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
                # Preserve background as the stock bucket label even when the monthly component_source
                # is a bridge adjustment. This keeps the class-capital recursion as close as possible
                # to the old strategy: the bridge is simply an extra background flow, not a new asset type.
                bucket_to_pool[bucket] = "background" if bucket.startswith("background::") else str(sub["pool_type"].iloc[0])

        for bucket in active_buckets:
            begin_stock = float(state.get(bucket, 0.0))
            delta_now = float(dep_m[bucket])
            investment = float(fm.loc[fm["engine_bucket"] == bucket, "amount_kNIS"].sum())
            end_stock = (1.0 - delta_now) * begin_stock + investment

            class_rows.append({
                "entity": entity,
                "month": month,
                "year": y,
                "pool_type": bucket_to_pool[bucket],
                "asset_class_std": bucket_to_class[bucket],
                "engine_bucket": bucket,
                "begin_stock_kNIS": begin_stock,
                "dep_rate_monthly_used": delta_now,
                "investment_kNIS": investment,
                "reconciliation_adjustment_kNIS": 0.0,
                "end_stock_kNIS": end_stock,
            })
            state[bucket] = end_stock

        month_rows.append({
            "entity": entity,
            "month": month,
            "year": y,
            "end_stock_total_kNIS": float(sum(state.values())),
        })

    return (
        pd.DataFrame(class_rows),
        pd.DataFrame(month_rows),
        state,
        dep_m,
        bucket_to_class,
        bucket_to_pool,
    )


def run_monthly_engine(
    anchors: pd.DataFrame,
    pools_with_dep: pd.DataFrame,
    monthly_flows: pd.DataFrame,
    shares: pd.DataFrame,
    rules: pd.DataFrame,
    initial_bg_rows: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months_all = month_range(SAMPLE_START, SAMPLE_END)
    entities = sorted(set(pools_with_dep["entity"]).union(set(anchors["entity"])).union(set(monthly_flows["entity"])))

    class_rows = []
    month_rows = []
    recon_rows = []
    bridge_rows_all = []
    final_flow_rows = []

    initial_map = {r["entity"]: r for _, r in initial_bg_rows.iterrows()}

    for entity in entities:
        rr = rules.loc[rules["entity"] == entity].copy()
        opening_month = rr["opening_month"].iloc[0] if not rr.empty else ""
        pre_open_zero = rr["pre_open_zero_rule"].iloc[0] if not rr.empty else None

        flows_e_base = monthly_flows.loc[monthly_flows["entity"] == entity].copy()
        flows_e_final = flows_e_base.copy()
        anchor_e = anchors.loc[anchors["entity"] == entity].copy().sort_values("year").reset_index(drop=True)

        state: Dict[str, float] = {}
        dep_map: Dict[str, float] = {}
        bucket_to_class: Dict[str, str] = {}
        bucket_to_pool: Dict[str, str] = {}

        init = initial_map.get(entity)
        if init is not None:
            bg_bucket = init["background_engine_bucket"]
            state[bg_bucket] = float(init["implied_initial_background_stock_kNIS"])
            dep_map[bg_bucket] = float(init["background_dep_rate_monthly"])
            bucket_to_class[bg_bucket] = str(init["background_asset_class_std"])
            bucket_to_pool[bg_bucket] = "background"

        hard_anchor_months = [f"{int(y)}-12" for y in anchor_e.loc[anchor_e["hard_anchor_flag"], "year"].tolist()]
        month_total_lookup = {}
        bridge_debug_by_year = {}

        if hard_anchor_months:
            # First hard anchor is still handled exactly by the original initial-background backsolve.
            first_anchor = hard_anchor_months[0]
            first_block_months = month_range(SAMPLE_START, first_anchor)
            block_class, block_month, state, dep_map, bucket_to_class, bucket_to_pool = simulate_entity_segment(
                entity, first_block_months, flows_e_final, pools_with_dep, state, dep_map, bucket_to_class, bucket_to_pool, opening_month, pre_open_zero
            )
            class_rows.append(block_class)
            month_rows.append(block_month)
            for _, r in block_month.iterrows():
                month_total_lookup[str(r["month"])] = float(r["end_stock_total_kNIS"])

            first_year = year_num(first_anchor)
            observed_first = float(anchor_e.loc[anchor_e["year"] == first_year, "stock_anchor_dec_kNIS"].iloc[0])
            provisional_first = float(month_total_lookup.get(first_anchor, 0.0))
            bridge_debug_by_year[first_year] = {
                "status": "initial_backsolve_anchor",
                "provisional_dec_stock_kNIS": provisional_first,
                "background_reconciliation_residual_kNIS": float(observed_first - provisional_first),
                "post_reconciliation_dec_stock_kNIS": provisional_first,
                "bridge_total_adjustment_kNIS": 0.0,
                "bridge_share_rule_used": "initial_background_backsolve",
                "bridge_effective_weight": np.nan,
                "bridge_scale_lambda_kNIS": np.nan,
                "block_start_month": SAMPLE_START,
                "block_end_month": first_anchor,
            }

            prev_anchor = first_anchor
            for current_anchor in hard_anchor_months[1:]:
                block_start = next_month(prev_anchor)
                block_months = month_range(block_start, current_anchor)

                prov_class, prov_month, prov_state, _, _, _ = simulate_entity_segment(
                    entity, block_months, flows_e_final, pools_with_dep, state.copy(), dep_map.copy(), bucket_to_class.copy(), bucket_to_pool.copy(), opening_month, pre_open_zero
                )
                provisional_dec = float(prov_month["end_stock_total_kNIS"].iloc[-1]) if not prov_month.empty else float(sum(state.values()))
                current_year = year_num(current_anchor)
                observed_anchor = float(anchor_e.loc[anchor_e["year"] == current_year, "stock_anchor_dec_kNIS"].iloc[0])
                residual = float(observed_anchor - provisional_dec)

                bridge_block_label = f"{entity}_{block_start}_to_{current_anchor}"
                bridge_rows, bridge_meta = build_bridge_rows_for_block(
                    entity=entity,
                    start_month=block_start,
                    end_month=current_anchor,
                    residual_to_anchor_kNIS=residual,
                    shares=shares,
                    rules=rules,
                    pools_with_dep=pools_with_dep,
                    block_label=bridge_block_label,
                )
                if not bridge_rows.empty:
                    bridge_rows_all.append(bridge_rows)
                    flows_e_final = pd.concat([flows_e_final, bridge_rows], ignore_index=True)
                    flows_e_final = flows_e_final.sort_values(["month", "engine_bucket", "component_source"]).reset_index(drop=True)

                final_block_class, final_block_month, state, dep_map, bucket_to_class, bucket_to_pool = simulate_entity_segment(
                    entity, block_months, flows_e_final, pools_with_dep, state, dep_map, bucket_to_class, bucket_to_pool, opening_month, pre_open_zero
                )
                class_rows.append(final_block_class)
                month_rows.append(final_block_month)
                for _, r in final_block_month.iterrows():
                    month_total_lookup[str(r["month"])] = float(r["end_stock_total_kNIS"])

                final_dec = float(final_block_month["end_stock_total_kNIS"].iloc[-1]) if not final_block_month.empty else float(sum(state.values()))
                bridge_debug_by_year[current_year] = {
                    "status": "bridged_to_anchor",
                    "provisional_dec_stock_kNIS": provisional_dec,
                    "background_reconciliation_residual_kNIS": residual,
                    "post_reconciliation_dec_stock_kNIS": final_dec,
                    "bridge_total_adjustment_kNIS": float(bridge_meta["bridge_total_adjustment_kNIS"]),
                    "bridge_share_rule_used": str(bridge_meta["share_rule_used"]),
                    "bridge_effective_weight": float(bridge_meta["effective_weight"]),
                    "bridge_scale_lambda_kNIS": float(bridge_meta["bridge_scale_lambda_kNIS"]),
                    "block_start_month": block_start,
                    "block_end_month": current_anchor,
                }
                prev_anchor = current_anchor

            if prev_anchor < SAMPLE_END:
                tail_start = next_month(prev_anchor)
                tail_months = month_range(tail_start, SAMPLE_END)
                tail_class, tail_month, state, dep_map, bucket_to_class, bucket_to_pool = simulate_entity_segment(
                    entity, tail_months, flows_e_final, pools_with_dep, state, dep_map, bucket_to_class, bucket_to_pool, opening_month, pre_open_zero
                )
                class_rows.append(tail_class)
                month_rows.append(tail_month)
                for _, r in tail_month.iterrows():
                    month_total_lookup[str(r["month"])] = float(r["end_stock_total_kNIS"])
        else:
            all_class, all_month, state, dep_map, bucket_to_class, bucket_to_pool = simulate_entity_segment(
                entity, months_all, flows_e_final, pools_with_dep, state, dep_map, bucket_to_class, bucket_to_pool, opening_month, pre_open_zero
            )
            class_rows.append(all_class)
            month_rows.append(all_month)
            for _, r in all_month.iterrows():
                month_total_lookup[str(r["month"])] = float(r["end_stock_total_kNIS"])

        final_flow_rows.append(flows_e_final)

        sample_years = sorted({year_num(m) for m in months_all})
        for y in sample_years:
            dec_month = f"{y}-12"
            final_dec_stock = float(month_total_lookup.get(dec_month, 0.0))
            ar = anchor_e.loc[anchor_e["year"] == y].copy()
            anchor_available = not ar.empty and pd.notna(ar["stock_anchor_dec_kNIS"].iloc[0])
            hard_anchor = bool(ar["hard_anchor_flag"].iloc[0]) if (anchor_available and "hard_anchor_flag" in ar.columns) else False
            observed_anchor = float(ar["stock_anchor_dec_kNIS"].iloc[0]) if anchor_available else np.nan
            anchor_policy = str(ar["stock_anchor_scope_note"].iloc[0]) if (anchor_available and "stock_anchor_scope_note" in ar.columns) else ""

            debug = bridge_debug_by_year.get(y, None)
            if debug is not None:
                status = debug["status"]
                provisional_dec_stock = float(debug["provisional_dec_stock_kNIS"])
                residual_out = float(debug["background_reconciliation_residual_kNIS"])
                post_out = float(debug["post_reconciliation_dec_stock_kNIS"])
                bridge_total = float(debug["bridge_total_adjustment_kNIS"])
                share_rule_out = str(debug["bridge_share_rule_used"])
                eff_w = debug["bridge_effective_weight"]
                lambda_out = debug["bridge_scale_lambda_kNIS"]
                block_start_out = debug["block_start_month"]
                block_end_out = debug["block_end_month"]
            elif anchor_available and not hard_anchor:
                status = "soft_anchor_diagnostic_only"
                provisional_dec_stock = final_dec_stock
                residual_out = np.nan
                post_out = final_dec_stock
                bridge_total = 0.0
                share_rule_out = ""
                eff_w = np.nan
                lambda_out = np.nan
                block_start_out = ""
                block_end_out = dec_month
            else:
                status = "propagated_no_anchor"
                provisional_dec_stock = final_dec_stock
                residual_out = np.nan
                post_out = final_dec_stock
                bridge_total = 0.0
                share_rule_out = ""
                eff_w = np.nan
                lambda_out = np.nan
                block_start_out = ""
                block_end_out = dec_month

            recon_rows.append({
                "entity": entity,
                "year": y,
                "december_month": dec_month,
                "anchor_observed_flag": bool(anchor_available),
                "hard_anchor_flag": bool(hard_anchor),
                "anchor_policy": anchor_policy if anchor_available else "",
                "observed_stock_anchor_kNIS": observed_anchor,
                "provisional_dec_stock_kNIS": provisional_dec_stock,
                "background_reconciliation_residual_kNIS": residual_out,
                "bridge_total_adjustment_kNIS": bridge_total,
                "bridge_share_rule_used": share_rule_out,
                "bridge_effective_weight": eff_w,
                "bridge_scale_lambda_kNIS": lambda_out,
                "bridge_block_start_month": block_start_out,
                "bridge_block_end_month": block_end_out,
                "post_reconciliation_dec_stock_kNIS": post_out,
                "final_dec_stock_kNIS": final_dec_stock,
                "status": status,
            })

    class_df = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame(columns=[
        "entity", "month", "year", "pool_type", "asset_class_std", "engine_bucket",
        "begin_stock_kNIS", "dep_rate_monthly_used", "investment_kNIS",
        "reconciliation_adjustment_kNIS", "end_stock_kNIS"
    ])
    month_df = pd.concat(month_rows, ignore_index=True) if month_rows else pd.DataFrame(columns=[
        "entity", "month", "year", "end_stock_total_kNIS"
    ])
    recon_df = pd.DataFrame(recon_rows)
    bridge_df = pd.concat(bridge_rows_all, ignore_index=True) if bridge_rows_all else pd.DataFrame(columns=monthly_flows.columns)
    final_flows_df = pd.concat(final_flow_rows, ignore_index=True) if final_flow_rows else monthly_flows.iloc[0:0].copy()

    class_df = class_df.sort_values(["entity", "month", "engine_bucket"]).reset_index(drop=True)
    month_df = month_df.sort_values(["entity", "month"]).reset_index(drop=True)
    recon_df = recon_df.sort_values(["entity", "year"]).reset_index(drop=True)
    bridge_df = bridge_df.sort_values(["entity", "month", "component_source"]).reset_index(drop=True)
    final_flows_df = final_flows_df.sort_values(["entity", "month", "engine_bucket", "component_source"]).reset_index(drop=True)

    return class_df, month_df, recon_df, bridge_df, final_flows_df


def build_engine_debug_summary(monthly_flows: pd.DataFrame, class_capital: pd.DataFrame, recon: pd.DataFrame, initial_bg: pd.DataFrame, project_milestones: pd.DataFrame, bridge_adjustments: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric, value, notes):
        rows.append({"metric": metric, "value": value, "notes": notes})

    add("monthly_flow_rows_final", int(len(monthly_flows)), "Rows in the final monthly class investment ledger after smooth bridge adjustments")
    add("monthly_capital_rows", int(len(class_capital)), "Rows in the final monthly class capital ledger")
    add("reconciliation_rows", int(len(recon)), "Entity-year December anchor diagnostics rows")
    add("initial_background_rows", int(len(initial_bg)), "Entities with a backsolved initial background stock")
    add("project_milestone_rows", int(len(project_milestones)), "Non-additive milestone rows passed through from Step 00")
    add("bridge_adjustment_rows", int(len(bridge_adjustments)), "Monthly bridge-adjustment rows added to the background margin")

    n_neg = int(initial_bg["is_materially_negative"].sum()) if not initial_bg.empty else 0
    add("materially_negative_initial_background_count", n_neg, "Count of entities with materially negative initial background stock")

    if not monthly_flows.empty and "share_rule_used" in monthly_flows.columns:
        fallback_rows = monthly_flows["share_rule_used"].astype(str).str.contains("fallback", case=False, na=False).sum()
    else:
        fallback_rows = 0
    add("rows_using_share_schedule_fallback", int(fallback_rows), "Monthly flow rows allocated with nearest-year or uniform fallback share rules")

    if not recon.empty and "status" in recon.columns:
        soft_rows = int((recon["status"] == "soft_anchor_diagnostic_only").sum())
        bridged_rows = int((recon["status"] == "bridged_to_anchor").sum())
    else:
        soft_rows = 0
        bridged_rows = 0
    add("soft_anchor_diagnostic_rows", soft_rows, "Entity-year December rows where a diagnostic-only anchor was available but not enforced")
    add("bridged_hard_anchor_rows", bridged_rows, "Hard-anchor years reached via smooth within-block bridge adjustments instead of a December stock plug")

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

    # Keep the stable v6 accounting core: annual totals remain authoritative and
    # positive dated events are netted out of annual pools before monthly allocation.
    pools_with_dep, dated_event_pool_netting = residualize_pools_for_positive_dated_events(pools_with_dep, events)

    monthly_flows_baseline = build_monthly_investment_ledger(pools_with_dep, events, shares, rules)
    monthly_flows_baseline = attach_dated_dep_rates(monthly_flows_baseline, dep)

    # Pre-bridge annual identity still refers to the baseline monthly flow ledger.
    annual_from_months = (
        monthly_flows_baseline.assign(gross_addition_kNIS=monthly_flows_baseline["amount_kNIS"].clip(lower=0.0))
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
        initial_rows.append(backsolve_initial_background_stock(entity, first_anchor, anchors, monthly_flows_baseline, pools_with_dep))
    initial_bg = pd.DataFrame(initial_rows)

    class_capital, entity_month_capital, recon, bridge_adjustments, monthly_flows_final = run_monthly_engine(
        anchors, pools_with_dep, monthly_flows_baseline, shares, rules, initial_bg
    )
    debug_summary = build_engine_debug_summary(monthly_flows_final, class_capital, recon, initial_bg, project_milestones, bridge_adjustments)

    write_tsv(monthly_flows_baseline, output_dir / "interpolation_01_monthly_class_investment_baseline.tsv")
    write_tsv(monthly_flows_final, output_dir / "interpolation_01_monthly_class_investment.tsv")
    write_tsv(class_capital, output_dir / "interpolation_01_monthly_class_capital.tsv")
    write_tsv(entity_month_capital, output_dir / "interpolation_01_monthly_entity_capital.tsv")
    write_tsv(bg_rates, output_dir / "interpolation_01_background_dep_rates.tsv")
    write_tsv(dated_event_pool_netting, output_dir / "interpolation_01_dated_event_pool_netting.tsv")
    write_tsv(initial_bg, output_dir / "interpolation_01_initial_stock_backsolve.tsv")
    write_tsv(bridge_adjustments, output_dir / "interpolation_01_bridge_adjustments.tsv")
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
            "monthly_flows_baseline": int(len(monthly_flows_baseline)),
            "monthly_flows_final": int(len(monthly_flows_final)),
            "bridge_adjustment_rows": int(len(bridge_adjustments)),
            "class_capital": int(len(class_capital)),
            "entity_month_capital": int(len(entity_month_capital)),
            "background_dep_rates": int(len(bg_rates)),
            "dated_event_pool_netting_rows": int(len(dated_event_pool_netting)),
            "initial_background_rows": int(len(initial_bg)),
            "year_end_reconciliation_rows": int(len(recon)),
        },
        "design_choices": {
            "annual_totals_authoritative": True,
            "end_of_month_investment_timing": True,
            "baseline_monthly_interpolation_preserved": True,
            "hard_anchor_residuals_converted_into_smooth_background_flow_bridges": True,
            "bridge_distribution_uses_existing_background_month_share_schedules": True,
            "hpc_operator_timing_uses_linear_backloaded_rule": True,
            "hpc_bridge_distribution_uses_same_linear_backloaded_rule": True,
            "missing_anchor_years_are_propagated_and_open_blocks_bridge_to_next_hard_anchor": True,
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
    print(f"Wrote: {output_dir / 'interpolation_01_monthly_class_investment_baseline.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_monthly_class_investment.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_monthly_class_capital.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_monthly_entity_capital.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_background_dep_rates.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_dated_event_pool_netting.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_initial_stock_backsolve.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_bridge_adjustments.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_year_end_reconciliation.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_engine_debug_summary.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_01_engine_manifest.json'}")
if __name__ == "__main__":
    main()

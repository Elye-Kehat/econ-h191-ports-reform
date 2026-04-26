#!/usr/bin/env python3
"""
interpolation_00_prepare_working_inputs_v7_bridge_ready.py

Purpose
-------
Prepare the finalized K input package for the monthly interpolation engine.

This version keeps the earlier v3 fixes and adds one more economically important
preprocessing decision:

Bridge-strategy note
---------
This file remains the stable preprocessing stage.

For the new smooth-bridge strategy, no substantive preprocessing change is needed here.
The economically sensitive input decisions still belong in Step 00, but the new smoothing
logic is a Step 01 engine change: we keep the same finalized annual pools, dated events,
depreciation lookup, and anchor metadata, and we change only how Step 01 distributes the
residual needed to land smoothly on the next hard annual stock anchor.

So this version is intentionally bridge-ready but logic-stable: it preserves the last working
Step 00 behavior and simply passes the same prepared inputs forward to the revised Step 01
engine.
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

DEFAULT_INPUT_DIR = "Data/K/Final Input Files"
DEFAULT_OUTPUT_DIR = "Data/K/Interpolation Output"

TRUE_STRINGS = {"1", "true", "t", "yes", "y"}
FALSE_STRINGS = {"0", "false", "f", "no", "n"}

REQUIRED_INPUT_FILES = {
    "anchors": "k_entity_year_anchors.tsv",
    "dep": "k_dep_lookup.tsv",
    "events": "k_dated_events.tsv",
    "pools": "k_annual_class_pools.tsv",
    "rules": "k_entity_rules.tsv",
    "shares": "k_month_shares.tsv",
}

ANCHOR_ALIASES = {
    "entity": ["entity", "owner", "firm", "operator"],
    "year": ["year", "fiscal_year"],
    "annual_total": [
        "I_annual_total_kNIS",
        "annual_total_kNIS",
        "annual_investment_kNIS",
        "annual_investment_total_kNIS",
        "annual_total_investment_kNIS",
        "gross_inv_kNIS",
        "investment_kNIS",
    ],
    "stock_anchor": [
        "K_anchor_dec_kNIS",
        "dec_stock_anchor_kNIS",
        "year_end_stock_anchor_kNIS",
        "net_ppe_close_kNIS",
        "stock_anchor_kNIS",
        "K_december_kNIS",
    ],
    "service_entry": [
        "service_entry_annual_kNIS",
        "explicit_service_entry_kNIS",
        "explicit_service_entry_annual_kNIS",
        "explicit_service_entry_annual_real_kNIS",
        "service_entry_kNIS",
        "transfer_to_ppe_kNIS",
        "annual_transfer_to_ppe_kNIS",
        "transfers_to_ppe_kNIS",
        "transfer_to_ppe_annual_kNIS",
        "transfer_to_ppe_annual_real_kNIS",
    ],
    "notes": ["notes", "anchor_notes"],
}

POOL_ALIASES = {
    "entity": ["entity", "owner", "firm", "operator"],
    "year": ["year", "fiscal_year"],
    "asset_class_std": ["asset_class_std", "asset_class", "class_std", "class"],
    "pool_type": ["pool_type", "component_type", "component", "annual_pool_type"],
    "annual_amount": ["annual_amount_kNIS", "amount_kNIS", "annual_kNIS", "value_kNIS"],
    "include_prod": [
        "include_in_productive_K",
        "include_productive",
        "productive_flag",
        "productive_capital_flag",
    ],
    "allocation_priority": ["allocation_priority", "priority"],
    "source_id": ["source_id", "record_id", "pool_id"],
    "notes": ["notes", "description"],
}

DEP_ALIASES = {
    "entity": ["entity", "owner", "firm", "operator"],
    "asset_class_std": ["asset_class_std", "asset_class", "class_std", "class"],
    "dep_rate_annual": [
        "dep_rate_annual",
        "annual_dep_rate",
        "annual_depreciation_rate",
        "annual_depreciation_rate_main_pct",
    ],
    "dep_rate_monthly": ["dep_rate_monthly", "monthly_dep_rate", "monthly_depreciation_rate"],
    "dep_method": ["dep_method", "depreciation_method"],
    "is_fallback": ["is_fallback", "fallback_flag"],
    "fallback_group": ["fallback_group", "group"],
    "source_id": ["source_id", "record_id", "dep_id"],
    "notes": ["notes", "description"],
}

RULE_ALIASES = {
    "entity": ["entity", "owner", "firm", "operator"],
    "opening_month": ["opening_month", "open_month", "operational_start_month", "first_operating_month"],
    "pre_open_zero": ["pre_open_zero_rule", "zero_before_opening", "pre_opening_zero_rule"],
    "allow_empty_dated_events": ["allow_empty_dated_events", "dated_events_optional"],
    "default_share_rule": ["default_share_rule", "share_rule"],
    "notes": ["notes", "description"],
}

EVENT_ALIASES = {
    "project_id": ["project_id", "event_id", "record_id"],
    "entity": ["entity", "owner", "firm", "operator"],
    "event_month": ["event_month", "month", "commission_month", "date_month"],
    "asset_class_std": ["asset_class_std", "asset_class", "class_std", "class"],
    "amount": ["amount_kNIS", "event_amount_kNIS", "signed_amount_kNIS", "annual_amount_kNIS"],
    "event_type": ["event_type", "kind", "direction", "flow_type"],
    "include_prod": [
        "include_in_productive_K",
        "include_productive",
        "productive_flag",
        "productive_capital_flag",
    ],
    "source_id": ["source_id", "record_id"],
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


# ---------------------------------------------------------------------------
# Non-additive milestone reclassification rules
# ---------------------------------------------------------------------------
# Why this exists:
# Some rows in the raw dated-events file are useful as project-timing evidence, but should not
# be treated as additive monthly capital-flow rows inside the engine.
#
# The main current case is the HPC Kishon West cranes project. The reviewed evidence suggests
# that the raw 100,000 headline amount is better interpreted as a broader multi-stage project
# amount than as one clean booked service-entry addition in a single 2023 month. Because this
# pipeline treats annual investment totals as authoritative, keeping that row as an additive dated
# event would double count it on top of the annual pools.
#
# Therefore we reclassify this row as a milestone in interpolation_00:
# - it is REMOVED from the additive working dated-events ledger that interpolation_01 should read
# - it is PRESERVED in a separate project-milestones output so the timing evidence is not lost
#
# The rule is intentionally narrow and explicit. We only match the exact project_id currently
# known to cause the accounting contradiction, rather than using loose keywords that could
# accidentally reclassify unrelated events.
NON_ADDITIVE_MILESTONE_PROJECT_IDS = {
    "HPC_KishonWest_NewCranes_2023",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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


def str_truthy(x: str) -> Optional[bool]:
    s = str(x).strip().lower()
    if s == "":
        return None
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    return None


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


def require_inputs(input_dir: Path) -> Dict[str, Path]:
    out = {}
    missing = []
    for k, fname in REQUIRED_INPUT_FILES.items():
        path = input_dir / fname
        out[k] = path
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required finalized input files:\n" + "\n".join(missing))
    return out


def prepare_anchors(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = rename_by_aliases(df_raw, ANCHOR_ALIASES, required_fields=["entity", "year", "annual_total"])
    for col in ["stock_anchor", "service_entry", "notes"]:
        if col not in df.columns:
            df[col] = ""

    df["entity"] = df["entity"].map(normalize_entity)
    df["year"] = as_num(df["year"]).astype("Int64")
    df["annual_total_all_kNIS"] = as_num(df["annual_total"])
    df["stock_anchor_dec_all_kNIS"] = as_num(df["stock_anchor"])
    df["service_entry_annual_kNIS"] = as_num(df["service_entry"])
    df["annual_total_kNIS"] = df["annual_total_all_kNIS"]
    df["stock_anchor_dec_kNIS"] = df["stock_anchor_dec_all_kNIS"]
    df["stock_anchor_scope_note"] = "raw_anchor_not_productive_adjusted"

    keep = [
        "entity",
        "year",
        "annual_total_all_kNIS",
        "annual_total_kNIS",
        "stock_anchor_dec_all_kNIS",
        "stock_anchor_dec_kNIS",
        "stock_anchor_scope_note",
        "service_entry_annual_kNIS",
        "notes",
    ]
    return df[keep].sort_values(["entity", "year"]).reset_index(drop=True)


def prepare_pools(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = rename_by_aliases(df_raw, POOL_ALIASES, required_fields=["entity", "year", "asset_class_std", "pool_type", "annual_amount"])
    for col in ["include_prod", "allocation_priority", "source_id", "notes"]:
        if col not in df.columns:
            df[col] = ""

    df["entity"] = df["entity"].map(normalize_entity)
    df["year"] = as_num(df["year"]).astype("Int64")
    df["asset_class_std"] = df["asset_class_std"].astype(str).str.strip()
    df["pool_type"] = df["pool_type"].astype(str).str.strip().str.lower()
    df["annual_amount_kNIS"] = as_num(df["annual_amount"]).fillna(0.0)
    df["allocation_priority"] = pd.to_numeric(df["allocation_priority"], errors="coerce").fillna(0).astype(int)
    df["include_in_productive_K"] = df["include_prod"].apply(str_truthy)

    keep = [
        "entity",
        "year",
        "asset_class_std",
        "pool_type",
        "annual_amount_kNIS",
        "include_in_productive_K",
        "allocation_priority",
        "source_id",
        "notes",
    ]
    return df[keep].sort_values(["entity", "year", "pool_type", "asset_class_std"]).reset_index(drop=True)


def prepare_dep(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = rename_by_aliases(df_raw, DEP_ALIASES, required_fields=["entity", "asset_class_std", "dep_rate_annual"])
    for col in ["dep_rate_monthly", "dep_method", "is_fallback", "fallback_group", "source_id", "notes"]:
        if col not in df.columns:
            df[col] = ""

    df["entity"] = df["entity"].map(normalize_entity)
    df["asset_class_std"] = df["asset_class_std"].astype(str).str.strip()
    annual = as_num(df["dep_rate_annual"])
    monthly = as_num(df["dep_rate_monthly"])
    monthly = monthly.where(monthly.notna(), 1.0 - (1.0 - annual) ** (1.0 / 12.0))
    df["dep_rate_annual"] = annual
    df["dep_rate_monthly"] = monthly
    df["is_fallback"] = df["is_fallback"].apply(str_truthy)

    keep = [
        "entity",
        "asset_class_std",
        "dep_rate_annual",
        "dep_rate_monthly",
        "dep_method",
        "is_fallback",
        "fallback_group",
        "source_id",
        "notes",
    ]
    return df[keep].sort_values(["entity", "asset_class_std", "source_id"]).reset_index(drop=True)


def prepare_rules(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = rename_by_aliases(df_raw, RULE_ALIASES, required_fields=["entity"])
    for col in ["opening_month", "pre_open_zero", "allow_empty_dated_events", "default_share_rule", "notes"]:
        if col not in df.columns:
            df[col] = ""
    df["entity"] = df["entity"].map(normalize_entity)
    df["opening_month"] = df["opening_month"].apply(normalize_month_str)
    df["pre_open_zero_rule"] = df["pre_open_zero"].apply(str_truthy)
    df["allow_empty_dated_events"] = df["allow_empty_dated_events"].apply(str_truthy)

    keep = [
        "entity",
        "opening_month",
        "pre_open_zero_rule",
        "allow_empty_dated_events",
        "default_share_rule",
        "notes",
    ]
    return df[keep].drop_duplicates(subset=["entity"]).sort_values(["entity"]).reset_index(drop=True)


def event_signed_amount(row: pd.Series) -> float:
    amt = float(row["amount_num"])
    et = str(row.get("event_type", "")).lower()
    if any(k in et for k in ["disposal", "sale", "writeoff", "retire", "derecognition", "remove", "transfer_out"]):
        return -abs(amt)
    return amt


def prepare_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = rename_by_aliases(
        df_raw,
        EVENT_ALIASES,
        required_fields=["project_id", "entity", "event_month", "asset_class_std", "amount"],
    )
    for col in ["event_type", "include_prod", "source_id", "notes"]:
        if col not in df.columns:
            df[col] = ""

    df["entity"] = df["entity"].map(normalize_entity)
    df["event_month"] = df["event_month"].apply(normalize_month_str)
    df["asset_class_std"] = df["asset_class_std"].astype(str).str.strip()
    df["amount_num"] = as_num(df["amount"]).fillna(0.0)
    df["signed_amount_kNIS"] = df.apply(event_signed_amount, axis=1)
    df["include_in_productive_K"] = df["include_prod"].apply(str_truthy)

    keep = [
        "project_id",
        "entity",
        "event_month",
        "asset_class_std",
        "event_type",
        "signed_amount_kNIS",
        "include_in_productive_K",
        "source_id",
        "notes",
    ]
    return df[keep].sort_values(["entity", "event_month", "project_id"]).reset_index(drop=True)


def prepare_shares(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = rename_by_aliases(
        df_raw,
        SHARE_ALIASES,
        required_fields=["entity", "year", "pool_type", "month_num", "share"],
    )
    if "notes" not in df.columns:
        df["notes"] = ""
    df["entity"] = df["entity"].map(normalize_entity)
    df["year"] = as_num(df["year"]).astype("Int64")
    df["pool_type"] = df["pool_type"].astype(str).str.strip().str.lower()
    df["month_num"] = as_num(df["month_num"]).astype("Int64")
    df["share"] = as_num(df["share"]).fillna(0.0)
    keep = ["entity", "year", "pool_type", "month_num", "share", "notes"]
    return df[keep].sort_values(["entity", "year", "pool_type", "month_num"]).reset_index(drop=True)


def normalize_match_text(*parts: str) -> Tuple[str, str]:
    raw = " ".join("" if p is None else str(p) for p in parts).lower()
    normalized = raw.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return raw, normalized


def is_right_of_use(asset_class_std: str, source_id: str = "", notes: str = "", pool_type: str = "") -> bool:
    raw_text, norm_text = normalize_match_text(asset_class_std, source_id, notes, pool_type)

    raw_explicit = [
        "right_of_use",
        "right-of-use",
        "right_of_use_assets",
        "lease_asset",
        "lease_assets",
    ]
    if any(tok in raw_text for tok in raw_explicit):
        return True

    phrase_explicit = [
        "right of use",
        "right of use asset",
        "right of use assets",
        "lease asset",
        "lease assets",
    ]
    if any(p in norm_text for p in phrase_explicit):
        return True

    if re.search(r"\brou\b", norm_text):
        return True

    return False


def productive_filter_pools(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keep_rows = []
    excluded_rows = []

    for _, row in df.iterrows():
        reason = None
        flag = row["include_in_productive_K"]
        if flag is False:
            reason = "non_productive_flag_false"
        elif is_right_of_use(row["asset_class_std"], row["source_id"], row["notes"], row["pool_type"]):
            reason = "right_of_use_excluded"

        if reason is None:
            keep_rows.append(row)
        else:
            rec = row.to_dict()
            rec["exclusion_reason"] = reason
            rec["source_table"] = "annual_pools"
            excluded_rows.append(rec)

    keep = pd.DataFrame(keep_rows, columns=df.columns).reset_index(drop=True)
    excluded = pd.DataFrame(excluded_rows)

    if not keep.empty:
        rou_left = keep.loc[
            keep.apply(
                lambda r: is_right_of_use(r["asset_class_std"], r["source_id"], r["notes"], r["pool_type"]),
                axis=1,
            )
        ].copy()
        if not rou_left.empty:
            raise ValueError(
                "Right-of-use rows still remain in working annual pools after productive filtering:\n"
                + rou_left[["entity", "year", "asset_class_std", "pool_type", "source_id"]].to_string(index=False)
            )

    return keep, excluded


def productive_filter_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keep_rows = []
    excluded_rows = []

    for _, row in df.iterrows():
        reason = None
        flag = row["include_in_productive_K"]
        if flag is False:
            reason = "non_productive_flag_false"
        elif is_right_of_use(row["asset_class_std"], row["source_id"], row["notes"], ""):
            reason = "right_of_use_excluded"

        if reason is None:
            keep_rows.append(row)
        else:
            rec = row.to_dict()
            rec["exclusion_reason"] = reason
            rec["source_table"] = "dated_events"
            excluded_rows.append(rec)

    return (
        pd.DataFrame(keep_rows, columns=df.columns).reset_index(drop=True),
        pd.DataFrame(excluded_rows),
    )



def adjust_anchors_to_productive_scope(anchors: pd.DataFrame, excluded_pool_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw annual anchors into productive-capital-consistent anchors.

    General rule
    ------------
    For most entities, productive annual totals equal raw annual totals minus any annual pool rows
    explicitly excluded from productive capital (for example right-of-use additions).

    SIPG special rule
    -----------------
    Bayport data in Panel C distinguish between:
    - CIP additions (construction spending and capitalised interest), and
    - Transfers to PPE (the part that actually enters productive fixed assets).

    For the preferred productive-K series, the annual flow that should enter the monthly engine is
    therefore Transfers to PPE, not CIP additions. So for SIPG we override:

        annual_total_productive_kNIS = explicit_service_entry_annual_kNIS

    with missing service-entry values interpreted as zero productive additions.

    SIPG stock-anchor rule
    ----------------------
    The SIPG stock-anchor basis used here is only a rough anchor based on cumulative transfers to PPE,
    not a true disclosed net-PPE stock. That cumulative figure is informative in:
    - years before productive service begins (zero anchor),
    - years in which transfers to PPE are positive (rough productive-service anchor).

    But once cumulative transfers stop changing, the flat cumulative figure is a gross cumulative
    commissioning total, not a valid depreciated stock anchor. For those flat years we leave the
    productive stock anchor blank so interpolation_01 can propagate the series rather than forcing
    the engine to reverse depreciation just to match a gross cumulative amount.
    """
    anchors = anchors.copy()

    if excluded_pool_rows.empty:
        anchors["annual_total_excluded_nonproductive_kNIS"] = 0.0
    else:
        tmp = excluded_pool_rows.copy()
        tmp["annual_amount_kNIS"] = as_num(tmp["annual_amount_kNIS"]).fillna(0.0)
        ex = (
            tmp.groupby(["entity", "year"], as_index=False)["annual_amount_kNIS"]
            .sum()
            .rename(columns={"annual_amount_kNIS": "annual_total_excluded_nonproductive_kNIS"})
        )
        anchors = anchors.merge(ex, on=["entity", "year"], how="left")
        anchors["annual_total_excluded_nonproductive_kNIS"] = anchors["annual_total_excluded_nonproductive_kNIS"].fillna(0.0)

    # Baseline productive annual total for non-SIPG entities.
    anchors["annual_total_productive_kNIS"] = anchors["annual_total_all_kNIS"] - anchors["annual_total_excluded_nonproductive_kNIS"]
    anchors["annual_total_kNIS"] = anchors["annual_total_productive_kNIS"]

    sipg_mask = anchors["entity"].eq("SIPG")
    if sipg_mask.any():
        sipg = anchors.loc[sipg_mask].copy().sort_values("year").reset_index(drop=True)

        # Missing service-entry values are interpreted as zero productive additions in the preferred spec.
        sipg["service_entry_annual_kNIS"] = sipg["service_entry_annual_kNIS"].fillna(0.0)

        # Productive annual flow for SIPG is transfers to PPE, not CIP additions.
        sipg["annual_total_productive_kNIS"] = sipg["service_entry_annual_kNIS"]
        sipg["annual_total_kNIS"] = sipg["annual_total_productive_kNIS"]

        # Rebuild cumulative transfers from annual transfer-to-PPE flows.
        sipg["sipg_cum_transfers_rebuilt_kNIS"] = sipg["service_entry_annual_kNIS"].cumsum()

        new_stock = []
        new_scope_note = []

        for _, row in sipg.iterrows():
            service_flow = float(row["service_entry_annual_kNIS"])
            cum_transfers = float(row["sipg_cum_transfers_rebuilt_kNIS"])

            if abs(cum_transfers) < 1e-12:
                # Before productive service starts, a zero productive anchor is informative and safe.
                new_stock.append(0.0)
                new_scope_note.append("sipg_zero_cumulative_transfers_anchor")
            elif service_flow > 0:
                # In years with positive transfers, cumulative transfers provide a rough productive-service anchor.
                new_stock.append(cum_transfers)
                new_scope_note.append("sipg_cumulative_transfers_service_year_anchor")
            else:
                # Flat cumulative transfers after the commissioning wave are not valid depreciated-stock anchors.
                new_stock.append(np.nan)
                new_scope_note.append("sipg_flat_cumulative_transfers_unanchored")

        sipg["stock_anchor_dec_kNIS"] = new_stock
        sipg["stock_anchor_scope_note"] = new_scope_note

        # Important implementation detail:
        # sipg was reset_index(drop=True) above, so its row index no longer matches the
        # original SIPG row positions inside anchors. Assigning DataFrames by label here
        # would align on index labels and silently write NaNs into the SIPG rows.
        # We therefore assign by VALUES back into the SIPG slice, preserving row order
        # rather than relying on index alignment.
        anchors.loc[sipg_mask, anchors.columns] = sipg[anchors.columns].to_numpy()

    return anchors


def build_sipg_cip_memo(anchors: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve the SIPG construction-side information in a memo table.

    This table is NOT an input to interpolation_01. Its purpose is transparency:
    - CIP additions remain visible for narrative and QA,
    - productive transfers to PPE remain visible as the preferred productive-flow concept,
    - the rebuilt cumulative-transfer anchors remain visible alongside the anchor policy used.
    """
    sipg = anchors.loc[anchors["entity"].eq("SIPG")].copy().sort_values("year").reset_index(drop=True)
    if sipg.empty:
        return pd.DataFrame(columns=[
            "entity",
            "year",
            "cip_additions_raw_kNIS",
            "productive_transfers_to_ppe_kNIS",
            "rebuilt_cumulative_transfers_kNIS",
            "productive_stock_anchor_used_kNIS",
            "productive_stock_anchor_policy",
        ])

    sipg["service_entry_annual_kNIS"] = sipg["service_entry_annual_kNIS"].fillna(0.0)
    sipg["rebuilt_cumulative_transfers_kNIS"] = sipg["service_entry_annual_kNIS"].cumsum()

    memo = pd.DataFrame({
        "entity": sipg["entity"],
        "year": sipg["year"],
        "cip_additions_raw_kNIS": sipg["annual_total_all_kNIS"],
        "productive_transfers_to_ppe_kNIS": sipg["service_entry_annual_kNIS"],
        "rebuilt_cumulative_transfers_kNIS": sipg["rebuilt_cumulative_transfers_kNIS"],
        "productive_stock_anchor_used_kNIS": sipg["stock_anchor_dec_kNIS"],
        "productive_stock_anchor_policy": sipg["stock_anchor_scope_note"],
    })

    return memo.reset_index(drop=True)


def build_sipg_service_entry_pools(anchors: pd.DataFrame, pools_kept_non_sipg_bg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pools = pools_kept_non_sipg_bg.copy()
    anchors = anchors.copy()

    drop_mask = (pools["entity"] == "SIPG") & (pools["pool_type"].isin(["background", "service_entry_undated"]))
    pools = pools.loc[~drop_mask].copy()

    sipg_anchors = anchors.loc[anchors["entity"] == "SIPG"].copy()

    service_rows = []
    background_rows = []
    positive_service_count = 0

    for _, ar in sipg_anchors.iterrows():
        year = int(ar["year"])
        annual_total = float(ar["annual_total_kNIS"]) if pd.notna(ar["annual_total_kNIS"]) else np.nan
        service_entry = float(ar["service_entry_annual_kNIS"]) if pd.notna(ar["service_entry_annual_kNIS"]) else 0.0

        if pd.isna(annual_total):
            continue

        service_amt = min(max(service_entry, 0.0), max(annual_total, 0.0))
        bg_amt = annual_total - service_amt

        if service_amt > 0:
            positive_service_count += 1

        service_rows.append({
            "entity": "SIPG",
            "year": year,
            "asset_class_std": "unknown_bayport_class",
            "pool_type": "service_entry_undated",
            "annual_amount_kNIS": round(service_amt, 6),
            "include_in_productive_K": True,
            "allocation_priority": 2,
            "source_id": "SIPG_DERIVED_SERVICE_ENTRY_V5",
            "notes": "Derived from SIPG transfer-to-PPE / service-entry anchor information; productive annual total is defined by transfers to PPE in v5",
        })
        background_rows.append({
            "entity": "SIPG",
            "year": year,
            "asset_class_std": "unknown_bayport_class",
            "pool_type": "background",
            "annual_amount_kNIS": round(bg_amt, 6),
            "include_in_productive_K": True,
            "allocation_priority": 3,
            "source_id": "SIPG_DERIVED_BACKGROUND_V5",
            "notes": "Residual SIPG background after derived service-entry split (expected to be zero in preferred productive SIPG build unless annual total exceeds service-entry)",
        })

    service_cols = [
        "entity",
        "year",
        "asset_class_std",
        "pool_type",
        "annual_amount_kNIS",
        "include_in_productive_K",
        "allocation_priority",
        "source_id",
        "notes",
    ]
    service_df = pd.DataFrame(service_rows, columns=service_cols)
    background_df = pd.DataFrame(background_rows, columns=service_cols)
    if not service_df.empty:
        service_df = service_df.sort_values(["year"]).reset_index(drop=True)
    if not background_df.empty:
        background_df = background_df.sort_values(["year"]).reset_index(drop=True)

    pools = pd.concat([pools, background_df, service_df], ignore_index=True)
    pools = pools.sort_values(["entity", "year", "pool_type", "asset_class_std", "source_id"]).reset_index(drop=True)

    if len(service_df) > 0 and positive_service_count == 0:
        raise ValueError(
            "SIPG service-entry pool derivation produced only zeros. "
            "This likely means the finalized anchor file uses a different transfer-to-PPE / service-entry "
            "column name than the alias map currently recognizes. Inspect k_entity_year_anchors.tsv and "
            "expand ANCHOR_ALIASES['service_entry'] if needed."
        )

    return pools, service_df


def build_working_dep_lookup(dep: pd.DataFrame, pools: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    dep = dep.copy()
    used_classes = set()

    for df in [pools, events]:
        if df.empty:
            continue
        used = df.loc[df["asset_class_std"].astype(str).str.strip() != "", ["entity", "asset_class_std"]]
        used_classes.update({(r.entity, r.asset_class_std) for r in used.itertuples(index=False)})

    dep_pairs = {(r.entity, r.asset_class_std) for r in dep[["entity", "asset_class_std"]].itertuples(index=False)}
    missing = sorted(used_classes - dep_pairs)
    if missing:
        pretty = "\n".join([f"  {e} / {c}" for e, c in missing])
        raise ValueError(
            "Working depreciation lookup is missing direct rows for productive classes used by pools or events:\n"
            + pretty
        )

    return dep.sort_values(["entity", "asset_class_std", "source_id"]).reset_index(drop=True)


def validate_share_table(shares: pd.DataFrame) -> List[Dict[str, object]]:
    bad = []
    g = shares.groupby(["entity", "year", "pool_type"], dropna=False)["share"].sum().reset_index()
    for _, row in g.iterrows():
        total = float(row["share"])
        if not np.isclose(total, 1.0, atol=1e-8):
            bad.append({
                "entity": row["entity"],
                "year": int(row["year"]),
                "pool_type": row["pool_type"],
                "share_sum": total,
            })
    return bad




def reclassify_non_additive_project_events(events_keep: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reclassify specific dated-event rows into a non-additive milestone table.

    Why this is done
    ----------------
    The monthly engine is built around the decomposition:

        annual total = dated flows + mapped-undated pools + background pools

    with annual investment totals treated as authoritative. That means a dated event should only
    remain in the additive monthly ledger if we are comfortable interpreting it as a genuinely
    booked capital flow that belongs on top of the annual residual pools.

    For the Kishon West cranes row, that interpretation is not appropriate. The row is useful as
    project timing evidence, but the 100,000 amount behaves like a project-level headline amount
    for a broader multi-stage program rather than a clean one-month booked addition. If we leave
    it in the additive dated-events ledger, interpolation_01 will try to add it on top of annual
    pools that already exhaust the authoritative annual total, producing an accounting contradiction.

    So this function moves those rows out of the additive ledger and into a milestone table.
    The milestone table is preserved for narrative, diagnostics, and future alternative designs.
    """

    if events_keep.empty:
        empty_events = pd.DataFrame(columns=events_keep.columns)
        milestone_cols = list(events_keep.columns) + [
            "milestone_reason_code",
            "milestone_interpretation",
            "milestone_policy",
        ]
        empty_milestones = pd.DataFrame(columns=milestone_cols)
        return empty_events, empty_milestones

    events = events_keep.copy()

    milestone_mask = events["project_id"].astype(str).isin(NON_ADDITIVE_MILESTONE_PROJECT_IDS)

    milestones = events.loc[milestone_mask].copy()
    working_events = events.loc[~milestone_mask].copy()

    if not milestones.empty:
        milestones["milestone_reason_code"] = "non_additive_project_headline_amount"
        milestones["milestone_interpretation"] = (
            "Timing evidence preserved, but excluded from additive monthly flow ledger because "
            "the amount is treated as a broader multi-stage project headline amount rather than "
            "a clean single-month booked service-entry flow."
        )
        milestones["milestone_policy"] = (
            "Preserve in milestone table; let annual pools and stock anchors carry the investment mass."
        )

    milestone_cols = list(events_keep.columns) + [
        "milestone_reason_code",
        "milestone_interpretation",
        "milestone_policy",
    ]
    milestones = milestones.reindex(columns=milestone_cols)

    return working_events.reset_index(drop=True), milestones.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = require_inputs(input_dir)

    raw_anchors = read_tsv(paths["anchors"])
    raw_dep = read_tsv(paths["dep"])
    raw_events = read_tsv(paths["events"])
    raw_pools = read_tsv(paths["pools"])
    raw_rules = read_tsv(paths["rules"])
    raw_shares = read_tsv(paths["shares"])

    anchors = prepare_anchors(raw_anchors)
    dep = prepare_dep(raw_dep)
    events = prepare_events(raw_events)
    pools = prepare_pools(raw_pools)
    _rules = prepare_rules(raw_rules)
    shares = prepare_shares(raw_shares)

    share_failures = validate_share_table(shares)
    if share_failures:
        raise ValueError(
            "Month-share table failed validation before preprocessing. Non-unit sums:\n"
            + "\n".join([str(x) for x in share_failures[:20]])
        )

    pools_keep, pools_excl = productive_filter_pools(pools)
    events_keep_raw, events_excl = productive_filter_events(events)

    # Reclassify known problematic project-headline dated events into a separate milestone table.
    # This is an economic interpretation decision, not a mechanical parsing step, which is why it
    # happens here in interpolation_00 rather than later in the recursion.
    working_events, project_milestones = reclassify_non_additive_project_events(events_keep_raw)

    anchors = adjust_anchors_to_productive_scope(anchors, pools_excl)
    sipg_cip_memo = build_sipg_cip_memo(anchors)

    # Keep a single exclusions file for all rows that are removed from the additive engine inputs.
    # We append milestone reclassifications here with a distinct reason code so the final working
    # dated-events ledger is fully auditable.
    milestone_excl = project_milestones.copy()
    if not milestone_excl.empty:
        milestone_excl["exclusion_reason"] = "reclassified_non_additive_milestone"
        milestone_excl["source_table"] = "dated_events"

    excluded_rows = pd.concat([pools_excl, events_excl, milestone_excl], ignore_index=True, sort=False)
    if excluded_rows.empty:
        excluded_rows = pd.DataFrame(columns=["source_table", "exclusion_reason"])

    working_pools, service_entry_df = build_sipg_service_entry_pools(anchors, pools_keep)
    working_dep = build_working_dep_lookup(dep, working_pools, working_events)

    prepared_anchors = anchors.sort_values(["entity", "year"]).reset_index(drop=True)

    write_tsv(excluded_rows, output_dir / "interpolation_00_excluded_rows.tsv")
    write_tsv(prepared_anchors, output_dir / "interpolation_00_prepared_anchors.tsv")
    write_tsv(working_pools, output_dir / "interpolation_00_working_annual_pools.tsv")
    write_tsv(working_dep, output_dir / "interpolation_00_working_dep_lookup.tsv")
    write_tsv(service_entry_df, output_dir / "interpolation_00_working_service_entry_pools.tsv")
    write_tsv(working_events, output_dir / "interpolation_00_working_dated_events.tsv")
    write_tsv(project_milestones, output_dir / "interpolation_00_project_milestones.tsv")
    write_tsv(sipg_cip_memo, output_dir / "interpolation_00_sipg_cip_memo.tsv")

    manifest = {
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "inputs_read": {k: str(v) for k, v in paths.items()},
        "rows": {
            "anchors_raw": int(len(raw_anchors)),
            "dep_raw": int(len(raw_dep)),
            "events_raw": int(len(raw_events)),
            "pools_raw": int(len(raw_pools)),
            "rules_raw": int(len(raw_rules)),
            "shares_raw": int(len(raw_shares)),
            "events_productive_kept_after_basic_filter": int(len(events_keep_raw)),
            "events_working_additive_kept": int(len(working_events)),
            "events_reclassified_to_milestones": int(len(project_milestones)),
            "pools_productive_kept": int(len(pools_keep)),
            "excluded_rows": int(len(excluded_rows)),
            "working_pools": int(len(working_pools)),
            "working_dep": int(len(working_dep)),
            "derived_sipg_service_rows": int(len(service_entry_df)),
            "derived_sipg_service_positive_rows": int((service_entry_df["annual_amount_kNIS"] > 0).sum()) if not service_entry_df.empty else 0,
            "sipg_cip_memo_rows": int(len(sipg_cip_memo)),
        },
        "decisions_applied": {
            "productive_capital_only": True,
            "right_of_use_excluded": True,
            "annual_totals_authoritative": True,
            "prepared_annual_totals_adjusted_to_productive_scope": True,
            "stock_anchors_left_raw_scope_unadjusted": False,
            "sipg_productive_annual_totals_defined_by_transfers_to_ppe": True,
            "sipg_stock_anchors_rebuilt_from_cumulative_transfers_when_informative": True,
            "sipg_service_entry_derived_from_anchors": True,
            "background_fallback_rates_deferred_to_interpolation_01": True,
            "non_additive_project_milestones_reclassified_in_00": True,
            "step01_smooth_bridge_strategy_expected": True,
        },
    }

    with open(output_dir / "interpolation_00_prepare_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"Wrote: {output_dir / 'interpolation_00_prepared_anchors.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_00_working_annual_pools.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_00_working_dep_lookup.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_00_working_service_entry_pools.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_00_working_dated_events.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_00_project_milestones.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_00_sipg_cip_memo.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_00_excluded_rows.tsv'}")
    print(f"Wrote: {output_dir / 'interpolation_00_prepare_manifest.json'}")


if __name__ == "__main__":
    main()

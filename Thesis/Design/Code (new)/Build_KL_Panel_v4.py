#!/usr/bin/env python3
"""Build_KL_Panel_v4.py

Construct a Model-2-safe monthly K/L panel for Haifa from the redesigned K pipeline.

Inputs
------
Labor:
  Data/L_proxy/L_Proxy.tsv

Capital (redesigned interpolation outputs):
  Data/K/Interpolation Output/interpolation_02_monthly_hpc.tsv
  Data/K/Interpolation Output/interpolation_02_monthly_sipg.tsv
  Data/K/Interpolation Output/interpolation_02_monthly_haifa_total.tsv

Outputs
-------
Canonical downstream panel:
  Data/KL/KL_Panel_monthly.tsv

Diagnostic panel:
  Data/KL/KL_Panel_monthly_diagnostic.tsv

Design choices
--------------
This version fixes two issues from earlier K/L builders:
1. It parses the redesigned K files, whose schema is typically
   [entity, month, K_productive_kNIS].
2. It writes the canonical downstream panel in a Model-2-safe way:
   exactly one row per entity-year-month for the three active Haifa entities:
     - Haifa--Legacy
     - Haifa--Bayport
     - Haifa port cluster

Series written to the canonical panel:
  - Haifa_Legacy_KL
      terminal-level: HPC K / Haifa-Legacy labor
  - Haifa_port_KL_SIPG_central
      terminal-level: SIPG K / Haifa-Bayport labor
  - Haifa_port_KL_cluster_central
      aggregate-level: Haifa total K / total Haifa labor
      clipped to 2021-08, consistent with the valid monthly aggregate window

The series_id labels intentionally preserve some legacy naming (especially SIPG/cluster)
for compatibility with downstream Model 1 / Model 2 normalization logic, while also
including an explicit canonical entity column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]

L_PROXY_PATH = THESIS_ROOT / "Data" / "L_proxy" / "L_Proxy.tsv"

K_OUT_DIR = THESIS_ROOT / "Data" / "K" / "Interpolation Output"
K_HPC_PATH = K_OUT_DIR / "interpolation_02_monthly_hpc.tsv"
K_SIPG_PATH = K_OUT_DIR / "interpolation_02_monthly_sipg.tsv"
K_CLUSTER_PATH = K_OUT_DIR / "interpolation_02_monthly_haifa_total.tsv"

KL_OUTPUT_DIR = THESIS_ROOT / "Data" / "KL"
KL_PANEL_PATH = KL_OUTPUT_DIR / "KL_Panel_monthly.tsv"
KL_DIAG_PATH = KL_OUTPUT_DIR / "KL_Panel_monthly_diagnostic.tsv"

CLUSTER_MAX_YM = 202108


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _find_col(columns: Iterable[str], candidates: list[str]) -> str | None:
    norm_map = {_norm(c): c for c in columns}
    for cand in candidates:
        hit = norm_map.get(_norm(cand))
        if hit is not None:
            return hit
    return None


def _require_col(columns: Iterable[str], candidates: list[str], *, label: str) -> str:
    hit = _find_col(columns, candidates)
    if hit is None:
        raise ValueError(f"{label}: none of the candidate columns {candidates} were found. Available columns: {list(columns)}")
    return hit


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_l_proxy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"L_Proxy.tsv not found at: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {"port", "terminal", "year", "month", "L_hours_i_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"L_Proxy.tsv is missing required columns: {missing}")

    df = df.copy()
    df["port"] = df["port"].astype(str).str.strip()
    df["terminal"] = df["terminal"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["L_hours_i_m"] = pd.to_numeric(df["L_hours_i_m"], errors="coerce")
    df = df.dropna(subset=["year", "month", "L_hours_i_m"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    return df


def load_single_k_series(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label}: K file not found at: {path}")

    df = pd.read_csv(path, sep="\t")
    month_col = _require_col(df.columns, ["month", "date", "period", "month_dt"], label=label)
    k_col = _require_col(
        df.columns,
        ["K_productive_kNIS", "K_productive", "K", "capital", "capital_stock"],
        label=label,
    )

    out = df.copy()
    dt = pd.to_datetime(out[month_col], errors="coerce")
    if dt.isna().all():
        raise ValueError(f"{label}: could not parse month column '{month_col}' as dates.")

    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["K"] = pd.to_numeric(out[k_col], errors="coerce")
    out = out.dropna(subset=["year", "month", "K"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)

    out = (
        out.groupby(["year", "month"], as_index=False)["K"]
        .sum()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def _finalize_kl(df: pd.DataFrame, *, series_id: str, entity: str, level: str, port: str, terminal: str | float, delta: str) -> pd.DataFrame:
    out = df.copy()

    good = (out["K"] > 0) & (out["L"] > 0)
    out = out.loc[good].copy()
    if out.empty:
        raise ValueError(f"{series_id}: no positive K/L rows remained after filtering.")

    out["KL"] = out["K"] / out["L"]
    out["log_KL"] = np.log(out["KL"])
    out["month_index"] = out["year"] * 100 + out["month"]
    out["series_id"] = series_id
    out["entity"] = entity
    out["level"] = level
    out["freq"] = "M"
    out["port"] = port
    out["terminal"] = terminal
    out["delta_scenario"] = delta

    cols = [
        "series_id", "entity", "level", "freq", "port", "terminal",
        "year", "month", "month_index", "delta_scenario", "K", "L", "KL", "log_KL",
    ]
    out = out[cols].sort_values(["series_id", "year", "month"]).reset_index(drop=True)
    return out


def build_terminal_kl(l_proxy: pd.DataFrame, k_df: pd.DataFrame, *, terminal_name: str, series_id: str, entity: str) -> pd.DataFrame:
    l_term = l_proxy[(l_proxy["port"] == "Haifa") & (l_proxy["terminal"] == terminal_name)].copy()
    if l_term.empty:
        raise ValueError(f"No labor rows found in L_Proxy for Haifa terminal '{terminal_name}'.")

    l_term = (
        l_term.groupby(["year", "month"], as_index=False)["L_hours_i_m"]
        .sum()
        .rename(columns={"L_hours_i_m": "L"})
    )

    merged = pd.merge(l_term, k_df, on=["year", "month"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"{series_id}: no overlapping months between labor and K series.")

    return _finalize_kl(
        merged,
        series_id=series_id,
        entity=entity,
        level="terminal",
        port="Haifa",
        terminal=terminal_name,
        delta="central",
    )


def build_cluster_kl(l_proxy: pd.DataFrame, k_df: pd.DataFrame) -> pd.DataFrame:
    l_port = (
        l_proxy[l_proxy["port"] == "Haifa"]
        .groupby(["year", "month"], as_index=False)["L_hours_i_m"]
        .sum()
        .rename(columns={"L_hours_i_m": "L"})
    )
    if l_port.empty:
        raise ValueError("No Haifa labor rows found in L_Proxy.")

    merged = pd.merge(l_port, k_df, on=["year", "month"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("Haifa cluster KL: no overlapping months between labor and K series.")

    merged["ym"] = merged["year"] * 100 + merged["month"]
    merged = merged[merged["ym"] <= CLUSTER_MAX_YM].copy()
    merged = merged.drop(columns=["ym"])
    if merged.empty:
        raise ValueError("Haifa cluster KL: no rows remained after applying CLUSTER_MAX_YM cutoff.")

    return _finalize_kl(
        merged,
        series_id="Haifa_port_KL_cluster_central",
        entity="Haifa port cluster",
        level="port",
        port="Haifa",
        terminal=np.nan,
        delta="central",
    )


# ---------------------------------------------------------------------
# QA / duplicate checks
# ---------------------------------------------------------------------

def assert_unique(df: pd.DataFrame, keys: list[str], *, label: str) -> None:
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys + [c for c in df.columns if c not in keys]].sort_values(keys)
        raise ValueError(
            f"{label}: duplicate rows found on keys {keys}. Sample:\n{sample.head(20).to_string(index=False)}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Build_KL_Panel_v4 (Haifa) ===")
    print(f"THESIS_ROOT           : {THESIS_ROOT}")
    print(f"Reading L_Proxy       : {L_PROXY_PATH}")
    print(f"Reading K HPC         : {K_HPC_PATH}")
    print(f"Reading K SIPG        : {K_SIPG_PATH}")
    print(f"Reading K Haifa total : {K_CLUSTER_PATH}")

    l_proxy = load_l_proxy(L_PROXY_PATH)
    k_hpc = load_single_k_series(K_HPC_PATH, label="HPC")
    k_sipg = load_single_k_series(K_SIPG_PATH, label="SIPG")
    k_cluster = load_single_k_series(K_CLUSTER_PATH, label="Haifa total")

    legacy = build_terminal_kl(
        l_proxy,
        k_hpc,
        terminal_name="Haifa-Legacy",
        series_id="Haifa_Legacy_KL",
        entity="Haifa--Legacy",
    )
    bayport = build_terminal_kl(
        l_proxy,
        k_sipg,
        terminal_name="Haifa-Bayport",
        series_id="Haifa_port_KL_SIPG_central",
        entity="Haifa--Bayport",
    )
    cluster = build_cluster_kl(l_proxy, k_cluster)

    canonical = pd.concat([legacy, bayport, cluster], ignore_index=True)
    canonical = canonical.sort_values(["entity", "year", "month"]).reset_index(drop=True)

    # Strict uniqueness for downstream safety
    assert_unique(canonical, ["series_id", "year", "month"], label="canonical K/L panel")
    assert_unique(canonical, ["entity", "year", "month"], label="Model-2 canonical K/L panel")

    # Diagnostic panel currently equals canonical by construction, but we still write it separately
    diagnostic = canonical.copy()

    KL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(KL_PANEL_PATH, sep="\t", index=False)
    diagnostic.to_csv(KL_DIAG_PATH, sep="\t", index=False)

    print(f"Wrote canonical KL panel     : {KL_PANEL_PATH}")
    print(f"Wrote diagnostic KL panel    : {KL_DIAG_PATH}")
    print(f"Rows written                 : {len(canonical)}")
    print("Entities written:")
    for ent in canonical["entity"].drop_duplicates().tolist():
        print(f"  - {ent}")


if __name__ == "__main__":
    main()

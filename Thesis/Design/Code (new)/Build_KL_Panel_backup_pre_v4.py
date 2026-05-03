#!/usr/bin/env python3
"""Build_KL_Panel_v2.py

Construct monthly Haifa K/L series using the *redesigned* K pipeline outputs from:

  Data/K/Interpolation Output/
    - interpolation_02_monthly_hpc.tsv
    - interpolation_02_monthly_ipc.tsv
    - interpolation_02_monthly_sipg.tsv
    - interpolation_02_monthly_haifa_total.tsv

and labor from:

  Data/L_proxy/L_Proxy.tsv

This version reads the redesigned K outputs directly and no longer depends on the old cluster file.

Output:
  Data/KL/KL_Panel_monthly.tsv

Series built:
1. Haifa_Legacy_KL                  (terminal-level, central K for HPC / legacy labor)
2. Haifa_port_KL_cluster_low
3. Haifa_port_KL_cluster_central
4. Haifa_port_KL_cluster_high
5. Haifa_port_KL_HPC_low / central / high
6. Haifa_port_KL_IPC_low / central / high
7. Haifa_port_KL_SIPG_low / central / high
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
K_IPC_PATH = K_OUT_DIR / "interpolation_02_monthly_ipc.tsv"
K_SIPG_PATH = K_OUT_DIR / "interpolation_02_monthly_sipg.tsv"
K_CLUSTER_PATH = K_OUT_DIR / "interpolation_02_monthly_haifa_total.tsv"

KL_OUTPUT_DIR = THESIS_ROOT / "Data" / "KL"
KL_PANEL_PATH = KL_OUTPUT_DIR / "KL_Panel_monthly.tsv"


# ---------------------------------------------------------------------
# Generic helpers
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


def _coerce_year_month(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    out = df.copy()

    year_col = _find_col(out.columns, ["year"])
    month_col = _find_col(out.columns, ["month", "month_num", "monthnumber"])
    month_index_col = _find_col(out.columns, ["month_index", "monthindex"])
    date_col = _find_col(
        out.columns,
        ["date", "month_dt", "month_date", "period_start", "period", "month_eom", "month_end"]
    )

    # If there is a month column that is actually date-like (as in interpolation_02_monthly_hpc.tsv),
    # treat it as the primary date source.
    month_is_datetime = False
    if month_col is not None and not pd.api.types.is_numeric_dtype(out[month_col]):
        parsed_month = pd.to_datetime(out[month_col], errors="coerce")
        if parsed_month.notna().any():
            month_is_datetime = True
            out["year"] = parsed_month.dt.year
            out["month"] = parsed_month.dt.month

    if not month_is_datetime:
        if year_col is None and date_col is None and month_index_col is None:
            raise ValueError(
                f"{label}: could not infer a year source. "
                f"Available columns: {list(out.columns)}"
            )

        if month_col is None:
            if date_col is not None:
                dt = pd.to_datetime(out[date_col], errors="coerce")
                if dt.notna().any():
                    out["year"] = dt.dt.year
                    out["month"] = dt.dt.month
                else:
                    raise ValueError(
                        f"{label}: found date-like column '{date_col}' but could not parse it."
                    )
            elif month_index_col is not None and year_col is not None:
                out["year"] = pd.to_numeric(out[year_col], errors="coerce")
                mi = pd.to_numeric(out[month_index_col], errors="coerce")
                out["month"] = (mi % 100).astype("Int64")
            elif month_index_col is not None:
                mi = pd.to_numeric(out[month_index_col], errors="coerce")
                out["year"] = (mi // 100).astype("Int64")
                out["month"] = (mi % 100).astype("Int64")
            else:
                raise ValueError(
                    f"{label}: could not infer month column. Available columns: {list(out.columns)}"
                )
        else:
            if year_col is not None:
                out["year"] = pd.to_numeric(out[year_col], errors="coerce")
            elif date_col is not None:
                dt = pd.to_datetime(out[date_col], errors="coerce")
                out["year"] = dt.dt.year
            else:
                raise ValueError(f"{label}: month was found but no usable year source exists.")

            month_raw = out[month_col]
            if pd.api.types.is_numeric_dtype(month_raw):
                out["month"] = pd.to_numeric(month_raw, errors="coerce")
            else:
                dt = pd.to_datetime(month_raw, errors="coerce")
                if dt.notna().any():
                    out["month"] = dt.dt.month
                    if year_col is None:
                        out["year"] = dt.dt.year
                else:
                    out["month"] = pd.to_numeric(month_raw, errors="coerce")

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")

    bad = out["year"].isna() | out["month"].isna()
    if bad.any():
        out = out.loc[~bad].copy()

    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)

    out = out[(out["month"] >= 1) & (out["month"] <= 12)].copy()

    if out.empty:
        raise ValueError(f"{label}: no valid year-month rows remained after parsing.")

    return out


def _extract_k_wide(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """
    Return standardized columns:
      year, month, K_low, K_central, K_high

    Supports any of:
      1) wide files with columns like K_low / K_central / K_high
      2) long files with scenario + K columns that can be pivoted
      3) single-series files like interpolation_02_monthly_hpc.tsv with
         columns ['entity', 'month', 'K_productive_kNIS'], in which case
         the same K path is copied into low/central/high for compatibility
         with downstream code that still expects scenario-specific series IDs.
    """
    out = _coerce_year_month(df, label=label)

    low_col = _find_col(out.columns, ["K_low", "k_low", "low"])
    cen_col = _find_col(out.columns, ["K_central", "k_central", "central", "K_mid", "k_mid", "mid"])
    high_col = _find_col(out.columns, ["K_high", "k_high", "high"])

    if all(c is not None for c in [low_col, cen_col, high_col]):
        keep = out[["year", "month", low_col, cen_col, high_col]].copy()
        keep.columns = ["year", "month", "K_low", "K_central", "K_high"]
    else:
        # First try the single productive-capital path written by interpolation_02 finalizer
        k_single_col = _find_col(
            out.columns,
            ["K_productive_kNIS", "k_productive_knis", "K_productive", "capital_productive", "K"]
        )
        if k_single_col is not None and _find_col(out.columns, ["scenario", "delta_scenario", "case"]) is None:
            keep = out[["year", "month", k_single_col]].copy()
            keep.columns = ["year", "month", "K_central"]
            keep["K_low"] = keep["K_central"]
            keep["K_high"] = keep["K_central"]
            keep = keep[["year", "month", "K_low", "K_central", "K_high"]]
        else:
            scenario_col = _find_col(out.columns, ["scenario", "delta_scenario", "case"])
            k_col = _find_col(out.columns, ["K", "k", "capital", "capital_stock", "K_productive_kNIS"])
            if scenario_col is None or k_col is None:
                raise ValueError(
                    f"{label}: could not find either wide K columns, a single productive K column, "
                    f"or a long scenario+K schema. Available columns: {list(out.columns)}"
                )

            tmp = out[["year", "month", scenario_col, k_col]].copy()
            tmp[scenario_col] = tmp[scenario_col].astype(str).str.strip().str.lower()
            tmp[scenario_col] = tmp[scenario_col].replace(
                {
                    "mid": "central",
                    "base": "central",
                    "baseline": "central",
                }
            )
            wide = (
                tmp.pivot_table(
                    index=["year", "month"],
                    columns=scenario_col,
                    values=k_col,
                    aggfunc="sum",
                )
                .reset_index()
            )
            need = ["low", "central", "high"]
            missing = [c for c in need if c not in wide.columns]
            if missing:
                raise ValueError(
                    f"{label}: long-form schema found, but missing scenarios {missing}. "
                    f"Available pivoted columns: {list(wide.columns)}"
                )
            keep = wide[["year", "month", "low", "central", "high"]].copy()
            keep.columns = ["year", "month", "K_low", "K_central", "K_high"]

    keep["K_low"] = pd.to_numeric(keep["K_low"], errors="coerce")
    keep["K_central"] = pd.to_numeric(keep["K_central"], errors="coerce")
    keep["K_high"] = pd.to_numeric(keep["K_high"], errors="coerce")

    keep = (
        keep.groupby(["year", "month"], as_index=False)[["K_low", "K_central", "K_high"]]
        .sum()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    return keep


# ---------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------

def load_l_proxy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"L_Proxy.tsv not found at: {path}")

    df = pd.read_csv(path, sep="\t")

    required_cols = {"port", "terminal", "year", "month", "L_hours_i_m"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"L_Proxy.tsv is missing required columns: {missing}")

    df["port"] = df["port"].astype(str).str.strip()
    df["terminal"] = df["terminal"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "month"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    return df


def load_k_series(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label}: K file not found at: {path}")

    df = pd.read_csv(path, sep="\t")
    out = _extract_k_wide(df, label=label)
    return out


# ---------------------------------------------------------------------
# K/L builders
# ---------------------------------------------------------------------

def build_haifa_legacy_kl(l_proxy: pd.DataFrame, k_hpc: pd.DataFrame) -> pd.DataFrame:
    l_hpc = l_proxy[
        (l_proxy["port"] == "Haifa") &
        (l_proxy["terminal"] == "Haifa-Legacy")
    ].copy()

    if l_hpc.empty:
        raise ValueError(
            'No rows in L_Proxy for port=="Haifa" & terminal=="Haifa-Legacy".'
        )

    merged = pd.merge(
        l_hpc,
        k_hpc[["year", "month", "K_central"]],
        on=["year", "month"],
        how="inner",
        validate="many_to_one",
    )

    if merged.empty:
        raise ValueError(
            "Merged Haifa-Legacy K/L is empty. "
            "There may be no overlapping months between L_Proxy and K."
        )

    merged = (
        merged.groupby(["port", "terminal", "year", "month"], as_index=False)[["L_hours_i_m", "K_central"]]
        .sum()
    )
    merged = merged.rename(columns={"L_hours_i_m": "L", "K_central": "K"})

    good = (merged["K"] > 0) & (merged["L"] > 0)
    merged = merged.loc[good].copy()

    merged["KL"] = merged["K"] / merged["L"]
    merged["log_KL"] = np.log(merged["KL"])
    merged["month_index"] = merged["year"] * 12 + merged["month"]
    merged["series_id"] = "Haifa_Legacy_KL"
    merged["level"] = "terminal"
    merged["freq"] = "M"

    out_cols = [
        "series_id", "level", "freq", "port", "terminal",
        "year", "month", "month_index", "K", "L", "KL", "log_KL"
    ]
    return merged[out_cols].sort_values(["series_id", "year", "month"]).reset_index(drop=True)


def _build_haifa_port_l_series(l_proxy: pd.DataFrame) -> pd.DataFrame:
    l_haifa_port = (
        l_proxy[l_proxy["port"] == "Haifa"]
        .groupby(["year", "month"], as_index=False)["L_hours_i_m"]
        .sum()
        .rename(columns={"L_hours_i_m": "L"})
    )

    if l_haifa_port.empty:
        raise ValueError("No Haifa labor rows found in L_Proxy.")

    return l_haifa_port


def _build_port_scenario_series(
    l_port: pd.DataFrame,
    k_df: pd.DataFrame,
    *,
    series_prefix: str,
) -> pd.DataFrame:
    merged = pd.merge(
        l_port,
        k_df[["year", "month", "K_low", "K_central", "K_high"]],
        on=["year", "month"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(f"{series_prefix}: merged K/L is empty.")

    scenario_map = {
        "low": f"{series_prefix}_low",
        "central": f"{series_prefix}_central",
        "high": f"{series_prefix}_high",
    }

    out_list = []
    for scen, series_id in scenario_map.items():
        tmp = merged.copy()
        tmp["K"] = tmp[f"K_{scen}"]

        good = (tmp["K"] > 0) & (tmp["L"] > 0)
        tmp = tmp.loc[good].copy()

        tmp["KL"] = tmp["K"] / tmp["L"]
        tmp["log_KL"] = np.log(tmp["KL"])
        tmp["month_index"] = tmp["year"] * 12 + tmp["month"]
        tmp["series_id"] = series_id
        tmp["level"] = "port"
        tmp["freq"] = "M"
        tmp["port"] = "Haifa"
        tmp["terminal"] = np.nan

        out_cols = [
            "series_id", "level", "freq", "port", "terminal",
            "year", "month", "month_index", "K", "L", "KL", "log_KL"
        ]
        out_list.append(
            tmp[out_cols].sort_values(["series_id", "year", "month"]).reset_index(drop=True)
        )

    return pd.concat(out_list, ignore_index=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Build_KL_Panel_v2 (Haifa) ===")
    print(f"THESIS_ROOT           : {THESIS_ROOT}")
    print(f"Reading L_Proxy       : {L_PROXY_PATH}")
    print(f"Reading K HPC         : {K_HPC_PATH}")
    print(f"Reading K IPC         : {K_IPC_PATH}")
    print(f"Reading K SIPG        : {K_SIPG_PATH}")
    print(f"Reading K Haifa total : {K_CLUSTER_PATH}")

    l_proxy = load_l_proxy(L_PROXY_PATH)
    k_hpc = load_k_series(K_HPC_PATH, label="HPC")
    k_ipc = load_k_series(K_IPC_PATH, label="IPC")
    k_sipg = load_k_series(K_SIPG_PATH, label="SIPG")
    k_cluster = load_k_series(K_CLUSTER_PATH, label="Haifa total")

    l_haifa_port = _build_haifa_port_l_series(l_proxy)

    haifa_legacy_kl = build_haifa_legacy_kl(l_proxy, k_hpc)
    haifa_cluster_kl_all = _build_port_scenario_series(
        l_haifa_port, k_cluster, series_prefix="Haifa_port_KL_cluster"
    )
    haifa_hpc_kl_all = _build_port_scenario_series(
        l_haifa_port, k_hpc, series_prefix="Haifa_port_KL_HPC"
    )
    haifa_ipc_kl_all = _build_port_scenario_series(
        l_haifa_port, k_ipc, series_prefix="Haifa_port_KL_IPC"
    )
    haifa_sipg_kl_all = _build_port_scenario_series(
        l_haifa_port, k_sipg, series_prefix="Haifa_port_KL_SIPG"
    )

    kl_panel = pd.concat(
        [
            haifa_legacy_kl,
            haifa_cluster_kl_all,
            haifa_hpc_kl_all,
            haifa_ipc_kl_all,
            haifa_sipg_kl_all,
        ],
        ignore_index=True,
    ).sort_values(["series_id", "year", "month"]).reset_index(drop=True)

    KL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kl_panel.to_csv(KL_PANEL_PATH, sep="\t", index=False)

    print(
        f"Wrote KL panel with {len(kl_panel)} rows and "
        f"{kl_panel['series_id'].nunique()} series_ids to: {KL_PANEL_PATH}"
    )


if __name__ == "__main__":
    main()

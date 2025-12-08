#!/usr/bin/env python3
"""
Build_KL_Panel.py

Constructs monthly K/L time series for Haifa using:
  - L_Proxy.tsv (terminal×month labor proxy),
  - 21_K_cluster_Haifa_HPC_IPC_SIPG_monthly.tsv (Haifa K tracks with SIPG).

Outputs a new panel:
  Data/KL/KL_Panel_monthly.tsv

Series built:

1. Haifa_Legacy_KL (terminal-level, central δ only)
   - port  = "Haifa"
   - terminal = "Haifa-Legacy"
   - level = "terminal"
   - K = K_central for company == "Haifa Port Company (legacy)"
   - L = L_hours_i_m from L_Proxy for that terminal
   - KL = K / L
   - log_KL = ln(KL)

2. Haifa_port_KL_cluster_low  (port-level, cluster K/L, low δ)
3. Haifa_port_KL_cluster_central  (port-level, cluster K/L, central δ)
4. Haifa_port_KL_cluster_high (port-level, cluster K/L, high δ)

   For all three cluster series:
   - port  = "Haifa"
   - terminal = NaN   (port-level series)
   - level = "port"
   - K = K_{scenario} for company == "Haifa cluster (HPC + IPC + SIPG)"
       (this is the cluster K from file 21, already HPC+IPC+SIPG)
   - L = sum of L_hours_i_m across all Haifa terminals each month
   - KL = K / L
   - log_KL = ln(KL)

This code is read-only with respect to inputs: it does not mutate any
existing files and only writes a new TSV in Data/KL/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths and basic configuration
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# Assumes this file lives in: THESIS_ROOT/Design/Code (new)/...
THESIS_ROOT = THIS_FILE.parents[2]

L_PROXY_PATH = THESIS_ROOT / "Data" / "L_Proxy" / "L_Proxy.tsv"
# NOTE the space in "K " matches your existing folder name
K_CLUSTER_PATH = THESIS_ROOT / "Data" / "K " / "21_K_cluster_Haifa_HPC_IPC_SIPG_monthly.tsv"

KL_OUTPUT_DIR = THESIS_ROOT / "Data" / "KL"
KL_PANEL_PATH = KL_OUTPUT_DIR / "KL_Panel_monthly.tsv"

# Company name constants (must match 21_K_cluster_... file)
HPC_COMPANY_NAME = "Haifa Port Company (legacy)"
CLUSTER_COMPANY_NAME = "Haifa cluster (HPC + IPC + SIPG)"


# ---------------------------------------------------------------------
# Helpers to load inputs
# ---------------------------------------------------------------------

def load_l_proxy(path: Path) -> pd.DataFrame:
    """
    Load terminal×month labor proxy from L_Proxy.tsv.

    Required columns:
      - port, terminal, year, month, L_hours_i_m

    We do NOT modify this file; we just read and standardize types.
    """
    if not path.exists():
        raise FileNotFoundError(f"L_Proxy file not found at: {path}")

    df = pd.read_csv(path, sep="\t")

    required_cols = {"port", "terminal", "year", "month", "L_hours_i_m"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"L_Proxy is missing required columns: {missing}")

    # Coerce year/month to integers
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype(int)

    # Quick duplicate check: there should be at most one row per (port, terminal, year, month)
    dup_mask = df.duplicated(subset=["port", "terminal", "year", "month"])
    if dup_mask.any():
        dup_count = int(dup_mask.sum())
        print(
            f"[WARN] L_Proxy has {dup_count} duplicated rows on "
            "(port,terminal,year,month). Keeping them, but you may want to inspect."
        )

    return df


def load_k_cluster(path: Path) -> pd.DataFrame:
    """
    Load Haifa K tracks from 21_K_cluster_Haifa_HPC_IPC_SIPG_monthly.tsv.

    Required columns:
      - port, company, year, month, K_low, K_central, K_high

    The 'month' column in the K file is a date-like string ("YYYY-MM-01").
    We parse it to recover integer month and keep the original column if needed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Haifa K cluster file not found at: {path}")

    df = pd.read_csv(path, sep="\t")

    required_cols = {"port", "company", "year", "month", "K_low", "K_central", "K_high"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"K cluster file is missing required columns: {missing}")

    # Standardize types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["month_dt"] = pd.to_datetime(df["month"])
    df["month"] = df["month_dt"].dt.month

    return df


# ---------------------------------------------------------------------
# K/L builders
# ---------------------------------------------------------------------

def build_haifa_legacy_kl(l_proxy: pd.DataFrame, k_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Build terminal-level K/L for Haifa-Legacy using:

      K: company == "Haifa Port Company (legacy)", K_central
      L: L_hours_i_m for port == "Haifa", terminal == "Haifa-Legacy"

    Output columns:
      series_id, level, freq, port, terminal, year, month, month_index,
      K, L, KL, log_KL
    """
    # Filter K for Haifa Port Company (legacy)
    k_hpc = k_cluster[k_cluster["company"] == HPC_COMPANY_NAME].copy()
    if k_hpc.empty:
        raise ValueError(
            f"No rows in K cluster file for company == '{HPC_COMPANY_NAME}'. "
            "Check that the file path and company labels match the K pipeline."
        )

    # Filter L_proxy for Haifa-Legacy terminal
    l_hpc = l_proxy[
        (l_proxy["port"] == "Haifa") &
        (l_proxy["terminal"] == "Haifa-Legacy")
    ].copy()

    if l_hpc.empty:
        raise ValueError(
            'No rows in L_Proxy for port=="Haifa" & terminal=="Haifa-Legacy". '
            "Check that L_Proxy.tsv is the same one used in the LP pipeline."
        )

    # Merge on (port, year, month)
    merged = pd.merge(
        l_hpc,
        k_hpc[["port", "year", "month", "K_central"]],
        on=["port", "year", "month"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "Merged Haifa-Legacy K/L is empty. "
            "There may be no overlapping months between L_Proxy and the K cluster file."
        )

    # Rename and compute K/L
    merged = merged.rename(columns={"L_hours_i_m": "L", "K_central": "K"})
    # Drop any rows where K<=0 or L<=0 (just in case)
    good = (merged["K"] > 0) & (merged["L"] > 0)
    if not good.all():
        dropped = int((~good).sum())
        print(
            f"[WARN] Dropping {dropped} Haifa-Legacy rows with non-positive K or L "
            "before computing log(K/L)."
        )
        merged = merged.loc[good].copy()

    merged["KL"] = merged["K"] / merged["L"]
    merged["log_KL"] = np.log(merged["KL"])

    # Compute month_index in the same way as LP_Panel_monthly
    merged["month_index"] = merged["year"] * 12 + merged["month"]

    # Attach series-level metadata
    merged["series_id"] = "Haifa_Legacy_KL"
    merged["level"] = "terminal"
    merged["freq"] = "M"

    out_cols = [
        "series_id",
        "level",
        "freq",
        "port",
        "terminal",
        "year",
        "month",
        "month_index",
        "K",
        "L",
        "KL",
        "log_KL",
    ]
    out = merged[out_cols].copy().sort_values(["series_id", "year", "month"])
    out.reset_index(drop=True, inplace=True)

    return out


def build_haifa_cluster_kl_scenarios(
    l_proxy: pd.DataFrame,
    k_cluster: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build three port-level (cluster) K/L series for Haifa, one for each δ-scenario:

      - Haifa_port_KL_cluster_low
      - Haifa_port_KL_cluster_central
      - Haifa_port_KL_cluster_high

    Using:

      K: company == "Haifa cluster (HPC + IPC + SIPG)", K_low / K_central / K_high
      L: sum of L_hours_i_m across all Haifa terminals each month

    Each series has:
      series_id, level, freq, port, terminal, year, month, month_index,
      K, L, KL, log_KL
    """
    # Filter to the cluster rows
    k_clust = k_cluster[k_cluster["company"] == CLUSTER_COMPANY_NAME].copy()
    if k_clust.empty:
        raise ValueError(
            f"No rows in K cluster file for company == '{CLUSTER_COMPANY_NAME}'. "
            "Check that the file path and company labels match the K pipeline."
        )

    # Aggregate L_proxy to the port-month level for Haifa
    l_haifa_port = (
        l_proxy[l_proxy["port"] == "Haifa"]
        .groupby(["port", "year", "month"], as_index=False)["L_hours_i_m"]
        .sum()
    )

    if l_haifa_port.empty:
        raise ValueError(
            "No L_proxy rows found for port=='Haifa'. "
            "Check that L_Proxy.tsv contains Haifa terminals."
        )

    # Merge port-level L with cluster K (all scenarios in one df)
    merged = pd.merge(
        l_haifa_port,
        k_clust[["port", "year", "month", "K_low", "K_central", "K_high"]],
        on=["port", "year", "month"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "Merged Haifa cluster K/L is empty. "
            "There may be no overlapping months between L_Proxy and the K cluster file."
        )

    merged = merged.rename(columns={"L_hours_i_m": "L"})

    # Build one output series per scenario
    scenario_map = {
        "low": "Haifa_port_KL_cluster_low",
        "central": "Haifa_port_KL_cluster_central",
        "high": "Haifa_port_KL_cluster_high",
    }

    out_list = []

    for scen, series_id in scenario_map.items():
        tmp = merged.copy()
        tmp["K"] = tmp[f"K_{scen}"]

        good = (tmp["K"] > 0) & (tmp["L"] > 0)
        if not good.all():
            dropped = int((~good).sum())
            print(
                f"[WARN] Dropping {dropped} Haifa-cluster ({scen}) rows with non-positive "
                "K or L before computing log(K/L)."
            )
            tmp = tmp.loc[good].copy()

        tmp["KL"] = tmp["K"] / tmp["L"]
        tmp["log_KL"] = np.log(tmp["KL"])

        tmp["month_index"] = tmp["year"] * 12 + tmp["month"]

        tmp["series_id"] = series_id
        tmp["level"] = "port"
        tmp["freq"] = "M"
        tmp["terminal"] = np.nan  # consistent with LP_Panel_monthly port-level rows

        out_cols = [
            "series_id",
            "level",
            "freq",
            "port",
            "terminal",
            "year",
            "month",
            "month_index",
            "K",
            "L",
            "KL",
            "log_KL",
        ]
        tmp = tmp[out_cols].copy().sort_values(["series_id", "year", "month"])
        tmp.reset_index(drop=True, inplace=True)

        out_list.append(tmp)

    out = pd.concat(out_list, ignore_index=True)
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Build_KL_Panel (Haifa) ===")
    print(f"THESIS_ROOT          : {THESIS_ROOT}")
    print(f"Reading L_Proxy      : {L_PROXY_PATH}")
    print(f"Reading Haifa K file : {K_CLUSTER_PATH}")

    l_proxy = load_l_proxy(L_PROXY_PATH)
    k_cluster = load_k_cluster(K_CLUSTER_PATH)

    print(f"Loaded L_Proxy rows  : {len(l_proxy):5d}")
    print(f"Loaded K rows        : {len(k_cluster):5d}")

    haifa_legacy_kl = build_haifa_legacy_kl(l_proxy, k_cluster)
    haifa_cluster_kl_all = build_haifa_cluster_kl_scenarios(l_proxy, k_cluster)

    print(
        f"Built Haifa-Legacy KL series with {len(haifa_legacy_kl)} "
        "terminal×month rows."
    )
    print(
        "Built Haifa cluster KL series (low/central/high) with "
        f"{len(haifa_cluster_kl_all)} port×month rows total "
        f"({haifa_cluster_kl_all['series_id'].nunique()} series)."
    )

    kl_panel = pd.concat(
        [haifa_legacy_kl, haifa_cluster_kl_all],
        ignore_index=True,
    ).sort_values(["series_id", "year", "month"])

    kl_panel.reset_index(drop=True, inplace=True)

    KL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kl_panel.to_csv(KL_PANEL_PATH, sep="\t", index=False)

    print("------------------------------------------------------------")
    print(f"Wrote K/L panel with {len(kl_panel)} rows to:")
    print(f"  {KL_PANEL_PATH}")
    print("Series breakdown (min/max year-month):")
    summary = (
        kl_panel.groupby("series_id")[["year", "month"]]
        .agg(["min", "max"])
        .sort_index()
    )
    print(summary)
    print("=== Build_KL_Panel: done ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

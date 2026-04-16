#!/usr/bin/env python3
"""Build_KL_Panel.py

Construct monthly K/L time series for Haifa using:

  - Data/L_Proxy/L_Proxy.tsv
      (terminal×month labor proxy with columns including:
       port, terminal, year, month, L_hours_i_m)
  - Data/K /21_K_cluster_Haifa_HPC_IPC_SIPG_monthly.tsv
      (monthly K tracks for:
         - Haifa Port Company (legacy)
         - Israel Ports Company (IPC)
         - SIPG Haifa Bayport
         - Haifa cluster (HPC + IPC + SIPG))

Output:

  Data/KL/KL_Panel_monthly.tsv

Series built:

1. Haifa_Legacy_KL  (terminal-level, central δ only)

   - port     = "Haifa"
   - terminal = "Haifa-Legacy"
   - level    = "terminal"
   - K        = K_central for company == "Haifa Port Company (legacy)"
   - L        = L_hours_i_m from L_Proxy for that terminal
   - KL       = K / L
   - log_KL   = ln(KL)

2. Haifa_port_KL_cluster_low     (port-level, cluster K/L, low δ)
3. Haifa_port_KL_cluster_central (port-level, cluster K/L, central δ)
4. Haifa_port_KL_cluster_high    (port-level, cluster K/L, high δ)

   - port     = "Haifa"
   - terminal = NaN   (port-level series)
   - level    = "port"
   - K        = K_low / K_central / K_high for company
                == "Haifa cluster (HPC + IPC + SIPG)"
   - L        = sum of L_hours_i_m across all Haifa terminals each month
   - KL       = K / L
   - log_KL   = ln(KL)

5. Entity-decomposed port-level K/L series for each δ scenario:

   - Haifa_port_KL_HPC_low,   Haifa_port_KL_HPC_central,   Haifa_port_KL_HPC_high
   - Haifa_port_KL_IPC_low,   Haifa_port_KL_IPC_central,   Haifa_port_KL_IPC_high
   - Haifa_port_KL_SIPG_low,  Haifa_port_KL_SIPG_central,  Haifa_port_KL_SIPG_high

   For all of these:

   - port     = "Haifa"
   - terminal = NaN   (port-level series)
   - level    = "port"
   - K        = K_low / K_central / K_high for the given entity
   - L        = the same Haifa port labor series used for the cluster:
                sum of L_hours_i_m across all Haifa terminals each month
   - KL       = K / L
   - log_KL   = ln(KL)

   By construction, for each scenario s ∈ {low, central, high} and month t:

       Haifa_port_KL_cluster_s(t)
         = Haifa_port_KL_HPC_s(t)
         + Haifa_port_KL_IPC_s(t)
         + Haifa_port_KL_SIPG_s(t)

This script is read-only with respect to inputs: it does not mutate any
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
IPC_COMPANY_NAME = "Israel Ports Company (Haifa cluster)"
SIPG_COMPANY_NAME = "SIPG Haifa Bayport"



# ---------------------------------------------------------------------
# Helpers to load inputs
# ---------------------------------------------------------------------

def load_l_proxy(path: Path) -> pd.DataFrame:
    """Load terminal×month labor proxy from L_Proxy.tsv.

    Required columns:
      - port, terminal, year, month, L_hours_i_m
    """
    if not path.exists():
        raise FileNotFoundError(f"L_Proxy.tsv not found at: {path}")

    df = pd.read_csv(path, sep="\t")

    required_cols = {"port", "terminal", "year", "month", "L_hours_i_m"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"L_Proxy.tsv is missing required columns: {missing}")

    # Standardize types we care about
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype(int)

    return df


def load_k_cluster(path: Path) -> pd.DataFrame:
    """Load Haifa K tracks from 21_K_cluster_Haifa_HPC_IPC_SIPG_monthly.tsv.

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


def _select_entity_company_rows(k_cluster: pd.DataFrame, entity_label: str) -> pd.DataFrame:
    """
    Robustly select rows from k_cluster for a given entity (HPC, IPC, SIPG)
    based on substring patterns in the 'company' column, to avoid brittle
    exact-string matches.
    """
    company_col = k_cluster["company"].astype(str)

    if entity_label == "HPC":
        patterns = ["Haifa Port Company", "HPC"]
    elif entity_label == "IPC":
        patterns = ["Israel Ports", "IPC"]
    elif entity_label == "SIPG":
        patterns = ["SIPG", "Bayport"]
    else:
        raise ValueError(f"Unknown entity_label '{entity_label}'")

    mask = False
    for pat in patterns:
        mask = mask | company_col.str.contains(pat, case=False, na=False)

    subset = k_cluster[mask].copy()

    if subset.empty:
        # Helpful debug: show what company names DO exist
        unique_companies = sorted(k_cluster["company"].astype(str).unique())
        raise ValueError(
            f"No K rows matched patterns {patterns} for entity '{entity_label}'. "
            f"Available company labels in K file are:\n{unique_companies}"
        )

    return subset




def build_haifa_legacy_kl(l_proxy: pd.DataFrame, k_cluster: pd.DataFrame) -> pd.DataFrame:
    """Build terminal-level K/L for Haifa-Legacy using:

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

    # Rename K and L columns
    merged = merged.rename(columns={"L_hours_i_m": "L", "K_central": "K"})

    # Drop any rows with non-positive K or L before logging
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


def _build_haifa_port_l_series(l_proxy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate L_Proxy to a port-level labor series for Haifa.

    Returns a DataFrame with columns:
      port, year, month, L
    where L is the sum of L_hours_i_m across all Haifa terminals.
    """
    l_haifa_port = (
        l_proxy[l_proxy["port"] == "Haifa"]
        .groupby(["port", "year", "month"], as_index=False)["L_hours_i_m"]
        .sum()
    )

    if l_haifa_port.empty:
        raise ValueError(
            "No L_Proxy rows found for port=='Haifa'. "
            "Check that L_Proxy.tsv contains Haifa terminals."
        )

    l_haifa_port = l_haifa_port.rename(columns={"L_hours_i_m": "L"})
    return l_haifa_port


def build_haifa_cluster_kl_scenarios(
    l_proxy: pd.DataFrame,
    k_cluster: pd.DataFrame,
) -> pd.DataFrame:
    """Build three port-level (cluster) K/L series for Haifa, one per δ-scenario.

    Series IDs:
      - Haifa_port_KL_cluster_low
      - Haifa_port_KL_cluster_central
      - Haifa_port_KL_cluster_high

    Using:

      K: company == "Haifa cluster (HPC + IPC + SIPG)", K_low / K_central / K_high
      L: sum of L_hours_i_m across all Haifa terminals each month

    Each output series has:
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
    l_haifa_port = _build_haifa_port_l_series(l_proxy)

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


def build_haifa_entity_kl_scenarios(
    l_proxy: pd.DataFrame,
    k_cluster: pd.DataFrame,
) -> pd.DataFrame:
    """Build port-level K/L series for HPC, IPC, and SIPG using Haifa port labor.

    For each entity e ∈ {HPC, IPC, SIPG} and each δ-scenario s ∈ {low, central, high},
    construct a series with ID:

      - Haifa_port_KL_<ENTITY>_<s>

    where:

      - K is the entity-specific capital K_s from the K cluster file
        (rows with company matching the appropriate *_COMPANY_NAME).
      - L is the Haifa port labor series (sum of L_hours_i_m across Haifa terminals).

    This ensures that, for every month t and scenario s,

        Haifa_port_KL_cluster_s(t)
          = Haifa_port_KL_HPC_s(t)
          + Haifa_port_KL_IPC_s(t)
          + Haifa_port_KL_SIPG_s(t).

    Output columns match the cluster builder:
      series_id, level, freq, port, terminal, year, month, month_index,
      K, L, KL, log_KL.
    """
    # Shared Haifa port labor series
    l_haifa_port = _build_haifa_port_l_series(l_proxy)

    # Configuration for each entity: (label, company_name)
    entity_configs = [
        ("HPC", HPC_COMPANY_NAME),
        ("IPC", IPC_COMPANY_NAME),
        ("SIPG", SIPG_COMPANY_NAME),
    ]

    scenario_suffixes = ["low", "central", "high"]

    out_list: list[pd.DataFrame] = []

    for entity_label, company_name in entity_configs:
        # 1. Take only this entity's K rows
        k_ent_raw = k_cluster[k_cluster["company"] == company_name].copy()
        if k_ent_raw.empty:
            raise ValueError(
                "No rows in K cluster file for company == '{name}'. "
                "Check that the company label matches the K pipeline."
                .format(name=company_name)
            )

        # 2. Ensure uniqueness on (port, year, month) by aggregating over duplicates
        k_ent = (
            k_ent_raw
            .groupby(["port", "year", "month"], as_index=False)[
                ["K_low", "K_central", "K_high"]
            ]
            .sum()
        )

        # 3. Merge with Haifa port labor; now it really is one-to-one
        merged = pd.merge(
            l_haifa_port,
            k_ent[["port", "year", "month", "K_low", "K_central", "K_high"]],
            on=["port", "year", "month"],
            how="inner",
            validate="one_to_one",
        )

        if merged.empty:
            raise ValueError(
                "Merged Haifa {ent} K/L is empty. "
                "There may be no overlapping months between L_Proxy and the K cluster file."
                .format(ent=entity_label)
            )

        for scen in scenario_suffixes:
            series_id = f"Haifa_port_KL_{entity_label}_{scen}"
            tmp = merged.copy()
            tmp["K"] = tmp[f"K_{scen}"]

            good = (tmp["K"] > 0) & (tmp["L"] > 0)
            if not good.all():
                dropped = int((~good).sum())
                print(
                    f"[WARN] Dropping {dropped} Haifa-{entity_label} ({scen}) rows with "
                    "non-positive K or L before computing log(K/L)."
                )
                tmp = tmp.loc[good].copy()

            tmp["KL"] = tmp["K"] / tmp["L"]
            tmp["log_KL"] = np.log(tmp["KL"])

            tmp["month_index"] = tmp["year"] * 12 + tmp["month"]

            tmp["series_id"] = series_id
            tmp["level"] = "port"
            tmp["freq"] = "M"
            tmp["terminal"] = np.nan

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
    haifa_entity_kl_all = build_haifa_entity_kl_scenarios(l_proxy, k_cluster)

    print(
        f"Built Haifa-Legacy KL series with {len(haifa_legacy_kl)} "
        "terminal×month rows."
    )
    print(
        "Built Haifa cluster KL series (low/central/high) with "
        f"{len(haifa_cluster_kl_all)} port×month rows total "
        f"({haifa_cluster_kl_all['series_id'].nunique()} series)."
    )
    print(
        "Built Haifa HPC/IPC/SIPG entity KL series (low/central/high) with "
        f"{len(haifa_entity_kl_all)} port×month rows total "
        f"({haifa_entity_kl_all['series_id'].nunique()} series)."
    )

    # Concatenate all series into a single KL panel
    kl_panel = pd.concat(
        [haifa_legacy_kl, haifa_cluster_kl_all, haifa_entity_kl_all],
        ignore_index=True,
    )

    kl_panel = kl_panel.sort_values(
        ["series_id", "year", "month"]
    ).reset_index(drop=True)

    # Ensure output directory exists and write TSV
    KL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kl_panel.to_csv(KL_PANEL_PATH, sep="\t", index=False)

    print(
        "Wrote KL panel with "
        f"{len(kl_panel)} rows and "
        f"{kl_panel['series_id'].nunique()} distinct series_ids "
        f"to: {KL_PANEL_PATH}"
    )


if __name__ == "__main__":
    main()

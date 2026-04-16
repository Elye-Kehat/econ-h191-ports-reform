"""
build_model2_panels.py

Purpose
-------
Constructs analysis-ready panels for Model 2 (elasticities and mediation).

This script builds:
  1) A terminal-month panel for Haifa terminals (Legacy, Bayport) with
     alternative depreciation scenarios for K/L (delta = 4%, 6%, 8%).
     -> Written to: Design/Output (new)/Model_2A/model2a_terminal_panel.tsv

  2) A port-cluster monthly panel for Haifa (LP and K/L cluster) with the same
     depreciation scenarios.
     -> Written to: Design/Output (new)/Model_2B/model2b_cluster_panel.tsv

Inputs
------
- Data/LP/LP_Panel_monthly.tsv
- Data/KL/KL_Panel_monthly.tsv

Assumptions about series_id naming (from your KL panel construction):
- *_low     corresponds to delta = 0.04 (low depreciation -> higher K, higher K/L)
- *_central corresponds to delta = 0.06
- *_high    corresponds to delta = 0.08

The script is intentionally conservative: it only builds Haifa panels (since
Ashdod K/L is unavailable).

Usage
-----
python build_model2_panels.py

"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


###############################################################################
# Helpers: paths, parsing, logging
###############################################################################

def find_thesis_root(start: Optional[Path] = None) -> Path:
    """
    Walk up from this file (or 'start') until we find a directory that
    looks like the thesis repo root (has Data/ and Design/).
    """
    if start is None:
        start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise FileNotFoundError("Could not locate thesis root (expected folders: Data/, Design/).")


def month_to_date(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=int(month), day=1)


def scenario_from_series_id(series_id: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Returns (scenario_label, delta_float) from a KL series_id suffix.
    """
    if series_id.endswith("_low"):
        return "low", 0.04
    if series_id.endswith("_central"):
        return "central", 0.06
    if series_id.endswith("_high"):
        return "high", 0.08
    return None, None


def safe_log(x: pd.Series) -> pd.Series:
    """
    Compute log(x) safely: returns NaN where x<=0 or missing.
    """
    x = pd.to_numeric(x, errors="coerce")
    out = np.where((x > 0) & np.isfinite(x), np.log(x), np.nan)
    return pd.Series(out, index=x.index, dtype="float64")


###############################################################################
# Build panels
###############################################################################

def build_terminal_panel(lp: pd.DataFrame, kl: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long terminal-month panel for Haifa terminals and depreciation scenarios.

    Output columns (minimal set):
      - terminal: "Haifa--Legacy" / "Haifa--Bayport"
      - delta, scenario
      - year, month, date
      - LP, log_LP
      - K, L, KL, log_KL
      - t_index (within terminal × delta)
    """

    # --- LP: pick Haifa terminal LP series
    lp_map = {
        "Haifa--Legacy": "Haifa_Legacy_Q",
        "Haifa--Bayport": "Haifa_SIPG_Q",
    }

    lp_use = lp[lp["series_id"].isin(lp_map.values())].copy()
    if lp_use.empty:
        raise ValueError("LP panel does not contain expected Haifa terminal series_id(s).")

    inv_lp_map = {v: k for k, v in lp_map.items()}
    lp_use["terminal"] = lp_use["series_id"].map(inv_lp_map)

    # Standardize and compute log(LP) if needed
    if "LP" not in lp_use.columns:
        raise ValueError("LP_Panel_monthly.tsv is missing required column 'LP'.")
    lp_use["log_LP"] = lp_use.get("log_LP")
    if "log_LP" not in lp_use.columns or lp_use["log_LP"].isna().all():
        lp_use["log_LP"] = safe_log(lp_use["LP"])

    lp_use["date"] = [month_to_date(y, m) for y, m in zip(lp_use["year"], lp_use["month"])]
    lp_use = lp_use[["terminal", "year", "month", "date", "LP", "log_LP"]].copy()

    # --- KL: pick terminal-specific K/L series for Haifa (HPC legacy, SIPG entrant)
    # We treat these as "terminal K/L inputs" even though series_id name uses "port_".
    kl_targets = {
        "Haifa--Legacy": "Haifa_port_KL_HPC_",
        "Haifa--Bayport": "Haifa_port_KL_SIPG_",
    }

    kl_rows: List[pd.DataFrame] = []
    for terminal, prefix in kl_targets.items():
        df = kl[kl["series_id"].str.startswith(prefix)].copy()
        if df.empty:
            raise ValueError(f"KL panel missing expected series for {terminal} (prefix={prefix}).")
        df["terminal"] = terminal
        df["scenario"], df["delta"] = zip(*df["series_id"].map(scenario_from_series_id))
        df = df[df["delta"].notna()].copy()
        df["date"] = [month_to_date(y, m) for y, m in zip(df["year"], df["month"])]

        # standardize log_KL
        if "log_KL" not in df.columns or df["log_KL"].isna().all():
            if "KL" not in df.columns:
                raise ValueError("KL_Panel_monthly.tsv missing required column 'KL' and/or 'log_KL'.")
            df["log_KL"] = safe_log(df["KL"])

        keep = ["terminal", "delta", "scenario", "year", "month", "date", "K", "L", "KL", "log_KL"]
        for c in keep:
            if c not in df.columns:
                raise ValueError(f"KL_Panel_monthly.tsv missing required column '{c}'.")
        kl_rows.append(df[keep])

    kl_use = pd.concat(kl_rows, ignore_index=True)

    # --- Merge LP and KL by terminal × year × month
    out = lp_use.merge(
        kl_use,
        on=["terminal", "year", "month", "date"],
        how="inner",
        validate="one_to_many",
    )

    # Build t_index within terminal×delta (so each scenario is a clean TS)
    out = out.sort_values(["terminal", "delta", "date"]).reset_index(drop=True)
    out["t_index"] = out.groupby(["terminal", "delta"]).cumcount()

    return out


def build_cluster_panel(lp: pd.DataFrame, kl: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long monthly Haifa port-cluster panel with depreciation scenarios.

    Output columns:
      - delta, scenario
      - year, month, date
      - LP, log_LP
      - K, L, KL, log_KL
      - t_index (within delta)
    """

    # LP: Haifa port aggregate
    lp_port = lp[lp["series_id"].eq("Haifa_port_M")].copy()
    if lp_port.empty:
        raise ValueError("LP panel does not contain expected series_id 'Haifa_port_M' for Haifa port LP.")

    if "LP" not in lp_port.columns:
        raise ValueError("LP_Panel_monthly.tsv is missing required column 'LP'.")
    if "log_LP" not in lp_port.columns or lp_port["log_LP"].isna().all():
        lp_port["log_LP"] = safe_log(lp_port["LP"])
    lp_port["date"] = [month_to_date(y, m) for y, m in zip(lp_port["year"], lp_port["month"])]
    lp_port = lp_port[["year", "month", "date", "LP", "log_LP"]].copy()

    # KL: Haifa port cluster scenarios
    kl_port = kl[kl["series_id"].str.startswith("Haifa_port_KL_cluster_")].copy()
    if kl_port.empty:
        raise ValueError("KL panel missing expected Haifa cluster series (prefix=Haifa_port_KL_cluster_).")
    kl_port["scenario"], kl_port["delta"] = zip(*kl_port["series_id"].map(scenario_from_series_id))
    kl_port = kl_port[kl_port["delta"].notna()].copy()
    kl_port["date"] = [month_to_date(y, m) for y, m in zip(kl_port["year"], kl_port["month"])]

    if "log_KL" not in kl_port.columns or kl_port["log_KL"].isna().all():
        kl_port["log_KL"] = safe_log(kl_port["KL"])

    keep = ["delta", "scenario", "year", "month", "date", "K", "L", "KL", "log_KL"]
    kl_port = kl_port[keep].copy()

    out = lp_port.merge(kl_port, on=["year", "month", "date"], how="inner", validate="one_to_many")

    out = out.sort_values(["delta", "date"]).reset_index(drop=True)
    out["t_index"] = out.groupby(["delta"]).cumcount()

    return out


###############################################################################
# Main
###############################################################################

def main() -> None:
    print("=== build_model2_panels: starting ===")
    THESIS_ROOT = find_thesis_root()
    print("THESIS_ROOT:", THESIS_ROOT)

    lp_path = THESIS_ROOT / "Data" / "LP" / "LP_Panel_monthly.tsv"
    kl_path = THESIS_ROOT / "Data" / "KL" / "KL_Panel_monthly.tsv"

    print("Reading:", lp_path)
    lp = pd.read_csv(lp_path, sep="\t")
    print("  LP rows:", len(lp))

    print("Reading:", kl_path)
    kl = pd.read_csv(kl_path, sep="\t")
    print("  KL rows:", len(kl))

    term_panel = build_terminal_panel(lp, kl)
    cluster_panel = build_cluster_panel(lp, kl)

    # Output directories consistent with existing Model_2A.py and Model_2B.py
    out2a = THESIS_ROOT / "Design" / "Output (new)" / "Model_2A"
    out2b = THESIS_ROOT / "Design" / "Output (new)" / "Model_2B"
    out2a.mkdir(parents=True, exist_ok=True)
    out2b.mkdir(parents=True, exist_ok=True)

    term_path = out2a / "model2a_terminal_panel.tsv"
    clus_path = out2b / "model2b_cluster_panel.tsv"

    term_panel.to_csv(term_path, sep="\t", index=False)
    cluster_panel.to_csv(clus_path, sep="\t", index=False)

    print("Wrote terminal panel:", term_path, f"(rows={len(term_panel)})")
    print("Wrote cluster  panel:", clus_path, f"(rows={len(cluster_panel)})")
    print("=== build_model2_panels: done ===")


if __name__ == "__main__":
    main()
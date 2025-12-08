# 11_build_K_annual_IPC_PIM.py
#
# IPC-specific annual PIM step for the Haifa K-series pipeline.
#
# Purpose:
#   - Read IPC Step-1 real financials (10_haifa_IPC_financials_step1_real.tsv).
#   - Read depreciation scenarios meta (10_IPC_depreciation_choice.json).
#   - Construct a Perpetual Inventory Method (PIM) capital stock series
#     for IPC under low/central/high δ assumptions.
#   - Save:
#       * 11_K_B_annual_Haifa_IPC_PIM.tsv
#       * 11_K_B_annual_Haifa_IPC_PIM_sample.csv
#       * 11_IPC_PIM_meta.json

from pathlib import Path
import json

import pandas as pd
import numpy as np


# --------------------------------------------------------------------
# File paths (adjust if needed)
# --------------------------------------------------------------------

IPC_STEP1_REAL_PATH = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /10_haifa_IPC_financials_step1_real.tsv")
IPC_DEPR_META_PATH  = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /10_IPC_depreciation_choice.json")

OUT_ANNUAL_PIM_PATH   = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /11_K_B_annual_Haifa_IPC_PIM.tsv")
OUT_ANNUAL_PIM_SAMPLE = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /11_K_B_annual_Haifa_IPC_PIM_sample.csv")
OUT_PIM_META_PATH     = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /11_IPC_PIM_meta.json")


# --------------------------------------------------------------------
# Helper: load IPC Step-1 real financials
# --------------------------------------------------------------------

def load_ipc_real(fin_real_path: Path) -> pd.DataFrame:
    """Load IPC Step-1 real financials.

    Expected columns in 10_haifa_IPC_financials_step1_real.tsv:

        company, year,
        ppe_net_nom, additions_nom, depr_nom, disposals_nom,
        deflator,
        K_book_real, I_real, depr_real, disposals_real

    We only use the 'real' columns plus year/company.
    """

    df = pd.read_csv(fin_real_path, sep="\t")

    required_cols = [
        "company",
        "year",
        "K_book_real",
        "I_real",
        "depr_real",
        "disposals_real",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {fin_real_path}: {missing}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # There should be exactly one company (IPC), but we keep the column
    # so that the output is self-describing.
    return df[required_cols].sort_values("year")


# --------------------------------------------------------------------
# Helper: load IPC depreciation scenarios
# --------------------------------------------------------------------

def load_depr_meta(meta_path: Path):
    """Load IPC depreciation meta JSON.

    Expected structure:

        {
          "delta_scenarios": {
              "low": 0.04,
              "central": 0.06,
              "high": 0.08
          },
          "default_scenario": "central",
          "company": "Israel Ports Company (Haifa cluster)",
          ...
        }
    """

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "delta_scenarios" not in meta:
        raise ValueError(
            f"'delta_scenarios' not found in {meta_path}. "
            f"Keys present: {list(meta.keys())}"
        )
    scenarios = meta["delta_scenarios"]
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError(
            f"'delta_scenarios' must be a non-empty dict, got: {scenarios}"
        )
    default = meta.get("default_scenario", "central")
    return scenarios, default, meta


# --------------------------------------------------------------------
# Helper: run PIM for a single δ
# --------------------------------------------------------------------

def run_pim_for_delta(df: pd.DataFrame, delta: float) -> pd.Series:
    """Run a simple annual PIM recursion for a given δ.

    PIM recursion (net capital):

        K_{t} = (1 - δ) * K_{t-1} + I_t - D_t

    where:
        - K_{t-1} is net capital at the end of year t-1,
        - I_t is gross real investment during year t,
        - D_t is real disposals (in cost terms) during year t.

    We initialize K_{t0} at the book value in the first year (K_book_real),
    which anchors the PIM to the accounting series.
    """

    years = df["year"].to_numpy()
    I = df["I_real"].fillna(0.0).to_numpy()
    D = df["disposals_real"].fillna(0.0).to_numpy()
    K_book = df["K_book_real"].to_numpy()

    K_pim = np.zeros_like(K_book, dtype=float)

    # Initialize at first observed book value
    K_pim[0] = K_book[0]

    for t in range(1, len(years)):
        K_pim[t] = (1.0 - delta) * K_pim[t - 1] + I[t] - D[t]

    return pd.Series(K_pim, index=df.index)


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------

def run_ipc_annual_pim() -> pd.DataFrame:
    """Run annual PIM for IPC and write outputs to disk."""

    print("\n[IPC PIM] Loading IPC Step-1 real financials...")
    df_real = load_ipc_real(IPC_STEP1_REAL_PATH)
    print(f"  - Rows loaded: {len(df_real)}")
    print(f"  - Years: {df_real['year'].min()}–{df_real['year'].max()}")

    print("\n[IPC PIM] Loading depreciation scenarios...")
    depr_scenarios, default_scenario, meta_in = load_depr_meta(IPC_DEPR_META_PATH)
    print(f"  - Scenarios: {depr_scenarios}")
    print(f"  - Default scenario: {default_scenario}")

    # Prepare output DataFrame
    out = df_real.copy()

    # Run PIM for each scenario
    for name, delta in depr_scenarios.items():
        print(f"  - Running PIM for scenario '{name}' with δ = {delta:.4f}...")
        out[f"K_PIM_real_{name}"] = run_pim_for_delta(df_real, float(delta))

    # QA: gap between book K and central PIM
    central_col = "K_PIM_real_central"
    if central_col in out.columns:
        out["gap_book_minus_PIM_central"] = out["K_book_real"] - out[central_col]
    else:
        out["gap_book_minus_PIM_central"] = np.nan

    # Flags
    # flows_imputed_flag: True if any key real flow is missing in that year
    flow_cols = ["I_real", "depr_real", "disposals_real"]
    out["flows_imputed_flag"] = out[flow_cols].isna().any(axis=1)

    # gap_years_from_prev: difference from previous year (0 for first)
    out = out.sort_values("year")
    out["gap_years_from_prev"] = out["year"].diff().fillna(0).astype(int)

    # Save outputs
    out.to_csv(OUT_ANNUAL_PIM_PATH, sep="\t", index=False)
    out.head(10).to_csv(OUT_ANNUAL_PIM_SAMPLE, index=False)

    meta_out = {
        "input_step1_real_path": str(IPC_STEP1_REAL_PATH),
        "input_depr_meta_path": str(IPC_DEPR_META_PATH),
        "delta_scenarios": depr_scenarios,
        "default_scenario": default_scenario,
        "notes": (
            "Annual IPC PIM K series for Haifa cluster infrastructure. "
            "K_PIM_real_* are net real capital under the given δ assumptions. "
            "gap_book_minus_PIM_central = K_book_real - K_PIM_real_central."
        ),
    }
    OUT_PIM_META_PATH.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print("\n[IPC PIM] Done.")
    print(f"  - Annual PIM table saved to:   {OUT_ANNUAL_PIM_PATH}")
    print(f"  - Sample (first 10 rows) to:   {OUT_ANNUAL_PIM_SAMPLE}")
    print(f"  - PIM meta saved to:           {OUT_PIM_META_PATH}")

    print("\n[IPC PIM] Preview of annual IPC PIM K:")
    print(out.head())

    return out


if __name__ == "__main__":
    run_ipc_annual_pim()

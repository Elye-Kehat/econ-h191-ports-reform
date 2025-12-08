# 31_build_K_annual_SIPG_Bayport_PIM.py
#
# Step 2 for SIPG Haifa Bayport K-series:
#   - Read annual CIP summary in real 2019 NIS (thousands).
#   - Read depreciation scenarios and I-product definition.
#   - Run a Perpetual Inventory Method (PIM) with:
#         * no depreciation before the first "productive" year
#         * geometric depreciation thereafter
#   - Save:
#       * 31_K_B_annual_Haifa_SIPG_PIM.tsv
#       * 31_K_B_annual_Haifa_SIPG_PIM_sample.csv
#       * 31_SIPG_Bayport_PIM_meta.json
#
# This script is SIPG-specific and does not depend on HPC/IPC code.

from pathlib import Path
import json

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).resolve().parent

ANNUAL_PATH = DATA_DIR / "30_SIPG_Bayport_annual_CIP_real.tsv"
META_IN_PATH = DATA_DIR / "30_SIPG_Bayport_depreciation_choice.json"

OUT_PIM_PATH = DATA_DIR / "31_K_B_annual_Haifa_SIPG_PIM.tsv"
OUT_PIM_SAMPLE_PATH = DATA_DIR / "31_K_B_annual_Haifa_SIPG_PIM_sample.csv"
OUT_META_PATH = DATA_DIR / "31_SIPG_Bayport_PIM_meta.json"


def load_annual_cip_real(path: Path) -> pd.DataFrame:
    """Load annual SIPG Bayport CIP summary (Step 1 output)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Annual CIP summary not found at {path}. "
            "Run 30_build_SIPG_Bayport_step1_real.py first."
        )
    df = pd.read_csv(path, sep="\t")
    if "year" not in df.columns:
        raise ValueError(
            f"'year' column not found in {path}. Columns: {list(df.columns)}"
        )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    if "company" not in df.columns:
        df["company"] = "SIPG Haifa Bayport"
    return df.sort_values("year").reset_index(drop=True)


def load_meta(path: Path) -> dict:
    """Load depreciation scenarios and I-product choice."""
    if not path.exists():
        raise FileNotFoundError(
            f"Meta file not found at {path}. "
            "Run 30_build_SIPG_Bayport_step1_real.py (which writes the JSON)."
        )
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def run_sipg_annual_pim() -> pd.DataFrame:
    """Run multi-scenario PIM for SIPG Haifa Bayport."""
    print("\n[31_PIM] Loading annual CIP and meta...")
    annual = load_annual_cip_real(ANNUAL_PATH)
    meta = load_meta(META_IN_PATH)

    scenarios = meta.get("scenarios_available", {"central": 0.06})
    I_def = meta.get("I_productive_definition", "transfers_only")

    if I_def == "transfers_only":
        col_I = "T_real_kNIS"
    elif I_def == "all_I":
        col_I = "I_real_kNIS"
    else:
        raise ValueError(
            "I_productive_definition must be 'transfers_only' or 'all_I'. "
            f"Got '{I_def}'."
        )

    if col_I not in annual.columns:
        raise ValueError(
            f"Column '{col_I}' not found in annual CIP table. "
            f"Available columns: {list(annual.columns)}"
        )

    # Determine the first year in which we treat investment as "productive".
    # If meta provides 'year_dep_start', use that; otherwise, default to the
    # first year with positive I_prod, or the earliest year if none > 0.
    year_dep_start = meta.get("year_dep_start")
    if year_dep_start is None:
        positive = annual.loc[annual[col_I] > 0, "year"]
        if not positive.empty:
            year_dep_start = int(positive.min())
        else:
            year_dep_start = int(annual["year"].min())

    print(f"  - I_productive_definition: {I_def} (column: {col_I})")
    print(f"  - Depreciation starts in year_dep_start = {year_dep_start}")
    print(f"  - Scenarios: {scenarios}")

    df = annual.copy().sort_values("year").reset_index(drop=True)
    df["company"] = "SIPG Haifa Bayport"

    # Prepare output frame with baseline columns
    pim = df[["company", "year", "deflator"]].copy()
    # Keep both total I_real_kNIS and gross K_gross_real_kNIS for QA/reference
    if "I_real_kNIS" in df.columns:
        pim["I_real_kNIS"] = df["I_real_kNIS"]
    else:
        pim["I_real_kNIS"] = np.nan
    if "K_gross_real_kNIS" in df.columns:
        pim["K_gross_real_kNIS"] = df["K_gross_real_kNIS"]
    else:
        pim["K_gross_real_kNIS"] = np.nan

    # PIM recursion per scenario
    for scen, delta_annual in scenarios.items():
        K_vals = []
        Dep_vals = []
        K_prev = 0.0

        for _, row in df.iterrows():
            year = int(row["year"])
            I_y = float(row[col_I])

            if year < year_dep_start:
                # Construction phase: no depreciation yet
                K_new = K_prev + I_y
                Dep_y = 0.0
            else:
                # Productive phase: geometric depreciation
                K_new = (1.0 - float(delta_annual)) * K_prev + I_y
                Dep_y = float(delta_annual) * K_prev

            K_vals.append(K_new)
            Dep_vals.append(Dep_y)
            K_prev = K_new

        pim[f"K_PIM_real_{scen}"] = K_vals
        pim[f"Dep_PIM_real_{scen}"] = Dep_vals

    # Write outputs
    pim.to_csv(OUT_PIM_PATH, sep="\t", index=False)
    pim.head(10).to_csv(OUT_PIM_SAMPLE_PATH, sep="\t", index=False)

    print(f"\n[31_PIM] Annual SIPG PIM table saved to: {OUT_PIM_PATH}")
    print(f"         Sample (first 10 rows): {OUT_PIM_SAMPLE_PATH}")

    meta_out = {
        "description": (
            "Annual PIM for SIPG Haifa Bayport (real 2019 NIS, thousands). "
            "Construction phase with no depreciation before year_dep_start; "
            "geometric depreciation thereafter."
        ),
        "scenarios": scenarios,
        "I_productive_definition": I_def,
        "year_dep_start": year_dep_start,
        "source_step1": str(ANNUAL_PATH.name),
    }
    with open(OUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[31_PIM] PIM meta written to: {OUT_META_PATH}")
    print("\n[31_PIM] Preview:")
    print(pim.head())

    return pim


if __name__ == "__main__":
    run_sipg_annual_pim()

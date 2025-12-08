# 30_build_SIPG_Bayport_step1_real.py
#
# Step 1 for SIPG Haifa Bayport K-series:
#   - Read Bayport CIP table in RMB.
#   - Merge FX and OECD capital deflator.
#   - Construct annual real flows/stocks in 2019 NIS (thousands).
#   - Save:
#       * 30_SIPG_Bayport_annual_CIP_real.tsv
#       * 30_SIPG_Bayport_annual_CIP_real_sample.csv
#       * 30_SIPG_Bayport_depreciation_choice.json
#
# Assumptions:
#   - This script lives in Thesis/Data/K/.
#   - The CIP file is sipg_haifa_bayport_CIP_RMB.tsv (tab-separated, RMB).
#   - If fx_rmb_nis.tsv does NOT exist, but "CNY_ILS Historical Data.csv"
#     DOES exist in the same folder, this script will:
#         * build fx_rmb_nis.tsv from that daily file (annual averages),
#         * then use it.
#   - You use the same OECD QNA deflator CSV as the HPC K pipeline.

from pathlib import Path
import json

import pandas as pd
import numpy as np


# --------------------------------------------------------------------
# Paths (relative to this file)
# --------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent

CIP_PATH = DATA_DIR / "sipg_haifa_bayport_CIP_RMB.tsv"
FX_PATH = DATA_DIR / "fx_rmb_nis.tsv"

# Optional daily FX file (Investing.com or similar)
CNY_ILS_DAILY_PATH = DATA_DIR / "CNY_ILS Historical Data.csv"

# Same OECD QNA file used by 00_build_K_step1.py
DEFLATOR_PATH = DATA_DIR / (
    "OECD-DSD_NAMAIN1@DF_QNA_EXPENDITURE_INDICES-"
    "Q.Y.ISR.S1.S1.P51G._Z._T._Z.IX.DR.N.T0102.csv"
)

OUT_ANNUAL_PATH = DATA_DIR / "30_SIPG_Bayport_annual_CIP_real.tsv"
OUT_SAMPLE_PATH = DATA_DIR / "30_SIPG_Bayport_annual_CIP_real_sample.csv"
OUT_META_PATH = DATA_DIR / "30_SIPG_Bayport_depreciation_choice.json"


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def load_cip_data(cip_path: Path) -> pd.DataFrame:
    """Load SIPG Haifa Bayport CIP data in RMB and standardize."""
    if not cip_path.exists():
        raise FileNotFoundError(
            f"CIP file not found at {cip_path}. "
            "Expected a TSV file with Bayport CIP in RMB."
        )

    df = pd.read_csv(cip_path, sep="\t")

    required = [
        "year",
        "project_group",
        "budget_rmb",
        "cip_opening_rmb",
        "cip_additions_rmb",
        "cip_transfers_to_ppe_rmb",
        "cip_other_decreases_rmb",
        "cip_closing_rmb",
        "cip_cumul_capitalized_interest_rmb",
        "cip_current_capitalized_interest_rmb",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in CIP file: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # Ensure year is clean
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # Coerce all CIP amount columns to numeric (strings -> NaN)
    numeric_cols = [
        "budget_rmb",
        "cip_opening_rmb",
        "cip_additions_rmb",
        "cip_transfers_to_ppe_rmb",
        "cip_other_decreases_rmb",
        "cip_closing_rmb",
        "cip_cumul_capitalized_interest_rmb",
        "cip_current_capitalized_interest_rmb",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only Haifa Bayport-related rows (should be all, but be explicit).
    # Currently project_group contains e.g. 'Bayport main terminal',
    # 'Bayport ancillary systems'.
    return df


def build_fx_from_cny_ils_daily(
    daily_path: Path, out_path: Path
) -> pd.DataFrame:
    """
    Build fx_rmb_nis.tsv from a daily 'CNY_ILS Historical Data.csv' file.

    Expected structure of the daily file (typical Investing.com export):
      - 'Date' column: daily dates.
      - One numeric column with the CNY/ILS rate (ILS per 1 CNY), usually
        called 'Price'. If not found, we pick the first numeric column.

    Output:
      - TSV at out_path with columns:
            year, fx_avg_nis_per_rmb
    """
    if not daily_path.exists():
        raise FileNotFoundError(
            f"Daily CNY/ILS file not found at: {daily_path}\n"
            "Either create fx_rmb_nis.tsv manually or place a file named "
            "'CNY_ILS Historical Data.csv' (daily CNY/ILS data) in the same "
            "folder as this script."
        )

    print(f"[30_step1] fx_rmb_nis.tsv not found; building from {daily_path.name} ...")

    d = pd.read_csv(daily_path)

    if "Date" not in d.columns:
        raise ValueError(
            f"'Date' column not found in {daily_path}. "
            f"Columns found: {list(d.columns)}"
        )

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).copy()

    # Identify the rate column
    num_cols = d.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise ValueError(
            f"No numeric columns found in {daily_path}. "
            "Cannot infer FX rate column."
        )

    # Prefer 'Price' if present; otherwise, use the first numeric column.
    if "Price" in num_cols:
        rate_col = "Price"
    else:
        rate_col = num_cols[0]
        print(
            f"[30_step1] Using numeric column '{rate_col}' as FX rate "
            f"(CNY→ILS) from {daily_path.name}."
        )

    d["year"] = d["Date"].dt.year

    annual = (
        d.groupby("year", as_index=False)[rate_col]
        .mean()
        .rename(columns={rate_col: "fx_avg_nis_per_rmb"})
        .sort_values("year")
        .reset_index(drop=True)
    )

    # Save as TSV that load_fx_table can read
    annual.to_csv(out_path, sep="\t", index=False)

    print(f"[30_step1] Built annual FX table and saved to: {out_path}")
    print("[30_step1] Annual FX preview:")
    print(annual.head())

    return annual


def load_fx_table(fx_path: Path) -> pd.DataFrame:
    """Load RMB→NIS FX table, building it from daily data if needed."""
    if not fx_path.exists():
        # Try to build from daily CNY_ILS Historical Data.csv
        if CNY_ILS_DAILY_PATH.exists():
            fx_built = build_fx_from_cny_ils_daily(
                daily_path=CNY_ILS_DAILY_PATH,
                out_path=fx_path,
            )
            # Ensure we only keep the columns we need
            fx_built = fx_built[["year", "fx_avg_nis_per_rmb"]]
            return fx_built

        # If we get here, neither fx_rmb_nis.tsv nor the daily file exists
        raise FileNotFoundError(
            f"FX file not found at {fx_path} and daily CNY_ILS file "
            f"not found at {CNY_ILS_DAILY_PATH}.\n"
            "Either:\n"
            "  (a) place 'CNY_ILS Historical Data.csv' (daily CNY/ILS data) in "
            "      this folder, or\n"
            "  (b) manually create fx_rmb_nis.tsv with columns:\n"
            "        year, fx_avg_nis_per_rmb\n"
        )

    fx = pd.read_csv(fx_path, sep="\t")
    if "year" not in fx.columns or "fx_avg_nis_per_rmb" not in fx.columns:
        raise ValueError(
            "fx_rmb_nis.tsv must have columns 'year' and 'fx_avg_nis_per_rmb'. "
            f"Columns found: {list(fx.columns)}"
        )

    fx["year"] = pd.to_numeric(fx["year"], errors="coerce")
    fx = fx.dropna(subset=["year"]).copy()
    fx["year"] = fx["year"].astype(int)

    fx = fx[["year", "fx_avg_nis_per_rmb"]].drop_duplicates(
        subset=["year"], keep="last"
    )
    return fx


def load_deflator(defl_path: Path, base_year: int = 2019) -> pd.DataFrame:
    """
    Read the deflator file and return a [year, deflator] table.

    Supports two formats:

    1) OECD QNA CSV (your current file):
         - column 'period' like '1995-Q1'
         - one numeric column with the deflator index (rebased).
       We:
         - parse year from 'period'
         - rebase to base_year
         - average by year → [year, deflator]

    2) Simple [year, deflator] table:
         - column 'year'
         - at least one numeric column → used as deflator (assumed already rebased)
    """
    if not defl_path.exists():
        raise FileNotFoundError(
            f"Deflator file not found at: {defl_path}\n"
            "Check that the OECD QNA CSV (or a simple capital_deflator.csv) "
            "is present in Data/K/."
        )

    d = pd.read_csv(defl_path)

    # Case 1: OECD QNA with 'period'
    if "period" in d.columns:
        num_cols = d.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            candidates = [c for c in d.columns if c != "period"]
            if not candidates:
                raise ValueError(
                    "No numeric index column found in OECD deflator file."
                )
            idx_col = candidates[0]
        else:
            idx_col = num_cols[0]

        # Parse year and rebase
        d["year"] = d["period"].str.slice(0, 4).astype(int)
        base_val = d.loc[d["year"] == base_year, idx_col].mean()
        if pd.isna(base_val) or base_val == 0:
            raise ValueError(
                f"Could not find a valid index value for base_year={base_year} "
                f"in {defl_path}."
            )

        d["deflator"] = d[idx_col] / base_val
        out = (
            d.groupby("year", as_index=False)["deflator"]
            .mean()
            .sort_values("year")
            .reset_index(drop=True)
        )
        return out

    # Case 2: simple [year, deflator]
    if "year" not in d.columns:
        raise ValueError(
            "Deflator file must have either 'period' (OECD QNA) or 'year' column."
        )

    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d.dropna(subset=["year"]).copy()
    d["year"] = d["year"].astype(int)

    num_cols = d.select_dtypes(include="number").columns.tolist()
    # Remove 'year' from numeric candidates
    num_cols = [c for c in num_cols if c != "year"]
    if not num_cols:
        raise ValueError(
            "Could not find a numeric deflator column in the [year, deflator] file."
        )
    idx_col = num_cols[0]

    out = (
        d[["year", idx_col]]
        .rename(columns={idx_col: "deflator"})
        .drop_duplicates(subset=["year"])
        .sort_values("year")
        .reset_index(drop=True)
    )
    return out


def build_annual_cip_real(
    cip: pd.DataFrame,
    fx: pd.DataFrame,
    defl: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Bayport CIP to annual RMB flows and stocks,
    then convert to NIS and real 2019 NIS (thousands).
    """

    df = cip.copy()

    # Annual flows in RMB
    df["I_RMB"] = (
        df["cip_additions_rmb"].fillna(0.0)
        + df["cip_current_capitalized_interest_rmb"].fillna(0.0)
    )
    df["T_RMB"] = df["cip_transfers_to_ppe_rmb"].fillna(0.0)
    df["E_RMB"] = df["cip_other_decreases_rmb"].fillna(0.0)

    # Aggregate across project groups (main terminal + ancillary systems)
    annual = (
        df.groupby("year", as_index=False)
        .agg(
            budget_rmb=("budget_rmb", "sum"),
            cip_opening_rmb=("cip_opening_rmb", "sum"),
            cip_closing_rmb=("cip_closing_rmb", "sum"),
            I_RMB=("I_RMB", "sum"),
            T_RMB=("T_RMB", "sum"),
            E_RMB=("E_RMB", "sum"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )

    # CIP identity check (for debugging, not enforced)
    lhs = annual["cip_closing_rmb"]
    rhs = (
        annual["cip_opening_rmb"].fillna(0.0)
        + annual["I_RMB"]
        - annual["T_RMB"]
        - annual["E_RMB"]
    )
    diff = lhs - rhs
    if diff.abs().max() > 1e-3:
        print("\n[build_annual_cip_real] Warning: CIP identity does not hold exactly.")
        print(
            pd.DataFrame(
                {
                    "year": annual["year"],
                    "cip_closing_rmb": lhs,
                    "open_plus_flows_minus_T_minus_E": rhs,
                    "diff": diff,
                }
            )
        )

    # Cumulative completed works and “gross” Bayport capital at cost
    annual["K_completed_RMB"] = annual["T_RMB"].cumsum()
    annual["K_gross_RMB"] = annual["K_completed_RMB"] + annual["cip_closing_rmb"]

    # Merge FX
    annual = annual.merge(fx, on="year", how="left")
    if annual["fx_avg_nis_per_rmb"].isna().any():
        missing_years = annual.loc[
            annual["fx_avg_nis_per_rmb"].isna(), "year"
        ].unique()
        raise ValueError(
            f"No FX rate for years: {missing_years}. "
            "Update fx_rmb_nis.tsv (or the daily CNY_ILS file) to cover all "
            "Bayport years."
        )

    for col in ["I_RMB", "T_RMB", "K_completed_RMB", "K_gross_RMB"]:
        annual[col.replace("_RMB", "_NIS")] = (
            annual[col] * annual["fx_avg_nis_per_rmb"]
        )

    # Merge deflator
    annual = annual.merge(defl, on="year", how="left")
    if annual["deflator"].isna().any():
        missing_years = annual.loc[annual["deflator"].isna(), "year"].unique()
        raise ValueError(
            f"No deflator value for years: {missing_years}. "
            "Check that the OECD deflator covers the Bayport years."
        )

    # Real 2019 NIS
    for col in ["I_NIS", "T_NIS", "K_completed_NIS", "K_gross_NIS"]:
        annual[col.replace("_NIS", "_real")] = annual[col] / annual["deflator"]

    # Convert to thousands
    for col in ["I_real", "T_real", "K_completed_real", "K_gross_real"]:
        annual[col + "_kNIS"] = annual[col] / 1_000.0

    # Add company label
    annual["company"] = "SIPG Haifa Bayport"

    cols_out = [
        "company",
        "year",
        "deflator",
        "budget_rmb",
        "cip_opening_rmb",
        "cip_closing_rmb",
        "I_RMB",
        "T_RMB",
        "E_RMB",
        "I_real_kNIS",
        "T_real_kNIS",
        "K_completed_real_kNIS",
        "K_gross_real_kNIS",
    ]
    return annual[cols_out].sort_values("year").reset_index(drop=True)


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------

def run_step1(
    I_productive_definition: str = "transfers_only",
) -> pd.DataFrame:
    """
    Build annual SIPG Bayport CIP summary in real 2019 NIS (thousands).

    I_productive_definition controls which investment measure will be
    treated as "productive" in the PIM step (used to construct K):

        - 'transfers_only':  use T_real_kNIS (transfers to PPE only).
        - 'all_I':           use I_real_kNIS (all CIP additions + cap. interest).

    The choice is recorded in 30_SIPG_Bayport_depreciation_choice.json
    and interpreted by 31_build_K_annual_SIPG_Bayport_PIM.py.
    """
    if I_productive_definition not in {"transfers_only", "all_I"}:
        raise ValueError(
            "I_productive_definition must be 'transfers_only' or 'all_I', "
            f"got '{I_productive_definition}'."
        )

    print("\n[30_step1] Loading CIP, FX, and deflator...")
    cip = load_cip_data(CIP_PATH)
    fx = load_fx_table(FX_PATH)
    defl = load_deflator(DEFLATOR_PATH)

    print(f"  - CIP rows: {len(cip)}; years: {cip['year'].min()}–{cip['year'].max()}")
    print(f"  - FX rows:  {len(fx)}; years: {fx['year'].min()}–{fx['year'].max()}")
    print(f"  - Deflator years: {defl['year'].min()}–{defl['year'].max()}")

    annual = build_annual_cip_real(cip, fx, defl)

    # Write outputs
    annual.to_csv(OUT_ANNUAL_PATH, sep="\t", index=False)
    annual.head(10).to_csv(OUT_SAMPLE_PATH, sep="\t", index=False)

    print(f"\n[30_step1] Annual SIPG Bayport CIP (real) saved to: {OUT_ANNUAL_PATH}")
    print(f"           Sample (first 10 rows): {OUT_SAMPLE_PATH}")

    # Depreciation scenarios meta
    depr_scenarios = {
        "low": 0.04,
        "central": 0.06,
        "high": 0.08,
    }
    meta = {
        "description": "Depreciation scenarios and I-product definition for SIPG Haifa Bayport.",
        "scenarios_available": depr_scenarios,
        "I_productive_definition": I_productive_definition,
        # year_dep_start can be added/overridden manually later if desired.
    }
    with open(OUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[30_step1] Depreciation-choice meta written to: {OUT_META_PATH}")
    print("\n[30_step1] Preview of annual table:")
    print(annual.head())

    return annual


if __name__ == "__main__":
    # Default: treat only transfers to PPE as "productive" investment in PIM.
    run_step1(I_productive_definition="transfers_only")

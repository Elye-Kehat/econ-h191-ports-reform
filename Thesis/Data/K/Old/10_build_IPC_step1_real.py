# 10_build_IPC_step1_real.py
#
# IPC-specific Step 1 for the Haifa K-series pipeline.
#
# Purpose:
#   - Read Haifa IPC raw financial data (two CSV tables).
#   - Build an annual nominal financials table (net PPE, depreciation, investment, disposals).
#   - Merge in an annual capital deflator and create real-valued flows/stocks.
#   - Save:
#       * 10_haifa_IPC_financials_step1_real.tsv  (full real financials)
#       * 10_haifa_IPC_financials_step1_sample.csv (preview)
#       * 10_IPC_depreciation_choice.json         (δ scenarios for the IPC PIM step)
#
# This script is designed to live in Thesis/Data/K/, alongside the HPC K scripts,
# but it does NOT modify or require the existing HPC code. It is fully IPC-specific.

from pathlib import Path
import json

import pandas as pd
import numpy as np


# --------------------------------------------------------------------
# File paths (adjust if needed)
# --------------------------------------------------------------------

IPC_TABLE1_PATH = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /haifa_ipc_raw_table1.csv")
IPC_TABLE2_PATH = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /haifa_ipc_raw_table2.csv")
DATA_DIR = Path(__file__).resolve().parent
DEFLATOR_PATH = DATA_DIR / (
    "OECD-DSD_NAMAIN1@DF_QNA_EXPENDITURE_INDICES-"
    "Q.Y.ISR.S1.S1.P51G._Z._T._Z.IX.DR.N.T0102.csv"
)

OUT_FIN_REAL_PATH   = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /10_haifa_IPC_financials_step1_real.tsv")
OUT_DEBUG_SAMPLE    = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /10_haifa_IPC_financials_step1_sample.csv")
OUT_DEPR_META_PATH  = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /10_IPC_depreciation_choice.json")

IPC_COMPANY_NAME = "Israel Ports Company (Haifa cluster)"


# --------------------------------------------------------------------
# Helper: load IPC raw tables and build nominal annual financials
# --------------------------------------------------------------------

def load_ipc_nominal(table1_path: Path, table2_path: Path) -> pd.DataFrame:
    """Load IPC Haifa raw tables and construct an annual nominal panel.

    Expected structure:

    haifa_ipc_raw_table1.csv:
        year, asset_class, cost_open, additions, transfers, cap_interest,
        disposals, cost_close, accdep_open, depreciation_expense,
        accdep_disposals, accdep_close, net_ppe_close

        - We use only rows where asset_class == 'total'.
        - net_ppe_close is net PPE at year-end (nominal, thousands of NIS).
        - disposals is the *change in cost* due to disposals (likely negative).

    haifa_ipc_raw_table2.csv:
        year, dep_amort_th_nis, purchase_ppe_cf_th_nis,
              capitalized_interest_ppe_th_nis,
              proceeds_sale_ppe_cf_th_nis,
              approx_gross_ppe_investment_th_nis

        - dep_amort_th_nis is total depreciation & amortization (nominal).
        - approx_gross_ppe_investment_th_nis is a *positive* approximation
          to gross physical investment in PPE for the year.

    Returns a DataFrame with columns:

        company, year,
        ppe_net_nom, additions_nom, depr_nom, disposals_nom

    in nominal thousands of NIS.
    """

    # ---- Table 1: year × asset_class (we only want total) ----------------
    t1 = pd.read_csv(table1_path)

    # Basic sanity check
    for col in ["year", "asset_class", "net_ppe_close"]:
        if col not in t1.columns:
            raise ValueError(
                f"Expected column '{col}' not found in {table1_path}. "
                f"Columns present: {list(t1.columns)}"
            )

    t1_total = t1.loc[t1["asset_class"] == "total"].copy()

    # Some IPC files may have missing disposals; treat as 0 for the K flows
    if "disposals" not in t1_total.columns:
        t1_total["disposals"] = 0.0

    t1_total["disposals"] = pd.to_numeric(t1_total["disposals"], errors="coerce")

    # We treat 'disposals' as a cost-level flow (often negative). For the
    # PIM, we only care about the magnitude of disposals, not the sign.
    t1_total["disposals_cost_nom"] = t1_total["disposals"].fillna(0.0).abs()

    t1_total["year"] = pd.to_numeric(t1_total["year"], errors="coerce")
    t1_total = t1_total.dropna(subset=["year"])
    t1_total["year"] = t1_total["year"].astype(int)

    t1_total = t1_total[["year", "net_ppe_close", "disposals_cost_nom"]].rename(
        columns={
            "net_ppe_close": "ppe_net_nom"
        }
    )

    # ---- Table 2: annual CF-based flows ----------------------------------
    t2 = pd.read_csv(table2_path)

    for col in ["year", "dep_amort_th_nis", "approx_gross_ppe_investment_th_nis"]:
        if col not in t2.columns:
            raise ValueError(
                f"Expected column '{col}' not found in {table2_path}. "
                f"Columns present: {list(t2.columns)}"
            )

    t2["year"] = pd.to_numeric(t2["year"], errors="coerce")
    t2 = t2.dropna(subset=["year"])
    t2["year"] = t2["year"].astype(int)

    t2 = t2[["year", "dep_amort_th_nis", "approx_gross_ppe_investment_th_nis"]].rename(
        columns={
            "dep_amort_th_nis": "depr_nom",
            "approx_gross_ppe_investment_th_nis": "additions_nom",
        }
    )

    # ---- Merge & finalize nominal IPC panel ------------------------------
    df = pd.merge(t1_total, t2, on="year", how="inner").sort_values("year")

    df["company"] = IPC_COMPANY_NAME

    # Disposals flow for PIM (nominal, thousands of NIS)
    df["disposals_nom"] = df["disposals_cost_nom"]

    # Reorder columns
    df = df[
        [
            "company",
            "year",
            "ppe_net_nom",
            "additions_nom",
            "depr_nom",
            "disposals_nom",
        ]
    ].copy()

    return df


# --------------------------------------------------------------------
# Helper: build annual capital deflator
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Helper: build annual capital deflator (using OECD QNA file)
# --------------------------------------------------------------------

def load_deflator(deflator_path: Path, base_year: int = 2019) -> pd.DataFrame:
    """
    Construct an annual capital deflator from the OECD QNA file.

    Expected structure of the deflator file
    (same file used by 00_build_K_step1.py):

        period, <one numeric index column>

    where:
        - 'period' looks like '1995-Q1', '1995-Q2', ...
        - the numeric column is an index of GFCF prices.

    We:
      - parse year from 'period',
      - compute an annual mean of the index,
      - rebase so that deflator[base_year] = 1.0.

    Returns a DataFrame with columns: year, deflator.
    """

    print(f"[load_deflator] Looking for deflator at: {deflator_path}")
    if not deflator_path.exists():
        raise FileNotFoundError(
            f"Deflator file not found at: {deflator_path}\n"
            "Check that the OECD QNA deflator file is in the same folder as this script."
        )

    d = pd.read_csv(deflator_path)

    # ------------------------------------------------------------------
    # Case 1: OECD QNA file with 'period' like '1995-Q1'
    # ------------------------------------------------------------------
    if "period" in d.columns:
        # Identify index column (there should be exactly one numeric column)
        num_cols = d.select_dtypes(include="number").columns.tolist()
        if len(num_cols) == 0:
            # fall back: treat any non-'period' column as numeric index
            candidates = [c for c in d.columns if c != "period"]
            if not candidates:
                raise ValueError(
                    "No numeric index column found in OECD deflator file."
                )
            idx_col = candidates[0]
            d[idx_col] = pd.to_numeric(d[idx_col], errors="coerce")
        else:
            idx_col = num_cols[0]

        d = d.rename(columns={idx_col: "index_raw"})

        # Parse year from 'period' (e.g. '1995-Q1' → 1995)
        per = d["period"].astype(str)
        d["year"] = per.str.slice(0, 4).astype(int)

        # Annual mean of the index, then rebase
        annual_raw = d.groupby("year", as_index=False)["index_raw"].mean()

        # Use base_year if available; otherwise fall back to median year
        if base_year not in annual_raw["year"].values:
            print(
                f"[load_deflator] Base year {base_year} not found in deflator data; "
                "falling back to median year."
            )
            base_year = int(annual_raw["year"].median())

        base_val = float(
            annual_raw.loc[annual_raw["year"] == base_year, "index_raw"].iloc[0]
        )
        annual_raw["deflator"] = annual_raw["index_raw"] / base_val

        annual = (
            annual_raw[["year", "deflator"]]
            .sort_values("year")
            .reset_index(drop=True)
        )

        print(
            f"[load_deflator] Parsed OECD QNA deflator for years "
            f"{annual['year'].min()}–{annual['year'].max()} "
            f"(base_year={base_year}, deflator(base)=1.0)"
        )
        return annual

    # ------------------------------------------------------------------
    # Fallback: simple [year, deflator] table (not your current case)
    # ------------------------------------------------------------------
    if "year" not in d.columns:
        for alt in ["Year", "YEAR"]:
            if alt in d.columns:
                d = d.rename(columns={alt: "year"})
                break

    if "year" not in d.columns:
        raise ValueError(
            f"'period' or 'year' column not found in deflator file. "
            f"Columns: {list(d.columns)}"
        )

    num_cols = d.select_dtypes(include="number").columns.tolist()
    if "year" in num_cols:
        num_cols.remove("year")

    if len(num_cols) == 0:
        raise ValueError("No numeric deflator column found in deflator file.")
    deflator_col = num_cols[0]

    annual = (
        d[["year", deflator_col]]
        .rename(columns={deflator_col: "deflator"})
        .assign(year=lambda x: x["year"].astype(int))
        .sort_values("year")
        .reset_index(drop=True)
    )

    print(
        f"[load_deflator] Loaded annual deflator for years "
        f"{annual['year'].min()}–{annual['year'].max()} "
        f"(column used: '{deflator_col}', assumed already rebased)"
    )
    return annual[["year", "deflator"]]


# --------------------------------------------------------------------
# Helper: attach real-valued flows/stocks
# --------------------------------------------------------------------

def attach_real_values(df_nom: pd.DataFrame, defl: pd.DataFrame) -> pd.DataFrame:
    """Merge annual deflator and create real-valued columns.

    Expects df_nom with:
        company, year, ppe_net_nom, additions_nom, depr_nom, disposals_nom

    and defl with:
        year, deflator
    """

    df = df_nom.copy()
    df = pd.merge(df, defl, on="year", how="left")

    if df["deflator"].isna().any():
        missing_years = df.loc[df["deflator"].isna(), "year"].unique().tolist()
        raise ValueError(
            f"Missing deflator for years: {missing_years}. "
            "Check capital_deflator.csv and base_year."
        )

    df["K_book_real"] = df["ppe_net_nom"] / df["deflator"]
    df["I_real"] = df["additions_nom"] / df["deflator"]
    df["depr_real"] = df["depr_nom"] / df["deflator"]
    df["disposals_real"] = df["disposals_nom"].fillna(0.0) / df["deflator"]

    return df


# --------------------------------------------------------------------
# Depreciation scenarios meta
# --------------------------------------------------------------------

def make_depreciation_meta(path: Path) -> None:
    """Write a simple JSON with IPC depreciation scenarios.

    This mirrors the spirit of 00_depreciation_choice.json used for HPC,
    but is IPC-specific and lives separately to avoid accidental conflicts.
    """

    meta = {
        "note": (
            "IPC depreciation scenarios for Haifa infrastructure. "
            "Rates are annual geometric depreciation used in the PIM."
        ),
        "delta_scenarios": {
            "low": 0.04,
            "central": 0.06,
            "high": 0.08,
        },
        "default_scenario": "central",
        "company": IPC_COMPANY_NAME,
    }

    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------

def run_step1() -> pd.DataFrame:
    """Execute IPC Step 1 and write outputs to disk."""

    print("\n[IPC Step 1] Loading IPC nominal financials...")
    ipc_nom = load_ipc_nominal(IPC_TABLE1_PATH, IPC_TABLE2_PATH)
    print(f"  - Loaded {len(ipc_nom)} annual rows from IPC tables.")

    print("\n[IPC Step 1] Loading capital deflator and constructing annual deflator...")
    defl = load_deflator(DEFLATOR_PATH, base_year=2019)
    print(f"  - Deflator years available: {defl['year'].min()}–{defl['year'].max()}")

    print("\n[IPC Step 1] Attaching real-valued flows/stocks...")
    ipc_real = attach_real_values(ipc_nom, defl)

    # Save real financials
    ipc_real.to_csv(OUT_FIN_REAL_PATH, sep="\t", index=False)
    ipc_real.head(10).to_csv(OUT_DEBUG_SAMPLE, index=False)

    # Save depreciation meta
    make_depreciation_meta(OUT_DEPR_META_PATH)

    print("\n[IPC Step 1] Done.")
    print(f"  - Real-valued IPC financials saved to: {OUT_FIN_REAL_PATH}")
    print(f"  - Sample (first 10 rows) saved to:   {OUT_DEBUG_SAMPLE}")
    print(f"  - Depreciation meta saved to:        {OUT_DEPR_META_PATH}")

    print("\n[IPC Step 1] Preview of IPC real financials:")
    print(ipc_real.head())

    return ipc_real


if __name__ == "__main__":
    run_step1()

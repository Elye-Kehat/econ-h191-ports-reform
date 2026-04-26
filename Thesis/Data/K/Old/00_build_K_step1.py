# 00_build_K_step1.py
#
# Steps 1.1–1.3 of the K-construction pipeline:
#   - load & standardize financials
#   - merge deflator and create real-valued flows/stocks
#   - choose depreciation rate via toggle
#
# This script assumes it lives in: Thesis/Data/K/

from pathlib import Path
import json

import pandas as pd


# ====================================================================
# Paths (relative to this file)
# ====================================================================

# Directory that contains this script: should be Thesis/Data/K
DATA_DIR = Path(__file__).resolve().parent

FIN_PATH = DATA_DIR / "haifa_financials_raw.tsv"

# NEW: use OECD QNA deflator file directly
DEFLATOR_PATH = DATA_DIR / (
    "OECD-DSD_NAMAIN1@DF_QNA_EXPENDITURE_INDICES-"
    "Q.Y.ISR.S1.S1.P51G._Z._T._Z.IX.DR.N.T0102.csv"
)
# If you ever go back to a simple [year, deflator] file, you can
# instead point DEFLATOR_PATH to "capital_deflator.csv".

OUT_FIN_REAL_PATH = DATA_DIR / "00_haifa_financials_step1_real.tsv"
OUT_DEBUG_SAMPLE_PATH = DATA_DIR / "00_haifa_financials_step1_sample.csv"
OUT_META_PATH = DATA_DIR / "00_depreciation_choice.json"


# ====================================================================
# Step 1.3 – depreciation toggle
# ====================================================================

DEPR_SCENARIOS = {
    "low": 0.04,      # 4% annual geometric depreciation
    "central": 0.06,  # 6% (default)
    "high": 0.08      # 8%
    # you can add more labels if you want
}


def get_delta(scenario="central"):
    """
    Return an annual depreciation rate δ.

    scenario can be:
    - "low", "central", "high" (keys in DEPR_SCENARIOS)
    - a float (e.g. 0.065 for 6.5%)
    """
    if isinstance(scenario, (int, float)):
        return float(scenario)

    if scenario not in DEPR_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Use one of {list(DEPR_SCENARIOS.keys())} or pass a numeric δ."
        )
    return DEPR_SCENARIOS[scenario]


# ====================================================================
# Step 1.1 – load & standardize financial data
# ====================================================================

def load_financials(fin_path: Path) -> pd.DataFrame:
    """
    Read haifa_financials_raw.tsv and standardize column names
    based on the actual structure of your file.

    Raw columns (from your TSV), among others:
      - company
      - year
      - ppe_net_nominal_thousands_nis_eoy
      - depr_amort_total_thousands_nis_cf_adj
      - purchase_of_fixed_assets_thousands_nis_cashflow
      - proceeds_from_realization_fixed_assets_thousands_nis
    """
    print(f"[load_financials] Looking for financials at: {fin_path}")
    if not fin_path.exists():
        raise FileNotFoundError(
            f"Financials file not found at: {fin_path}\n"
            "Check that the file is in Data/K/ and named 'haifa_financials_raw.tsv'."
        )

    df = pd.read_csv(fin_path, sep="\t")

    # Map raw columns → internal names used downstream
    rename_map = {
        "ppe_net_nominal_thousands_nis_eoy": "ppe_net_nom",
        "purchase_of_fixed_assets_thousands_nis_cashflow": "additions_nom",
        "depr_amort_total_thousands_nis_cf_adj": "depr_nom",
        "proceeds_from_realization_fixed_assets_thousands_nis": "disposals_nom",
    }
    df = df.rename(columns=rename_map)

    # ---- Clean up YEAR -------------------------------------------------
    if "year" not in df.columns:
        raise ValueError(
            f"'year' column not found in financials. Columns: {list(df.columns)}"
        )

    df["year_raw"] = df["year"]  # keep for debugging
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    bad_year = df[df["year"].isna()]
    if not bad_year.empty:
        print("\n[load_financials] Dropping rows with non-numeric year values:")
        print(bad_year[["year_raw"]].drop_duplicates())
        df = df[df["year"].notna()].copy()

    df["year"] = df["year"].astype(int)

    # ---- Force key value columns to numeric ----------------------------
    num_cols = ["ppe_net_nom", "additions_nom", "depr_nom", "disposals_nom"]
    for col in num_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check required columns exist after renaming
    required = ["year", "ppe_net_nom", "additions_nom", "depr_nom"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in financials: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Keep core columns for PIM, plus some metadata
    keep_cols = [
        c
        for c in [
            "company",
            "year",
            "ppe_net_nom",
            "additions_nom",
            "depr_nom",
            "disposals_nom",
            "purchase_of_intangible_assets_thousands_nis_cashflow",
            "revaluation_or_basis_flag",
            "source_pdf",
            "source_pages",
            "units_note",
        ]
        if c in df.columns
    ]
    df = df[keep_cols].copy()
    df = df.sort_values("year")

    print(
        f"[load_financials] Loaded {len(df)} rows for years "
        f"{df['year'].min()}–{df['year'].max()}"
    )
    print("[load_financials] Columns after standardization:", list(df.columns))

    # Quick sign sanity check for additions & disposals
    print("\n[load_financials] Sign check (means, after coercion):")
    print("  additions_nom mean:", df["additions_nom"].mean())
    if "disposals_nom" in df.columns:
        print("  disposals_nom mean:", df["disposals_nom"].mean())

    # Show rows where additions_nom is NaN so you can inspect later
    bad_add = df[df["additions_nom"].isna()]
    if not bad_add.empty:
        print(
            "\n[load_financials] WARNING: some rows have non-numeric additions_nom; "
            "they were coerced to NaN. Rows:"
        )
        print(bad_add[["year", "company", "additions_nom"]])

    return df


# ====================================================================
# Step 1.2 – load deflator & create real flows/stocks
# ====================================================================

def load_deflator(defl_path: Path) -> pd.DataFrame:
    """
    Read the deflator file and return an annual GFCF deflator.

    Supports two formats:

    1) OECD QNA CSV (your current file):
         - column 'period' like '1995-Q1'
         - one numeric column with the deflator index (rebased).
       We:
         - parse year from 'period'
         - rebase to base_year=2019
         - average by year → [year, deflator]

    2) Simple [year, deflator] table:
         - column 'year'
         - at least one numeric column → used as deflator (assumed already rebased)
    """
    print(f"[load_deflator] Looking for deflator at: {defl_path}")
    if not defl_path.exists():
        raise FileNotFoundError(
            f"Deflator file not found at: {defl_path}\n"
            "Check that the file is in Data/K/."
        )

    d = pd.read_csv(defl_path)

    # ------------------------------------------------------------------
    # Case 1: OECD QNA file with 'period' like '1995-Q1'
    # ------------------------------------------------------------------
    if "period" in d.columns:
        # Identify index column (there should be exactly one numeric col)
        num_cols = d.select_dtypes(include="number").columns.tolist()
        if len(num_cols) == 0:
            # fall back: treat the non-period column as numeric
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

        # Parse year from 'period' (e.g. '1995-Q1')
        per = d["period"].astype(str)
        d["year"] = per.str.slice(0, 4).astype(int)

        # Annual mean of the index, then rebase
        annual_raw = d.groupby("year", as_index=False)["index_raw"].mean()
        base_year = 2019
        if base_year not in annual_raw["year"].values:
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
            f"[load_deflator] Parsed OECD QNA deflator "
            f"for years {annual['year'].min()}–{annual['year'].max()} "
            f"(base_year={base_year}, deflator(base)=1.0)"
        )
        return annual

    # ------------------------------------------------------------------
    # Case 2: simple [year, deflator] table
    # ------------------------------------------------------------------
    if "year" not in d.columns:
        for alt in ["Year", "YEAR"]:
            if alt in d.columns:
                d = d.rename(columns={alt: "year"})
                break

    if "year" not in d.columns:
        raise ValueError(
            f"'year' column not found in deflator file. Columns: {list(d.columns)}"
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


def attach_real_values(fin_df: pd.DataFrame, defl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge financials with deflator and create real-valued variables.

    - K_book_real: real net PPE at year-end
    - I_real: real annual additions (investment)
    - depr_real: real depreciation
    - disposals_real: real disposals (if present)
    """
    df = fin_df.merge(defl_df, on="year", how="left")
    if df["deflator"].isna().any():
        missing_years = df.loc[df["deflator"].isna(), "year"].tolist()
        raise ValueError(f"Missing deflator for years: {missing_years}")

    df["K_book_real"] = df["ppe_net_nom"] / df["deflator"]
    df["I_real"] = df["additions_nom"] / df["deflator"]
    df["depr_real"] = df["depr_nom"] / df["deflator"]

    if "disposals_nom" in df.columns:
        df["disposals_real"] = df["disposals_nom"] / df["deflator"]

    return df


# ====================================================================
# Driver for Steps 1.1–1.3
# ====================================================================

def run_step1(depr_scenario="central"):
    # 1.1: load & standardize financials
    fin = load_financials(FIN_PATH)

    # 1.2: load deflator and create real values
    defl = load_deflator(DEFLATOR_PATH)
    fin_real = attach_real_values(fin, defl)

    # 1.3: pick depreciation rate δ via toggle
    delta = get_delta(depr_scenario)

    # Save full real-valued financials
    fin_real.to_csv(OUT_FIN_REAL_PATH, sep="\t", index=False)

    # Also save a small sample (first 10 rows) for quick inspection
    fin_real.head(10).to_csv(OUT_DEBUG_SAMPLE_PATH, index=False)

    meta = {
        "depreciation_scenario": depr_scenario,
        "delta": delta,
        "scenarios_available": DEPR_SCENARIOS,
    }
    with OUT_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n[run_step1] Step 1 complete.")
    print(f"  - Real-valued financials saved to: {OUT_FIN_REAL_PATH}")
    print(f"  - Sample (first 10 rows) saved to: {OUT_DEBUG_SAMPLE_PATH}")
    print(f"  - Depreciation choice saved to:   {OUT_META_PATH}")
    print(f"  - Chosen δ = {delta:.4f} ({depr_scenario})")

    print("\n[run_step1] Preview of real-valued financials:")
    print(fin_real.head())

    return fin_real, delta


if __name__ == "__main__":
    # Change this to "low", "central", "high", or a numeric δ if you want
    run_step1(depr_scenario="central")

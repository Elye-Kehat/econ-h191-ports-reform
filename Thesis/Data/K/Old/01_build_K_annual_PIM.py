from pathlib import Path
import json
import pandas as pd
import numpy as np


# =====================================================================
# Paths and configuration
# =====================================================================

DATA_DIR = Path(__file__).resolve().parent

# Step-1 outputs
FIN_REAL_PATH = DATA_DIR / "00_haifa_financials_step1_real.tsv"
DEPR_META_PATH = DATA_DIR / "00_depreciation_choice.json"

# Step-2 outputs (annual PIM backbone)
OUT_PIM_PATH = DATA_DIR / "01_K_B_annual_Haifa_PIM.tsv"
OUT_PIM_SAMPLE_PATH = DATA_DIR / "01_K_B_annual_Haifa_PIM_sample.csv"
OUT_PIM_CONFIG_PATH = DATA_DIR / "01_PIM_config.json"

# Toggle: whether to include intangibles in PIM investment (if the CF line exists)
INCLUDE_INTANGIBLES_IN_PIM = False


# =====================================================================
# Helpers: load inputs
# =====================================================================

def load_financials_step1(fin_path: Path) -> pd.DataFrame:
    """
    Load the real-valued financials produced by 00_build_K_step1.py.

    Expected key columns (in addition to metadata like company, source_pdf, etc.):

      - year               : integer financial year
      - company            : company name (Haifa Port Company (legacy))
      - K_book_real        : real net PPE at year-end (wealth stock)
      - I_real             : real cash-flow line (purchase_of_fixed_assets / deflator),
                             signed according to the CF convention
      - depr_real          : real depreciation expense (for information)
      - disposals_real     : real value of disposals (for information / netting)
      - deflator           : price index used in step 1

    We will then:
      - keep only the columns needed for PIM and provenance
      - coerce numeric types for core value columns
      - sort by (company, year)
    """
    print(f"[load_financials_step1] Reading real-valued financials from: {fin_path}")
    if not fin_path.exists():
        raise FileNotFoundError(
            f"Real financials file not found at: {fin_path}\n"
            "Run 00_build_K_step1.py first so that '00_haifa_financials_step1_real.tsv' exists."
        )

    df = pd.read_csv(fin_path, sep="\t")

    required_cols = ["year", "company", "K_book_real", "I_real", "depr_real", "disposals_real"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in 00_haifa_financials_step1_real.tsv: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    # Keep only the columns we actually need for annual PIM plus some provenance
    keep_cols = [
        "company",
        "year",
        "K_book_real",
        "I_real",   # signed CF-based real series from step 1
        "depr_real",
        "disposals_real",
        "ppe_net_nom",
        "additions_nom",
        "depr_nom",
        "disposals_nom",
        "deflator",
        "purchase_of_intangible_assets_thousands_nis_cashflow",
        "source_pdf",
        "source_pages",
        "revaluation_or_basis_flag",
        "units_note",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Ensure numeric types on key value columns (deflator included)
    for col in ["year", "K_book_real", "I_real", "depr_real", "disposals_real", "deflator"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["company", "year"]).reset_index(drop=True)

    # For now we expect a single company; if more appear later this will still behave sensibly.
    companies = df["company"].dropna().unique().tolist()
    print(f"[load_financials_step1] Found companies: {companies}")
    print(f"[load_financials_step1] Years present (raw): {sorted(df['year'].dropna().unique())}")

    return df


def load_depreciation_meta(meta_path: Path):
    """
    Load the depreciation choice produced by 00_build_K_step1.py.

    The JSON produced by 00_build_K_step1.py is expected to look like:

        {
          "depreciation_scenario": "central",
          "delta": 0.06,
          "scenarios_available": {
              "low": 0.04,
              "central": 0.06,
              "high": 0.08
          }
        }

    We will:
      - use 'scenarios_available' to compute PIM paths for all scenarios
      - report which scenario was used in step 1 (for traceability)
    """
    print(f"[load_depreciation_meta] Reading depreciation metadata from: {meta_path}")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Depreciation metadata not found at: {meta_path}\n"
            "Run 00_build_K_step1.py first so that '00_depreciation_choice.json' exists."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    scenario_chosen = meta.get("depreciation_scenario", "central")
    delta_chosen = meta.get("delta")
    scenarios_available = meta.get("scenarios_available")

    if scenarios_available is None:
        # Fall back to using only the chosen scenario if older metadata format
        if delta_chosen is None:
            raise ValueError(
                "Depreciation metadata missing both 'scenarios_available' and 'delta'.\n"
                "Delete 00_depreciation_choice.json and rerun 00_build_K_step1.py."
            )
        scenarios_available = {scenario_chosen: float(delta_chosen)}

    # Coerce to float just in case
    scenarios_available = {k: float(v) for k, v in scenarios_available.items()}

    print(f"[load_depreciation_meta] Scenario chosen in step 1: '{scenario_chosen}' (δ = {delta_chosen})")
    print(f"[load_depreciation_meta] Scenarios available for PIM: {scenarios_available}")

    return scenario_chosen, delta_chosen, scenarios_available


# =====================================================================
# Core PIM logic
# =====================================================================

def build_annual_pim(df_real: pd.DataFrame, depr_scenarios: dict) -> pd.DataFrame:
    """
    Construct annual PIM capital stock series for each depreciation scenario.

    Inputs
    ------
    df_real : DataFrame
        Real-valued financials (output of step 1) with at least:
          - company
          - year
          - K_book_real       (wealth stock at year-end, in real terms)
          - I_real            (real cash-flow line, signed)
          - disposals_real    (real value of disposals; may be NaN)
          - deflator          (real deflator used in step 1)
    depr_scenarios : dict
        Mapping from scenario name (e.g. "low", "central", "high") to
        an annual depreciation rate δ (float).

    Output
    ------
    DataFrame with one row per (company, year) including:
      - K_book_real
      - I_real_cf_signed  : real CF series (can be negative)
      - I_real_pim        : non-negative PIM investment (gross acquisitions)
      - disposals_real, depr_real
      - K_PIM_real_{scenario} for each scenario
      - gap_book_minus_PIM_{scenario} = K_book_real - K_PIM_real_{scenario}
      - flows_imputed_flag : True if either (a) I_real_pim or disposals_real
                             were missing and replaced with 0, or
                             (b) there is a multi-year gap and we implicitly
                                 assume zero flows in the intermediate years.
      - gap_years_from_prev: number of years between this year and previous
                             observation (e.g. 2 means we skipped one year,
                             like 2021→2023 with no 2022 K_book_real row).
    """
    if df_real.empty:
        raise ValueError("df_real is empty; nothing to do.")

    df_real = df_real.sort_values(["company", "year"]).reset_index(drop=True)

    required_cols = ["company", "year", "K_book_real", "I_real", "disposals_real", "depr_real", "deflator"]
    missing = [c for c in required_cols if c not in df_real.columns]
    if missing:
        raise ValueError(
            "df_real is missing required columns for PIM: "
            f"{missing}. Available columns: {list(df_real.columns)}"
        )

    # -----------------------------------------------------------------
    # Build PIM investment series I_real_pim:
    #  - Start from I_real (real CF line, signed).
    #  - Take absolute value to get gross acquisitions.
    #  - Optionally add intangibles (deflated) if requested.
    # -----------------------------------------------------------------
    df_real = df_real.copy()

    # Signed CF series for reference
    df_real["I_real_cf_signed"] = df_real["I_real"]

    # Base PIM investment: gross acquisitions from CF (abs of real CF line)
    I_gross = df_real["I_real"].abs()

    # Optional: add intangibles (deflated) if column exists and toggle is on
    if INCLUDE_INTANGIBLES_IN_PIM and "purchase_of_intangible_assets_thousands_nis_cashflow" in df_real.columns:
        int_nom = pd.to_numeric(
            df_real["purchase_of_intangible_assets_thousands_nis_cashflow"],
            errors="coerce"
        ).fillna(0.0)
        int_real = int_nom / df_real["deflator"]
        df_real["intangibles_real_pim"] = int_real
        I_gross = I_gross + int_real.abs()
        print("[build_annual_pim] Including intangibles in PIM investment.")

    df_real["I_real_pim"] = I_gross

    results = []

    for company, df_c in df_real.groupby("company"):
        df_c = df_c.sort_values("year").reset_index(drop=True)

        # Drop years with no wealth stock (e.g. 2017 scaffolding row with K_book_real NaN).
        df_c = df_c[df_c["K_book_real"].notna()].copy()
        if df_c.empty:
            continue

        years = df_c["year"].tolist()
        print(f"[build_annual_pim] Company '{company}': years with K_book_real = {years}")

        # -----------------------------------------------------------------
        # Initialise PIM: anchor the first year to the observed wealth stock.
        # -----------------------------------------------------------------
        first_row = df_c.iloc[0]
        first_year = int(first_row["year"])
        K_book0 = float(first_row["K_book_real"])

        # Running PIM stock for each scenario.
        K_prev = {name: K_book0 for name in depr_scenarios.keys()}

        # First-year record: PIM = book (no recursion because we don't know K_{t-1}).
        base_row = {
            "company": company,
            "year": first_year,
            "K_book_real": K_book0,
            "I_real_cf_signed": float(first_row["I_real_cf_signed"]) if not pd.isna(first_row["I_real_cf_signed"]) else np.nan,
            "I_real_pim": float(first_row["I_real_pim"]) if not pd.isna(first_row["I_real_pim"]) else np.nan,
            "disposals_real": float(first_row["disposals_real"]) if not pd.isna(first_row["disposals_real"]) else np.nan,
            "depr_real": float(first_row["depr_real"]) if not pd.isna(first_row["depr_real"]) else np.nan,
            "flows_imputed_flag": False,
            "gap_years_from_prev": 0,
        }
        # Alias for convenience (PIM investment series)
        base_row["I_real"] = base_row["I_real_pim"]

        for scen, delta in depr_scenarios.items():
            base_row[f"K_PIM_real_{scen}"] = K_prev[scen]
            base_row[f"gap_book_minus_PIM_{scen}"] = base_row["K_book_real"] - base_row[f"K_PIM_real_{scen}"]
        results.append(base_row)

        # -----------------------------------------------------------------
        # Subsequent years: apply PIM recursion, allowing for multi-year gaps.
        # -----------------------------------------------------------------
        for idx in range(1, len(df_c)):
            row = df_c.iloc[idx]
            year = int(row["year"])
            K_book = float(row["K_book_real"])

            I_pim = row["I_real_pim"]
            D_real = row["disposals_real"]

            flows_imputed = False
            if pd.isna(I_pim):
                I_used = 0.0
                flows_imputed = True
            else:
                I_used = float(I_pim)

            if pd.isna(D_real):
                D_used = 0.0
                flows_imputed = True or flows_imputed
            else:
                D_used = float(D_real)

            gap_years = year - int(df_c.iloc[idx - 1]["year"])
            if gap_years < 1:
                raise ValueError(
                    f"Non-positive year gap encountered between {df_c.iloc[idx - 1]['year']} and {year} "
                    f"for company '{company}'."
                )

            # If there is a multi-year gap (e.g. 2021→2023),
            # we implicitly assume zero flows in the missing intermediate year(s).
            # Mark this row as having imputed flows, even if the current year's
            # I_pim / D_real are observed.
            if gap_years > 1:
                flows_imputed = True

            out_row = {
                "company": company,
                "year": year,
                "K_book_real": K_book,
                "I_real_cf_signed": float(row["I_real_cf_signed"]) if not pd.isna(row["I_real_cf_signed"]) else np.nan,
                "I_real_pim": float(I_pim) if not pd.isna(I_pim) else np.nan,
                "disposals_real": float(D_real) if not pd.isna(D_real) else np.nan,
                "depr_real": float(row["depr_real"]) if not pd.isna(row["depr_real"]) else np.nan,
                "flows_imputed_flag": bool(flows_imputed),
                "gap_years_from_prev": int(gap_years),
            }
            # Alias: I_real = I_real_pim (PIM investment series)
            out_row["I_real"] = out_row["I_real_pim"]

            for scen, delta in depr_scenarios.items():
                # Apply depreciation for any missing intermediate years (with zero flows)
                K_tmp = K_prev[scen]
                for _ in range(1, gap_years):
                    K_tmp = (1.0 - delta) * K_tmp

                # Final step: depreciation plus observed flows in 'year'
                K_new = (1.0 - delta) * K_tmp + I_used - D_used
                K_prev[scen] = K_new

                out_row[f"K_PIM_real_{scen}"] = K_new
                out_row[f"gap_book_minus_PIM_{scen}"] = K_book - K_new

            results.append(out_row)

    if not results:
        raise ValueError("No PIM results were generated; check that df_real contains valid data.")

    df_pim = pd.DataFrame(results)
    df_pim = df_pim.sort_values(["company", "year"]).reset_index(drop=True)

    return df_pim


# =====================================================================
# Main entry point
# =====================================================================

def run_annual_pim():
    """
    Orchestrate step 2:
      - load real-valued financials (step 1 output)
      - load depreciation scenarios (step 1 metadata)
      - build annual PIM K series for each scenario
      - save outputs to TSV/CSV and JSON
    """
    df_real = load_financials_step1(FIN_REAL_PATH)
    scenario_chosen, delta_chosen, depr_scenarios = load_depreciation_meta(DEPR_META_PATH)

    df_pim = build_annual_pim(df_real, depr_scenarios)

    # Save full table
    df_pim.to_csv(OUT_PIM_PATH, sep="\t", index=False)
    print(f"[run_annual_pim] Saved annual PIM table to: {OUT_PIM_PATH}")

    # Save a small preview for quick inspection
    df_pim.head(10).to_csv(OUT_PIM_SAMPLE_PATH, index=False)
    print(f"[run_annual_pim] Saved sample (first 10 rows) to: {OUT_PIM_SAMPLE_PATH}")

    # Save configuration / metadata for traceability
    pim_meta = {
        "description": "Annual PIM capital stock for Haifa Port Company (legacy), Track B.",
        "input_financials_file": str(FIN_REAL_PATH.name),
        "input_depreciation_meta": str(DEPR_META_PATH.name),
        "depreciation_scenario_step1": scenario_chosen,
        "delta_step1": delta_chosen,
        "depreciation_scenarios_used": depr_scenarios,
        "include_intangibles_in_pim": INCLUDE_INTANGIBLES_IN_PIM,
        "output_pim_file": str(OUT_PIM_PATH.name),
        "output_pim_sample_file": str(OUT_PIM_SAMPLE_PATH.name),
    }

    with open(OUT_PIM_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(pim_meta, f, indent=2, ensure_ascii=False)

    print(f"[run_annual_pim] Saved PIM configuration metadata to: {OUT_PIM_CONFIG_PATH}")

    # Simple console preview
    print("\n[run_annual_pim] Preview of annual PIM results:")
    print(df_pim)


if __name__ == "__main__":
    run_annual_pim()

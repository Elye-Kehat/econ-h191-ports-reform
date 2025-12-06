from pathlib import Path
import json
import pandas as pd
import numpy as np


# =====================================================================
# Paths and configuration
# =====================================================================

DATA_DIR = Path(__file__).resolve().parent

# Step-1 / Step-2 inputs
ANNUAL_PIM_PATH = DATA_DIR / "01_K_B_annual_Haifa_PIM.tsv"

# Step-3 outputs (monthly Path A: PIM + linear)
OUT_MONTHLY_PATH = DATA_DIR / "02_K_B_monthly_Haifa_PIM_lin.tsv"
OUT_MONTHLY_SAMPLE_PATH = DATA_DIR / "02_K_B_monthly_Haifa_PIM_lin_sample.csv"
OUT_MONTHLY_META_PATH = DATA_DIR / "02_PIM_lin_config.json"


# =====================================================================
# Helpers: load annual PIM backbone
# =====================================================================

def load_annual_pim(pim_path: Path) -> pd.DataFrame:
    """
    Load the annual PIM capital stock table produced by 01_build_K_annual_PIM.py.

    Expected columns (from 01_... script):
      - company
      - year
      - K_book_real
      - I_real_cf_signed
      - I_real_pim
      - disposals_real
      - depr_real
      - flows_imputed_flag
      - gap_years_from_prev
      - K_PIM_real_low
      - K_PIM_real_central
      - K_PIM_real_high
      - gap_book_minus_PIM_low / central / high

    This function just ensures the basics and returns the DataFrame sorted by (company, year).
    """
    print(f"[load_annual_pim] Reading annual PIM from: {pim_path}")
    if not pim_path.exists():
        raise FileNotFoundError(
            f"Annual PIM file not found at: {pim_path}\n"
            "Run 01_build_K_annual_PIM.py first so that '01_K_B_annual_Haifa_PIM.tsv' exists."
        )

    df = pd.read_csv(pim_path, sep="\t")

    required = ["company", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Annual PIM file is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).sort_values(["company", "year"]).reset_index(drop=True)

    print(f"[load_annual_pim] Companies: {df['company'].dropna().unique().tolist()}")
    print(f"[load_annual_pim] Years in PIM backbone: {sorted(df['year'].dropna().unique().tolist())}")

    return df


# =====================================================================
# Core logic: build monthly Path A (PIM + linear)
# =====================================================================

def build_monthly_pim_linear(df_pim: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a monthly K path (Path A) by linearly interpolating between
    annual PIM stocks at December of each year.

    For each company:
      1. Take annual K_PIM_real_* at year-end y.
      2. Create anchor dates at 1 Dec of each year y.
      3. Build a monthly date range from first to last anchor.
      4. Interpolate K in levels between anchors for each scenario:
           K_PIM_lin_{scenario}
      5. Add flags:
           - imputed_2022: 1 if calendar year == 2022, else 0
           - flows_imputed_annual: annual PIM flows_imputed_flag mapped to that year
           - gap_years_from_prev_annual: annual gap_years_from_prev mapped to that year

    Output columns:
      - port
      - company
      - operator_or_owner
      - month (Timestamp)
      - year (int)
      - imputed_2022
      - flows_imputed_annual
      - gap_years_from_prev_annual
      - K_PIM_lin_low / central / high
      - K_PIM_real_low / central / high (anchor values, mostly filled on Dec)
      - any remaining columns from the merge (for traceability)
    """
    if df_pim.empty:
        raise ValueError("df_pim is empty; nothing to do.")

    # Identify the annual PIM stock columns (scenarios)
    scenario_cols = [c for c in df_pim.columns if c.startswith("K_PIM_real_")]
    if not scenario_cols:
        raise ValueError(
            "No 'K_PIM_real_*' columns found in annual PIM file.\n"
            "Check that 01_build_K_annual_PIM.py created K_PIM_real_low/central/high."
        )

    # Scenario names, e.g. ["low", "central", "high"]
    scenarios = [c.replace("K_PIM_real_", "") for c in scenario_cols]

    df_pim = df_pim.sort_values(["company", "year"]).reset_index(drop=True)

    monthly_frames = []

    for company, df_c in df_pim.groupby("company"):
        df_c = df_c.sort_values("year").reset_index(drop=True)
        if df_c.empty:
            continue

        # -----------------------------------------------------------------
        # Build anchor table: one row per year with December date
        # -----------------------------------------------------------------
        # We expect flows_imputed_flag and gap_years_from_prev from 01_...
        missing_flags = [c for c in ["flows_imputed_flag", "gap_years_from_prev"] if c not in df_c.columns]
        if missing_flags:
            raise ValueError(
                f"Annual PIM DataFrame for company '{company}' is missing columns: {missing_flags}"
            )

        anchors = df_c[["year"] + scenario_cols + ["flows_imputed_flag", "gap_years_from_prev"]].copy()
        anchors["company"] = company

        # Anchor at December 1 of each year y
        anchors["month"] = pd.to_datetime(anchors["year"].astype(int).astype(str) + "-12-01")

        start_month = anchors["month"].min()
        end_month = anchors["month"].max()

        # Monthly skeleton from first to last December
        months = pd.date_range(start=start_month, end=end_month, freq="MS")
        monthly = pd.DataFrame({"month": months})
        monthly["company"] = company

        # Merge anchors into the monthly skeleton
        monthly = monthly.merge(
            anchors[["company", "month"] + scenario_cols + ["year", "flows_imputed_flag", "gap_years_from_prev"]],
            on=["company", "month"],
            how="left",
        )

        # Calendar year for each month
        monthly["year"] = monthly["month"].dt.year

        # Map annual flags down to months (by calendar year)
        # flows_imputed_flag: True/False at the annual PIM level
        annual_flag_map = df_c.set_index("year")["flows_imputed_flag"].to_dict()
        annual_gap_map = df_c.set_index("year")["gap_years_from_prev"].to_dict()

        monthly["flows_imputed_annual"] = monthly["year"].map(annual_flag_map).fillna(False).astype(bool)
        monthly["gap_years_from_prev_annual"] = monthly["year"].map(annual_gap_map).fillna(0).astype(int)

        # Use DatetimeIndex so interpolate(method="time") works
        monthly = monthly.set_index("month")

        # -----------------------------------------------------------------
        # Interpolate each scenario in levels between December anchors
        # -----------------------------------------------------------------
        for scen_name in scenarios:
            anchor_col = f"K_PIM_real_{scen_name}"
            lin_col = f"K_PIM_lin_{scen_name}"

            if anchor_col not in monthly.columns:
                raise ValueError(f"Expected anchor column '{anchor_col}' not found in monthly DataFrame.")

            # Linear interpolation in time between anchor points
            monthly[lin_col] = monthly[anchor_col].interpolate(method="time")

        # Restore month as a column
        monthly = monthly.reset_index()

        # -----------------------------------------------------------------
        # Add port/operator labels and 2022 imputation flag
        # -----------------------------------------------------------------
        monthly["port"] = "Haifa"
        monthly["operator_or_owner"] = "Haifa Port Company (legacy)"

        # Mark 2022 as imputed structurally (no direct annual K anchor)
        monthly["imputed_2022"] = (monthly["year"] == 2022).astype(int)

        monthly_frames.append(monthly)

    if not monthly_frames:
        raise ValueError("No monthly rows generated; check annual PIM input.")

    df_monthly = pd.concat(monthly_frames, ignore_index=True)

    # ---------------------------------------------------------------------
    # Column ordering: front matter, flags, PIM-lin K, anchor K, everything else
    # ---------------------------------------------------------------------
    cols_front = ["port", "company", "operator_or_owner", "month", "year"]
    cols_flags = ["imputed_2022", "flows_imputed_annual", "gap_years_from_prev_annual"]

    scenario_lin_cols = [c for c in df_monthly.columns if c.startswith("K_PIM_lin_")]
    scenario_anchor_cols = [c for c in df_monthly.columns if c.startswith("K_PIM_real_")]

    other_cols = [
        c for c in df_monthly.columns
        if c not in cols_front + cols_flags + scenario_lin_cols + scenario_anchor_cols
    ]

    ordered_cols = cols_front + cols_flags + scenario_lin_cols + scenario_anchor_cols + other_cols
    df_monthly = df_monthly[ordered_cols].sort_values(["company", "month"]).reset_index(drop=True)

    return df_monthly


# =====================================================================
# Main entry point
# =====================================================================

def run_monthly_pim_linear():
    """
    Orchestrate Step 2 (Path A: PIM + linear):

      - Load annual PIM backbone from 01_K_B_annual_Haifa_PIM.tsv
      - Build monthly PIM-linear K for each depreciation scenario
      - Save:
          02_K_B_monthly_Haifa_PIM_lin.tsv          (full monthly series)
          02_K_B_monthly_Haifa_PIM_lin_sample.csv   (first 24 rows)
          02_PIM_lin_config.json                    (metadata)
    """
    df_pim = load_annual_pim(ANNUAL_PIM_PATH)
    df_monthly = build_monthly_pim_linear(df_pim)

    # Full output
    df_monthly.to_csv(OUT_MONTHLY_PATH, sep="\t", index=False)
    print(f"[run_monthly_pim_linear] Saved monthly PIM-linear K to: {OUT_MONTHLY_PATH}")

    # Short preview for quick inspection
    df_monthly.head(24).to_csv(OUT_MONTHLY_SAMPLE_PATH, index=False)
    print(f"[run_monthly_pim_linear] Saved sample (first 24 rows) to: {OUT_MONTHLY_SAMPLE_PATH}")

    # Metadata / config for traceability
    meta = {
        "description": (
            "Monthly K series for Haifa Port Company (legacy) based on annual PIM "
            "with linear interpolation between December year-end stocks (Path A)."
        ),
        "input_annual_pim_file": ANNUAL_PIM_PATH.name,
        "output_monthly_file": OUT_MONTHLY_PATH.name,
        "output_monthly_sample_file": OUT_MONTHLY_SAMPLE_PATH.name,
        "port": "Haifa",
        "company": "Haifa Port Company (legacy)",
        "interpolation_method": "linear in levels between December annual PIM stocks using pandas.interpolate(method='time')",
        "flags": {
            "imputed_2022_rule": "imputed_2022 = 1 for all months with calendar year 2022; 0 otherwise.",
            "flows_imputed_annual": "annual flows_imputed_flag from PIM, mapped to months by year.",
            "gap_years_from_prev_annual": "annual gap_years_from_prev from PIM, mapped to months by year.",
        },
    }

    with open(OUT_MONTHLY_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[run_monthly_pim_linear] Saved config metadata to: {OUT_MONTHLY_META_PATH}")

    # Console preview
    print("\n[run_monthly_pim_linear] Preview of monthly PIM-linear K:")
    print(df_monthly.head(24))


if __name__ == "__main__":
    run_monthly_pim_linear()

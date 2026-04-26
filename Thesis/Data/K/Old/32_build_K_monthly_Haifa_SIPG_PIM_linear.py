# 32_build_K_monthly_Haifa_SIPG_PIM_linear.py
#
# Step 3 for SIPG Haifa Bayport K-series:
#   - Read annual SIPG PIM K table (31_K_B_annual_Haifa_SIPG_PIM.tsv).
#   - Treat K_PIM_real_* as December-end anchors for each year.
#   - Interpolate to a monthly series:
#         * zero K before first positive annual anchor
#         * log-linear interpolation between positive annual anchors
#   - Save:
#       * 32_K_B_monthly_Haifa_SIPG_PIM_lin.tsv
#       * 32_K_B_monthly_Haifa_SIPG_PIM_lin_sample.csv
#       * 32_SIPG_Bayport_PIM_lin_meta.json
#
# This is analogous to the IPC PIM-lin script but simplified and
# adapted to the SIPG-specific annual PIM schema.

from pathlib import Path
import json

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).resolve().parent

SIPG_ANNUAL_PIM_PATH = DATA_DIR / "31_K_B_annual_Haifa_SIPG_PIM.tsv"

OUT_MONTHLY_PATH = DATA_DIR / "32_K_B_monthly_Haifa_SIPG_PIM_lin.tsv"
OUT_SAMPLE_PATH = DATA_DIR / "32_K_B_monthly_Haifa_SIPG_PIM_lin_sample.csv"
OUT_META_PATH = DATA_DIR / "32_SIPG_Bayport_PIM_lin_meta.json"


def load_sipg_annual_pim(path: Path) -> pd.DataFrame:
    """Load annual SIPG PIM table."""
    if not path.exists():
        raise FileNotFoundError(
            f"SIPG annual PIM file not found at {path}. "
            "Run 31_build_K_annual_SIPG_Bayport_PIM.py first."
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


def build_monthly_k(df_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Construct monthly SIPG K series from annual PIM anchors.

    - K_PIM_real_* are treated as December-end stocks.
    - For each scenario:
         * K=0 for all months before the first year with positive K.
         * From the first positive December onward, we log-linearly
           interpolate between annual anchors (December of each year).
    """
    df = df_annual.copy().sort_values("year").reset_index(drop=True)

    years = df["year"].to_numpy()
    start_year = int(years[0])
    end_year = int(years[-1])

    # Full monthly date range (Jan of first year to Dec of last year)
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    out = pd.DataFrame({"month": dates})
    out["year"] = out["month"].dt.year

    annual_lookup = df.set_index("year")

    # Identify scenario names from columns like K_PIM_real_low, K_PIM_real_central, etc.
    scenario_names = []
    for col in df.columns:
        if col.startswith("K_PIM_real_"):
            scenario_names.append(col.replace("K_PIM_real_", ""))
    scenario_names = sorted(set(scenario_names))

    if not scenario_names:
        raise ValueError(
            "No K_PIM_real_* columns found in SIPG annual PIM table. "
            f"Columns: {list(df.columns)}"
        )

    def interpolate_scenario(annual_series: pd.Series) -> pd.Series:
        """
        Interpolate annual series (indexed by year) to monthly.

        - Set K=0 for all months before the first positive annual K.
        - For years with positive K, treat December K as anchors and
          log-linearly interpolate between anchors.
        """
        # Build result series indexed by full monthly dates
        result = pd.Series(0.0, index=dates)

        # Annual series indexed by year
        s = annual_series.copy()
        s.index = s.index.astype(int)

        years_arr = s.index.to_numpy()
        vals = s.to_numpy()

        # Identify first year with positive K
        pos_mask = vals > 0
        if not pos_mask.any():
            # All zeros: return zeros
            return result

        first_pos_year = int(years_arr[pos_mask][0])

        # Anchors: all years with positive K (from first_pos_year onwards)
        anchor_mask = years_arr >= first_pos_year
        anchor_years = years_arr[anchor_mask]
        anchor_vals = vals[anchor_mask]

        # Build monthly index from Dec(first_pos_year) to Dec(end_year)
        anchor_dates = pd.to_datetime([f"{int(y)}-12-01" for y in anchor_years])
        anchor = pd.DataFrame({"date": anchor_dates, "K": anchor_vals}).set_index("date")

        full_idx = pd.date_range(anchor.index.min(), f"{end_year}-12-01", freq="MS")
        full = anchor.reindex(full_idx)

        # Log-linear interpolation for positive segment
        full["logK"] = np.log(full["K"])
        full["logK"] = full["logK"].interpolate(method="time")
        full["logK"] = full["logK"].ffill().bfill()
        full["K_interp"] = np.exp(full["logK"])

        # Assign interpolated values for months >= Dec(first_pos_year)
        result.loc[full.index] = full["K_interp"].to_numpy()

        # Months before Dec(first_pos_year) remain 0.
        return result

    # Compute monthly K_PIM_lin_* for each scenario
    for name in scenario_names:
        col = f"K_PIM_real_{name}"
        if col not in annual_lookup.columns:
            raise ValueError(
                f"Column '{col}' not found in annual PIM table. "
                f"Available columns: {list(annual_lookup.columns)}"
            )
        series_y = annual_lookup[col]
        K_monthly = interpolate_scenario(series_y)
        out[f"K_PIM_lin_{name}"] = K_monthly.to_numpy()

    # Attach company / port metadata
    # (Assume a single company per table; if multiple, you can group later)
    out["company"] = "SIPG Haifa Bayport"
    out["port"] = "Haifa"
    out["operator_or_owner"] = "SIPG Haifa Bayport (private operator)"

    # Reorder columns
    k_cols = [c for c in out.columns if c.startswith("K_PIM_lin_")]
    cols = ["port", "company", "operator_or_owner", "month", "year"] + k_cols
    out = out[cols].sort_values(["company", "month"]).reset_index(drop=True)

    return out


def run_sipg_monthly_pim_lin() -> pd.DataFrame:
    """Build monthly SIPG K series from annual PIM and write outputs."""
    print("\n[32_PIM_lin] Loading annual SIPG PIM table...")
    df_annual = load_sipg_annual_pim(SIPG_ANNUAL_PIM_PATH)
    print(f"  - Rows: {len(df_annual)}")
    print(f"  - Years: {df_annual['year'].min()}–{df_annual['year'].max()}")

    df_monthly = build_monthly_k(df_annual)

    df_monthly.to_csv(OUT_MONTHLY_PATH, sep="\t", index=False)
    df_monthly.head(10).to_csv(OUT_SAMPLE_PATH, sep="\t", index=False)

    print(f"\n[32_PIM_lin] Monthly SIPG K (PIM-lin) saved to: {OUT_MONTHLY_PATH}")
    print(f"             Sample (first 10 rows): {OUT_SAMPLE_PATH}")

    # Simple meta file
    k_cols = [c for c in df_monthly.columns if c.startswith("K_PIM_lin_")]
    meta = {
        "description": (
            "Monthly PIM-linear K for SIPG Haifa Bayport (real 2019 NIS, thousands). "
            "K_PIM_real_* treated as December anchors; zero K before first positive "
            "annual anchor; log-linear interpolation thereafter."
        ),
        "annual_pim_source": str(SIPG_ANNUAL_PIM_PATH.name),
        "columns_K": k_cols,
        "port": "Haifa",
        "company": "SIPG Haifa Bayport",
    }
    with open(OUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[32_PIM_lin] Meta written to: {OUT_META_PATH}")
    print("\n[32_PIM_lin] Preview:")
    print(df_monthly.head())

    return df_monthly


if __name__ == "__main__":
    run_sipg_monthly_pim_lin()

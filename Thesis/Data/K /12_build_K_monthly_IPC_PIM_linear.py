# 12_build_K_monthly_IPC_PIM_linear.py
#
# IPC-specific monthly K construction (Path A: PIM + log-linear interpolation).
#
# Purpose:
#   - Read IPC annual PIM output (11_K_B_annual_Haifa_IPC_PIM.tsv).
#   - Construct a monthly date index for Haifa.
#   - Log-linearly interpolate K_PIM_real_* between annual anchors to get
#     monthly K_PIM_lin_* series.
#   - Save:
#       * 12_K_B_monthly_Haifa_IPC_PIM_lin.tsv
#       * 12_K_B_monthly_Haifa_IPC_PIM_lin_sample.csv
#       * 12_IPC_PIM_lin_meta.json

from pathlib import Path
import json

import pandas as pd
import numpy as np


# --------------------------------------------------------------------
# File paths (adjust if needed)
# --------------------------------------------------------------------

IPC_ANNUAL_PIM_PATH   = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /11_K_B_annual_Haifa_IPC_PIM.tsv")
OUT_MONTHLY_PATH      = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /12_K_B_monthly_Haifa_IPC_PIM_lin.tsv")
OUT_MONTHLY_SAMPLE    = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /12_K_B_monthly_Haifa_IPC_PIM_lin_sample.csv")
OUT_PIM_LIN_META_PATH = Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Data/K /12_IPC_PIM_lin_meta.json")

PORT_NAME        = "Haifa"
IPC_COMPANY_NAME = "Israel Ports Company (Haifa cluster)"
IPC_OWNER_ROLE   = "Israel Ports Company (landlord / infrastructure owner)"


# --------------------------------------------------------------------
# Helper: load annual IPC PIM table
# --------------------------------------------------------------------

def load_ipc_annual_pim(path: Path) -> pd.DataFrame:
    """Load IPC annual PIM K table.

    Expected columns:

        company, year,
        K_book_real, I_real, depr_real, disposals_real,
        K_PIM_real_low, K_PIM_real_central, K_PIM_real_high,
        gap_book_minus_PIM_central,
        flows_imputed_flag, gap_years_from_prev
    """

    df = pd.read_csv(path, sep="\t")

    required_cols = [
        "company",
        "year",
        "K_PIM_real_low",
        "K_PIM_real_central",
        "K_PIM_real_high",
        "flows_imputed_flag",
        "gap_years_from_prev",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # Keep only the IPC company (even if extra rows exist by accident)
    df = df[df["company"] == IPC_COMPANY_NAME].copy()

    return df.sort_values("year")


# --------------------------------------------------------------------
# Helper: build monthly K via log-linear interpolation
# --------------------------------------------------------------------

def build_monthly_k(df_annual: pd.DataFrame) -> pd.DataFrame:
    """Construct IPC monthly K series from annual PIM anchors.

    We treat K_PIM_real_* as December-end anchors for each year and
    log-linearly interpolate between years to get monthly values.

    For months within the first year before December (Jan–Nov), we hold K
    constant at the first year's anchor. This keeps the implementation
    simple and is usually innocuous given slow-moving capital stocks.
    """

    df = df_annual.copy().sort_values("year").reset_index(drop=True)

    years = df["year"].to_numpy()
    start_year = int(years[0])
    end_year = int(years[-1])
    

    # Build monthly date range from Jan of first year to Dec of last year
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")

    out = pd.DataFrame({"month": dates})
    out["year"] = out["month"].dt.year

    # Map annual K to December anchors
    # We assume annual K_PIM_real_* are December-end stocks.
    annual_lookup = df.set_index("year")

    def interpolate_log_lin(annual_series: pd.Series) -> pd.Series:
        """Log-linear interpolate annual series to monthly.

        annual_series is indexed by year (e.g. 2018, 2019, ..., 2024),
        containing K_t values (December-end stocks).
        """

        # Build a temporary DataFrame with December anchor dates
        anchor_dates = pd.to_datetime(
            [f"{int(y)}-12-01" for y in annual_series.index]
        )
        anchor = pd.DataFrame(
            {
                "date": anchor_dates,
                "K": annual_series.to_numpy(),
            }
        ).set_index("date")

        # Reindex to full monthly frequency
        full_idx = pd.date_range(
            anchor.index.min(), anchor.index.max(), freq="MS"
        )
        full = anchor.reindex(full_idx)

        # Take logs for interpolation
        full["logK"] = np.log(full["K"])

        # Interpolate missing monthly logs linearly
        full["logK"] = full["logK"].interpolate(method="time")

        # Exponentiate back to levels
        full["K_interp"] = np.exp(full["logK"])

        # For months before the first anchor (if any), hold the first value constant
        full["K_interp"] = full["K_interp"].ffill()

        return full["K_interp"]

    # Build a continuous monthly index for interpolation
    monthly_index = pd.date_range(
        f"{start_year}-12-01", f"{end_year}-12-01", freq="MS"
    )

    # We create Series indexed by year for each scenario
    scenarios = {}
    for name in ["low", "central", "high"]:
        col = f"K_PIM_real_{name}"
        if col not in annual_lookup.columns:
            raise ValueError(
                f"Column '{col}' not found in annual PIM table. "
                f"Available columns: {list(annual_lookup.columns)}"
            )
        scenarios[name] = annual_lookup[col]

    # Interpolate for each scenario
    # Note: we call interpolate_log_lin on the annual Series indexed by year.
    for name, series_y in scenarios.items():
        series_y = series_y.copy()
        series_y.index = series_y.index.astype(int)
        K_monthly = interpolate_log_lin(series_y)
        # Align with full range Jan(first_year)–Dec(last_year)
        K_monthly = K_monthly.reindex(dates, method="bfill")
        out[f"K_PIM_lin_{name}"] = K_monthly.to_numpy()

    # Carry annual flags down to all months of that year
    out = out.merge(
        df[["year", "flows_imputed_flag", "gap_years_from_prev"]],
        on="year",
        how="left",
    ).rename(
        columns={
            "flows_imputed_flag": "flows_imputed_annual",
            "gap_years_from_prev": "gap_years_from_prev_annual",
        }
    )

    # Also keep annual K_PIM_real_* at December months (NaN otherwise)
    for name in ["low", "central", "high"]:
        col_real = f"K_PIM_real_{name}"
        col_out = f"K_PIM_real_{name}"
        out[col_out] = np.nan

    for _, row in df.iterrows():
        y = int(row["year"])
        dec_date = pd.Timestamp(f"{y}-12-01")
        mask = out["month"] == dec_date
        for name in ["low", "central", "high"]:
            col_real = f"K_PIM_real_{name}"
            if col_real in df.columns:
                out.loc[mask, col_real] = row[col_real]

    return out


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------

def run_ipc_monthly_pim_lin() -> pd.DataFrame:
    """Build monthly IPC K series from annual PIM and write outputs."""

    print("\n[IPC Monthly PIM-Lin] Loading annual IPC PIM table...")
    df_annual = load_ipc_annual_pim(IPC_ANNUAL_PIM_PATH)
    print(f"  - Annual rows: {len(df_annual)}")
    print(f"  - Years: {df_annual['year'].min()}–{df_annual['year'].max()}")

    print("\n[IPC Monthly PIM-Lin] Building monthly K series...")
    df_monthly = build_monthly_k(df_annual)

    # Attach port, company, operator/owner labels
    df_monthly.insert(0, "port", PORT_NAME)
    df_monthly.insert(1, "company", IPC_COMPANY_NAME)
    df_monthly.insert(2, "operator_or_owner", IPC_OWNER_ROLE)

    # Save outputs
    df_monthly.to_csv(OUT_MONTHLY_PATH, sep="\t", index=False)
    df_monthly.head(24).to_csv(OUT_MONTHLY_SAMPLE, index=False)

    meta = {
        "input_annual_pim_path": str(IPC_ANNUAL_PIM_PATH),
        "port": PORT_NAME,
        "company": IPC_COMPANY_NAME,
        "operator_or_owner": IPC_OWNER_ROLE,
        "notes": (
            "IPC monthly K series for Haifa, built via log-linear interpolation "
            "of annual PIM K values. K_PIM_lin_* are the main monthly K series."
        ),
    }
    OUT_PIM_LIN_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[IPC Monthly PIM-Lin] Done.")
    print(f"  - Monthly IPC K saved to:       {OUT_MONTHLY_PATH}")
    print(f"  - Monthly sample (24 rows) to:  {OUT_MONTHLY_SAMPLE}")
    print(f"  - Meta saved to:                {OUT_PIM_LIN_META_PATH}")

    print("\n[IPC Monthly PIM-Lin] Preview of IPC monthly K:")
    print(df_monthly.head())

    return df_monthly


if __name__ == "__main__":
    run_ipc_monthly_pim_lin()

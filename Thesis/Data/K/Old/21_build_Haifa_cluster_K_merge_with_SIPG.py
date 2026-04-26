# 21_build_Haifa_cluster_K_merge_with_SIPG.py
#
# Build a Haifa cluster K panel by combining:
#   - HPC+IPC info from 20_K_cluster_Haifa_HPC_IPC_monthly.tsv
#   - SIPG monthly K from 32_K_B_monthly_Haifa_SIPG_PIM_lin.tsv
#
# Outputs:
#   - 21_K_cluster_Haifa_HPC_IPC_SIPG_monthly.tsv
#
# The output is in "long" format with four company values:
#   1) Haifa Port Company (legacy)
#   2) Israel Ports Company (Haifa cluster)
#   3) SIPG Haifa Bayport
#   4) Haifa cluster (HPC + IPC + SIPG)
#
# Each row has:
#   port, company, operator_or_owner, month, year,
#   K_low, K_central, K_high,
#   K_HPC_low, K_HPC_central, K_HPC_high,
#   K_IPC_low, K_IPC_central, K_IPC_high,
#   K_SIPG_low, K_SIPG_central, K_SIPG_high,
#   K_cluster_low, K_cluster_central, K_cluster_high
#
# Notes:
#   - We use 20_K_cluster_Haifa_HPC_IPC_monthly.tsv as the "base" for HPC+IPC.
#     For each (port, month, year), the HPC rows in that file already contain:
#       * K_HPC_* (Path B)
#       * K_IPC_* (IPC PIM-lin)
#       * K_cluster_* (HPC + IPC)
#   - We then add SIPG K and build a new cluster:
#       K_cluster_* = K_HPC_* + K_IPC_* + K_SIPG_*
#
# Run this script from the Thesis repo root, e.g.:
#   (.venv) python "Data/K /21_build_Haifa_cluster_K_merge_with_SIPG.py"

from pathlib import Path
import pandas as pd


# --------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------

# Existing HPC+IPC cluster file (from 20_build_Haifa_cluster_K_merge.py)
HPC_IPC_CLUSTER_PATH = Path("Data/K /20_K_cluster_Haifa_HPC_IPC_monthly.tsv")

# SIPG monthly K (PIM-lin) from step 32
SIPG_MONTHLY_PIM_PATH = Path("Data/K /32_K_B_monthly_Haifa_SIPG_PIM_lin.tsv")

# Output
OUT_CLUSTER_PATH = Path("Data/K /21_K_cluster_Haifa_HPC_IPC_SIPG_monthly.tsv")
OUT_SAMPLE_PATH = Path("Data/K /21_K_cluster_Haifa_HPC_IPC_SIPG_monthly_sample.csv")

# Names (must match your existing files)
HPC_COMPANY_NAME = "Haifa Port Company (legacy)"
HPC_OPERATOR_NAME = "Haifa Port Company (legacy operator)"

IPC_COMPANY_NAME = "Israel Ports Company (Haifa cluster)"
IPC_OPERATOR_NAME = "Israel Ports Company (landlord / infrastructure owner)"

SIPG_COMPANY_NAME = "SIPG Haifa Bayport"
SIPG_OPERATOR_NAME = "SIPG Haifa Bayport (private operator)"

CLUSTER_COMPANY_NAME = "Haifa cluster (HPC + IPC + SIPG)"
CLUSTER_OPERATOR_NAME = "Haifa port cluster (HPC + IPC + SIPG)"


# --------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------

def load_hpc_ipc_base(path: Path) -> pd.DataFrame:
    """
    Load the HPC+IPC cluster file from step 20 and keep only one row per
    (port, month, year) with the decomposition K_HPC_* and K_IPC_*.

    We do this by filtering to rows where company == HPC_COMPANY_NAME,
    because those rows already carry both HPC and IPC K columns.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"HPC+IPC cluster file not found at: {path}\n"
            "Run 20_build_Haifa_cluster_K_merge.py first."
        )

    df = pd.read_csv(path, sep="\t")

    required = [
        "port",
        "company",
        "operator_or_owner",
        "month",
        "year",
        "K_HPC_low",
        "K_HPC_central",
        "K_HPC_high",
        "K_IPC_low",
        "K_IPC_central",
        "K_IPC_high",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            "Make sure you're using the updated 20_K_cluster_Haifa_HPC_IPC_monthly.tsv."
        )

    # Keep only HPC rows; they already contain the IPC and HPC decomposition.
    df_hpc = df[df["company"] == HPC_COMPANY_NAME].copy()
    if df_hpc.empty:
        raise ValueError(
            f"No rows found for company '{HPC_COMPANY_NAME}' in {path}."
        )

    df_hpc["month"] = pd.to_datetime(df_hpc["month"], errors="coerce")
    df_hpc["year"] = pd.to_numeric(df_hpc["year"], errors="coerce").astype("Int64")

    # One row per port-month-year with HPC and IPC K
    df_hpc = df_hpc[
        [
            "port",
            "month",
            "year",
            "K_HPC_low",
            "K_HPC_central",
            "K_HPC_high",
            "K_IPC_low",
            "K_IPC_central",
            "K_IPC_high",
        ]
    ].copy()

    return df_hpc


def load_sipg_monthly(path: Path) -> pd.DataFrame:
    """
    Load SIPG monthly K (PIM-lin) and return a table with:
      port, month, year, K_SIPG_low, K_SIPG_central, K_SIPG_high
    """
    if not path.exists():
        raise FileNotFoundError(
            f"SIPG monthly K file not found at: {path}\n"
            "Run 32_build_K_monthly_Haifa_SIPG_PIM_linear.py first."
        )

    df = pd.read_csv(path, sep="\t")

    required = [
        "port",
        "company",
        "operator_or_owner",
        "month",
        "year",
        "K_PIM_lin_low",
        "K_PIM_lin_central",
        "K_PIM_lin_high",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            "Make sure 32_K_B_monthly_Haifa_SIPG_PIM_lin.tsv has the expected columns."
        )

    # (Optional) filter to SIPG company explicitly, in case other rows appear later
    df = df[df["company"] == SIPG_COMPANY_NAME].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for company '{SIPG_COMPANY_NAME}' in {path}."
        )

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df = df[
        [
            "port",
            "month",
            "year",
            "K_PIM_lin_low",
            "K_PIM_lin_central",
            "K_PIM_lin_high",
        ]
    ].rename(
        columns={
            "K_PIM_lin_low": "K_SIPG_low",
            "K_PIM_lin_central": "K_SIPG_central",
            "K_PIM_lin_high": "K_SIPG_high",
        }
    )

    return df


# --------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------

def build_haifa_cluster_with_sipg() -> pd.DataFrame:
    """Combine HPC, IPC, and SIPG K into a Haifa cluster (HPC + IPC + SIPG)."""

    print("\n[Haifa Cluster + SIPG] Loading HPC+IPC base (from step 20)...")
    base = load_hpc_ipc_base(HPC_IPC_CLUSTER_PATH)
    print(
        f"  - Base rows (HPC-only view, with IPC decomposition): {len(base)}, "
        f"years: {base['year'].min()}–{base['year'].max()}"
    )

    print("\n[Haifa Cluster + SIPG] Loading SIPG monthly K (PIM-lin)...")
    sipg = load_sipg_monthly(SIPG_MONTHLY_PIM_PATH)
    print(
        f"  - SIPG rows: {len(sipg)}, "
        f"years: {sipg['year'].min()}–{sipg['year'].max()}"
    )

    # Merge on (port, month, year) using inner join to restrict to common date range
    print("\n[Haifa Cluster + SIPG] Merging base and SIPG on (port, month, year)...")
    merged = base.merge(
        sipg,
        on=["port", "month", "year"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No overlapping months between HPC+IPC base and SIPG series. "
            "Check the date ranges in 20_... and 32_...."
        )

    print(
        f"  - Merged rows: {len(merged)}, "
        f"years: {merged['year'].min()}–{merged['year'].max()}"
    )

    # Compute cluster K = HPC + IPC + SIPG, for each δ scenario
    for scen in ["low", "central", "high"]:
        merged[f"K_cluster_{scen}"] = (
            merged[f"K_HPC_{scen}"]
            + merged[f"K_IPC_{scen}"]
            + merged[f"K_SIPG_{scen}"]
        )

    # ----------------------------------------------------------------
    # Build long panel with four company blocks
    # ----------------------------------------------------------------
    long_rows = []

    # 1) HPC rows
    hpc = merged.copy()
    hpc["company"] = HPC_COMPANY_NAME
    hpc["operator_or_owner"] = HPC_OPERATOR_NAME
    hpc["K_low"] = hpc["K_HPC_low"]
    hpc["K_central"] = hpc["K_HPC_central"]
    hpc["K_high"] = hpc["K_HPC_high"]
    long_rows.append(hpc)

    # 2) IPC rows
    ipc = merged.copy()
    ipc["company"] = IPC_COMPANY_NAME
    ipc["operator_or_owner"] = IPC_OPERATOR_NAME
    ipc["K_low"] = ipc["K_IPC_low"]
    ipc["K_central"] = ipc["K_IPC_central"]
    ipc["K_high"] = ipc["K_IPC_high"]
    long_rows.append(ipc)

    # 3) SIPG rows
    sipg_rows = merged.copy()
    sipg_rows["company"] = SIPG_COMPANY_NAME
    sipg_rows["operator_or_owner"] = SIPG_OPERATOR_NAME
    sipg_rows["K_low"] = sipg_rows["K_SIPG_low"]
    sipg_rows["K_central"] = sipg_rows["K_SIPG_central"]
    sipg_rows["K_high"] = sipg_rows["K_SIPG_high"]
    long_rows.append(sipg_rows)

    # 4) Cluster rows (HPC + IPC + SIPG)
    cluster = merged.copy()
    cluster["company"] = CLUSTER_COMPANY_NAME
    cluster["operator_or_owner"] = CLUSTER_OPERATOR_NAME
    cluster["K_low"] = cluster["K_cluster_low"]
    cluster["K_central"] = cluster["K_cluster_central"]
    cluster["K_high"] = cluster["K_cluster_high"]
    long_rows.append(cluster)

    # Concatenate and order columns
    long = pd.concat(long_rows, ignore_index=True)

    # Sort for readability
    long = long.sort_values(
        ["port", "year", "month", "company"], ignore_index=True
    )

    cols_out = [
        "port",
        "company",
        "operator_or_owner",
        "month",
        "year",
        "K_low",
        "K_central",
        "K_high",
        "K_HPC_low",
        "K_HPC_central",
        "K_HPC_high",
        "K_IPC_low",
        "K_IPC_central",
        "K_IPC_high",
        "K_SIPG_low",
        "K_SIPG_central",
        "K_SIPG_high",
        "K_cluster_low",
        "K_cluster_central",
        "K_cluster_high",
    ]

    # Ensure all columns exist
    missing_out = [c for c in cols_out if c not in long.columns]
    if missing_out:
        raise ValueError(
            f"Internal error: expected columns missing in assembled panel: {missing_out}"
        )

    long = long[cols_out]

    # Save outputs
    long.to_csv(OUT_CLUSTER_PATH, sep="\t", index=False)
    long.head(20).to_csv(OUT_SAMPLE_PATH, sep="\t", index=False)

    print("\n[Haifa Cluster + SIPG] Done.")
    print(f"  - Cluster K panel (HPC + IPC + SIPG) saved to: {OUT_CLUSTER_PATH}")
    print(f"  - Sample (first 20 rows) saved to: {OUT_SAMPLE_PATH}")

    print("\n[Haifa Cluster + SIPG] Preview:")
    print(long.head())

    return long


if __name__ == "__main__":
    build_haifa_cluster_with_sipg()

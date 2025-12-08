"""
20_build_Haifa_cluster_K_merge.py

Build a Haifa cluster K panel by combining:

  - HPC monthly K (Path B, all depreciation scenarios) from
        03_K_B_monthly_Haifa_PathB.tsv
  - IPC monthly K (PIM-lin, all depreciation scenarios) from
        12_K_B_monthly_Haifa_IPC_PIM_lin.tsv

The output is in "long" format with three company values:
  1) Haifa Port Company (legacy)
  2) Israel Ports Company (Haifa cluster)
  3) Haifa cluster (HPC + IPC)

Each row has:
  port, company, operator_or_owner, month, year,
  K_low, K_central, K_high,
  plus decomposition columns:
    K_HPC_low/central/high,
    K_IPC_low/central/high,
    K_cluster_low/central/high

Notes:
  - This script assumes your HPC Path B file has columns:
        K_PathB_low, K_PathB_central, K_PathB_high
    and a 'company' value exactly equal to
        'Haifa Port Company (legacy)'.
  - It assumes your IPC monthly file has columns:
        K_PIM_lin_low, K_PIM_lin_central, K_PIM_lin_high
    and a 'company' value exactly equal to
        'Israel Ports Company (Haifa cluster)'.

  Adjust HPC/IPC constants below if your local files use slightly
  different names.
"""

from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------
# File paths & configuration
# --------------------------------------------------------------------

HPC_PATHB_PATH      = Path("Data/K /03_K_B_monthly_Haifa_PathB.tsv")
IPC_MONTHLY_PIMPATH = Path("Data/K /12_K_B_monthly_Haifa_IPC_PIM_lin.tsv")

OUT_CLUSTER_PATH = Path("Data/K /20_K_cluster_Haifa_HPC_IPC_monthly.tsv")

HPC_COMPANY_NAME = "Haifa Port Company (legacy)"
IPC_COMPANY_NAME = "Israel Ports Company (Haifa cluster)"
CLUSTER_COMPANY_NAME = "Haifa cluster (HPC + IPC)"

# Depreciation scenarios we expect in the input files
SCENARIOS = ["low", "central", "high"]

# Mapping from scenario -> column names in the input files
HPC_K_COLS = {s: f"K_PathB_{s}" for s in SCENARIOS}
IPC_K_COLS = {s: f"K_PIM_lin_{s}" for s in SCENARIOS}


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def load_hpc_pathb(path: Path) -> pd.DataFrame:
    """
    Load HPC Path B monthly K and keep the main K columns for all scenarios.

    Expected columns in 03_K_B_monthly_Haifa_PathB.tsv include:

        port, company, operator_or_owner, month, year,
        K_PathB_low, K_PathB_central, K_PathB_high, ...

    We:
      - filter to HPC_COMPANY_NAME,
      - keep port, company, operator_or_owner, month, year,
        and K_PathB_<s> for all scenarios s,
      - rename K_PathB_<s> -> K_HPC_<s>.
    """
    df = pd.read_csv(path, sep="\t")

    required = ["port", "company", "operator_or_owner", "month", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )

    for scen, col in HPC_K_COLS.items():
        if col not in df.columns:
            raise ValueError(
                f"Expected HPC K column '{col}' for scenario '{scen}' "
                f"in {path}, but it is missing. Available columns: {list(df.columns)}"
            )

    df = df[df["company"] == HPC_COMPANY_NAME].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for company '{HPC_COMPANY_NAME}' in {path}. "
            f"Check HPC_COMPANY_NAME or the file contents."
        )

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    keep_cols = ["port", "company", "operator_or_owner", "month", "year"] + list(HPC_K_COLS.values())
    df = df[keep_cols].copy()

    rename_map = {col: f"K_HPC_{scen}" for scen, col in HPC_K_COLS.items()}
    df = df.rename(columns=rename_map)

    return df


def load_ipc_monthly(path: Path) -> pd.DataFrame:
    """
    Load IPC monthly K (PIM-lin) and keep the main K columns for all scenarios.

    Expected columns in 12_K_B_monthly_Haifa_IPC_PIM_lin.tsv include:

        port, company, operator_or_owner, month, year,
        K_PIM_lin_low, K_PIM_lin_central, K_PIM_lin_high, ...

    We:
      - filter to IPC_COMPANY_NAME,
      - keep port, company, operator_or_owner, month, year,
        and K_PIM_lin_<s> for all scenarios s,
      - rename K_PIM_lin_<s> -> K_IPC_<s>.
    """
    df = pd.read_csv(path, sep="\t")

    required = ["port", "company", "operator_or_owner", "month", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}"
        )

    for scen, col in IPC_K_COLS.items():
        if col not in df.columns:
            raise ValueError(
                f"Expected IPC K column '{col}' for scenario '{scen}' "
                f"in {path}, but it is missing. Available columns: {list(df.columns)}"
            )

    df = df[df["company"] == IPC_COMPANY_NAME].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for company '{IPC_COMPANY_NAME}' in {path}. "
            f"Check IPC_COMPANY_NAME or the file contents."
        )

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    keep_cols = ["port", "company", "operator_or_owner", "month", "year"] + list(IPC_K_COLS.values())
    df = df[keep_cols].copy()

    rename_map = {col: f"K_IPC_{scen}" for scen, col in IPC_K_COLS.items()}
    df = df.rename(columns=rename_map)

    return df


# --------------------------------------------------------------------
# Main procedure
# --------------------------------------------------------------------

def build_haifa_cluster_panel() -> pd.DataFrame:
    """Combine HPC Path B and IPC PIM-lin into a Haifa cluster K panel
    for all depreciation scenarios (low / central / high)."""

    print("\n[Haifa Cluster K] Loading HPC Path B monthly K (all scenarios)...")
    hpc = load_hpc_pathb(HPC_PATHB_PATH)
    print(f"  - HPC rows: {len(hpc)}")
    print(f"  - HPC months: {hpc['month'].min().date()}–{hpc['month'].max().date()}")

    print("\n[Haifa Cluster K] Loading IPC monthly PIM-lin K (all scenarios)...")
    ipc = load_ipc_monthly(IPC_MONTHLY_PIMPATH)
    print(f"  - IPC rows: {len(ipc)}")
    print(f"  - IPC months: {ipc['month'].min().date()}–{ipc['month'].max().date()}")

    key_cols = ["port", "month", "year"]

    hpc_keep = key_cols + [f"K_HPC_{s}" for s in SCENARIOS]
    ipc_keep = key_cols + [f"K_IPC_{s}" for s in SCENARIOS]

    merged = pd.merge(
        hpc[hpc_keep],
        ipc[ipc_keep],
        on=key_cols,
        how="outer",
        suffixes=("_hpc", "_ipc"),
    )

    # Drop months where we don't trust either IPC or HPC K (Jan–Nov 2018)
    bad_early_2018 = (merged["year"] == 2018) & (merged["month"] < pd.Timestamp("2018-12-01"))
    merged = merged.loc[~bad_early_2018].copy()

    # Compute cluster total K for each scenario
    for scen in SCENARIOS:
        merged[f"K_cluster_{scen}"] = merged[[f"K_HPC_{scen}", f"K_IPC_{scen}"]].sum(
            axis=1,
            min_count=1,
        )

    # ------------------------------------------------------------------
    # Build "long" panel with three company categories
    # ------------------------------------------------------------------

    # 1) HPC rows
    hpc_long = merged.copy()
    hpc_long["company"] = HPC_COMPANY_NAME
    hpc_long["operator_or_owner"] = "Haifa Port Company (legacy operator)"
    for scen in SCENARIOS:
        hpc_long[f"K_{scen}"] = hpc_long[f"K_HPC_{scen}"]

    # 2) IPC rows
    ipc_long = merged.copy()
    ipc_long["company"] = IPC_COMPANY_NAME
    ipc_long["operator_or_owner"] = "Israel Ports Company (landlord / infrastructure owner)"
    for scen in SCENARIOS:
        ipc_long[f"K_{scen}"] = ipc_long[f"K_IPC_{scen}"]

    # 3) Cluster total rows
    cluster_long = merged.copy()
    cluster_long["company"] = CLUSTER_COMPANY_NAME
    cluster_long["operator_or_owner"] = "Haifa port cluster (HPC + IPC)"
    for scen in SCENARIOS:
        cluster_long[f"K_{scen}"] = cluster_long[f"K_cluster_{scen}"]

    long = pd.concat([hpc_long, ipc_long, cluster_long], ignore_index=True)

    # For backward compatibility, "K_central" is our main K
    # (already present in K_central; this just makes the intention explicit)
    # long["K_central"] already exists via the loop above.

    # Column ordering
    keep_cols = [
        "port",
        "company",
        "operator_or_owner",
        "month",
        "year",
    ]

    # Generic scenario columns
    keep_cols += [f"K_{s}" for s in SCENARIOS]

    # Decomposition columns
    keep_cols += [f"K_HPC_{s}" for s in SCENARIOS]
    keep_cols += [f"K_IPC_{s}" for s in SCENARIOS]
    keep_cols += [f"K_cluster_{s}" for s in SCENARIOS]

    keep_cols = [c for c in keep_cols if c in long.columns]

    long = long[keep_cols].copy()
    long = long.sort_values(["company", "month"]).reset_index(drop=True)

    # Save
    long.to_csv(OUT_CLUSTER_PATH, sep="\t", index=False)

    print("\n[Haifa Cluster K] Done.")
    print(f"  - Cluster K panel (all scenarios) saved to: {OUT_CLUSTER_PATH}")

    print("\n[Haifa Cluster K] Preview:")
    print(long.head())

    return long


if __name__ == "__main__":
    build_haifa_cluster_panel()

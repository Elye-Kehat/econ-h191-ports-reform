"""
03_build_K_monthly_PathB_lumpy.py

Construct Path B monthly K series for Haifa Port Company (legacy) for
ALL depreciation scenarios available in the annual PIM backbone
(e.g. low / central / high):

  - Uses annual PIM backbone from 01_K_B_annual_Haifa_PIM.tsv
  - Uses monthly skeleton (dates + flags) from 02_K_B_monthly_Haifa_PIM_lin.tsv
  - Uses Haifa_big_projects.csv to overlay lumpy “big project” capital
  - For each scenario s, decomposes annual PIM K into:
        K_PIM_y^s = K_big_y + K_bg_y^s
    then builds a smooth monthly background K_bg,t^s via interpolation
  - Sums K_big,t + K_bg,t^s to get monthly K_PathB,t^s

Output:
  - 03_K_B_monthly_Haifa_PathB.tsv
  - 03_K_B_monthly_Haifa_PathB_sample.csv
  - 03_PathB_config.json
  - 03_K_B_PathB_QA.tsv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ======================================================================
# Paths and basic config
# ======================================================================

DATA_DIR = Path(__file__).resolve().parent

# Inputs
ANNUAL_PIM_PATH = DATA_DIR / "01_K_B_annual_Haifa_PIM.tsv"
MONTHLY_PIM_LIN_PATH = DATA_DIR / "02_K_B_monthly_Haifa_PIM_lin.tsv"
BIG_PROJECTS_PATH = DATA_DIR / "Haifa_big_projects.csv"
STEP1_REAL_PATH = DATA_DIR / "00_haifa_financials_step1_real.tsv"  # for deflator if we need to deflate project costs

# Outputs
OUT_PATHB_MONTHLY = DATA_DIR / "03_K_B_monthly_Haifa_PathB.tsv"
OUT_PATHB_SAMPLE = DATA_DIR / "03_K_B_monthly_Haifa_PathB_sample.csv"
OUT_PATHB_CONFIG = DATA_DIR / "03_PathB_config.json"
OUT_PATHB_QA = DATA_DIR / "03_K_B_PathB_QA.tsv"

# Port / operator labels (hard-coded for this script)
PORT_NAME = "Haifa"
COMPANY_NAME = "Haifa Port Company (legacy)"
OPERATOR_NAME = "Haifa Port Company (legacy)"  # same for now

# Default “aggregate PPE” asset life in years if project-level life is missing
DEFAULT_ASSET_LIFE_YEARS = 28.0


# ======================================================================
# Helper: load annual PIM backbone (Step 2 output)
# ======================================================================

def load_annual_pim(pim_path: Path) -> pd.DataFrame:
    """
    Load the annual PIM capital stock table produced by 01_build_K_annual_PIM.py.

    We expect at least:
      - company
      - year
      - K_PIM_real_central

    but we ALSO allow additional scenario columns such as:
      - K_PIM_real_low
      - K_PIM_real_high

    The function:
      - filters to Haifa Port Company (legacy)
      - coerces year to int
      - sorts by year
    """
    print(f"[load_annual_pim] Reading annual PIM from: {pim_path}")
    if not pim_path.exists():
        raise FileNotFoundError(
            f"Annual PIM file not found at: {pim_path}\n"
            "Run 01_build_K_annual_PIM.py first."
        )

    df = pd.read_csv(pim_path, sep="\t")

    required = ["company", "year", "K_PIM_real_central"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Annual PIM file is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    # Filter to Haifa legacy company
    df = df[df["company"] == COMPANY_NAME].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for company '{COMPANY_NAME}' in annual PIM file."
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)

    print(f"[load_annual_pim] Years in PIM backbone: {df['year'].dropna().astype(int).tolist()}")
    return df


# ======================================================================
# Helper: load monthly skeleton (from Path A / 02)
# ======================================================================

def load_monthly_skeleton(monthly_path: Path) -> pd.DataFrame:
    """
    Load the monthly skeleton from 02_K_B_monthly_Haifa_PIM_lin.tsv.

    We use this for:
      - monthly date index
      - port/company/operator labels
      - flags: imputed_2022, flows_imputed_annual, gap_years_from_prev_annual

    We do *not* use the Path A K levels here.
    """
    print(f"[load_monthly_skeleton] Reading monthly skeleton from: {monthly_path}")
    if not monthly_path.exists():
        raise FileNotFoundError(
            f"Monthly PIM-linear file not found at: {monthly_path}\n"
            "Run 02_build_K_monthly_PIM_linear.py first."
        )

    df = pd.read_csv(monthly_path, sep="\t")

    required = ["port", "company", "month", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Monthly PIM-linear file is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    # Filter to Haifa legacy
    df = df[(df["port"] == PORT_NAME) & (df["company"] == COMPANY_NAME)].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for port '{PORT_NAME}' and company '{COMPANY_NAME}' "
            f"in {monthly_path.name}"
        )

    # Ensure 'month' is datetime and 'year' is coherent
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year

    # If these columns are missing, create defaults
    if "operator_or_owner" not in df.columns:
        df["operator_or_owner"] = OPERATOR_NAME
    if "imputed_2022" not in df.columns:
        df["imputed_2022"] = (df["year"] == 2022).astype(int)
    if "flows_imputed_annual" not in df.columns:
        df["flows_imputed_annual"] = False
    if "gap_years_from_prev_annual" not in df.columns:
        df["gap_years_from_prev_annual"] = 0

    df = df.sort_values("month").reset_index(drop=True)

    print(f"[load_monthly_skeleton] Months: {df['month'].min().date()} → {df['month'].max().date()}")
    return df


# ======================================================================
# Helper: deflator-by-year for project cost deflation
# ======================================================================

def load_deflator_by_year(step1_real_path: Path) -> dict:
    """
    Load deflator values from 00_haifa_financials_step1_real.tsv.

    We expect:
      - year
      - deflator

    We will:
      - filter to Haifa Port Company (legacy)
      - build a dict year -> deflator (mean or first if repeated)
    """
    print(f"[load_deflator_by_year] Reading step1 real financials from: {step1_real_path}")
    if not step1_real_path.exists():
        raise FileNotFoundError(
            f"Step1 real financials file not found at: {step1_real_path}\n"
            "Run 00_build_K_step1.py first so that '00_haifa_financials_step1_real.tsv' exists."
        )

    df = pd.read_csv(step1_real_path, sep="\t")
    if "year" not in df.columns or "deflator" not in df.columns:
        raise ValueError(
            "00_haifa_financials_step1_real.tsv must contain 'year' and 'deflator' columns."
        )

    df = df[df.get("company", "") == COMPANY_NAME] if "company" in df.columns else df
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "deflator"])

    year_def = (
        df.sort_values("year")
          .drop_duplicates(subset=["year"])
          .set_index("year")["deflator"]
          .to_dict()
    )

    print(f"[load_deflator_by_year] Deflator years available: {sorted(year_def.keys())}")
    return year_def


# ======================================================================
# Helper: load big project table
# ======================================================================

def load_big_projects(
    projects_path: Path,
    deflator_by_year: dict,
) -> pd.DataFrame:
    """
    Load and prepare Haifa big projects table.

    Expected / preferred columns in Haifa_big_projects.csv:
      - project_id
      - port
      - operator_owner (or operator_or_owner / company / owner)
      - commissioning_year
      - commissioning_month
      - expected_end_year
      - expected_end_month
      - start_year
      - start_month
      - investment_year_fs / investment_year
      - amount_nominal_th_nis (nominal project cost, HPC share)
      - asset_life_years (optional)
      - relevant_for_K_B or similar Path B flag (optional)

    This function:
      - filters to Haifa + HPC-related projects
      - keeps only Path B relevant rows (if such a column exists)
      - infers a Timestamp 'commissioning_month' with fallbacks
      - parses investment_year from investment_year_fs (first 4-digit year),
        with fallback to start_year if needed
      - computes cost_real_thousands_nis using deflator_by_year
      - fills missing asset_life_years with DEFAULT_ASSET_LIFE_YEARS
    """
    print(f"[load_big_projects] Reading big projects from: {projects_path}")
    if not projects_path.exists():
        raise FileNotFoundError(
            f"Big projects file not found at: {projects_path}\n"
            "Populate Haifa_big_projects.csv according to your capital projects report."
        )

    df = pd.read_csv(projects_path)

    # 1. Filter to Haifa
    if "port" not in df.columns:
        raise ValueError("Haifa_big_projects.csv must contain a 'port' column.")
    df = df[df["port"] == PORT_NAME].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for port '{PORT_NAME}' in Haifa_big_projects.csv."
        )

    df_port_only = df.copy()

    # 2. Filter to HPC-related projects (fuzzy match)
    operator_col = None
    for cand in ["operator_or_owner", "operator_owner", "company", "owner"]:
        if cand in df.columns:
            operator_col = cand
            break

    if operator_col is not None:
        op_str = df[operator_col].astype(str).str.upper()
        mask_hpc = op_str.str.contains("HAIFA PORT COMPANY") | op_str.str.contains(r"\bHPC\b")
        df_hpc = df[mask_hpc].copy()

        if df_hpc.empty:
            print(
                f"[load_big_projects] WARNING: No rows matched HPC using column "
                f"'{operator_col}'. Keeping all rows for port='{PORT_NAME}'."
            )
            df = df_port_only
        else:
            df = df_hpc
            print(
                f"[load_big_projects] Filtered to {len(df)} HPC-related projects "
                f"using fuzzy match on column '{operator_col}'."
            )
    else:
        print(
            "[load_big_projects] No operator column found "
            "(operator_or_owner/operator_owner/company/owner); "
            "using all rows for port=Haifa."
        )

    # 3. Path B relevance flag
    include_col = None
    for cand in ["include_in_pathB", "include_pathB", "pathB_flag", "include_in_path_b", "relevant_for_K_B"]:
        if cand in df.columns:
            include_col = cand
            break
    if include_col is not None:
        df[include_col] = (
            df[include_col]
            .astype(str)
            .str.upper()
            .map({"1": 1, "TRUE": 1, "0": 0, "FALSE": 0})
            .fillna(1)
            .astype(int)
        )
        df = df[df[include_col] == 1].copy()
        print(f"[load_big_projects] Filtered to Path B relevant projects using column '{include_col}'.")
    else:
        print("[load_big_projects] No Path B flag column found; assuming all rows are Path B projects.")

    # 4. project_id
    if "project_id" not in df.columns:
        df["project_id"] = df.index.astype(str)
        print("[load_big_projects] 'project_id' missing; using row index as project_id.")

    # 5. Investment year (parse early so commissioning can fall back to it)
    invest_col = None
    for cand in ["investment_year", "investment_year_fs", "capex_year_fs"]:
        if cand in df.columns:
            invest_col = cand
            break
    if invest_col is None:
        raise ValueError(
            "Haifa_big_projects.csv must contain an investment year column "
            "like 'investment_year' or 'investment_year_fs'."
        )

    raw_year = df[invest_col].astype(str)
    year_str = raw_year.str.extract(r"(\d{4})")[0]
    df["investment_year"] = pd.to_numeric(year_str, errors="coerce").astype("Int64")

    for fallback_col in ["start_year", "expected_end_year", "commissioning_year"]:
        if fallback_col in df.columns:
            mask_missing = df["investment_year"].isna()
            df.loc[mask_missing, "investment_year"] = pd.to_numeric(
                df.loc[mask_missing, fallback_col],
                errors="coerce"
            ).astype("Int64")

    if df["investment_year"].isna().any():
        bad = df[df["investment_year"].isna()][["project_id", invest_col]]
        raise ValueError(
            "Some projects have invalid investment_year values even after parsing. "
            f"Offending projects: {bad.to_dict(orient='records')}"
        )

    # 6. Commissioning month (with fallbacks)
    def _to_int_or_none(x):
        try:
            if pd.isna(x):
                return None
            s = str(x).strip()
            if s == "":
                return None
            return int(float(s))
        except Exception:
            return None

    def _infer_commissioning_ts(row: pd.Series) -> pd.Timestamp:
        """
        Infer commissioning date as YYYY-MM-01 using:
          1) commissioning_year + commissioning_month
          2) expected_end_year + expected_end_month (default month=12)
          3) start_year + start_month (default month=6)
          4) investment_year (default month=12)
        """
        y = _to_int_or_none(row.get("commissioning_year"))
        m = _to_int_or_none(row.get("commissioning_month"))

        if y is None or m is None:
            y_e = _to_int_or_none(row.get("expected_end_year"))
            m_e = _to_int_or_none(row.get("expected_end_month"))
            if y_e is not None:
                y = y_e
                m = m_e if m_e is not None else 12

        if y is None or m is None:
            y_s = _to_int_or_none(row.get("start_year"))
            m_s = _to_int_or_none(row.get("start_month"))
            if y_s is not None:
                y = y_s
                m = m_s if m_s is not None else 6

        if y is None or m is None:
            y_i = _to_int_or_none(row.get("investment_year"))
            if y_i is not None:
                y = y_i
                m = 12

        if y is None or m is None:
            return pd.NaT

        m = max(1, min(12, m))
        try:
            return pd.Timestamp(year=int(y), month=int(m), day=1)
        except Exception:
            return pd.NaT

    df["commissioning_month"] = df.apply(_infer_commissioning_ts, axis=1)

    if df["commissioning_month"].isna().any():
        bad = df[df["commissioning_month"].isna()][[
            "project_id",
            "commissioning_year",
            "commissioning_month",
            "expected_end_year",
            "expected_end_month",
            "start_year",
            "start_month",
            "investment_year",
        ]]
        raise ValueError(
            "Some projects have invalid commissioning_month dates even after applying "
            "fallbacks (commissioning → expected_end → start → investment). Offending projects: "
            f"{bad.to_dict(orient='records')}"
        )

    # 7. Real cost (deflating nominal)
    col_cost_real = None
    for cand in ["cost_real_thousands_nis", "amount_real_thousands_nis", "real_cost_th_nis"]:
        if cand in df.columns:
            col_cost_real = cand
            break

    if col_cost_real is not None:
        df["cost_real_thousands_nis"] = pd.to_numeric(df[col_cost_real], errors="coerce")
    else:
        col_cost_nom = None
        for cand in ["cost_nominal_thousands_nis", "amount_nominal_th_nis", "project_cost_thousands_nis"]:
            if cand in df.columns:
                col_cost_nom = cand
                break
        if col_cost_nom is None:
            raise ValueError(
                "Haifa_big_projects.csv must contain either a real-cost column "
                "('cost_real_thousands_nis') or a nominal-cost column "
                "('amount_nominal_th_nis' / 'cost_nominal_thousands_nis')."
            )

        df[col_cost_nom] = pd.to_numeric(df[col_cost_nom], errors="coerce")
        mask_bad = df[col_cost_nom].isna()
        if mask_bad.any():
            bad = df.loc[mask_bad, ["project_id", col_cost_nom]]
            print(
                "[load_big_projects] WARNING: Dropping projects with missing nominal cost "
                "(they will be treated as background PIM investment only): "
                f"{bad.to_dict(orient='records')}"
            )
            df = df[~mask_bad].copy()

        if df.empty:
            raise ValueError(
                "After dropping projects with missing nominal costs, no Path B projects remain. "
                "Provide at least one project with a usable cost."
            )

        real_costs = []
        for _, row in df.iterrows():
            proj_id = row["project_id"]
            year = int(row["investment_year"])
            if year not in deflator_by_year:
                all_years = sorted(deflator_by_year.keys())
                nearest_year = min(all_years, key=lambda y: abs(int(y) - year))
                print(
                    f"[load_big_projects] WARNING: deflator for investment_year={year} "
                    f"not found for project '{proj_id}'. Using nearest available year {nearest_year}."
                )
                defl = deflator_by_year[nearest_year]
            else:
                defl = deflator_by_year[year]
            real_costs.append(row[col_cost_nom] / defl)

        df["cost_real_thousands_nis"] = real_costs
        print("[load_big_projects] Computed 'cost_real_thousands_nis' from nominal costs and deflator.")

    # 8. Asset life handling
    if "asset_life_years" in df.columns:
        df["asset_life_years"] = pd.to_numeric(df["asset_life_years"], errors="coerce")
    else:
        df["asset_life_years"] = np.nan

    df["asset_life_years"] = df["asset_life_years"].fillna(DEFAULT_ASSET_LIFE_YEARS)

    # 9. Final columns
    keep_cols = [
        "project_id",
        "commissioning_month",
        "investment_year",
        "cost_real_thousands_nis",
        "asset_life_years",
    ]
    if "asset_class" in df.columns:
        keep_cols.append("asset_class")

    df = df[keep_cols].copy()
    df = df.sort_values("commissioning_month").reset_index(drop=True)

    print(f"[load_big_projects] Loaded {len(df)} Path B projects.")
    return df


# ======================================================================
# Helper: build monthly K_t for a single project
# ======================================================================

def build_project_K_ts(project_row: pd.Series, monthly_index: pd.DatetimeIndex) -> pd.Series:
    """
    Given a single project and a monthly date index, construct the project’s
    capital stock time series K_{p,t}:

      - 0 before commissioning_month
      - jump by cost_real_thousands_nis at commissioning_month
      - geometric depreciation thereafter with monthly rate derived from
        1 / asset_life_years (annual)

    Returns:
      pd.Series indexed by monthly_index, with capital levels for this project.
    """
    proj_id = project_row["project_id"]
    t0 = pd.to_datetime(project_row["commissioning_month"])
    cost_real = float(project_row["cost_real_thousands_nis"])
    life_years = float(project_row["asset_life_years"])

    if life_years <= 0:
        raise ValueError(
            f"Project '{proj_id}' has non-positive asset_life_years={life_years}."
        )

    delta_annual = 1.0 / life_years
    delta_month = 1.0 - (1.0 - delta_annual) ** (1.0 / 12.0)

    K_values = np.zeros(len(monthly_index), dtype=float)

    if t0 not in monthly_index:
        nearest_pos = np.argmin(np.abs(monthly_index - t0))
        t0_effective = monthly_index[nearest_pos]
        print(
            f"[build_project_K_ts] WARNING: commissioning_month={t0.date()} for project '{proj_id}' "
            f"not in monthly index. Using nearest index month {t0_effective.date()} instead."
        )
        t0 = t0_effective

    start_idx = int(np.where(monthly_index == t0)[0][0])

    K_values[start_idx] = cost_real

    for i in range(start_idx + 1, len(monthly_index)):
        K_values[i] = (1.0 - delta_month) * K_values[i - 1]

    return pd.Series(K_values, index=monthly_index, name=f"K_proj_{proj_id}")


def build_all_projects_big_K(
    projects_df: pd.DataFrame,
    monthly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build the total K_big,t series by summing K_p,t across all projects.

    Also compute:
      - lumpy_event_month : 1 if any project commissions in that month
      - num_projects_commissioned : count of projects commissioning in that month

    Returns a DataFrame with:
      - month
      - K_big_central   (scenario-invariant big-project capital)
      - lumpy_event_month
      - num_projects_commissioned
    """
    print("[build_all_projects_big_K] Building per-project and aggregate K_big,t...")

    K_big = np.zeros(len(monthly_index), dtype=float)
    event_counts = np.zeros(len(monthly_index), dtype=int)

    for _, row in projects_df.iterrows():
        proj_id = row["project_id"]
        K_p_series = build_project_K_ts(row, monthly_index)
        K_big += K_p_series.to_numpy()

        t0 = pd.to_datetime(row["commissioning_month"])
        if t0 in monthly_index:
            pos = int(np.where(monthly_index == t0)[0][0])
            event_counts[pos] += 1

    df_big = pd.DataFrame(
        {
            "month": monthly_index,
            "K_big_central": K_big,
            "num_projects_commissioned": event_counts,
        }
    )
    df_big["lumpy_event_month"] = (df_big["num_projects_commissioned"] > 0).astype(int)

    print("[build_all_projects_big_K] Done building K_big,t.")
    return df_big


# ======================================================================
# Helper: compute annual K_big_y and K_bg_y^s from annual PIM
# ======================================================================

def compute_annual_big_and_bg_K(
    df_big_monthly: pd.DataFrame,
    df_pim_annual: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    For each year y in df_pim_annual, compute:

      - K_big_y  = K_big_central at December of year y (scenario-invariant)
      - For each PIM scenario s (e.g. low, central, high):
            K_bg_y^s = K_PIM_real_y^s - K_big_y

    Returns:
      - annual_bg_df : DataFrame with columns
            year
            K_big_y
            flows_imputed_flag
            gap_years_from_prev
            K_PIM_real_<s> (copied from df_pim_annual)
            K_bg_y_<s>
      - scenarios : list of scenario names, e.g. ["low", "central", "high"]

    If any K_bg_y^s < 0, we warn and clamp to a small positive epsilon.
    """
    print("[compute_annual_big_and_bg_K] Computing annual K_big_y and K_bg_y for all scenarios...")

    df_big_monthly = df_big_monthly.copy()
    df_big_monthly["month"] = pd.to_datetime(df_big_monthly["month"])
    df_big_monthly["year"] = df_big_monthly["month"].dt.year

    scenario_cols = [c for c in df_pim_annual.columns if c.startswith("K_PIM_real_")]
    if not scenario_cols:
        raise ValueError(
            "Annual PIM file does not contain any 'K_PIM_real_*' columns; "
            "expected at least 'K_PIM_real_central'."
        )
    scenarios = [c.replace("K_PIM_real_", "") for c in scenario_cols]

    rows = []
    eps = 1e-6

    for _, row in df_pim_annual.iterrows():
        year = int(row["year"])

        month_dec = pd.Timestamp(f"{year}-12-01")
        df_dec = df_big_monthly[df_big_monthly["month"] == month_dec]
        if df_dec.empty:
            raise ValueError(
                f"No monthly big-K row found for December {year}-12-01 when computing annual K_big_y."
            )
        K_big_y = float(df_dec["K_big_central"].iloc[0])

        out_row = {
            "year": year,
            "K_big_y": K_big_y,
        }

        if "flows_imputed_flag" in df_pim_annual.columns:
            out_row["flows_imputed_flag"] = bool(row["flows_imputed_flag"])
        else:
            out_row["flows_imputed_flag"] = False

        if "gap_years_from_prev" in df_pim_annual.columns:
            out_row["gap_years_from_prev"] = int(row["gap_years_from_prev"])
        else:
            out_row["gap_years_from_prev"] = 0

        for scen in scenarios:
            col_pim = f"K_PIM_real_{scen}"
            if col_pim not in df_pim_annual.columns:
                raise ValueError(
                    f"Expected column '{col_pim}' in annual PIM data but it is missing."
                )
            K_pim_s = float(row[col_pim])
            K_bg_y_s = K_pim_s - K_big_y
            if K_bg_y_s < 0:
                print(
                    f"[compute_annual_big_and_bg_K] WARNING: Background K is negative for "
                    f"year {year}, scenario '{scen}' (K_bg_y = {K_bg_y_s:.2f}). "
                    f"Clamping to epsilon = {eps}."
                )
                K_bg_y_s = eps

            out_row[col_pim] = K_pim_s
            out_row[f"K_bg_y_{scen}"] = K_bg_y_s

        rows.append(out_row)

    annual_bg_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    print("[compute_annual_big_and_bg_K] Done computing annual background K for all scenarios.")
    return annual_bg_df, scenarios


# ======================================================================
# Helper: build monthly background K_bg,t^s by log-linear interpolation
# ======================================================================

def build_monthly_background_K_bg(
    annual_bg_df: pd.DataFrame,
    monthly_index: pd.DatetimeIndex,
    scenarios: list,
) -> pd.DataFrame:
    """
    Build smooth monthly background capital series K_bg,t^s for each scenario s
    using log-linear interpolation between annual background K_bg_y^s at
    December year-ends.

    Returns a DataFrame with:
      - month
      - K_bg_<s>  for each scenario s (e.g. K_bg_low, K_bg_central, K_bg_high)
    """
    print("[build_monthly_background_K_bg] Building monthly K_bg,t for all scenarios via log-linear interpolation...")

    df_monthly = pd.DataFrame({"month": monthly_index}).set_index("month")

    for scen in scenarios:
        col_bg_y = f"K_bg_y_{scen}"
        if col_bg_y not in annual_bg_df.columns:
            raise ValueError(
                f"Annual background DataFrame missing column '{col_bg_y}' "
                f"for scenario '{scen}'."
            )

        anchors = annual_bg_df[["year", col_bg_y]].copy()
        anchors["month"] = pd.to_datetime(anchors["year"].astype(int).astype(str) + "-12-01")
        anchors = anchors.set_index("month")

        tmp = df_monthly.copy()
        tmp["K_bg_anchor"] = anchors[col_bg_y]

        log_K = np.log(tmp["K_bg_anchor"])
        log_K_interpolated = log_K.interpolate(method="time")

        df_monthly[f"K_bg_{scen}"] = np.exp(log_K_interpolated)

    df_monthly = df_monthly.reset_index()
    print("[build_monthly_background_K_bg] Done building K_bg,t for all scenarios.")
    return df_monthly


# ======================================================================
# Helper: combine big and background to form Path B monthly K
# ======================================================================

def combine_big_and_bg(
    df_monthly_skel: pd.DataFrame,
    df_big_monthly: pd.DataFrame,
    df_bg_monthly: pd.DataFrame,
    df_pim_annual: pd.DataFrame,
    scenarios: list,
) -> pd.DataFrame:
    """
    Combine:
      - monthly skeleton (dates, flags, labels, PIM anchors from 02)
      - monthly big-K (K_big_central, lumpy_event_month, num_projects_commissioned)
      - monthly background K_bg_<s> for each scenario s

    to produce the Path B monthly K series for each scenario:

      K_PathB_<s>_t = K_big_central_t + K_bg_<s>_t

    For backward compatibility we also keep:
      - K_PathB_central
      - K_bg_central
      - K_PIM_real_central
      - gap_PathB_minus_PIM_central (central scenario only)
    """
    print("[combine_big_and_bg] Combining big and background K to form Path B for all scenarios...")

    df = df_monthly_skel.copy()
    df["month"] = pd.to_datetime(df["month"])

    # 1. Merge big-K monthly
    df_big = df_big_monthly.copy()
    df_big["month"] = pd.to_datetime(df_big["month"])
    df = df.merge(df_big, on="month", how="left")

    # 2. Merge background K_bg,t for all scenarios
    df_bg = df_bg_monthly.copy()
    df_bg["month"] = pd.to_datetime(df_bg["month"])
    df = df.merge(df_bg, on="month", how="left")

    # Fill missing big project capital with 0
    df["K_big_central"] = df["K_big_central"].fillna(0.0)

    # 3. Total Path B K for each scenario
    for scen in scenarios:
        col_bg = f"K_bg_{scen}"
        if col_bg not in df.columns:
            raise ValueError(
                f"Background monthly DataFrame missing column '{col_bg}' "
                f"for scenario '{scen}'."
            )
        df[col_bg] = df[col_bg].fillna(0.0)
        col_pathB = f"K_PathB_{scen}"
        df[col_pathB] = df["K_big_central"] + df[col_bg]

    # 4. QA anchors: ensure we have K_PIM_real_central in df
    if "K_PIM_real_central" not in df.columns:
        anchor = df_pim_annual[["year", "K_PIM_real_central"]].copy()
        anchor["month"] = pd.to_datetime(anchor["year"].astype(int).astype(str) + "-12-01")
        df = df.merge(anchor[["month", "K_PIM_real_central"]], on="month", how="left")

    # gap for central scenario only
    df["gap_PathB_minus_PIM_central"] = df[f"K_PathB_central"] - df["K_PIM_real_central"]

    # 5. Set identifiers and order columns
    df["port"] = PORT_NAME
    df["company"] = COMPANY_NAME
    df["operator_or_owner"] = OPERATOR_NAME

    cols_front = ["port", "company", "operator_or_owner", "month", "year"]
    cols_flags = ["imputed_2022", "flows_imputed_annual", "gap_years_from_prev_annual"]
    cols_events = ["lumpy_event_month", "num_projects_commissioned"]

    cols_K = ["K_big_central", "K_PIM_real_central", "gap_PathB_minus_PIM_central"]
    for scen in scenarios:
        cols_K.append(f"K_bg_{scen}")
    for scen in scenarios:
        cols_K.append(f"K_PathB_{scen}")

    other_cols = [
        c
        for c in df.columns
        if c not in cols_front + cols_flags + cols_K + cols_events
    ]

    ordered = cols_front + cols_flags + cols_K + cols_events + other_cols
    df = df[ordered].sort_values(["company", "month"]).reset_index(drop=True)

    print("[combine_big_and_bg] Done combining; Path B monthly K (all scenarios) is ready.")
    return df


# ======================================================================
# Main orchestration
# ======================================================================

def run_pathB():
    """
    Orchestrate Path B construction for ALL depreciation scenarios:

      1. Load annual PIM backbone.
      2. Load monthly skeleton from Path A.
      3. Build deflator-by-year from Step1 real financials.
      4. Load big project table and compute real costs + asset lives.
      5. Build monthly big-K (K_big,t) and lumpy event flags.
      6. Compute annual K_big_y and K_bg_y^s from PIM for each scenario s.
      7. Build monthly background K_bg,t^s by log-linear interpolation.
      8. Combine big and background into Path B monthly K for each scenario.
      9. Save outputs and basic QA (central scenario).

    Output:
      - 03_K_B_monthly_Haifa_PathB.tsv
      - 03_K_B_monthly_Haifa_PathB_sample.csv
      - 03_PathB_config.json
      - 03_K_B_PathB_QA.tsv (year-end QA, central scenario)
    """
    # 1. Annual PIM backbone
    df_pim = load_annual_pim(ANNUAL_PIM_PATH)

    # 2. Monthly skeleton (dates + flags)
    df_monthly_skel = load_monthly_skeleton(MONTHLY_PIM_LIN_PATH)
    monthly_index = pd.DatetimeIndex(df_monthly_skel["month"].sort_values().unique())

    # 3. Deflator-by-year from step1 real financials
    deflator_by_year = load_deflator_by_year(STEP1_REAL_PATH)

    # 4. Big project table
    projects_df = load_big_projects(BIG_PROJECTS_PATH, deflator_by_year)

    # 5. Monthly big-K
    df_big_monthly = build_all_projects_big_K(projects_df, monthly_index)

    # 6. Annual K_big_y and K_bg_y^s
    annual_bg_df, scenarios = compute_annual_big_and_bg_K(df_big_monthly, df_pim)

    # 7. Monthly background K_bg,t^s
    df_bg_monthly = build_monthly_background_K_bg(annual_bg_df, monthly_index, scenarios)

    # 8. Combine into Path B monthly K (all scenarios)
    df_pathB = combine_big_and_bg(
        df_monthly_skel,
        df_big_monthly,
        df_bg_monthly,
        df_pim,
        scenarios,
    )

    # 9a. Save full monthly Path B series
    df_pathB.to_csv(OUT_PATHB_MONTHLY, sep="\t", index=False)
    print(f"[run_pathB] Saved Path B monthly K to: {OUT_PATHB_MONTHLY}")

    # 9b. Save sample preview
    df_pathB.head(24).to_csv(OUT_PATHB_SAMPLE, index=False)
    print(f"[run_pathB] Saved Path B sample (first 24 rows) to: {OUT_PATHB_SAMPLE}")

    # 9c. Save simple QA at annual anchors (Decembers, central scenario)
    qa_rows = []
    for _, row in df_pim.iterrows():
        year = int(row["year"])
        dec_month = pd.Timestamp(f"{year}-12-01")
        row_dec = df_pathB[df_pathB["month"] == dec_month]
        if row_dec.empty:
            continue
        K_pim_central = float(row["K_PIM_real_central"])
        K_pathB_central = float(row_dec["K_PathB_central"].iloc[0])
        gap = K_pathB_central - K_pim_central
        qa_rows.append(
            {
                "year": year,
                "K_PIM_real_central": K_pim_central,
                "K_PathB_central_dec": K_pathB_central,
                "gap_PathB_minus_PIM_central_dec": gap,
            }
        )
    if qa_rows:
        df_qa = pd.DataFrame(qa_rows).sort_values("year")
        df_qa.to_csv(OUT_PATHB_QA, sep="\t", index=False)
        print(f"[run_pathB] Saved annual QA table to: {OUT_PATHB_QA}")
    else:
        print("[run_pathB] WARNING: No QA rows generated (no matching Decembers?).")

    # 9d. Save config / metadata
    config = {
        "description": (
            "Path B monthly capital stock for Haifa Port Company (legacy), "
            "built from annual PIM (Track B) plus lumpy big projects and "
            "smooth background K, for all depreciation scenarios."
        ),
        "inputs": {
            "annual_pim_file": ANNUAL_PIM_PATH.name,
            "monthly_pim_lin_file": MONTHLY_PIM_LIN_PATH.name,
            "big_projects_file": BIG_PROJECTS_PATH.name,
            "step1_real_file": STEP1_REAL_PATH.name,
        },
        "outputs": {
            "pathB_monthly_file": OUT_PATHB_MONTHLY.name,
            "pathB_monthly_sample_file": OUT_PATHB_SAMPLE.name,
            "pathB_QA_file": OUT_PATHB_QA.name,
        },
        "port": PORT_NAME,
        "company": COMPANY_NAME,
        "operator_or_owner": OPERATOR_NAME,
        "defaults": {
            "default_asset_life_years": DEFAULT_ASSET_LIFE_YEARS,
            "per_project_depreciation_rule": "delta_annual = 1 / asset_life_years; monthly delta from geometric equivalence",
            "background_interpolation": "log-linear between annual K_bg_y^s at December year-ends for each scenario s",
        },
    }
    with open(OUT_PATHB_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[run_pathB] Saved Path B config metadata to: {OUT_PATHB_CONFIG}")

    print("\n[run_pathB] Preview of Path B monthly K (first 24 rows):")
    print(df_pathB.head(24))


if __name__ == "__main__":
    run_pathB()

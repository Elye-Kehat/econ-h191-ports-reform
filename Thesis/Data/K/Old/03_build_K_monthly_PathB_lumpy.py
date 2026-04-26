"""
03_build_K_monthly_PathB_lumpy.py  (FULL REWRITE)

Goal
----
Construct a "Path B" monthly capital stock series for Haifa Port Company (legacy)
that respects *annual* PIM year-end anchors, while allowing *within-year* timing
to be lumpy using a curated "big projects" table.

Core decomposition (for each depreciation scenario s in annual PIM):
    K_PIM_y^s  (December, observed/constructed annual anchor)
      = K_big_y (scenario-invariant, built from project timing + per-project depreciation)
      + K_bg_y^s (residual background)

Then interpolate K_bg_y^s smoothly within year, and form:
    K_PathB_t^s = K_big_t + K_bg_t^s

Key fixes vs the prior version
------------------------------
1) No more "nearest month snapping" that can pull post-sample projects into last sample month.
2) Pre-sample projects enter at a *depreciated* level at sample start.
3) Dispose/transfer/non-PPE projects excluded by default (rule-based, auditable).
4) Commissioning-month inference is explicit + stored with a "commission_source".
5) Asset life parsing from strings is explicit (no silent default without a flag).
6) Strong sanity-check prints + QA outputs.

Inputs (expected in same folder as this script)
-----------------------------------------------
- 01_K_B_annual_Haifa_PIM.tsv
- 02_K_B_monthly_Haifa_PIM_lin.tsv
- Haifa_big_projects.csv
- 00_haifa_financials_step1_real.tsv   (only needed if project real costs are not provided)

Outputs
-------
- 03_K_B_monthly_Haifa_PathB.tsv
- 03_K_B_monthly_Haifa_PathB_sample.csv
- 03_PathB_config.json
- 03_K_B_PathB_QA.tsv
- 03_PathB_projects_QA.tsv   (NEW, project-level audit table)
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

DATA_DIR = Path(__file__).resolve().parent

# Inputs
ANNUAL_PIM_PATH = DATA_DIR / "01_K_B_annual_Haifa_PIM.tsv"
MONTHLY_PIM_LIN_PATH = DATA_DIR / "02_K_B_monthly_Haifa_PIM_lin.tsv"
BIG_PROJECTS_PATH = DATA_DIR / "Haifa_big_projects.csv"
STEP1_REAL_PATH = DATA_DIR / "00_haifa_financials_step1_real.tsv"  # optional: only if nominal -> real needed

# Outputs
OUT_PATHB_MONTHLY = DATA_DIR / "03_K_B_monthly_Haifa_PathB.tsv"
OUT_PATHB_SAMPLE = DATA_DIR / "03_K_B_monthly_Haifa_PathB_sample.csv"
OUT_PATHB_CONFIG = DATA_DIR / "03_PathB_config.json"
OUT_PATHB_QA = DATA_DIR / "03_K_B_PathB_QA.tsv"
OUT_PROJECTS_QA = DATA_DIR / "03_PathB_projects_QA.tsv"

# Labels
PORT_NAME = "Haifa"
COMPANY_NAME = "Haifa Port Company (legacy)"
OPERATOR_NAME = "Haifa Port Company (legacy)"

# Default asset life (years) if we cannot parse life from project row
DEFAULT_ASSET_LIFE_YEARS = 28.0

# Project inclusion rules (conservative by default)
# - include only these event types as "additions" to operator capital
INCLUDE_EVENT_TYPES = {
    "addition",
    "addition_and_upgrade",
    # If you later decide waterfront relocation is PPE, add "relocation" here.
    # "relocation",
}

# Always exclude these event types (even if user mistakenly flags as relevant)
EXCLUDE_EVENT_TYPES = {
    "disposal",
    "operating_project_not_ppe",
}

# If a project is outside sample bounds, we do not include it in the monthly series.
# (We still show it in project QA.)
EXCLUDE_POST_SAMPLE = True

# Threshold for suspicious "big jumps" without a commission event (in your K units)
SUSPICIOUS_JUMP_THRESHOLD = 1e-6  # set tiny because jumps should coincide with event flags


# ======================================================================
# Utility helpers
# ======================================================================

def _norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def _to_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None

def _months_between(t0: pd.Timestamp, t1: pd.Timestamp) -> int:
    """Number of whole months from t0 to t1 (t1 - t0), assuming both are 1st of month."""
    return (t1.year - t0.year) * 12 + (t1.month - t0.month)

def _as_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def _parse_year_from_fs(s: str) -> Optional[int]:
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else None

def _parse_boolish(x) -> int:
    s = _norm_str(x).upper()
    if s in {"1", "TRUE", "T", "YES", "Y"}:
        return 1
    if s in {"0", "FALSE", "F", "NO", "N"}:
        return 0
    # default to 1 if ambiguous (your prior behavior)
    return 1

def _parse_life_years(raw) -> Tuple[float, str]:
    """
    Parse asset life (years) robustly.

    Returns (life_years_used, life_source).
    life_source ∈ {"numeric", "parsed_from_string", "default"}
    """
    if pd.isna(raw):
        return DEFAULT_ASSET_LIFE_YEARS, "default"

    # If already numeric-ish
    try:
        val = float(raw)
        if np.isfinite(val) and val > 0:
            return float(val), "numeric"
    except Exception:
        pass

    s = _norm_str(raw)
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if nums:
        # conservative default: pick max (slower depreciation => higher K)
        life = max(float(n) for n in nums)
        if life > 0:
            return life, "parsed_from_string"

    return DEFAULT_ASSET_LIFE_YEARS, "default"

def _delta_month_from_life(life_years: float) -> float:
    """Geometric equivalence: annual delta = 1/life, convert to monthly delta."""
    life_years = float(life_years)
    if life_years <= 0:
        raise ValueError(f"asset life must be positive; got {life_years}")
    delta_annual = 1.0 / life_years
    return 1.0 - (1.0 - delta_annual) ** (1.0 / 12.0)


# ======================================================================
# Load inputs
# ======================================================================

def load_annual_pim(pim_path: Path) -> pd.DataFrame:
    print(f"[load_annual_pim] Reading annual PIM: {pim_path}")
    if not pim_path.exists():
        raise FileNotFoundError(
            f"Missing {pim_path.name}. You must run 01_build_K_annual_PIM.py first "
            f"or place the output TSV next to this script."
        )

    df = pd.read_csv(pim_path, sep="\t")
    required = {"company", "year", "K_PIM_real_central"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Annual PIM missing columns {missing}. Columns found: {list(df.columns)}")

    df = df[df["company"] == COMPANY_NAME].copy()
    if df.empty:
        raise ValueError(f"No rows for company='{COMPANY_NAME}' in annual PIM file.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)

    scenario_cols = [c for c in df.columns if c.startswith("K_PIM_real_")]
    scenarios = [c.replace("K_PIM_real_", "") for c in scenario_cols]
    if "central" not in scenarios:
        raise ValueError("Annual PIM must contain K_PIM_real_central at minimum.")

    print(f"[load_annual_pim] Years: {df['year'].astype(int).tolist()}")
    print(f"[load_annual_pim] Scenarios: {scenarios}")
    return df


def load_monthly_skeleton(monthly_path: Path) -> pd.DataFrame:
    print(f"[load_monthly_skeleton] Reading monthly skeleton: {monthly_path}")
    if not monthly_path.exists():
        raise FileNotFoundError(
            f"Missing {monthly_path.name}. You must run 02_build_K_monthly_PIM_linear.py first "
            f"or place the output TSV next to this script."
        )

    df = pd.read_csv(monthly_path, sep="\t")
    required = {"port", "company", "month"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Monthly skeleton missing columns {missing}. Columns found: {list(df.columns)}")

    df = df[(df["port"] == PORT_NAME) & (df["company"] == COMPANY_NAME)].copy()
    if df.empty:
        raise ValueError(f"No rows for port='{PORT_NAME}', company='{COMPANY_NAME}' in monthly skeleton.")

    df["month"] = pd.to_datetime(df["month"])
    df["month"] = df["month"].map(_as_month_start)
    df["year"] = df["month"].dt.year

    # Ensure flags exist
    if "operator_or_owner" not in df.columns:
        df["operator_or_owner"] = OPERATOR_NAME
    if "imputed_2022" not in df.columns:
        df["imputed_2022"] = (df["year"] == 2022).astype(int)
    if "flows_imputed_annual" not in df.columns:
        df["flows_imputed_annual"] = False
    if "gap_years_from_prev_annual" not in df.columns:
        df["gap_years_from_prev_annual"] = 0

    df = df.sort_values("month").reset_index(drop=True)

    # check full monthly grid (optional but helpful)
    full_grid = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    missing_months = sorted(set(full_grid) - set(df["month"]))
    if missing_months:
        print(f"[load_monthly_skeleton] WARNING: skeleton is missing {len(missing_months)} months.")
        print(f"  First few missing: {[d.date() for d in missing_months[:6]]}")

    print(f"[load_monthly_skeleton] Sample months: {df['month'].min().date()} → {df['month'].max().date()} ({len(df)} rows)")
    return df


def load_deflator_by_year(step1_path: Path) -> Dict[int, float]:
    """
    Only used if projects do not provide real costs.
    """
    print(f"[load_deflator_by_year] Reading Step1 deflators: {step1_path}")
    if not step1_path.exists():
        raise FileNotFoundError(
            f"Need deflators to convert nominal project costs to real, but {step1_path.name} is missing. "
            f"Either provide it or add a real-cost column in Haifa_big_projects.csv."
        )

    df = pd.read_csv(step1_path, sep="\t")
    if "year" not in df.columns or "deflator" not in df.columns:
        raise ValueError(f"{step1_path.name} must include 'year' and 'deflator' columns.")

    if "company" in df.columns:
        df = df[df["company"] == COMPANY_NAME].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "deflator"]).sort_values("year")

    out = df.drop_duplicates("year").set_index("year")["deflator"].to_dict()
    out = {int(k): float(v) for k, v in out.items()}
    print(f"[load_deflator_by_year] Years available: {sorted(out.keys())}")
    return out


# ======================================================================
# Projects: load, infer commission, cost, filter, QA
# ======================================================================

def infer_commission_month(row: pd.Series) -> Tuple[pd.Timestamp, str]:
    """
    Infer commissioning month (YYYY-MM-01) + source string.

    Priority:
      1) commissioning_year + commissioning_month
      2) expected_end_year + expected_end_month (default month=12)
      3) start_year + start_month (default month=6)
      4) investment_year (default month=12)
    """
    y = _to_int(row.get("commissioning_year"))
    m = _to_int(row.get("commissioning_month"))
    if y is not None and m is not None:
        m = max(1, min(12, m))
        return pd.Timestamp(year=y, month=m, day=1), "commissioning"

    y = _to_int(row.get("expected_end_year"))
    m = _to_int(row.get("expected_end_month"))
    if y is not None:
        m = max(1, min(12, m if m is not None else 12))
        return pd.Timestamp(year=y, month=m, day=1), "expected_end"

    y = _to_int(row.get("start_year"))
    m = _to_int(row.get("start_month"))
    if y is not None:
        m = max(1, min(12, m if m is not None else 6))
        return pd.Timestamp(year=y, month=m, day=1), "start"

    y = _to_int(row.get("investment_year"))
    if y is not None:
        return pd.Timestamp(year=y, month=12, day=1), "investment_year"

    return pd.NaT, "missing"


def load_and_prepare_projects(
    projects_path: Path,
    sample_min_month: pd.Timestamp,
    sample_max_month: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - projects_included: projects that feed into K_big_t construction (in-sample OR pre-sample)
      - projects_qa: full project QA (including excluded + post-sample)
    """
    print(f"[load_projects] Reading projects: {projects_path}")
    if not projects_path.exists():
        raise FileNotFoundError(f"Missing {projects_path.name}.")

    df = pd.read_csv(projects_path)

    # basic checks
    if "port" not in df.columns:
        raise ValueError("Haifa_big_projects.csv must have a 'port' column.")
    df = df[df["port"] == PORT_NAME].copy()
    if df.empty:
        raise ValueError("No projects for port=Haifa in Haifa_big_projects.csv")

    if "project_id" not in df.columns:
        df["project_id"] = df.index.astype(str)

    # PathB relevance
    rel_col = "relevant_for_K_B" if "relevant_for_K_B" in df.columns else None
    if rel_col:
        df["_pathb_flag"] = df[rel_col].apply(_parse_boolish)
    else:
        df["_pathb_flag"] = 1

    # investment year parse
    invest_col = None
    for cand in ["investment_year", "investment_year_fs", "capex_year_fs"]:
        if cand in df.columns:
            invest_col = cand
            break
    if invest_col is None:
        raise ValueError("Projects file must include investment_year or investment_year_fs.")

    raw = df[invest_col].astype(str)
    yr = raw.map(_parse_year_from_fs)
    df["investment_year"] = pd.Series(yr, index=df.index).astype("Int64")

    # fallbacks
    for fallback in ["start_year", "expected_end_year", "commissioning_year"]:
        if fallback in df.columns:
            mask = df["investment_year"].isna()
            df.loc[mask, "investment_year"] = pd.to_numeric(df.loc[mask, fallback], errors="coerce").astype("Int64")

    if df["investment_year"].isna().any():
        bad = df[df["investment_year"].isna()][["project_id", invest_col]]
        raise ValueError(f"Some projects have no usable investment_year: {bad.to_dict(orient='records')}")

    # event type normalization
    if "event_type" in df.columns:
        df["_event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    else:
        df["_event_type"] = ""

    # cost columns
    # Prefer real cost if already present
    real_col = None
    for cand in ["cost_real_thousands_nis", "amount_real_thousands_nis", "real_cost_th_nis"]:
        if cand in df.columns:
            real_col = cand
            break

    nom_col = None
    for cand in ["amount_nominal_th_nis", "cost_nominal_thousands_nis", "project_cost_thousands_nis"]:
        if cand in df.columns:
            nom_col = cand
            break

    if real_col is None and nom_col is None:
        raise ValueError("Projects file must include a real cost column or a nominal cost column.")

    # asset life parse
    if "asset_life_years" in df.columns:
        life_parsed = df["asset_life_years"].apply(_parse_life_years)
        df["_life_years_used"] = life_parsed.map(lambda t: t[0])
        df["_life_source"] = life_parsed.map(lambda t: t[1])
    else:
        df["_life_years_used"] = DEFAULT_ASSET_LIFE_YEARS
        df["_life_source"] = "default"

    # commission month inference
    comm = df.apply(infer_commission_month, axis=1)
    df["_commission_month"] = comm.map(lambda t: t[0])
    df["_commission_source"] = comm.map(lambda t: t[1])
    if df["_commission_month"].isna().any():
        bad = df[df["_commission_month"].isna()][["project_id", "commissioning_year", "commissioning_month", "expected_end_year", "start_year", invest_col]]
        raise ValueError(f"Some projects have no commissioning month even after fallbacks: {bad.to_dict(orient='records')}")

    df["_commission_month"] = pd.to_datetime(df["_commission_month"]).map(_as_month_start)

    # In-sample classification
    df["_timing_class"] = "in_sample"
    df.loc[df["_commission_month"] < sample_min_month, "_timing_class"] = "pre_sample"
    df.loc[df["_commission_month"] > sample_max_month, "_timing_class"] = "post_sample"

    # Compute real cost
    df["_cost_nominal"] = pd.to_numeric(df[nom_col], errors="coerce") if nom_col else np.nan
    df["_cost_real"] = pd.to_numeric(df[real_col], errors="coerce") if real_col else np.nan
    df["_cost_source"] = "provided_real" if real_col else "deflated_nominal"

    if real_col is None:
        # need deflators
        defl = load_deflator_by_year(STEP1_REAL_PATH)
        real_vals = []
        for _, r in df.iterrows():
            y = int(r["investment_year"])
            if y not in defl:
                avail = sorted(defl.keys())
                nearest = min(avail, key=lambda yy: abs(yy - y))
                print(f"[load_projects] WARNING: deflator missing for year={y} (proj {r['project_id']}). Using nearest {nearest}.")
                d = defl[nearest]
            else:
                d = defl[y]
            if pd.isna(r["_cost_nominal"]):
                real_vals.append(np.nan)
            else:
                real_vals.append(float(r["_cost_nominal"]) / float(d))
        df["_cost_real"] = real_vals

    # exclusion rules (auditable)
    df["_excluded"] = 0
    df["_exclude_reason"] = ""

    # not pathB relevant
    mask = df["_pathb_flag"] != 1
    df.loc[mask, "_excluded"] = 1
    df.loc[mask, "_exclude_reason"] = "not_relevant_for_pathB"

    # event type exclusion
    mask = df["_event_type"].isin(EXCLUDE_EVENT_TYPES)
    df.loc[mask, "_excluded"] = 1
    df.loc[mask, "_exclude_reason"] = df.loc[mask, "_exclude_reason"].where(df.loc[mask, "_exclude_reason"] != "", "excluded_event_type")

    # must be in include list (if event_type present)
    # If event_type missing/blank: treat as "unknown" and exclude (conservative)
    mask_unknown = (df["_event_type"] == "")
    df.loc[mask_unknown, "_excluded"] = 1
    df.loc[mask_unknown, "_exclude_reason"] = df.loc[mask_unknown, "_exclude_reason"].where(df.loc[mask_unknown, "_exclude_reason"] != "", "missing_event_type")

    mask_not_in_include = (~df["_event_type"].isin(INCLUDE_EVENT_TYPES)) & (df["_event_type"] != "")
    df.loc[mask_not_in_include, "_excluded"] = 1
    df.loc[mask_not_in_include, "_exclude_reason"] = df.loc[mask_not_in_include, "_exclude_reason"].where(df.loc[mask_not_in_include, "_exclude_reason"] != "", "event_type_not_in_include_set")

    # missing cost => exclude from K_big (it becomes background implicitly)
    mask_cost_missing = df["_cost_real"].isna()
    df.loc[mask_cost_missing, "_excluded"] = 1
    df.loc[mask_cost_missing, "_exclude_reason"] = df.loc[mask_cost_missing, "_exclude_reason"].where(df.loc[mask_cost_missing, "_exclude_reason"] != "", "missing_cost")

    # post-sample exclusion
    if EXCLUDE_POST_SAMPLE:
        mask_post = (df["_timing_class"] == "post_sample")
        df.loc[mask_post, "_excluded"] = 1
        df.loc[mask_post, "_exclude_reason"] = df.loc[mask_post, "_exclude_reason"].where(df.loc[mask_post, "_exclude_reason"] != "", "post_sample")

    # build included subset (pre_sample + in_sample only, excluded removed)
    included = df[(df["_excluded"] == 0) & (df["_timing_class"].isin(["pre_sample", "in_sample"]))].copy()

    # build QA table (all rows)
    qa_cols = [
        "project_id",
        "project_name_short" if "project_name_short" in df.columns else None,
        "project_type" if "project_type" in df.columns else None,
        "event_type" if "event_type" in df.columns else None,
        "asset_class" if "asset_class" in df.columns else None,
        "investment_year",
        "_commission_month",
        "_commission_source",
        "_timing_class",
        "_cost_real",
        "_cost_source",
        "_life_years_used",
        "_life_source",
        "_pathb_flag",
        "_excluded",
        "_exclude_reason",
    ]
    qa_cols = [c for c in qa_cols if c is not None and c in df.columns or c.startswith("_")]

    projects_qa = df.copy()
    # Ensure these exist for printing
    if "project_name_short" not in projects_qa.columns:
        projects_qa["project_name_short"] = ""

    projects_qa_out = projects_qa[[
        "project_id",
        "project_name_short",
        "project_type" if "project_type" in projects_qa.columns else "project_name_short",
        "event_type" if "event_type" in projects_qa.columns else "project_name_short",
        "asset_class" if "asset_class" in projects_qa.columns else "project_name_short",
        "investment_year",
        "_commission_month",
        "_commission_source",
        "_timing_class",
        "_cost_real",
        "_cost_source",
        "_life_years_used",
        "_life_source",
        "_pathb_flag",
        "_excluded",
        "_exclude_reason",
    ]].copy()
    # normalize placeholder columns if missing
    for col in ["project_type", "event_type", "asset_class"]:
        if col not in projects_qa_out.columns:
            projects_qa_out[col] = ""

    # prints
    print("\n[load_projects] --- PROJECTS SUMMARY ---")
    print(f"  Total rows (Haifa): {len(df)}")
    print(f"  PathB-flagged: {int(df['_pathb_flag'].sum())} (flag parsing default=1 for ambiguous)")
    print(f"  Included in K_big: {len(included)}")
    print(f"    in-sample: {int((included['_timing_class']=='in_sample').sum())}")
    print(f"    pre-sample: {int((included['_timing_class']=='pre_sample').sum())}")
    print(f"  Excluded: {int((df['_excluded']==1).sum())}")
    if (df["_excluded"] == 1).any():
        print("  Exclusion reasons counts:")
        print(df.loc[df["_excluded"] == 1, "_exclude_reason"].value_counts().to_string())

    # Save project QA
    projects_qa_out.to_csv(OUT_PROJECTS_QA, sep="\t", index=False)
    print(f"[load_projects] Saved project QA: {OUT_PROJECTS_QA}")

    return included, projects_qa_out


# ======================================================================
# Build K_big,t
# ======================================================================

def build_project_series(
    proj_id: str,
    cost_real: float,
    life_years: float,
    commission_month: pd.Timestamp,
    timing_class: str,
    monthly_index: pd.DatetimeIndex,
    sample_min_month: pd.Timestamp,
    sample_max_month: pd.Timestamp,
) -> pd.Series:
    """
    Construct monthly project capital stock series under geometric depreciation.

    - pre_sample: initialize at sample start with depreciation already applied
    - in_sample: jump at commissioning month
    - post_sample: should not be called (excluded earlier); returns zeros if called
    """
    delta_m = _delta_month_from_life(life_years)

    K = np.zeros(len(monthly_index), dtype=float)

    if timing_class == "post_sample":
        return pd.Series(K, index=monthly_index, name=f"K_proj_{proj_id}")

    if timing_class == "pre_sample":
        # asset exists before sample; bring in at depreciated level at sample start
        age = _months_between(commission_month, sample_min_month)
        # if commission_month is after sample_min_month unexpectedly, treat as in-sample
        if age < 0:
            age = 0
        K0 = float(cost_real) * ((1.0 - delta_m) ** age)
        K[0] = K0
        for i in range(1, len(monthly_index)):
            K[i] = (1.0 - delta_m) * K[i - 1]
        return pd.Series(K, index=monthly_index, name=f"K_proj_{proj_id}")

    # in_sample
    t0 = commission_month
    if t0 < sample_min_month or t0 > sample_max_month:
        # should not happen given classification
        return pd.Series(K, index=monthly_index, name=f"K_proj_{proj_id}")

    if t0 not in monthly_index:
        # if skeleton missing some months, we "ceiling" to next available month
        future = monthly_index[monthly_index >= t0]
        if len(future) == 0:
            return pd.Series(K, index=monthly_index, name=f"K_proj_{proj_id}")
        t0_eff = future[0]
        print(f"[build_project_series] WARNING: {proj_id} commission {t0.date()} not in index, using next available {t0_eff.date()}")
        t0 = t0_eff

    start_idx = int(np.where(monthly_index == t0)[0][0])
    K[start_idx] = float(cost_real)
    for i in range(start_idx + 1, len(monthly_index)):
        K[i] = (1.0 - delta_m) * K[i - 1]
    return pd.Series(K, index=monthly_index, name=f"K_proj_{proj_id}")


def build_K_big(
    included_projects: pd.DataFrame,
    monthly_index: pd.DatetimeIndex,
    sample_min_month: pd.Timestamp,
    sample_max_month: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, List[str]]]:
    """
    Returns:
      - df_big: month, K_big_central, num_projects_commissioned, lumpy_event_month
      - commission_map: month -> list of project_ids commissioned in-sample that month
    """
    K_big = np.zeros(len(monthly_index), dtype=float)
    event_counts = np.zeros(len(monthly_index), dtype=int)
    commission_map: Dict[pd.Timestamp, List[str]] = {}

    print("\n[build_K_big] Building K_big,t from included projects...")
    for _, r in included_projects.iterrows():
        pid = str(r["project_id"])
        cost = float(r["_cost_real"])
        life = float(r["_life_years_used"])
        comm = pd.to_datetime(r["_commission_month"])
        timing = str(r["_timing_class"])

        series = build_project_series(
            proj_id=pid,
            cost_real=cost,
            life_years=life,
            commission_month=comm,
            timing_class=timing,
            monthly_index=monthly_index,
            sample_min_month=sample_min_month,
            sample_max_month=sample_max_month,
        )
        K_big += series.to_numpy()

        if timing == "in_sample":
            # commission month must be within sample; count event
            if comm in monthly_index:
                idx = int(np.where(monthly_index == comm)[0][0])
                event_counts[idx] += 1
                commission_map.setdefault(comm, []).append(pid)

    df_big = pd.DataFrame({
        "month": monthly_index,
        "K_big_central": K_big,
        "num_projects_commissioned": event_counts,
    })
    df_big["lumpy_event_month"] = (df_big["num_projects_commissioned"] > 0).astype(int)

    print(f"[build_K_big] Done. K_big range: {df_big['K_big_central'].min():.3f} → {df_big['K_big_central'].max():.3f}")
    print(f"[build_K_big] Total in-sample commissioning events: {int(df_big['num_projects_commissioned'].sum())}")
    return df_big, commission_map


# ======================================================================
# Annual decomposition and monthly background interpolation
# ======================================================================

def compute_annual_big_and_bg(
    df_big_monthly: pd.DataFrame,
    df_pim_annual: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    At each year y (December), compute K_big_y and residual K_bg_y^s for each scenario s.
    Clamps negative background to epsilon and prints warnings.
    """
    scenario_cols = [c for c in df_pim_annual.columns if c.startswith("K_PIM_real_")]
    scenarios = [c.replace("K_PIM_real_", "") for c in scenario_cols]

    eps = 1e-6
    rows = []

    for _, r in df_pim_annual.iterrows():
        y = int(r["year"])
        dec = pd.Timestamp(year=y, month=12, day=1)

        dec_row = df_big_monthly[df_big_monthly["month"] == dec]
        if dec_row.empty:
            raise ValueError(f"No monthly row for December {dec.date()} in big-K series.")

        K_big_y = float(dec_row["K_big_central"].iloc[0])

        out = {"year": y, "K_big_y": K_big_y}

        # carry any audit cols in annual PIM if present (keeps prior behavior)
        for col in ["flows_imputed_flag", "gap_years_from_prev"]:
            if col in df_pim_annual.columns:
                out[col] = r[col]

        for scen in scenarios:
            col = f"K_PIM_real_{scen}"
            K_pim = float(r[col])
            K_bg = K_pim - K_big_y
            if K_bg < 0:
                print(
                    f"[compute_annual_big_and_bg] WARNING: negative background at {y} scen={scen}: "
                    f"K_bg={K_bg:.3f} (K_pim={K_pim:.3f}, K_big={K_big_y:.3f}). Clamping to {eps}."
                )
                K_bg = eps
            out[col] = K_pim
            out[f"K_bg_y_{scen}"] = K_bg

        rows.append(out)

    annual_bg = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return annual_bg, scenarios


def interpolate_monthly_bg(
    annual_bg: pd.DataFrame,
    monthly_index: pd.DatetimeIndex,
    scenarios: List[str],
) -> pd.DataFrame:
    """
    Log-linear interpolation of K_bg between December anchors for each scenario.
    """
    df_month = pd.DataFrame({"month": monthly_index}).set_index("month")

    for scen in scenarios:
        anchor = annual_bg[["year", f"K_bg_y_{scen}"]].copy()
        anchor["month"] = pd.to_datetime(anchor["year"].astype(int).astype(str) + "-12-01")
        anchor = anchor.set_index("month")

        tmp = df_month.copy()
        tmp["K_bg_anchor"] = anchor[f"K_bg_y_{scen}"]

        # interpolate in logs over time
        logK = np.log(tmp["K_bg_anchor"])
        logK_i = logK.interpolate(method="time")
        df_month[f"K_bg_{scen}"] = np.exp(logK_i)

    return df_month.reset_index()


# ======================================================================
# Combine into PathB output
# ======================================================================

def combine_output(
    df_skel: pd.DataFrame,
    df_big: pd.DataFrame,
    df_bg: pd.DataFrame,
    df_pim_annual: pd.DataFrame,
    scenarios: List[str],
) -> pd.DataFrame:
    df = df_skel.copy()
    df["month"] = pd.to_datetime(df["month"]).map(_as_month_start)

    df_big = df_big.copy()
    df_big["month"] = pd.to_datetime(df_big["month"]).map(_as_month_start)

    df_bg = df_bg.copy()
    df_bg["month"] = pd.to_datetime(df_bg["month"]).map(_as_month_start)

    df = df.merge(df_big, on="month", how="left")
    df = df.merge(df_bg, on="month", how="left")

    df["K_big_central"] = df["K_big_central"].fillna(0.0)

    for scen in scenarios:
        df[f"K_bg_{scen}"] = df[f"K_bg_{scen}"].fillna(0.0)
        df[f"K_PathB_{scen}"] = df["K_big_central"] + df[f"K_bg_{scen}"]

    # Ensure we have a central anchor column (existing skeleton usually carries many)
    if "K_PIM_real_central" not in df.columns:
        anchor = df_pim_annual[["year", "K_PIM_real_central"]].copy()
        anchor["month"] = pd.to_datetime(anchor["year"].astype(int).astype(str) + "-12-01")
        df = df.merge(anchor[["month", "K_PIM_real_central"]], on="month", how="left")

    df["gap_PathB_minus_PIM_central"] = df["K_PathB_central"] - df["K_PIM_real_central"]

    # Set identifiers
    df["port"] = PORT_NAME
    df["company"] = COMPANY_NAME
    df["operator_or_owner"] = OPERATOR_NAME

    # Order columns to preserve compatibility with your existing output
    cols_front = ["port", "company", "operator_or_owner", "month", "year"]
    cols_flags = ["imputed_2022", "flows_imputed_annual", "gap_years_from_prev_annual"]
    cols_events = ["lumpy_event_month", "num_projects_commissioned"]

    cols_K = ["K_big_central", "K_PIM_real_central", "gap_PathB_minus_PIM_central"]
    for scen in scenarios:
        cols_K.append(f"K_bg_{scen}")
    for scen in scenarios:
        cols_K.append(f"K_PathB_{scen}")

    # keep any other columns from skeleton (K_PIM_lin_*, etc.) to avoid downstream breakage
    other_cols = [c for c in df.columns if c not in cols_front + cols_flags + cols_events + cols_K]
    df = df[cols_front + cols_flags + cols_K + cols_events + other_cols].sort_values("month").reset_index(drop=True)

    return df


# ======================================================================
# Sanity checks / prints
# ======================================================================

def sanity_checks(
    df_pathB: pd.DataFrame,
    df_pim_annual: pd.DataFrame,
    commission_map: Dict[pd.Timestamp, List[str]],
):
    print("\n[sanity] --- YEAR-END ANCHOR CHECK (central) ---")
    qa_rows = []
    for _, r in df_pim_annual.iterrows():
        y = int(r["year"])
        dec = pd.Timestamp(year=y, month=12, day=1)
        row = df_pathB[df_pathB["month"] == dec]
        if row.empty:
            print(f"[sanity] WARNING: missing December row for {y}-12-01")
            continue
        pim = float(r["K_PIM_real_central"])
        pathb = float(row["K_PathB_central"].iloc[0])
        big = float(row["K_big_central"].iloc[0])
        share = big / pathb if pathb != 0 else np.nan
        gap = pathb - pim
        qa_rows.append({
            "year": y,
            "K_PIM_real_central": pim,
            "K_PathB_central_dec": pathb,
            "gap_dec": gap,
            "K_big_dec": big,
            "share_big_in_total_dec": share,
        })
    df_qa = pd.DataFrame(qa_rows).sort_values("year")
    print(df_qa.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
    df_qa.to_csv(OUT_PATHB_QA, sep="\t", index=False)
    print(f"[sanity] Saved annual QA: {OUT_PATHB_QA}")

    # Check that K_big jumps correspond to events (in-sample)
    print("\n[sanity] --- CHECK: big jumps coincide with commission events ---")
    df = df_pathB.sort_values("month").reset_index(drop=True).copy()
    df["dK_big"] = df["K_big_central"].diff().fillna(0.0)

    suspicious = df[(df["dK_big"] > SUSPICIOUS_JUMP_THRESHOLD) & (df["lumpy_event_month"] == 0)]
    if suspicious.empty:
        print("[sanity] OK: No positive K_big jumps without event flags.")
    else:
        print("[sanity] WARNING: Found K_big jumps without event flags:")
        show = suspicious[["month", "K_big_central", "dK_big", "lumpy_event_month", "num_projects_commissioned"]].copy()
        print(show.to_string(index=False))
        # Try to diagnose: list any projects that were commissioned that month (should be none)
        for _, rr in suspicious.iterrows():
            m = pd.to_datetime(rr["month"])
            pids = commission_map.get(m, [])
            print(f"  Month {m.date()} commission_map projects: {pids}")

    # Print commissioning months and projects (in-sample only)
    print("\n[sanity] --- IN-SAMPLE COMMISSION EVENTS ---")
    if commission_map:
        for m in sorted(commission_map.keys()):
            print(f"  {m.date()} : {commission_map[m]}")
    else:
        print("  (none)")

    # Last-month check
    last = df["month"].max()
    last_row = df[df["month"] == last].iloc[0]
    print("\n[sanity] --- LAST MONTH SUMMARY ---")
    print(f"  last month: {pd.to_datetime(last).date()}")
    print(f"  K_big: {last_row['K_big_central']:.6f}")
    print(f"  events that month: {int(last_row['num_projects_commissioned'])} (flag={int(last_row['lumpy_event_month'])})")


# ======================================================================
# Main
# ======================================================================

def run_pathB():
    df_pim = load_annual_pim(ANNUAL_PIM_PATH)
    df_skel = load_monthly_skeleton(MONTHLY_PIM_LIN_PATH)

    monthly_index = pd.DatetimeIndex(df_skel["month"].sort_values().unique())
    sample_min = pd.to_datetime(monthly_index.min())
    sample_max = pd.to_datetime(monthly_index.max())

    print("\n[run_pathB] --- SAMPLE WINDOW ---")
    print(f"  sample_min_month = {sample_min.date()}")
    print(f"  sample_max_month = {sample_max.date()}")
    print(f"  N months          = {len(monthly_index)}")

    included_projects, projects_qa = load_and_prepare_projects(
        BIG_PROJECTS_PATH,
        sample_min_month=sample_min,
        sample_max_month=sample_max,
    )

    # Build K_big,t
    df_big, commission_map = build_K_big(
        included_projects=included_projects,
        monthly_index=monthly_index,
        sample_min_month=sample_min,
        sample_max_month=sample_max,
    )

    # Annual decomposition + background interpolation
    annual_bg, scenarios = compute_annual_big_and_bg(df_big, df_pim)
    df_bg = interpolate_monthly_bg(annual_bg, monthly_index, scenarios)

    # Combine
    df_pathB = combine_output(df_skel, df_big, df_bg, df_pim, scenarios)

    # Save outputs
    df_pathB.to_csv(OUT_PATHB_MONTHLY, sep="\t", index=False)
    df_pathB.head(24).to_csv(OUT_PATHB_SAMPLE, index=False)

    # Config metadata
    cfg = {
        "description": "Path B monthly K for Haifa Port Company (legacy): annual PIM anchors + lumpy projects + residual background interpolation.",
        "inputs": {
            "annual_pim_file": ANNUAL_PIM_PATH.name,
            "monthly_skeleton_file": MONTHLY_PIM_LIN_PATH.name,
            "big_projects_file": BIG_PROJECTS_PATH.name,
            "step1_real_file_used_if_needed": STEP1_REAL_PATH.name,
        },
        "outputs": {
            "pathB_monthly_file": OUT_PATHB_MONTHLY.name,
            "pathB_sample_file": OUT_PATHB_SAMPLE.name,
            "annual_QA_file": OUT_PATHB_QA.name,
            "project_QA_file": OUT_PROJECTS_QA.name,
        },
        "defaults": {
            "default_asset_life_years": DEFAULT_ASSET_LIFE_YEARS,
            "include_event_types": sorted(list(INCLUDE_EVENT_TYPES)),
            "exclude_event_types": sorted(list(EXCLUDE_EVENT_TYPES)),
            "exclude_post_sample_projects": EXCLUDE_POST_SAMPLE,
            "big_project_depreciation": "per-project delta_annual = 1/life_years (life parsed); monthly delta via geometric equivalence",
            "background_interpolation": "log-linear between December anchors for each scenario",
        },
    }
    with open(OUT_PATHB_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n[run_pathB] Saved: {OUT_PATHB_MONTHLY}")
    print(f"[run_pathB] Saved: {OUT_PATHB_SAMPLE}")
    print(f"[run_pathB] Saved: {OUT_PATHB_CONFIG}")
    print(f"[run_pathB] Saved: {OUT_PROJECTS_QA}")

    # Sanity checks / prints
    sanity_checks(df_pathB, df_pim, commission_map)

    print("\n[run_pathB] --- PREVIEW (first 12 rows) ---")
    print(df_pathB.head(12).to_string(index=False))


if __name__ == "__main__":
    run_pathB()

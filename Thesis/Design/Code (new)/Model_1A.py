#!/usr/bin/env python
"""
Model_1A.py

Event-study for log labor productivity ln(LP) using LP_Panel_monthly.tsv,
with not-yet-treated windows taken directly from Table 1 of the thesis.

Design:
    - For each (reform, regression target) row in Table 1:
        * Mark treatment rows exactly as in the "Treatment months" column.
        * Mark control rows exactly as in the "Control months" column.
        * Define event_time for treated rows using month_index - event_month_index.
        * Set event_time = -1 for all control observations (reference bin).
        * Run ln(LP) on event_time dummies + unit FE + time FE.
    - Specifications:
        * "baseline": event_time dummies + unit FE + time FE.
        * "porttr": add port-specific linear time trends.
        * "tr_shocks": add port-specific linear trends + exogenous shock controls.
    - Export, for each specification separately (and also pooled across specs):
        * A tidy table of dynamic betas beta_m for each event_time m,
          including N(m) (number of unit×month observations at that m).
        * A tidy table of window-average betas beta_[a,b].
        * A tidy table of pre-trend F-tests based on coarse pre bins
          over m ∈ [-12, -2], using HC1-style (unclustered OLS) covariance
          for the Wald test.

The idea is that:
    - Main coefficient SEs are cluster-robust at the series (terminal) level.
    - Pre-trend F-tests are descriptive diagnostics using simple OLS
      covariance to avoid small-cluster pathologies.
    - Wild-cluster bootstrap for inference (if desired) will be applied
      later in a separate script that consumes these TSVs.

Outputs are written to: Thesis/Design/Output (new)/Model_1A.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ----------------------------------------------------------------------
# 0. Paths
# ----------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]  # .../Thesis
LP_PANEL_PATH = THESIS_ROOT / "Data" / "LP" / "LP_Panel_monthly.tsv"

# Output directory: put Model 1A files in a dedicated subfolder
OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1A"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Helper data structures
# ----------------------------------------------------------------------

@dataclass
class Window:
    label: str              # e.g. "Haifa port", "Haifa-Bayport"
    start: Tuple[int, int]  # (year, month) inclusive
    end:   Tuple[int, int]  # (year, month) inclusive


@dataclass
class Spec:
    """One row of Table 1: a (reform, regression target) pair."""
    reform: str             # e.g. "haifa_comp"
    target: str             # e.g. "Haifa-Bayport terminal"
    event_year: int         # calendar year of the reform (for event_time)
    event_month: int        # calendar month of the reform
    treat_windows: List[Window]   # Table 1 "Treatment months"
    control_windows: List[Window] # Table 1 "Control months"


@dataclass
class SpecWithFE:
    """
    One estimation specification applied to a given Table-1 row.

    spec_name:
        "baseline"  -> log_LP ~ event_time dummies + unit FE + time FE
        "porttr"    -> baseline + port-specific linear time trends
        "tr_shocks" -> porttr + exogenous shock controls
    """
    spec: Spec
    spec_name: str
    include_port_trends: bool = False
    include_shocks: bool = False


# ----------------------------------------------------------------------
# 2. Mapping from Table-1 labels to LP_Panel filters
# ----------------------------------------------------------------------

LABEL_FILTERS: Dict[str, Dict[str, str]] = {
    # Port-level monthly series
    "Haifa port":  {"level": "port",    "port": "Haifa"},
    "Ashdod port": {"level": "port",    "port": "Ashdod"},

    # Terminal-level series (monthly-expanded from quarterly LP)
    "Haifa-Bayport":   {"level": "terminal", "port": "Haifa",  "terminal": "Haifa-Bayport"},
    "Haifa-Legacy":    {"level": "terminal", "port": "Haifa",  "terminal": "Haifa-Legacy"},
    "Ashdod-HCT":      {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-HCT"},
    "Ashdod-Legacy":   {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-Legacy"},
}


''''
# ----------------------------------------------------------------------
# 3. Table-1 specs (NYT windows)
# ----------------------------------------------------------------------

NYT_SPECS: List[Spec] = [

    # --- Haifa competition entry (Bayport opens 09-2021) ---
    # Extend to 09-2023 so max post = 24 (m=24 at 2023-09)
    Spec(
        reform="haifa_comp",
        target="Haifa-Bayport terminal",
        event_year=2021,
        event_month=9,  # Bayport opens 09-2021
        treat_windows=[
            Window("Haifa port",      (2019, 9), (2021, 8)),
            Window("Haifa-Bayport",   (2021, 9), (2023, 9)),
        ],
        control_windows=[
            Window("Ashdod port",     (2019, 9), (2021, 7)),
            Window("Ashdod-Legacy",   (2021, 8), (2023, 9)),
            Window("Ashdod-HCT",      (2021, 8), (2023, 9)),
        ],
    ),

    Spec(
        reform="haifa_comp",
        target="Haifa-Legacy terminal",
        event_year=2021,
        event_month=9,
        treat_windows=[
            Window("Haifa port",      (2019, 9), (2021, 8)),
            Window("Haifa-Legacy",    (2021, 9), (2023, 9)),
        ],
        control_windows=[
            Window("Ashdod port",     (2019, 9), (2021, 7)),
            Window("Ashdod-Legacy",   (2021, 8), (2023, 9)),
            Window("Ashdod-HCT",      (2021, 8), (2023, 9)),
        ],
    ),

    # --- Ashdod competition entry (HCT effective 11-2022) ---
    # (unchanged — not central to your Haifa mediation tables)
    Spec(
        reform="ashdod_comp",
        target="Ashdod-HCT terminal",
        event_year=2022,
        event_month=11,  # HCT effective 11-2022
        treat_windows=[
            Window("Ashdod port",     (2020, 11), (2021, 7)),
            Window("Ashdod-HCT",      (2021, 8),  (2023, 9)),
        ],
        control_windows=[
            Window("Haifa port",      (2020, 11), (2021, 8)),
            Window("Haifa-Bayport",   (2021, 9),  (2023, 9)),
            Window("Haifa-Legacy",    (2021, 9),  (2023, 9)),
        ],
    ),

    Spec(
        reform="ashdod_comp",
        target="Ashdod-Legacy terminal",
        event_year=2022,
        event_month=11,
        treat_windows=[
            Window("Ashdod port",     (2020, 11), (2021, 7)),
            Window("Ashdod-Legacy",   (2021, 8),  (2023, 9)),
        ],
        control_windows=[
            Window("Haifa port",      (2020, 11), (2021, 8)),
            Window("Haifa-Bayport",   (2021, 9),  (2023, 9)),
            Window("Haifa-Legacy",    (2021, 9),  (2023, 9)),
        ],
    ),

    # --- Haifa privatization (Haifa-Legacy sold 01-2023) ---
    # EXTENDED: end at 2024-12 so max post = 23 (m=23 at 2024-12)
    Spec(
        reform="haifa_priv",
        target="Haifa-Legacy terminal",
        event_year=2023,
        event_month=1,  # sale in 01-2023
        treat_windows=[
            Window("Haifa port",      (2021, 1), (2021, 8)),
            Window("Haifa-Legacy",    (2021, 9), (2024, 12)),
        ],
        control_windows=[
            Window("Haifa-Bayport",   (2021, 9), (2024, 12)),
            Window("Ashdod port",     (2021, 1), (2021, 7)),
            Window("Ashdod-Legacy",   (2021, 8), (2024, 12)),
            Window("Ashdod-HCT",      (2021, 8), (2024, 12)),
        ],
    ),

    # --- Haifa privatization placebo: Bayport treated at 01-2023 ---
    Spec(
        reform="haifa_priv",
        target="Haifa-Bayport terminal",
        event_year=2023,
        event_month=1,
        treat_windows=[
            Window("Haifa port",      (2021, 1), (2021, 8)),
            Window("Haifa-Bayport",   (2021, 9), (2024, 12)),
        ],
        control_windows=[
            Window("Haifa-Legacy",    (2021, 9), (2024, 12)),
            Window("Ashdod port",     (2021, 1), (2021, 7)),
            Window("Ashdod-Legacy",   (2021, 8), (2024, 12)),
            Window("Ashdod-HCT",      (2021, 8), (2024, 12)),
        ],
    ),

]
'''


# ----------------------------------------------------------------------
# 3. NYT specs (updated Table: Haifa reform clocks only)
# ----------------------------------------------------------------------

NYT_SPECS: List[Spec] = [

    # --- Haifa competition entry (Bayport opens 09-2021) ---
    # NEW: truncate at 10-2022 so Ashdod controls are excluded after Ashdod entry (11-2022).
    Spec(
        reform="haifa_comp",
        target="Haifa-Bayport terminal",
        event_year=2021,
        event_month=9,  # Bayport opens 09-2021
        treat_windows=[
            Window("Haifa port",      (2019, 9), (2021, 8)),   # pre
            Window("Haifa-Bayport",   (2021, 9), (2022, 10)),  # post (truncated)
        ],
        control_windows=[
            Window("Ashdod port",     (2019, 9), (2021, 7)),   # pre (port-level)
            Window("Ashdod-Legacy",   (2021, 8), (2022, 10)),  # terminals (truncated)
            Window("Ashdod-HCT",      (2021, 8), (2022, 10)),
        ],
    ),

    Spec(
        reform="haifa_comp",
        target="Haifa-Legacy terminal",
        event_year=2021,
        event_month=9,
        treat_windows=[
            Window("Haifa port",      (2019, 9), (2021, 8)),   # pre
            Window("Haifa-Legacy",    (2021, 9), (2022, 10)),  # post (truncated)
        ],
        control_windows=[
            Window("Ashdod port",     (2019, 9), (2021, 7)),   # pre (port-level)
            Window("Ashdod-Legacy",   (2021, 8), (2022, 10)),  # terminals (truncated)
            Window("Ashdod-HCT",      (2021, 8), (2022, 10)),
        ],
    ),

    # --- Haifa privatization (Haifa-Legacy sold 01-2023) ---
    # NEW: sample is exactly 01-2022 to 09-2023 (pre: 01-2022..12-2022; post: 01-2023..09-2023).
    Spec(
        reform="haifa_priv",
        target="Haifa-Legacy terminal",
        event_year=2023,
        event_month=1,  # sale in 01-2023
        treat_windows=[
            Window("Haifa-Legacy",    (2022, 1), (2023, 9)),
        ],
        control_windows=[
            Window("Haifa-Bayport",   (2022, 1), (2023, 9)),
            Window("Ashdod-Legacy",   (2022, 1), (2023, 9)),
            Window("Ashdod-HCT",      (2022, 1), (2023, 9)),
        ],
    ),

]

# ----------------------------------------------------------------------
# 4. Optional shock controls
# ----------------------------------------------------------------------

EXPLICIT_SHOCK_COLS: List[str] = [
    # Explicitly known shock controls constructed in load_lp_panel().
    "covid_shock",
    "war_shock",
]


def get_shock_control_cols(df: pd.DataFrame) -> List[str]:
    """Return the list of shock-control columns present in df."""
    if EXPLICIT_SHOCK_COLS:
        return [c for c in EXPLICIT_SHOCK_COLS if c in df.columns]
    # Fallback: guess based on column name
    return [c for c in df.columns if "shock" in c.lower()]


# ----------------------------------------------------------------------
# 5. Load panel and basic helpers
# ----------------------------------------------------------------------

def load_lp_panel(path: Path) -> pd.DataFrame:
    """
    Load LP_Panel_monthly.tsv.

    We keep *all* rows (monthly port + terminal), and only
    restrict using the explicit Table-1 windows. That way we never
    accidentally drop relevant rows via freq/level filters.
    """
    print(f"Reading monthly LP panel from: {path}")
    df = pd.read_csv(path, sep="\t")

    print(f"Loaded {len(df)} rows from LP_Panel_monthly.tsv.")

    # Ensure basic columns exist
    required_cols = {"year", "month", "month_index", "LP",
                     "series_id", "level", "freq", "port", "terminal"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"LP_Panel_monthly.tsv is missing required columns: {missing}")

    # Basic typing
    for col in ["year", "month", "month_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if (df["LP"] <= 0).any():
        raise ValueError("LP contains non-positive values; cannot take logs.")

    df["log_LP"] = np.log(df["LP"])

    # ------------------------------------------------------------------
    # Shock controls: COVID (2020–21) and late-2023 war shock
    # ------------------------------------------------------------------
    covid_mask = df["year"].between(2020, 2021, inclusive="both")
    covid_mask = covid_mask.fillna(False)
    df["covid_shock"] = covid_mask.astype(int)

    war_mask = (df["year"] > 2023) | ((df["year"] == 2023) & (df["month"] >= 10))
    war_mask = war_mask.fillna(False)
    df["war_shock"] = war_mask.astype(int)

    # IDs for FE
    df["unit_id"] = df["series_id"]         # terminal or port series
    df["time_id"] = df["month_index"]
    # Cluster at the series_id level (more than two clusters); WCB for
    # inference will be handled downstream.
    df["cluster_id"] = df["series_id"]

    return df


def build_year_month_to_index(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    """
    Build a mapping (year, month) -> month_index.

    We assume every (year, month) pair corresponds to a single month_index
    across all series, which is how LP_Panel is constructed.
    """
    mapping: Dict[Tuple[int, int], int] = {}
    grouped = df.groupby(["year", "month"])["month_index"].unique()

    for (y, m), idxs in grouped.items():
        if len(idxs) != 1:
            raise ValueError(
                f"(year={y}, month={m}) has multiple month_index values: {idxs}. "
                "LP_Panel should use a single global month_index per month."
            )
        mapping[(y, m)] = int(idxs[0])

    print(f"Built (year, month) -> month_index mapping for {len(mapping)} months.")
    return mapping


def get_series_id_for_label(df: pd.DataFrame, label: str) -> str:
    """
    Resolve a Table-1 label (e.g. 'Haifa port') into the corresponding
    series_id in LP_Panel_monthly, using LABEL_FILTERS.

    We require exactly one series_id match; otherwise we raise a clear error
    so you can adjust LABEL_FILTERS.
    """
    if label not in LABEL_FILTERS:
        raise KeyError(f"Label '{label}' not found in LABEL_FILTERS.")

    conds = LABEL_FILTERS[label]
    mask = pd.Series(True, index=df.index)
    for col, val in conds.items():
        mask &= (df[col] == val)

    sids = df.loc[mask, "series_id"].unique()

    if len(sids) == 0:
        unique = df[["series_id", "level", "freq", "port", "terminal"]].drop_duplicates()
        raise ValueError(
            f"No series_id found for label '{label}' with filters {conds}.\n"
            f"Check LABEL_FILTERS. Here are some unique series from LP_Panel_monthly:\n"
            f"{unique.head(20)}"
        )
    if len(sids) > 1:
        raise ValueError(
            f"Label '{label}' and filters {conds} matched multiple series_ids: {sids}.\n"
            "Please refine LABEL_FILTERS to pick a unique series."
        )

    return sids[0]


def year_month_to_index(mapping: Dict[Tuple[int, int], int], ym: Tuple[int, int]) -> int:
    """Convenience: (year, month) -> month_index via precomputed mapping."""
    y, m = ym
    if (y, m) not in mapping:
        raise KeyError(f"(year={y}, month={m}) not found in (year,month)->month_index mapping.")
    return mapping[(y, m)]


# ----------------------------------------------------------------------
# 6. Build estimation sample for one Spec
# ----------------------------------------------------------------------

def build_es_sample(df: pd.DataFrame, spec: Spec,
                    ym_to_idx: Dict[Tuple[int, int], int]) -> pd.DataFrame:
    """
    Given the full LP panel and one Spec (Table-1 row), build the
    event-study sample:

        - df_es['in_sample'] = True only for the exact treatment + control windows
          listed in Table 1 for this spec.
        - df_es['treated']   = True only for treatment windows.
        - df_es['event_time'] = month_index - event_index for treated rows;
                                = -1 for controls.

    Returns df_es (a copy of df) restricted to in_sample == True, with:
        - 'port'       (Haifa / Ashdod) inferred from series_id
        - 'time_index' (alias for time_id)
        - 'port_trend' (signed linear trend used when include_port_trends=True)
    """
    df_es = df.copy()
    df_es["in_sample"] = False
    df_es["treated"] = False

    # 1. Mark treatment windows
    for w in spec.treat_windows:
        sid = get_series_id_for_label(df_es, w.label)
        start_idx = year_month_to_index(ym_to_idx, w.start)
        end_idx   = year_month_to_index(ym_to_idx, w.end)

        mask = (
            (df_es["series_id"] == sid) &
            df_es["month_index"].between(start_idx, end_idx)
        )

        df_es.loc[mask, "in_sample"] = True
        df_es.loc[mask, "treated"] = True

    # 2. Mark control windows
    for w in spec.control_windows:
        sid = get_series_id_for_label(df_es, w.label)
        start_idx = year_month_to_index(ym_to_idx, w.start)
        end_idx   = year_month_to_index(ym_to_idx, w.end)

        mask = (
            (df_es["series_id"] == sid) &
            df_es["month_index"].between(start_idx, end_idx)
        )

        df_es.loc[mask, "in_sample"] = True
        # treated remains False for controls

    # Keep only rows in the NYT windows
    df_es = df_es[df_es["in_sample"]].copy()

    # 3. Define event_time
    event_index = year_month_to_index(ym_to_idx, (spec.event_year, spec.event_month))

    df_es["event_time"] = -1  # default for controls
    treated_mask = df_es["treated"]
    df_es.loc[treated_mask, "event_time"] = (
        df_es.loc[treated_mask, "month_index"] - event_index
    )

    # 4. Create simple port identifier from series_id (assumes it starts with "Haifa"/"Ashdod")
    df_es["port"] = df_es["series_id"].str.extract(r"^(Haifa|Ashdod)", expand=False)
    if df_es["port"].isna().any():
        bad = df_es.loc[df_es["port"].isna(), "series_id"].unique()
        raise ValueError(
            "Could not infer port from series_id for some rows. "
            f"Expected series_id to start with 'Haifa' or 'Ashdod'. Offending ids: {bad}"
        )

    # 5. Time index and port-specific trend (for +PortTr specs)
    df_es["time_index"] = df_es["time_id"]

    df_es["port_trend"] = 0.0
    for port_name in df_es["port"].unique():
        mask_port = df_es["port"] == port_name
        # Multiply by +1 for Haifa, -1 for Ashdod so the two trends are independent.
        slope_sign = 1.0 if port_name == "Haifa" else -1.0
        df_es.loc[mask_port, "port_trend"] = df_es.loc[mask_port, "time_index"] * slope_sign

    return df_es


# ----------------------------------------------------------------------
# 7. Run regression and extract dynamic betas
# ----------------------------------------------------------------------

def _parse_event_time_from_param_name(name: str) -> Optional[int]:
    """
    Given a parameter name from statsmodels, return the corresponding
    event_time (int) if this is an event_time coefficient, else None.

    With the formula:
        C(event_time, Treatment(reference=-1))
    statsmodels/patsy generate names like:
        'C(event_time, Treatment(reference=-1))[T.0]'
        'C(event_time, Treatment(reference=-1))[T.1]'
        'C(event_time, Treatment(reference=-1))[T.-12]'
    """
    prefix = "C(event_time, Treatment(reference=-1))[T."
    if not name.startswith(prefix):
        return None

    # Extract the part after the prefix, strip the trailing ']'
    m_str = name[len(prefix):].rstrip("]")

    try:
        m_val = int(m_str)
    except ValueError:
        return None

    return m_val


def run_event_study(
    df_es: pd.DataFrame,
    spec_with_fe: SpecWithFE,
    shock_cols: Optional[List[str]] = None,
):
    """
    Run the Model 1A regression on df_es under a given specification:

        log_LP_it = sum_{m != -1} beta_m 1{event_time_it = m}
                    + unit FE + time FE
                    [+ port-specific linear trends]
                    [+ shock controls]
                    + epsilon_it

    Standard errors for beta_m and window-averages use series-clustered
    covariance. Pre-trend F-tests, however, will use an HC1-style
    unclustered covariance (constructed from OLS) purely as a diagnostic.

    Returns
    -------
    result : statsmodels RegressionResults
    n_by_event_time : dict[int, int]
        Mapping from event_time m -> number of unit×month observations
        at that event month in the estimation sample.
    """
    spec = spec_with_fe.spec
    spec_name = spec_with_fe.spec_name
    include_port_trends = spec_with_fe.include_port_trends
    include_shocks = spec_with_fe.include_shocks

    df_es = df_es.copy()

    # Patsy/statsmodels don't like pandas' nullable Int64Dtype()
    # for categorical variables inside C(). Coerce the *numeric* IDs;
    # unit_id is a string and should stay that way.
    for col in ["event_time", "time_id"]:
        if pd.api.types.is_integer_dtype(df_es[col].dtype):
            df_es[col] = df_es[col].astype("int64")

    # Build the regression formula
    formula = "log_LP ~ C(event_time, Treatment(reference=-1)) + C(unit_id) + C(time_id)"

    if include_port_trends:
        formula += " + port_trend"

    used_shock_cols: List[str] = []
    if include_shocks and shock_cols:
        for col in shock_cols:
            if col in df_es.columns:
                used_shock_cols.append(col)
        if used_shock_cols:
            formula += " + " + " + ".join(used_shock_cols)
        else:
            print(f"[WARN] Spec '{spec_name}' requested shock controls "
                  f"but none of the candidate columns {shock_cols} "
                  f"are present in the estimation sample. Running without shocks.")
            include_shocks = False  # effectively

    print(f"  Running spec='{spec_name}' "
          f"(+PortTr={include_port_trends}, +Shocks={include_shocks})")

    # Count observations per event_time for Appendix N(m)
    n_by_event_time = (
        df_es.groupby("event_time")["unit_id"]
        .size()
        .to_dict()
    )

    model = smf.ols(formula=formula, data=df_es)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df_es["cluster_id"]},
    )

    return result, n_by_event_time


def extract_dynamic_betas(
    result,
    spec_with_fe: SpecWithFE,
    n_by_event_time: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """
    Extract beta_m for each event_time m (excluding reference m=-1)
    into a tidy DataFrame, including N(m) if provided.
    """
    spec = spec_with_fe.spec
    rows = []
    params = result.params
    bse = result.bse
    pvals = result.pvalues

    n_map = n_by_event_time or {}

    for name, beta in params.items():
        m_val = _parse_event_time_from_param_name(name)
        if m_val is None:
            continue

        se = float(bse.get(name, np.nan))
        t_stat = beta / se if (not np.isnan(se) and se != 0) else np.nan
        pval = float(pvals.get(name, np.nan))
        n_event = float(n_map.get(m_val, np.nan))

        row = {
            "reform": spec.reform,
            "target": spec.target,
            "spec_name": spec_with_fe.spec_name,
            "event_time": m_val,
            "beta": float(beta),
            "se": se,
            "t": t_stat,
            "pvalue": pval,
            "n_event_obs": n_event,   # N(m) for Appendix tables
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 8. Compute window averages beta_[a,b]
# ----------------------------------------------------------------------

# Coarse windows for the tables; note that avg_pre matches the pre-trend window
PRETREND_MIN = -12
PRETREND_MAX = -2

WINDOWS: Dict[str, Tuple[int, int]] = {
    # window_name: (a, b) in event_time units (months)
    "avg_pre":    (PRETREND_MIN, PRETREND_MAX),
    "post_1yr":   (1, 12),
    "post_2yrs":  (1, 24),
    "full_post":  (1, 999),  # interpreted as [1, max observed post m]
}

# For pre-trend F-test: coarse pre bins inside [PRETREND_MIN, PRETREND_MAX]
PRETREND_BINS: List[Tuple[int, int]] = [
    (PRETREND_MIN, -7),  # e.g. [-12, -7]
    (-6, PRETREND_MAX),  # e.g. [-6, -2]
]


def compute_window_averages(result, spec_with_fe: SpecWithFE) -> pd.DataFrame:
    """
    Compute window-average betas beta_[a,b] as linear combinations of beta_m.
    """
    spec = spec_with_fe.spec
    params = result.params
    cov = result.cov_params()

    # Map m (int) -> parameter name
    m_to_name: Dict[int, str] = {}
    for name in params.index:
        m_val = _parse_event_time_from_param_name(name)
        if m_val is None:
            continue
        m_to_name[m_val] = name

    if not m_to_name:
        return pd.DataFrame([])

    available_ms = sorted(m_to_name.keys())
    max_post_m = max((m for m in available_ms if m > 0), default=None)

    rows = []
    for wname, (a, b) in WINDOWS.items():
        # Resolve 'full_post'
        b_eff = b
        if max_post_m is not None:
            if b == 999:
                b_eff = max_post_m
            elif a >= 1:
                b_eff = min(b, max_post_m)


        ms_in_window = [m for m in available_ms if (m >= a and m <= b_eff)]
        if not ms_in_window:
            continue

        # Build weight vector over all params
        w = pd.Series(0.0, index=params.index)
        weight = 1.0 / len(ms_in_window)
        for m in ms_in_window:
            w[m_to_name[m]] = weight

        beta_w = float(np.dot(w.values, params.values))
        var_w = float(np.dot(w.values, np.dot(cov.values, w.values)))
        se_w = np.sqrt(var_w) if var_w >= 0 else np.nan

        row = {
            "reform": spec.reform,
            "target": spec.target,
            "spec_name": spec_with_fe.spec_name,
            "window": wname,
            "a": a,
            "b": b_eff,
            "beta": beta_w,
            "se": se_w,
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 9. Pre-trend F-test for aggregated leads in coarse pre bins
# ----------------------------------------------------------------------

def compute_pretrend_f_test(
    result,
    spec_with_fe: SpecWithFE,
    pre_min: int = PRETREND_MIN,
    pre_max: int = PRETREND_MAX,
    bin_edges: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    Compute a pre-trend test for coarse lead bins within [pre_min, pre_max].

    Steps:
      * identify event-time coefficients with m in [pre_min, pre_max] (leads),
      * define one or more coarse pre bins (e.g. [-12,-7] and [-6,-2]),
      * for each bin k, form the equal-weight average of the betas in that bin,
      * test joint H0: all bin averages = 0 via a Wald F-test,
      * construct the Wald test using an unclustered OLS covariance matrix:
          cov_unclustered = (X'X)^{-1} * mse_resid
        derived from the same regression.

    This keeps the main reported SEs cluster-robust, while using a
    simpler, better-behaved covariance for the pre-trend diagnostic
    (especially given the small number of clusters).
    """
    params = result.params
    param_names = list(params.index)
    k_params = len(param_names)

    # 1. Identify all event-time coefficients and select leads in [pre_min, pre_max]
    event_param_info: List[Tuple[int, int, str]] = []  # (m_val, idx, name)
    for j, name in enumerate(param_names):
        m_val = _parse_event_time_from_param_name(name)
        if m_val is None:
            continue
        if pre_min <= m_val <= pre_max:
            event_param_info.append((m_val, j, name))

    if not event_param_info:
        return pd.DataFrame([])

    n_leads_total = len(event_param_info)

    # 2. Define coarse bins
    bins = bin_edges if bin_edges is not None else PRETREND_BINS

    # Build R with one row per bin that has at least one coefficient
    R_rows: List[np.ndarray] = []
    bins_used: List[Tuple[int, int]] = []

    for (a, b) in bins:
        idxs_in_bin = [idx for (m_val, idx, _) in event_param_info if (m_val >= a and m_val <= b)]
        if not idxs_in_bin:
            continue
        r = np.zeros(k_params)
        w = 1.0 / len(idxs_in_bin)
        for j in idxs_in_bin:
            r[j] = w
        R_rows.append(r)
        bins_used.append((a, b))

    if not R_rows:
        # No bins actually contain any leads (shouldn't happen if event_param_info non-empty,
        # but guard anyway).
        return pd.DataFrame([])

    R = np.vstack(R_rows)
    n_restr = R.shape[0]

    # 3. Construct an unclustered (OLS-style) covariance matrix.
    try:
        cov_unclustered = np.asarray(result.normalized_cov_params) * float(result.mse_resid)
    except Exception:
        return pd.DataFrame([])

    # 4. Wald test with F-statistic using the unclustered covariance.
    try:
        # scalar=False to avoid the future-behavior warning; we then
        # safely coerce the outputs to scalars.
        wtest = result.wald_test(R, cov_p=cov_unclustered, use_f=True, scalar=False)

        f_raw = getattr(wtest, "fvalue", getattr(wtest, "statistic", np.nan))
        f_val = float(np.asarray(f_raw).ravel()[0])

        p_raw = getattr(wtest, "pvalue", np.nan)
        p_val = float(np.asarray(p_raw).ravel()[0])

        df_num_attr = getattr(wtest, "df_num", None)
        df_denom_attr = getattr(wtest, "df_denom", None)

        df_num = float(df_num_attr) if df_num_attr is not None else float(n_restr)
        df_denom = float(df_denom_attr) if df_denom_attr is not None else float(result.df_resid)
    except Exception:
        f_val = np.nan
        p_val = np.nan
        df_num = float(n_restr)
        df_denom = float(result.df_resid)

    row = {
        "reform": spec_with_fe.spec.reform,
        "target": spec_with_fe.spec.target,
        "spec": spec_with_fe.spec_name,
        "pre_min": float(pre_min),
        "pre_max": float(pre_max),
        "n_leads_total": float(n_leads_total),
        "n_bins_defined": float(len(bins)),
        "n_bins_used": float(len(bins_used)),
        "f_stat": f_val,
        "pvalue": p_val,
        "df_num": df_num,
        "df_denom": df_denom,
        "n_obs": int(result.nobs),
        "r2": float(result.rsquared),
    }
    return pd.DataFrame([row])


# ----------------------------------------------------------------------
# 10. Main driver
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full monthly panel
    df = load_lp_panel(LP_PANEL_PATH)

    # Precompute (year, month) -> month_index mapping once
    ym_to_idx = build_year_month_to_index(df)

    # Determine candidate shock-control columns up front
    shock_cols_all = get_shock_control_cols(df)
    if shock_cols_all:
        print(f"Detected shock-control columns: {shock_cols_all}")
    else:
        print("No shock-control columns detected; '+Tr&Shocks' specs will be skipped.")

    # Build the list of (specification x Table-1 row) combinations.
    SPECS_WITH_FE: List[SpecWithFE] = []
    for base_spec in NYT_SPECS:
        # Baseline
        SPECS_WITH_FE.append(
            SpecWithFE(spec=base_spec, spec_name="baseline",
                       include_port_trends=False, include_shocks=False)
        )
        # +PortTr
        SPECS_WITH_FE.append(
            SpecWithFE(spec=base_spec, spec_name="porttr",
                       include_port_trends=True, include_shocks=False)
        )
        # +Tr&Shocks (only if any shock controls exist at all)
        if shock_cols_all:
            SPECS_WITH_FE.append(
                SpecWithFE(spec=base_spec, spec_name="tr_shocks",
                           include_port_trends=True, include_shocks=True)
            )

    # Containers keyed by spec_name so we can export per-spec files
    dynamic_by_spec: Dict[str, List[pd.DataFrame]] = {}
    windows_by_spec: Dict[str, List[pd.DataFrame]] = {}
    pretrend_by_spec: Dict[str, List[pd.DataFrame]] = {}

    for spec_with_fe in SPECS_WITH_FE:
        spec = spec_with_fe.spec
        print(f"\n=== Running reform={spec.reform}, target={spec.target}, "
              f"spec={spec_with_fe.spec_name} ===")

        df_es = build_es_sample(df, spec, ym_to_idx)

        if df_es["treated"].sum() == 0:
            print(f"[WARN] No treated observations for reform={spec.reform}, "
                  f"target={spec.target}. Skipping.")
            continue

        print(f"Sample size: {len(df_es)} rows "
              f"({df_es['treated'].sum()} treated, "
              f"{len(df_es) - df_es['treated'].sum()} controls).")

        # Run the ES regression
        result, n_by_event_time = run_event_study(
            df_es,
            spec_with_fe,
            shock_cols=shock_cols_all,
        )

        # Collect outputs
        dyn = extract_dynamic_betas(result, spec_with_fe, n_by_event_time)
        win = compute_window_averages(result, spec_with_fe)
        pre = compute_pretrend_f_test(result, spec_with_fe)

        if not dyn.empty:
            dynamic_by_spec.setdefault(spec_with_fe.spec_name, []).append(dyn)
        if not win.empty:
            windows_by_spec.setdefault(spec_with_fe.spec_name, []).append(win)
        if not pre.empty:
            pretrend_by_spec.setdefault(spec_with_fe.spec_name, []).append(pre)

    # ------------------------------------------------------------------
    # Save per-specification TSVs
    # ------------------------------------------------------------------
    base_name = "model1a_lp"

    # Dynamic betas
    all_dynamic_frames: List[pd.DataFrame] = []
    for spec_name, frames in dynamic_by_spec.items():
        dynamic_df = pd.concat(frames, ignore_index=True)
        all_dynamic_frames.append(dynamic_df)
        dynamic_path = OUTPUT_DIR / f"{base_name}_dynamic_betas_{spec_name}.tsv"
        dynamic_df.to_csv(dynamic_path, sep="\t", index=False)
        print(f"Saved dynamic betas for spec '{spec_name}' to: {dynamic_path}")

    # Window averages
    all_window_frames: List[pd.DataFrame] = []
    for spec_name, frames in windows_by_spec.items():
        windows_df = pd.concat(frames, ignore_index=True)
        all_window_frames.append(windows_df)
        windows_path = OUTPUT_DIR / f"{base_name}_window_betas_{spec_name}.tsv"
        windows_df.to_csv(windows_path, sep="\t", index=False)
        print(f"Saved window-average betas for spec '{spec_name}' to: {windows_path}")

    # Pre-trend F-tests
    all_pretrend_frames: List[pd.DataFrame] = []
    for spec_name, frames in pretrend_by_spec.items():
        pretrend_df = pd.concat(frames, ignore_index=True)
        all_pretrend_frames.append(pretrend_df)
        pretrend_path = OUTPUT_DIR / f"{base_name}_pretrend_tests_{spec_name}.tsv"
        pretrend_df.to_csv(pretrend_path, sep="\t", index=False)
        print(f"Saved pre-trend F-tests for spec '{spec_name}' to: {pretrend_path}")

    # Also save pooled "all-specs" versions for convenience
    if all_dynamic_frames:
        dynamic_all = pd.concat(all_dynamic_frames, ignore_index=True)
        dynamic_all_path = OUTPUT_DIR / f"{base_name}_dynamic_betas_all.tsv"
        dynamic_all.to_csv(dynamic_all_path, sep="\t", index=False)
        print(f"Saved pooled dynamic betas (all specs) to: {dynamic_all_path}")

    if all_window_frames:
        windows_all = pd.concat(all_window_frames, ignore_index=True)
        windows_all_path = OUTPUT_DIR / f"{base_name}_window_betas_all.tsv"
        windows_all.to_csv(windows_all_path, sep="\t", index=False)
        print(f"Saved pooled window-average betas (all specs) to: {windows_all_path}")

    if all_pretrend_frames:
        pretrend_all = pd.concat(all_pretrend_frames, ignore_index=True)
        pretrend_all_path = OUTPUT_DIR / f"{base_name}_pretrend_tests_all.tsv"
        pretrend_all.to_csv(pretrend_all_path, sep="\t", index=False)
        print(f"Saved pooled pre-trend F-tests (all specs) to: {pretrend_all_path}")


if __name__ == "__main__":
    main()






# ----------------------------------------------------------------------
# DIAGNOSTIC NOTE (NYT windows update; keep for later when true hours arrive)
#
# After truncating the Haifa competition-entry samples to end 10-2022, the
# NYT design is now correct (Ashdod controls are excluded after 11-2022).
# However, the current LP proxy + tiny treated sample implies two known issues:
#
# (1) haifa_comp regressions often show R^2 ~ 1.000 and near-zero SEs.
#     Mechanically, dynamic ES coefficients are identified off a single treated
#     time series with N(m)=1 per event-time bin (plus many FE/dummies and
#     monthly-expanded quarterly values), so the model can (near-)perfectly fit.
#     Interpretation: treat p-values / pretrend F-tests as not informative here.
#
# (2) haifa_priv sometimes triggers statsmodels warnings:
#       "invalid value encountered in sqrt"
#     This arises from numerically unstable (non-PSD) cluster-robust covariance
#     estimates with very few effective clusters / near-collinearity.
#     Interpretation: clustered SEs can be unreliable in this small panel.
#
# We are deferring fixes (alternative inference + lower-dimensional ES / window
# bins / small-cluster corrections) until true monthly labor-hours data is in.
# ----------------------------------------------------------------------
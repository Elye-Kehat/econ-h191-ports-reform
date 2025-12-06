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
        * A tidy table of dynamic betas beta_m for each event_time m.
        * A tidy table of window-average betas beta_[a,b].
        * A tidy table of pre-trend F-tests for leads m <= -2.

The goal is that the resulting TSVs can be read almost directly into the
LaTeX tables for the main text and appendix.
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
# .../Thesis/Design/Code (new)/Model_1A.py  --> parents[2] = .../Thesis
THESIS_ROOT = THIS_FILE.parents[2]

LP_PANEL_PATH = THESIS_ROOT / "Data" / "LP" / "LP_Panel_monthly.tsv"
OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)"  # will be created if needed


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
# We use simple conditions on (level, port, terminal) to find the right
# series_id for each label. This is evaluated once at the beginning,
# and we assert that each label maps to exactly one series_id.
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


# ----------------------------------------------------------------------
# 3. Table-1 specs (NYT windows)
# ----------------------------------------------------------------------
# These are copied directly from your LaTeX table.
# Dates are written as (year, month); months are integers 1-12.
# ----------------------------------------------------------------------

NYT_SPECS: List[Spec] = [

    # --- Haifa competition entry (Bayport opens 09-2021) ---
    Spec(
        reform="haifa_comp",
        target="Haifa-Bayport terminal",
        event_year=2021,
        event_month=9,  # Bayport opens 09-2021
        treat_windows=[
            Window("Haifa port",      (2019, 9), (2021, 8)),
            Window("Haifa-Bayport",   (2021, 9), (2022, 10)),
        ],
        control_windows=[
            Window("Ashdod port",     (2019, 9), (2021, 7)),
            Window("Ashdod-Legacy",   (2021, 8), (2022, 10)),
            Window("Ashdod-HCT",      (2021, 8), (2022, 10)),
        ],
    ),

    Spec(
        reform="haifa_comp",
        target="Haifa-Legacy terminal",
        event_year=2021,
        event_month=9,
        treat_windows=[
            Window("Haifa port",      (2019, 9), (2021, 8)),
            Window("Haifa-Legacy",    (2021, 9), (2022, 10)),
        ],
        control_windows=[
            Window("Ashdod port",     (2019, 9), (2021, 7)),
            Window("Ashdod-Legacy",   (2021, 8), (2022, 10)),
            Window("Ashdod-HCT",      (2021, 8), (2022, 10)),
        ],
    ),

    # --- Ashdod competition entry (HCT effective 11-2022) ---
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
    Spec(
        reform="haifa_priv",
        target="Haifa-Legacy terminal",
        event_year=2023,
        event_month=1,  # sale in 01-2023
        treat_windows=[
            Window("Haifa port",      (2021, 1), (2021, 8)),
            Window("Haifa-Legacy",    (2021, 9), (2023, 9)),
        ],
        control_windows=[
            Window("Haifa-Bayport",   (2021, 9), (2023, 9)),
            Window("Ashdod port",     (2021, 1), (2021, 7)),
            Window("Ashdod-Legacy",   (2021, 8), (2023, 9)),
            Window("Ashdod-HCT",      (2021, 8), (2023, 9)),
        ],
    ),
]


# ----------------------------------------------------------------------
# 4. Optional shock controls
# ----------------------------------------------------------------------
# You can either:
#   (i) Fill EXPLICIT_SHOCK_COLS with the exact column names of your
#       shock controls in LP_Panel_monthly.tsv (e.g. "covid_Haifa",
#       "stevedore_strike_Ashdod"), or
#  (ii) Leave EXPLICIT_SHOCK_COLS empty and let the script auto-detect
#       any columns whose names contain the substring "shock".
# If no shock controls are found, the "tr_shocks" specification is
# skipped with a warning so that the script still runs on the current
# data.
# ----------------------------------------------------------------------

EXPLICIT_SHOCK_COLS: List[str] = [
    # Example:
    # "covid_Haifa",
    # "covid_Ashdod",
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

    # IDs for FE
    df["unit_id"] = df["series_id"]         # terminal or port series
    df["time_id"] = df["month_index"]
    df["cluster_id"] = df["port"]          # cluster at port level

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


def run_event_study(df_es: pd.DataFrame, spec_with_fe: SpecWithFE,
                    shock_cols: Optional[List[str]] = None):
    """
    Run the Model 1A regression on df_es under a given specification:

        log_LP_it = sum_{m != -1} beta_m 1{event_time_it = m}
                    + unit FE + time FE
                    [+ port-specific linear trends]
                    [+ shock controls]
                    + epsilon_it
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
        # Only include columns that actually exist in df_es
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

    model = smf.ols(formula=formula, data=df_es)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df_es["cluster_id"]},
    )

    return result


def extract_dynamic_betas(result, spec_with_fe: SpecWithFE) -> pd.DataFrame:
    """
    Extract beta_m for each event_time m (excluding reference m=-1)
    into a tidy DataFrame.
    """
    spec = spec_with_fe.spec
    rows = []
    params = result.params
    bse = result.bse
    pvals = result.pvalues

    for name, beta in params.items():
        m_val = _parse_event_time_from_param_name(name)
        if m_val is None:
            continue

        se = float(bse.get(name, np.nan))
        t_stat = beta / se if se not in (0, np.nan) else np.nan
        pval = float(pvals.get(name, np.nan))

        row = {
            "reform": spec.reform,
            "target": spec.target,
            "spec_name": spec_with_fe.spec_name,
            "event_time": m_val,
            "beta": float(beta),
            "se": se,
            "t": t_stat,
            "pvalue": pval,
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 8. Compute window averages beta_[a,b]
# ----------------------------------------------------------------------

WINDOWS: Dict[str, Tuple[int, int]] = {
    # window_name: (a, b) in event_time units (months)
    "avg_pre":    (-24, -2),
    "post_1yr":   (1, 12),
    "post_2yrs":  (1, 24),
    "full_post":  (1, 999),  # interpreted as [1, max observed post m]
}


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
        if b == 999 and max_post_m is not None:
            b_eff = max_post_m

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
# 9. Pre-trend F-test for leads m <= -2
# ----------------------------------------------------------------------

def compute_pretrend_f_test(result, spec_with_fe: SpecWithFE) -> pd.DataFrame:
    """
    Compute an F-test (Wald test) for the null that all lead coefficients
    (event_time m <= -2) are jointly zero.

    Returns a one-row DataFrame with columns:
        reform, target, spec_name, n_leads, f_stat, pvalue,
        df_num, df_denom, n_obs, r2
    """
    spec = spec_with_fe.spec
    params = result.params
    param_names = params.index.to_list()

    lead_param_indices: List[int] = []
    for j, name in enumerate(param_names):
        m_val = _parse_event_time_from_param_name(name)
        if m_val is not None and m_val <= -2:
            lead_param_indices.append(j)

    if not lead_param_indices:
        row = {
            "reform": spec.reform,
            "target": spec.target,
            "spec_name": spec_with_fe.spec_name,
            "n_leads": 0,
            "f_stat": np.nan,
            "pvalue": np.nan,
            "df_num": 0.0,
            "df_denom": float(result.df_resid),
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        }
        return pd.DataFrame([row])

    # Build restriction matrix R for H0: beta_leads = 0
    R = np.zeros((len(lead_param_indices), len(param_names)))
    for i, idx in enumerate(lead_param_indices):
        R[i, idx] = 1.0

    ftest = result.f_test(R)
    # fvalue and pvalue may be arrays; squash them
    f_val = float(np.squeeze(np.asarray(ftest.fvalue)))
    p_val = float(np.squeeze(np.asarray(ftest.pvalue)))

    row = {
        "reform": spec.reform,
        "target": spec.target,
        "spec_name": spec_with_fe.spec_name,
        "n_leads": float(len(lead_param_indices)),
        "f_stat": f_val,
        "pvalue": p_val,
        "df_num": float(len(lead_param_indices)),
        "df_denom": float(result.df_resid),
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
    # We always run "baseline" and "+PortTr" for all NYT_SPECS.
    # For "+Tr&Shocks", we only attempt it if at least one shock control
    # column exists in the data.
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
        result = run_event_study(df_es, spec_with_fe, shock_cols=shock_cols_all)

        # Collect outputs
        dyn = extract_dynamic_betas(result, spec_with_fe)
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

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
    - Export:
        * A tidy table of dynamic betas beta_m for each event_time m.
        * A tidy table of window-average betas beta_[a,b].

This script deliberately avoids clever abstractions so the logic
can be read straight alongside Table 1.
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
# .../Thesis/Design/Code (new)/Model_1A  --> parents[2] = .../Thesis
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
# 4. Load panel and basic helpers
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

    # Optional sanity check: unique (series_id, level, freq, port, terminal)
    # uniques = df[["series_id", "level", "freq", "port", "terminal"]].drop_duplicates()
    # print("Unique series (head):")
    # print(uniques.head(10))

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
# 5. Build estimation sample for one Spec
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

    Returns df_es (a copy of df) restricted to in_sample == True.
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

    return df_es


# ----------------------------------------------------------------------
# 6. Run regression and extract dynamic betas
# ----------------------------------------------------------------------

def run_event_study(df_es: pd.DataFrame, spec: Spec):
    """
    Run the Model 1A regression on df_es:

        log_LP_it = sum_{m != -1} beta_m 1{event_time_it = m}
                    + unit FE + time FE + epsilon_it
    """
    # We coerce key regressors to plain numpy int64 to avoid patsy choking
    # on pandas' nullable Int64Dtype.
    df_es = df_es.copy()

    # Patsy/statsmodels don't like pandas' nullable Int64Dtype()
    # for categorical variables inside C(). We only coerce the *numeric*
    # IDs; unit_id is a string and should stay that way.
    for col in ["event_time", "time_id"]:
        if pd.api.types.is_integer_dtype(df_es[col].dtype):
            df_es[col] = df_es[col].astype("int64")

    formula = "log_LP ~ C(event_time, Treatment(reference=-1)) + C(unit_id) + C(time_id)"
    model = smf.ols(formula=formula, data=df_es)

    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df_es["cluster_id"]},
    )

    return result


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


def extract_dynamic_betas(result, spec: Spec) -> pd.DataFrame:
    """
    Extract beta_m for each event_time m (excluding reference m=-1)
    into a tidy DataFrame.
    """
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
# 7. Compute window averages beta_[a,b]
# ----------------------------------------------------------------------

WINDOWS = {
    # window_name: (a, b) in event_time units (months)
    "avg_pre":    (-24, -2),
    "post_1yr":   (1, 12),
    "post_2yrs":  (1, 24),
    "full_post":  (1, 999),  # interpreted as [1, max observed post m]
}


def compute_window_averages(result, spec: Spec) -> pd.DataFrame:
    """
    Compute window-average betas beta_[a,b] as linear combinations of beta_m.
    """
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
# 8. Main driver
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full monthly panel
    df = load_lp_panel(LP_PANEL_PATH)

    # Precompute (year, month) -> month_index mapping once
    ym_to_idx = build_year_month_to_index(df)

    all_dynamic = []
    all_windows = []

    for spec in NYT_SPECS:
        print(f"\n=== Running spec: reform={spec.reform}, target={spec.target} ===")

        df_es = build_es_sample(df, spec, ym_to_idx)

        if df_es["treated"].sum() == 0:
            print(f"[WARN] No treated observations for reform={spec.reform}, "
                  f"target={spec.target}. Skipping.")
            continue

        print(f"Sample size: {len(df_es)} rows "
              f"({df_es['treated'].sum()} treated, "
              f"{len(df_es) - df_es['treated'].sum()} controls).")

        result = run_event_study(df_es, spec)

        dyn = extract_dynamic_betas(result, spec)
        win = compute_window_averages(result, spec)

        all_dynamic.append(dyn)
        all_windows.append(win)

    # Save outputs
    if all_dynamic:
        dynamic_df = pd.concat(all_dynamic, ignore_index=True)
        dynamic_path = OUTPUT_DIR / "model1a_lp_dynamic_betas.tsv"
        dynamic_df.to_csv(dynamic_path, sep="\t", index=False)
        print(f"\nSaved dynamic betas to: {dynamic_path}")

    if all_windows:
        windows_df = pd.concat(all_windows, ignore_index=True)
        windows_path = OUTPUT_DIR / "model1a_lp_window_betas.tsv"
        windows_df.to_csv(windows_path, sep="\t", index=False)
        print(f"Saved window-average betas to: {windows_path}")


if __name__ == "__main__":
    main()

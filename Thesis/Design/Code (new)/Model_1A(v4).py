#!/usr/bin/env python
"""
Model_1A(v4).py

Revised v4 for Model 1A (ln(LP)).

What this revision does
-----------------------
1) Preserves the v4 output-contract change:
      - static DiD results live in dedicated *_static_betas_*.tsv files
      - dynamic/window/pretrend results remain separate

2) Revises the competition design to match the agreed identification logic:
      - remove entrant-terminal competition specs from the main estimation universe
      - competition is now evaluated for:
            * incumbent / legacy path
            * aggregate path (hook kept in code, disabled unless upstream aggregate
              LP series already exist in LP_Panel_monthly.tsv)

3) Removes the ad hoc on-the-fly aggregate LP builder from this econometrics file.
   Aggregate specs remain scaffolded but are only activated if maintained aggregate
   series already exist upstream in LP_Panel_monthly.tsv and ENABLE_AGGREGATE_SPECS
   is set to True.

4) Fixes the incumbent-path identification problem in competition specs without
   inventing new data series. For competition analyses only, the pre-reform port row
   and the post-reform legacy-terminal row are assigned the same analysis unit id,
   reflecting the incumbent operator path:
      - Haifa port (pre) + Haifa-Legacy (post) -> "haifa_incumbent"
      - Ashdod port (pre) + Ashdod-Legacy (post) -> "ashdod_incumbent"
   This is analysis-side relabeling of observed series, not a synthetic data series.

5) Keeps strict NYT competition asymmetric, as required by identification:
      - Haifa competition only in NYT
      - Ashdod competition only in conventional TWFE

6) Cleans up default outputs:
      - default main-table specs are only baseline and +port trends
      - tr_shocks is disabled by default (can be re-enabled as robustness)
      - static outputs are written only for TWFE (Panel A), not for NYT

7) Skips estimation for specs with no treated post observations.

Notes
-----
- This file intentionally changes the estimation universe relative to the earlier v4.
- The old v2/v3 table-builder still will not work; a new v4 table-builder is needed.
"""

from __future__ import annotations

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
THESIS_ROOT = THIS_FILE.parents[2]
LP_PANEL_PATH = THESIS_ROOT / "Data" / "LP" / "LP_Panel_monthly.tsv"

OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1A"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)


# ----------------------------------------------------------------------
# 1. Configuration flags
# ----------------------------------------------------------------------

# Aggregate hooks stay in the code architecture, but are disabled by default until
# upstream LP files contain defensible maintained aggregate series.
ENABLE_AGGREGATE_SPECS = False

# Robustness layer only. Main-table architecture uses baseline and +port trends.
INCLUDE_SHOCK_SPECS = False

# Keep static outputs only for the conventional/TWFE benchmark layer.
WRITE_STATIC_FOR_NYT = False
WRITE_STATIC_FOR_TWFE = True


# ----------------------------------------------------------------------
# 2. Helper data structures
# ----------------------------------------------------------------------

@dataclass
class Window:
    label: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    analysis_unit_label: Optional[str] = None


@dataclass
class Spec:
    table_group: str
    reform: str
    target: str
    target_key: str
    event_year: int
    event_month: int
    treat_windows: List[Window]
    control_windows: List[Window]


@dataclass
class SpecWithFE:
    spec: Spec
    spec_name: str
    include_port_trends: bool = False
    include_shocks: bool = False


# ----------------------------------------------------------------------
# 3. Label mapping
# ----------------------------------------------------------------------

LABEL_FILTERS: Dict[str, Dict[str, str]] = {
    "Haifa port": {"level": "port", "port": "Haifa", "terminal": ""},
    "Ashdod port": {"level": "port", "port": "Ashdod", "terminal": ""},

    "Haifa-Bayport": {"level": "terminal", "port": "Haifa", "terminal": "Haifa-Bayport"},
    "Haifa-Legacy": {"level": "terminal", "port": "Haifa", "terminal": "Haifa-Legacy"},
    "Ashdod-HCT": {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-HCT"},
    "Ashdod-Legacy": {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-Legacy"},

    # These are only used if upstream aggregate series truly exist in the LP panel.
    "Haifa aggregate": {"series_id": "haifa_aggregate_port"},
    "Ashdod aggregate": {"series_id": "ashdod_aggregate_port"},
}


# ----------------------------------------------------------------------
# 4. Spec builders
# ----------------------------------------------------------------------

EXPLICIT_SHOCK_COLS: List[str] = ["covid_shock", "war_shock"]


def get_shock_control_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in EXPLICIT_SHOCK_COLS if c in df.columns]


def make_spec(
    table_group: str,
    reform: str,
    target: str,
    target_key: str,
    event_year: int,
    event_month: int,
    treat_windows: List[Window],
    control_windows: List[Window],
) -> Spec:
    return Spec(
        table_group = table_group,
        reform = reform,
        target = target,
        target_key = target_key,
        event_year = event_year,
        event_month = event_month,
        treat_windows = treat_windows,
        control_windows = control_windows,
    )


def aggregate_series_available(df: pd.DataFrame, port_name: str) -> bool:
    sid = f"{port_name.lower()}_aggregate_port"
    return sid in set(df["series_id"].dropna().astype(str).unique())


def build_nyt_specs(df: pd.DataFrame) -> List[Spec]:
    specs: List[Spec] = []

    # --------------------------------------------------
    # Competition (strict NYT: Haifa only)
    # --------------------------------------------------
    specs.append(
        make_spec(
            table_group = "competition",
            reform = "haifa_comp",
            target = "Haifa-Legacy",
            target_key = "haifa_legacy",
            event_year = 2021,
            event_month = 9,
            treat_windows = [
                Window("Haifa port", (2019, 9), (2021, 8), "haifa_incumbent"),
                Window("Haifa-Legacy", (2021, 9), (2022, 10), "haifa_incumbent"),
            ],
            control_windows = [
                Window("Ashdod port", (2019, 9), (2021, 8), "ashdod_incumbent"),
                Window("Ashdod-Legacy", (2021, 9), (2022, 10), "ashdod_incumbent"),
            ],
        )
    )

    if ENABLE_AGGREGATE_SPECS and aggregate_series_available(df, "Haifa") and aggregate_series_available(df, "Ashdod"):
        specs.append(
            make_spec(
                table_group = "competition",
                reform = "haifa_comp",
                target = "Haifa aggregate",
                target_key = "haifa_aggregate",
                event_year = 2021,
                event_month = 9,
                treat_windows = [
                    Window("Haifa port", (2019, 9), (2021, 8), "haifa_aggregate"),
                    Window("Haifa aggregate", (2021, 9), (2022, 10), "haifa_aggregate"),
                ],
                control_windows = [
                    Window("Ashdod port", (2019, 9), (2021, 8), "ashdod_aggregate"),
                    Window("Ashdod aggregate", (2021, 9), (2022, 10), "ashdod_aggregate"),
                ],
            )
        )

    # --------------------------------------------------
    # Privatization (Haifa legacy treated; Bayport placebo; optional aggregate)
    # --------------------------------------------------
    specs.append(
        make_spec(
            table_group = "privatization",
            reform = "haifa_priv",
            target = "Haifa-Legacy",
            target_key = "haifa_legacy",
            event_year = 2023,
            event_month = 1,
            treat_windows = [
                Window("Haifa-Legacy", (2022, 1), (2023, 9), "haifa_legacy"),
            ],
            control_windows = [
                Window("Haifa-Bayport", (2022, 1), (2023, 9), "haifa_bayport"),
                Window("Ashdod-Legacy", (2022, 1), (2023, 9), "ashdod_legacy"),
                Window("Ashdod-HCT", (2022, 1), (2023, 9), "ashdod_hct"),
            ],
        )
    )
    specs.append(
        make_spec(
            table_group = "privatization",
            reform = "haifa_priv",
            target = "Haifa-Bayport",
            target_key = "haifa_bayport_placebo",
            event_year = 2023,
            event_month = 1,
            treat_windows = [
                Window("Haifa-Bayport", (2022, 1), (2023, 9), "haifa_bayport"),
            ],
            control_windows = [
                Window("Haifa-Legacy", (2022, 1), (2023, 9), "haifa_legacy"),
                Window("Ashdod-Legacy", (2022, 1), (2023, 9), "ashdod_legacy"),
                Window("Ashdod-HCT", (2022, 1), (2023, 9), "ashdod_hct"),
            ],
        )
    )

    if ENABLE_AGGREGATE_SPECS and aggregate_series_available(df, "Haifa") and aggregate_series_available(df, "Ashdod"):
        specs.append(
            make_spec(
                table_group = "privatization",
                reform = "haifa_priv",
                target = "Haifa aggregate",
                target_key = "haifa_aggregate",
                event_year = 2023,
                event_month = 1,
                treat_windows = [
                    Window("Haifa aggregate", (2022, 1), (2023, 9), "haifa_aggregate"),
                ],
                control_windows = [
                    Window("Ashdod aggregate", (2022, 1), (2023, 9), "ashdod_aggregate"),
                ],
            )
        )

    return specs


def build_twfe_specs(df: pd.DataFrame) -> List[Spec]:
    specs: List[Spec] = []

    # --------------------------------------------------
    # Competition, Haifa clock
    # --------------------------------------------------
    specs.append(
        make_spec(
            table_group = "competition",
            reform = "haifa_comp",
            target = "Haifa-Legacy",
            target_key = "haifa_legacy",
            event_year = 2021,
            event_month = 9,
            treat_windows = [
                Window("Haifa port", (2019, 9), (2021, 8), "haifa_incumbent"),
                Window("Haifa-Legacy", (2021, 9), (2099, 12), "haifa_incumbent"),
            ],
            control_windows = [
                Window("Ashdod port", (2019, 9), (2021, 8), "ashdod_incumbent"),
                Window("Ashdod-Legacy", (2021, 9), (2099, 12), "ashdod_incumbent"),
            ],
        )
    )

    if ENABLE_AGGREGATE_SPECS and aggregate_series_available(df, "Haifa") and aggregate_series_available(df, "Ashdod"):
        specs.append(
            make_spec(
                table_group = "competition",
                reform = "haifa_comp",
                target = "Haifa aggregate",
                target_key = "haifa_aggregate",
                event_year = 2021,
                event_month = 9,
                treat_windows = [
                    Window("Haifa port", (2019, 9), (2021, 8), "haifa_aggregate"),
                    Window("Haifa aggregate", (2021, 9), (2099, 12), "haifa_aggregate"),
                ],
                control_windows = [
                    Window("Ashdod port", (2019, 9), (2021, 8), "ashdod_aggregate"),
                    Window("Ashdod aggregate", (2021, 9), (2099, 12), "ashdod_aggregate"),
                ],
            )
        )

    # --------------------------------------------------
    # Competition, Ashdod clock
    # --------------------------------------------------
    specs.append(
        make_spec(
            table_group = "competition",
            reform = "ashdod_comp",
            target = "Ashdod-Legacy",
            target_key = "ashdod_legacy",
            event_year = 2022,
            event_month = 11,
            treat_windows = [
                Window("Ashdod port", (2020, 11), (2022, 10), "ashdod_incumbent"),
                Window("Ashdod-Legacy", (2022, 11), (2099, 12), "ashdod_incumbent"),
            ],
            control_windows = [
                Window("Haifa port", (2020, 11), (2021, 8), "haifa_incumbent"),
                Window("Haifa-Legacy", (2021, 9), (2099, 12), "haifa_incumbent"),
            ],
        )
    )

    if ENABLE_AGGREGATE_SPECS and aggregate_series_available(df, "Haifa") and aggregate_series_available(df, "Ashdod"):
        specs.append(
            make_spec(
                table_group = "competition",
                reform = "ashdod_comp",
                target = "Ashdod aggregate",
                target_key = "ashdod_aggregate",
                event_year = 2022,
                event_month = 11,
                treat_windows = [
                    Window("Ashdod port", (2020, 11), (2022, 10), "ashdod_aggregate"),
                    Window("Ashdod aggregate", (2022, 11), (2099, 12), "ashdod_aggregate"),
                ],
                control_windows = [
                    Window("Haifa port", (2020, 11), (2021, 8), "haifa_aggregate"),
                    Window("Haifa aggregate", (2021, 9), (2099, 12), "haifa_aggregate"),
                ],
            )
        )

    # --------------------------------------------------
    # Privatization, Haifa clock
    # --------------------------------------------------
    specs.append(
        make_spec(
            table_group = "privatization",
            reform = "haifa_priv",
            target = "Haifa-Legacy",
            target_key = "haifa_legacy",
            event_year = 2023,
            event_month = 1,
            treat_windows = [
                Window("Haifa-Legacy", (2022, 1), (2099, 12), "haifa_legacy"),
            ],
            control_windows = [
                Window("Haifa-Bayport", (2022, 1), (2099, 12), "haifa_bayport"),
                Window("Ashdod-Legacy", (2022, 1), (2099, 12), "ashdod_legacy"),
                Window("Ashdod-HCT", (2022, 1), (2099, 12), "ashdod_hct"),
            ],
        )
    )
    specs.append(
        make_spec(
            table_group = "privatization",
            reform = "haifa_priv",
            target = "Haifa-Bayport",
            target_key = "haifa_bayport_placebo",
            event_year = 2023,
            event_month = 1,
            treat_windows = [
                Window("Haifa-Bayport", (2022, 1), (2099, 12), "haifa_bayport"),
            ],
            control_windows = [
                Window("Haifa-Legacy", (2022, 1), (2099, 12), "haifa_legacy"),
                Window("Ashdod-Legacy", (2022, 1), (2099, 12), "ashdod_legacy"),
                Window("Ashdod-HCT", (2022, 1), (2099, 12), "ashdod_hct"),
            ],
        )
    )

    if ENABLE_AGGREGATE_SPECS and aggregate_series_available(df, "Haifa") and aggregate_series_available(df, "Ashdod"):
        specs.append(
            make_spec(
                table_group = "privatization",
                reform = "haifa_priv",
                target = "Haifa aggregate",
                target_key = "haifa_aggregate",
                event_year = 2023,
                event_month = 1,
                treat_windows = [
                    Window("Haifa aggregate", (2022, 1), (2099, 12), "haifa_aggregate"),
                ],
                control_windows = [
                    Window("Ashdod aggregate", (2022, 1), (2099, 12), "ashdod_aggregate"),
                ],
            )
        )

    return specs


# ----------------------------------------------------------------------
# 5. Load panel
# ----------------------------------------------------------------------

def load_lp_panel(path: Path) -> pd.DataFrame:
    print(f"Reading monthly LP panel from: {path}")
    df = pd.read_csv(path, sep = "\t")
    print(f"Loaded {len(df)} rows from LP_Panel_monthly.tsv.")

    required_cols = {"year", "month", "month_index", "LP", "series_id", "level", "port", "terminal"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"LP_Panel_monthly.tsv is missing required columns: {missing}")

    for col in ["year", "month", "month_index"]:
        df[col] = pd.to_numeric(df[col], errors = "coerce").astype("Int64")

    df["terminal"] = df["terminal"].fillna("")

    n_missing_lp = int(df["LP"].isna().sum())
    if n_missing_lp > 0:
        print(f"Dropping {n_missing_lp} rows with missing LP before log transform.")
        df = df[np.isfinite(df["LP"])].copy()

    if (df["LP"] <= 0).any():
        bad = df.loc[df["LP"] <= 0, ["series_id", "year", "month", "LP"]].head(10)
        raise ValueError(f"LP contains non-positive values; cannot take logs. Sample:\n{bad}")

    df["log_LP"] = np.log(df["LP"])

    covid_mask = df["year"].between(2020, 2021, inclusive = "both").fillna(False)
    df["covid_shock"] = covid_mask.astype(int)

    war_mask = ((df["year"] > 2023) | ((df["year"] == 2023) & (df["month"] >= 10))).fillna(False)
    df["war_shock"] = war_mask.astype(int)

    # raw identifiers remain available; analysis-side relabeling happens later inside build_es_sample()
    df["unit_id"] = df["series_id"]
    df["time_id"] = df["month_index"]
    df["cluster_id"] = df["series_id"]

    return df


# ----------------------------------------------------------------------
# 6. Time helpers + series lookup
# ----------------------------------------------------------------------

def build_year_month_to_index(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    mapping: Dict[Tuple[int, int], int] = {}
    grouped = df.groupby(["year", "month"])["month_index"].unique()
    for (y, m), idxs in grouped.items():
        idxs = [x for x in idxs if pd.notna(x)]
        if len(idxs) != 1:
            raise ValueError(f"(year={y}, month={m}) has multiple month_index values: {idxs}")
        mapping[(int(y), int(m))] = int(idxs[0])
    print(f"Built (year, month) -> month_index mapping for {len(mapping)} months.")
    return mapping


def _ym_serial(ym: Tuple[int, int]) -> int:
    y, m = ym
    return int(y) * 12 + int(m)


def year_month_to_index_strict(mapping: Dict[Tuple[int, int], int], ym: Tuple[int, int]) -> int:
    if ym not in mapping:
        raise KeyError(f"(year={ym[0]}, month={ym[1]}) not found in mapping.")
    return mapping[ym]


def year_month_to_index_clamped(mapping: Dict[Tuple[int, int], int], ym: Tuple[int, int]) -> int:
    if ym in mapping:
        return mapping[ym]

    target = _ym_serial(ym)
    keys = list(mapping.keys())
    serials = np.array([_ym_serial(k) for k in keys], dtype = int)

    ok = serials <= target
    if not ok.any():
        raise KeyError(f"No available month <= (year={ym[0]}, month={ym[1]}) in mapping.")

    best_serial = serials[ok].max()
    best_key = keys[int(np.where(serials == best_serial)[0][0])]
    return mapping[best_key]


def get_series_id_for_label(df: pd.DataFrame, label: str) -> str:
    if label not in LABEL_FILTERS:
        raise KeyError(f"Label '{label}' not found in LABEL_FILTERS.")

    conds = LABEL_FILTERS[label]
    mask = pd.Series(True, index = df.index)
    for col, val in conds.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' from LABEL_FILTERS is not in the data frame.")
        mask &= (df[col] == val)

    sids = df.loc[mask, "series_id"].dropna().astype(str).unique()
    if len(sids) == 0:
        unique = df[["series_id", "level", "port", "terminal"]].drop_duplicates()
        raise ValueError(
            f"No series_id found for label '{label}' with filters {conds}.\n"
            f"Some unique series in LP_Panel:\n{unique.head(30)}"
        )
    if len(sids) > 1:
        raise ValueError(f"Label '{label}' matched multiple series_ids: {sids}. Refine LABEL_FILTERS.")
    return str(sids[0])


# ----------------------------------------------------------------------
# 7. Build estimation sample for one spec
# ----------------------------------------------------------------------

def build_es_sample(
    df: pd.DataFrame,
    spec: Spec,
    ym_to_idx: Dict[Tuple[int, int], int],
    clamp_windows: bool,
) -> pd.DataFrame:
    df_es = df.copy()
    df_es["in_sample"] = False
    df_es["treated"] = False
    df_es["analysis_unit_id"] = df_es["series_id"].astype(str)

    ym_to_index = year_month_to_index_clamped if clamp_windows else year_month_to_index_strict

    for w in spec.treat_windows:
        sid = get_series_id_for_label(df_es, w.label)
        start_idx = ym_to_index(ym_to_idx, w.start)
        end_idx = ym_to_index(ym_to_idx, w.end)
        mask = (df_es["series_id"] == sid) & df_es["month_index"].between(start_idx, end_idx)
        df_es.loc[mask, "in_sample"] = True
        df_es.loc[mask, "treated"] = True
        if w.analysis_unit_label is not None:
            df_es.loc[mask, "analysis_unit_id"] = w.analysis_unit_label

    for w in spec.control_windows:
        sid = get_series_id_for_label(df_es, w.label)
        start_idx = ym_to_index(ym_to_idx, w.start)
        end_idx = ym_to_index(ym_to_idx, w.end)
        mask = (df_es["series_id"] == sid) & df_es["month_index"].between(start_idx, end_idx)
        df_es.loc[mask, "in_sample"] = True
        if w.analysis_unit_label is not None:
            df_es.loc[mask, "analysis_unit_id"] = w.analysis_unit_label

    df_es = df_es[df_es["in_sample"]].copy().reset_index(drop = True)
    if df_es.empty:
        return df_es

    event_index = ym_to_index(ym_to_idx, (spec.event_year, spec.event_month))

    df_es["event_time"] = -1
    treated_mask = df_es["treated"]
    df_es.loc[treated_mask, "event_time"] = df_es.loc[treated_mask, "month_index"] - event_index

    df_es["time_index"] = df_es["time_id"]
    df_es["port_trend"] = 0.0
    for port_name in sorted(df_es["port"].dropna().unique()):
        mask_port = df_es["port"] == port_name
        slope_sign = 1.0 if port_name == "Haifa" else -1.0
        df_es.loc[mask_port, "port_trend"] = df_es.loc[mask_port, "time_index"] * slope_sign

    # Use analysis-side unit relabeling for FE regression. Raw series_id remains the cluster id.
    df_es["unit_id"] = df_es["analysis_unit_id"].astype(str)
    df_es["j"] = df_es["event_time"]
    return df_es


# ----------------------------------------------------------------------
# 8. Regression helpers
# ----------------------------------------------------------------------

def _parse_event_time_from_param_name(name: str) -> Optional[int]:
    prefix = "C(event_time, Treatment(reference=-1))[T."
    if not name.startswith(prefix):
        return None
    m_str = name[len(prefix):].rstrip("]")
    try:
        return int(m_str)
    except ValueError:
        return None


def _coerce_for_patsy(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns and pd.api.types.is_integer_dtype(out[col].dtype):
            out[col] = out[col].astype("int64")
    return out


def fit_clustered_ols(formula: str, data: pd.DataFrame):
    model = smf.ols(formula = formula, data = data, missing = "drop")
    used_idx = pd.Index(model.data.row_labels)
    groups = data.loc[used_idx, "cluster_id"].to_numpy()

    if len(groups) != model.exog.shape[0]:
        raise ValueError(
            f"Cluster vector length {len(groups)} does not match model row count {model.exog.shape[0]}."
        )

    nobs, k_params = model.exog.shape
    n_clusters = pd.Series(groups).nunique()

    if n_clusters < 2:
        print("[WARN] Fewer than 2 clusters available; falling back to HC1.")
        return model.fit(cov_type = "HC1")

    if nobs <= k_params:
        print(
            f"[WARN] Saturated or near-saturated design detected "
            f"(nobs={nobs}, k_params={k_params}). "
            f"Retrying clustered SE without finite-sample correction."
        )
        return model.fit(
            cov_type = "cluster",
            cov_kwds = {
                "groups": groups,
                "use_correction": False,
            },
        )

    try:
        return model.fit(
            cov_type = "cluster",
            cov_kwds = {
                "groups": groups,
            },
        )
    except ZeroDivisionError:
        print(
            "[WARN] Cluster small-sample correction failed; "
            "retrying without finite-sample correction."
        )
        return model.fit(
            cov_type = "cluster",
            cov_kwds = {
                "groups": groups,
                "use_correction": False,
            },
        )
    

def fit_hc1_ols(formula: str, data: pd.DataFrame):
    model = smf.ols(formula = formula, data = data, missing = "drop")
    return model.fit(cov_type = "HC1")


def run_event_study(df_es: pd.DataFrame, spec_with_fe: SpecWithFE, shock_cols: List[str]):
    df_es = _coerce_for_patsy(df_es.copy(), ["event_time", "time_id"])

    formula = "log_LP ~ C(event_time, Treatment(reference=-1)) + C(unit_id) + C(time_id)"
    if spec_with_fe.include_port_trends:
        formula += " + port_trend"

    if spec_with_fe.include_shocks and shock_cols:
        used_shocks = [c for c in shock_cols if c in df_es.columns]
        if used_shocks:
            formula += " + " + " + ".join(used_shocks)
            

    n_by_event_time = df_es.groupby("event_time")["unit_id"].size().to_dict()
    result = fit_clustered_ols(formula = formula, data = df_es)
    return result, n_by_event_time


STATIC_HORIZONS: Dict[str, Optional[int]] = {
    "full_post": None,
    "post_y1": 12,
    "post_y1_2": 24,
}


def subset_for_static_horizon(df_es: pd.DataFrame, event_index: int, horizon_end: Optional[int]) -> pd.DataFrame:
    df_out = df_es.copy()
    if horizon_end is None:
        return df_out
    max_month = int(event_index + horizon_end)
    return df_out[df_out["month_index"] <= max_month].copy()


def run_static_did(
    df_es: pd.DataFrame,
    spec_with_fe: SpecWithFE,
    shock_cols: List[str],
    event_index: int,
    horizon_name: str,
    horizon_end: Optional[int],
):
    df_did = subset_for_static_horizon(df_es, event_index, horizon_end)
    df_did = _coerce_for_patsy(df_did, ["time_id"])

    if df_did.empty:
        return None

    n_post_treated = int(((df_did["treated"]) & (df_did["month_index"] >= (event_index + 1))).sum())
    if n_post_treated == 0:
        return None

    df_did["treated_int"] = df_did["treated"].astype(int)
    df_did["post_in_horizon"] = ((df_did["month_index"] >= (event_index + 1))).astype(int)
    df_did["treated_post"] = df_did["treated_int"] * df_did["post_in_horizon"]

    formula = "log_LP ~ treated_post + C(unit_id) + C(time_id)"
    if spec_with_fe.include_port_trends:
        formula += " + port_trend"

    if spec_with_fe.include_shocks and shock_cols:
        used_shocks = [c for c in shock_cols if c in df_did.columns]
        if used_shocks:
            formula += " + " + " + ".join(used_shocks)

    res = fit_clustered_ols(formula = formula, data = df_did)
    beta = float(res.params.get("treated_post", np.nan))
    se = float(res.bse.get("treated_post", np.nan))
    pval = float(res.pvalues.get("treated_post", np.nan))
    se_type = "cluster"

    if not np.isfinite(se):
        res_hc1 = fit_hc1_ols(formula = formula, data = df_did)
        beta = float(res_hc1.params.get("treated_post", np.nan))
        se = float(res_hc1.bse.get("treated_post", np.nan))
        pval = float(res_hc1.pvalues.get("treated_post", np.nan))
        res = res_hc1
        se_type = "HC1_fallback"

    treated_post_js = df_did.loc[
        (df_did["treated"]) & (df_did["month_index"] >= (event_index + 1)),
        "month_index"
    ] - event_index

    if len(treated_post_js) == 0:
        return None

    max_post_supported = int(treated_post_js.max())
    if horizon_end is not None:
        max_post_supported = min(max_post_supported, int(horizon_end))

    return {
        "horizon": horizon_name,
        "a": 1,
        "b": max_post_supported,
        "beta": beta,
        "se": se,
        "pvalue": pval,
        "n_obs": int(res.nobs),
        "r2": float(res.rsquared),
        "n_treated": int(df_did["treated"].sum()),
        "n_control": int((~df_did["treated"]).sum()),
        "n_post_treated": n_post_treated,
        "se_type": se_type,
    }


def extract_dynamic_betas(result, spec_with_fe: SpecWithFE, n_by_event_time: Dict[int, int], design_name: str) -> pd.DataFrame:
    spec = spec_with_fe.spec
    rows = []

    for name, beta in result.params.items():
        j = _parse_event_time_from_param_name(name)
        if j is None:
            continue
        se = float(result.bse.get(name, np.nan))
        t_stat = float(beta / se) if (np.isfinite(se) and se != 0) else np.nan
        pval = float(result.pvalues.get(name, np.nan))
        rows.append(
            {
                "design": design_name,
                "table_group": spec.table_group,
                "reform": spec.reform,
                "target": spec.target,
                "target_key": spec.target_key,
                "spec_name": spec_with_fe.spec_name,
                "event_time": int(j),
                "j": int(j),
                "beta": float(beta),
                "se": se,
                "t": t_stat,
                "pvalue": pval,
                "n_event_obs": float(n_by_event_time.get(int(j), np.nan)),
                "n_obs": int(result.nobs),
                "r2": float(result.rsquared),
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 9. Window averages from event-study betas
# ----------------------------------------------------------------------

PRETREND_MIN = -12
PRETREND_MAX = -2

WINDOWS: Dict[str, Tuple[int, int]] = {
    "avg_pre": (PRETREND_MIN, PRETREND_MAX),
    "post_y1": (1, 12),
    "post_y1_2": (1, 24),
    "full_post": (1, 999),
}

PRETREND_BINS: List[Tuple[int, int]] = [
    (PRETREND_MIN, -7),
    (-6, PRETREND_MAX),
]


def compute_window_averages(result, spec_with_fe: SpecWithFE, design_name: str) -> pd.DataFrame:
    spec = spec_with_fe.spec
    params = result.params
    cov = result.cov_params()

    j_to_name: Dict[int, str] = {}
    for name in params.index:
        j = _parse_event_time_from_param_name(name)
        if j is not None:
            j_to_name[int(j)] = name

    if not j_to_name:
        return pd.DataFrame([])

    available_js = sorted(j_to_name.keys())
    max_post_j = max((j for j in available_js if j > 0), default = None)

    rows = []
    for wname, (a, b) in WINDOWS.items():
        b_eff = b
        if max_post_j is not None:
            if b == 999:
                b_eff = int(max_post_j)
            elif a >= 1:
                b_eff = int(min(b, max_post_j))

        js = [j for j in available_js if (j >= a and j <= b_eff)]
        if not js:
            continue

        w = pd.Series(0.0, index = params.index)
        weight = 1.0 / len(js)
        for j in js:
            w[j_to_name[j]] = weight

        beta_w = float(np.dot(w.values, params.values))
        var_w = float(np.dot(w.values, np.dot(cov.values, w.values)))
        se_w = float(np.sqrt(var_w)) if var_w >= 0 else np.nan

        rows.append(
            {
                "design": design_name,
                "table_group": spec.table_group,
                "reform": spec.reform,
                "target": spec.target,
                "target_key": spec.target_key,
                "spec_name": spec_with_fe.spec_name,
                "window": wname,
                "a": int(a),
                "b": int(b_eff),
                "beta": beta_w,
                "se": se_w,
                "n_obs": int(result.nobs),
                "r2": float(result.rsquared),
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 10. Pretrend test
# ----------------------------------------------------------------------

def compute_pretrend_f_test(result, spec_with_fe: SpecWithFE, design_name: str) -> pd.DataFrame:
    params = result.params
    names = list(params.index)
    k = len(names)

    event_param_info: List[Tuple[int, int]] = []
    for idx, name in enumerate(names):
        j = _parse_event_time_from_param_name(name)
        if j is not None and PRETREND_MIN <= j <= PRETREND_MAX:
            event_param_info.append((int(j), idx))

    if not event_param_info:
        return pd.DataFrame([])

    R_rows: List[np.ndarray] = []
    bins_used: List[Tuple[int, int]] = []
    for (a, b) in PRETREND_BINS:
        idxs = [idx for (j, idx) in event_param_info if a <= j <= b]
        if not idxs:
            continue
        r = np.zeros(k)
        w = 1.0 / len(idxs)
        for j_idx in idxs:
            r[j_idx] = w
        R_rows.append(r)
        bins_used.append((a, b))

    if not R_rows:
        return pd.DataFrame([])

    R = np.vstack(R_rows)
    n_restr = int(R.shape[0])

    try:
        cov_unclustered = np.asarray(result.normalized_cov_params) * float(result.mse_resid)
        wtest = result.wald_test(R, cov_p = cov_unclustered, use_f = True, scalar = False)
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

    return pd.DataFrame(
        [
            {
                "design": design_name,
                "table_group": spec_with_fe.spec.table_group,
                "reform": spec_with_fe.spec.reform,
                "target": spec_with_fe.spec.target,
                "target_key": spec_with_fe.spec.target_key,
                "spec_name": spec_with_fe.spec_name,
                "pre_min": float(PRETREND_MIN),
                "pre_max": float(PRETREND_MAX),
                "n_leads_total": float(len(event_param_info)),
                "n_bins_defined": float(len(PRETREND_BINS)),
                "n_bins_used": float(len(bins_used)),
                "f_stat": f_val,
                "pvalue": p_val,
                "df_num": df_num,
                "df_denom": df_denom,
                "n_obs": int(result.nobs),
                "r2": float(result.rsquared),
            }
        ]
    )


# ----------------------------------------------------------------------
# 11. Driver helpers
# ----------------------------------------------------------------------

def build_specs_with_fe(base_specs: List[Spec], shock_cols_all: List[str]) -> List[SpecWithFE]:
    out: List[SpecWithFE] = []
    for s in base_specs:
        out.append(SpecWithFE(spec = s, spec_name = "baseline", include_port_trends = False, include_shocks = False))
        out.append(SpecWithFE(spec = s, spec_name = "porttr", include_port_trends = True, include_shocks = False))
        if INCLUDE_SHOCK_SPECS and shock_cols_all:
            out.append(SpecWithFE(spec = s, spec_name = "tr_shocks", include_port_trends = True, include_shocks = True))
    return out


def make_static_rows(
    df_es: pd.DataFrame,
    spec_with_fe: SpecWithFE,
    shock_cols_all: List[str],
    event_index: int,
    design_name: str,
) -> pd.DataFrame:
    rows = []
    for horizon_name, horizon_end in STATIC_HORIZONS.items():
        out = run_static_did(
            df_es = df_es,
            spec_with_fe = spec_with_fe,
            shock_cols = shock_cols_all,
            event_index = event_index,
            horizon_name = horizon_name,
            horizon_end = horizon_end,
        )
        if out is None:
            continue
        rows.append(
            {
                "design": design_name,
                "table_group": spec_with_fe.spec.table_group,
                "reform": spec_with_fe.spec.reform,
                "target": spec_with_fe.spec.target,
                "target_key": spec_with_fe.spec.target_key,
                "spec_name": spec_with_fe.spec_name,
                **out,
            }
        )
    return pd.DataFrame(rows)


def run_design(
    df: pd.DataFrame,
    ym_to_idx: Dict[Tuple[int, int], int],
    shock_cols_all: List[str],
    base_specs: List[Spec],
    design_name: str,
    clamp_windows: bool,
    write_per_spec: bool,
    suffix: str,
    run_static: bool,
) -> None:
    print(f"\n==================== {design_name} run ====================")
    specs_with_fe = build_specs_with_fe(base_specs, shock_cols_all)

    dynamic_by_spec: Dict[str, List[pd.DataFrame]] = {}
    windows_by_spec: Dict[str, List[pd.DataFrame]] = {}
    pretrend_by_spec: Dict[str, List[pd.DataFrame]] = {}
    static_by_spec: Dict[str, List[pd.DataFrame]] = {}

    for spec_with_fe in specs_with_fe:
        spec = spec_with_fe.spec
        print(
            f"\n=== [{design_name}] table={spec.table_group}, reform={spec.reform}, "
            f"target={spec.target}, spec={spec_with_fe.spec_name} ==="
        )

        df_es = build_es_sample(df, spec, ym_to_idx, clamp_windows = clamp_windows)
        if df_es.empty or df_es["treated"].sum() == 0:
            print("[WARN] No treated observations. Skipping.")
            continue

        if not ((df_es["treated"]) & (df_es["event_time"] >= 1)).any():
            print("[WARN] No treated post observations. Skipping.")
            continue

        n_treat = int(df_es["treated"].sum())
        n_ctrl = int((~df_es["treated"]).sum())
        print(f"Sample size: {len(df_es)} rows ({n_treat} treated, {n_ctrl} controls).")

        if clamp_windows:
            event_index = year_month_to_index_clamped(ym_to_idx, (spec.event_year, spec.event_month))
        else:
            event_index = year_month_to_index_strict(ym_to_idx, (spec.event_year, spec.event_month))

        es_res, n_by_j = run_event_study(df_es, spec_with_fe, shock_cols_all)
        dyn = extract_dynamic_betas(es_res, spec_with_fe, n_by_j, design_name)
        win = compute_window_averages(es_res, spec_with_fe, design_name)
        pre = compute_pretrend_f_test(es_res, spec_with_fe, design_name)
        sta = pd.DataFrame([])
        if run_static:
            sta = make_static_rows(df_es, spec_with_fe, shock_cols_all, event_index, design_name)

        if not dyn.empty:
            dynamic_by_spec.setdefault(spec_with_fe.spec_name, []).append(dyn)
        if not win.empty:
            windows_by_spec.setdefault(spec_with_fe.spec_name, []).append(win)
        if not pre.empty:
            pretrend_by_spec.setdefault(spec_with_fe.spec_name, []).append(pre)
        if run_static and not sta.empty:
            static_by_spec.setdefault(spec_with_fe.spec_name, []).append(sta)

    base_name = "model1a_lp"
    pooled = {"dynamic": [], "window": [], "pretrend": [], "static": []}

    if write_per_spec:
        for spec_name, frames in dynamic_by_spec.items():
            ddf = pd.concat(frames, ignore_index = True)
            pooled["dynamic"].append(ddf)
            path = OUTPUT_DIR / f"{base_name}_dynamic_betas_{spec_name}{suffix}.tsv"
            ddf.to_csv(path, sep = "\t", index = False)
            print(f"Saved dynamic betas ({design_name}, spec={spec_name}) to: {path}")

        for spec_name, frames in windows_by_spec.items():
            wdf = pd.concat(frames, ignore_index = True)
            pooled["window"].append(wdf)
            path = OUTPUT_DIR / f"{base_name}_window_betas_{spec_name}{suffix}.tsv"
            wdf.to_csv(path, sep = "\t", index = False)
            print(f"Saved window betas ({design_name}, spec={spec_name}) to: {path}")

        for spec_name, frames in pretrend_by_spec.items():
            pdf = pd.concat(frames, ignore_index = True)
            pooled["pretrend"].append(pdf)
            path = OUTPUT_DIR / f"{base_name}_pretrend_tests_{spec_name}{suffix}.tsv"
            pdf.to_csv(path, sep = "\t", index = False)
            print(f"Saved pretrend tests ({design_name}, spec={spec_name}) to: {path}")

        if run_static:
            for spec_name, frames in static_by_spec.items():
                sdf = pd.concat(frames, ignore_index = True)
                pooled["static"].append(sdf)
                path = OUTPUT_DIR / f"{base_name}_static_betas_{spec_name}{suffix}.tsv"
                sdf.to_csv(path, sep = "\t", index = False)
                print(f"Saved static betas ({design_name}, spec={spec_name}) to: {path}")

    if dynamic_by_spec:
        if not pooled["dynamic"]:
            pooled["dynamic"] = [pd.concat(frames, ignore_index = True) for frames in dynamic_by_spec.values()]
        dyn_all = pd.concat(pooled["dynamic"], ignore_index = True)
        path = OUTPUT_DIR / f"{base_name}_dynamic_betas_all{suffix}.tsv"
        dyn_all.to_csv(path, sep = "\t", index = False)
        print(f"Saved pooled dynamic betas ({design_name}) to: {path}")

    if windows_by_spec:
        if not pooled["window"]:
            pooled["window"] = [pd.concat(frames, ignore_index = True) for frames in windows_by_spec.values()]
        win_all = pd.concat(pooled["window"], ignore_index = True)
        path = OUTPUT_DIR / f"{base_name}_window_betas_all{suffix}.tsv"
        win_all.to_csv(path, sep = "\t", index = False)
        print(f"Saved pooled window betas ({design_name}) to: {path}")

    if pretrend_by_spec:
        if not pooled["pretrend"]:
            pooled["pretrend"] = [pd.concat(frames, ignore_index = True) for frames in pretrend_by_spec.values()]
        pre_all = pd.concat(pooled["pretrend"], ignore_index = True)
        path = OUTPUT_DIR / f"{base_name}_pretrend_tests_all{suffix}.tsv"
        pre_all.to_csv(path, sep = "\t", index = False)
        print(f"Saved pooled pretrend tests ({design_name}) to: {path}")

    if run_static and static_by_spec:
        if not pooled["static"]:
            pooled["static"] = [pd.concat(frames, ignore_index = True) for frames in static_by_spec.values()]
        sta_all = pd.concat(pooled["static"], ignore_index = True)
        path = OUTPUT_DIR / f"{base_name}_static_betas_all{suffix}.tsv"
        sta_all.to_csv(path, sep = "\t", index = False)
        print(f"Saved pooled static betas ({design_name}) to: {path}")


def clear_model1a_outputs(output_dir: Path) -> None:
    patterns = [
        "model1a_lp_dynamic_betas_*.tsv",
        "model1a_lp_window_betas_*.tsv",
        "model1a_lp_pretrend_tests_*.tsv",
        "model1a_lp_static_betas_*.tsv",
    ]

    removed = 0
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok = True)
            removed += 1

    print(f"Cleared {removed} old Model 1A output files from: {output_dir}")


def debug_nyt_privatization(
    df: pd.DataFrame,
    ym_to_idx: Dict[Tuple[int, int], int],
) -> None:
    print("\n==================== DEBUG: NYT privatization ====================\n")

    nyt_specs = build_nyt_specs(df)
    priv_specs = [s for s in nyt_specs if s.reform == "haifa_priv"]

    for spec in priv_specs:
        print(f"--- target = {spec.target} | target_key = {spec.target_key} ---")
        df_es = build_es_sample(
            df = df,
            spec = spec,
            ym_to_idx = ym_to_idx,
            clamp_windows = True,
        )

        if df_es.empty:
            print("Sample is empty.\n")
            continue

        print("Rows by series_id x treated:")
        print(
            df_es.groupby(["series_id", "treated"])
                 .size()
                 .rename("n")
                 .reset_index()
                 .to_string(index = False)
        )

        print("\nRows by unit_id x treated:")
        print(
            df_es.groupby(["unit_id", "treated"])
                 .size()
                 .rename("n")
                 .reset_index()
                 .to_string(index = False)
        )

        tmp = df_es.copy()
        tmp["post"] = (tmp["month_index"] >= ym_to_idx[(spec.event_year, spec.event_month)]).astype(int)

        print("\nMean log_LP by unit_id x post:")
        print(
            tmp.groupby(["unit_id", "post"])["log_LP"]
               .mean()
               .rename("mean_log_LP")
               .reset_index()
               .to_string(index = False)
        )

        print("\nMean log_LP by series_id x post:")
        print(
            tmp.groupby(["series_id", "post"])["log_LP"]
               .mean()
               .rename("mean_log_LP")
               .reset_index()
               .to_string(index = False)
        )

        print("\n")

# ----------------------------------------------------------------------
# 12. Main
# ----------------------------------------------------------------------

def main() -> None:
    clear_model1a_outputs(OUTPUT_DIR)

    df = load_lp_panel(LP_PANEL_PATH)
    ym_to_idx = build_year_month_to_index(df)

    debug_nyt_privatization(df = df, ym_to_idx = ym_to_idx) #temporary debug for NYT privatization spec
    
    shock_cols_all = get_shock_control_cols(df)

    if shock_cols_all:
        print(f"Detected shock-control columns: {shock_cols_all}")
    else:
        print("No shock-control columns detected.")

    if ENABLE_AGGREGATE_SPECS:
        available = []
        if aggregate_series_available(df, "Haifa"):
            available.append("Haifa")
        if aggregate_series_available(df, "Ashdod"):
            available.append("Ashdod")
        print(f"Aggregate specs requested. Upstream aggregate series present for: {available}")
    else:
        print("Aggregate specs disabled by configuration. Aggregate hooks remain scaffolded in the code.")

    nyt_specs = build_nyt_specs(df)
    twfe_specs = build_twfe_specs(df)

    run_design(
        df = df,
        ym_to_idx = ym_to_idx,
        shock_cols_all = shock_cols_all,
        base_specs = nyt_specs,
        design_name = "NYT",
        clamp_windows = False,
        write_per_spec = True,
        suffix = "",
        run_static = WRITE_STATIC_FOR_NYT,
    )

    run_design(
        df = df,
        ym_to_idx = ym_to_idx,
        shock_cols_all = shock_cols_all,
        base_specs = twfe_specs,
        design_name = "TWFE",
        clamp_windows = True,
        write_per_spec = True,
        suffix = "_twfe",
        run_static = WRITE_STATIC_FOR_TWFE,
    )




if __name__ == "__main__":
    main()





# =============================================================================
# EVALUATION NOTE AFTER v4 PATCH TEST RUN + NYT PRIVATIZATION DEBUG
#
# Summary:
# Model_1A(v4) is now behaving as intended at the estimator-logic level.
# The earlier concern about identical NYT privatization estimates for
# Haifa-Legacy and Haifa-Bayport was investigated with an explicit debug pass.
# The debug results show that this is NOT a coding bug in Model 1A, but rather
# a consequence of the current interim LP data construction.
#
# What worked:
# 1. Output cleanup worked correctly: stale Model 1A TSVs were deleted before run.
# 2. The script completed end-to-end and wrote all expected output families:
#    dynamic, window, pretrend, and TWFE static betas.
# 3. The Ashdod TWFE competition clock fix appears successful. In the saved
#    static output, Ashdod competition now shows post-period counts consistent
#    with a Nov 2022 reform date, rather than the old mistaken 2021 split.
# 4. Saturation warnings, R^2 = 1.0 in some competition specs, and extremely
#    small or undefined clustered SEs are consistent with the current temporary
#    LP construction / expanded monthly panel and do not by themselves indicate
#    a coding bug.
#
# NYT privatization debug findings:
# 5. The treated assignment in the NYT privatization branch is correct.
#    - In the Haifa-Legacy spec, only Haifa-Legacy is treated.
#    - In the Haifa-Bayport spec, only Haifa-Bayport is treated.
#    - The control units remain the expected untreated terminal series.
# 6. The unit_id and series_id mappings are also correct; the code is not
#    accidentally collapsing the two privatization specs onto the same rows.
# 7. The reason the NYT privatization estimates are numerically identical is
#    in the current LP data structure:
#    - Haifa-Legacy and Haifa-Bayport have different levels of log_LP,
#      but exactly the same pre/post mean change in the debug sample.
#    - The two Ashdod control units also show exactly the same pre/post mean
#      change in the debug sample.
#    This means the privatization event-study is currently seeing nearly
#    perfectly parallel within-port changes, so switching the treated Haifa
#    terminal does not materially change the estimated dynamic effects.
# 8. Therefore, the identical NYT privatization estimates are best interpreted
#    as a feature of the interim LP construction, not a Model 1A logic error.
#
# Minor non-blocking notes:
# 9. A few NaN SE / p-value entries in dynamic outputs are expected under
#    singular covariance in isolated bins.
# 10. NaN pretrend F-tests for Haifa competition are also expected here because
#     the competition design is saturated enough that df_denom collapses to 0.
# 11. For privatization, post_y1_2 matching full_post is acceptable given the
#     limited post-treatment support in the current sample.
#
# Bottom line:
# Model 1A is now in acceptable shape for the current v4 pipeline.
# No further estimator-level patch is currently needed for Model 1A.
# Remaining limitations are primarily driven by the temporary LP data rather
# than by the regression code itself.
# =============================================================================
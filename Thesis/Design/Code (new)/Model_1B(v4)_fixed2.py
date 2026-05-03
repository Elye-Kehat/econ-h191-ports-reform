from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =============================================================================
# Model_1B(v4): ln(K/L) event-study with revised thesis architecture
#
# Purpose:
#   Implement the settled v4 Model 1B logic so that the K/L results line up
#   with the updated thesis tables.
#
# Core architecture:
#   - Competition table:
#       Panel A (TWFE static benchmark):
#           Haifa-Legacy, Ashdod-Legacy, and optional aggregates
#       Panel B (strict NYT event-study):
#           Haifa-Legacy, and optional Haifa aggregate only
#
#   - Privatization table:
#       Panel A (TWFE static benchmark):
#           Haifa-Legacy, Haifa-Bayport placebo, and optional Haifa aggregate
#       Panel B (strict NYT event-study):
#           same target set as above
#
# Main estimator in this file:
#   "baseline" = event-study / DiD with unit FE + calendar-month FE
#
# Companion files:
#   - Model_1B_relaxed(v4).py:
#       relaxed + port-trend alternative (used as the second column family in
#       the tables, labeled "Relaxed+Tr")
#   - Model_1B_to_tables(v4).py:
#       merges baseline + relaxed outputs into the v4 tables
#
# Outputs written here:
#   NYT (baseline):
#       model1b_kl_dynamic_betas_all.tsv
#       model1b_kl_window_betas_all.tsv
#       model1b_kl_pretrend_tests_all.tsv
#
#   TWFE (baseline):
#       model1b_kl_dynamic_betas_all_twfe.tsv
#       model1b_kl_window_betas_all_twfe.tsv
#       model1b_kl_pretrend_tests_all_twfe.tsv
#       model1b_kl_static_betas_all_twfe.tsv
# =============================================================================


MIN_EVENT_TIME = -12
MAX_EVENT_TIME = 24

ENABLE_AGGREGATE_SPECS = False
WRITE_STATIC_FOR_NYT = False
WRITE_STATIC_FOR_TWFE = True

# Lean-output option: write only pooled *_all files by default.
WRITE_SPLIT_SPEC_FILES = False

PRETREND_MIN = -12
PRETREND_MAX = -2
PRETREND_BINS = [(-12, -7), (-6, -2)]

WINDOWS = {
    "avg_pre": (PRETREND_MIN, PRETREND_MAX),
    "post_y1": (1, 12),
    "post_y1_2": (1, 24),
    "full_post": (1, 999),
}

STATIC_HORIZONS = {
    "full_post": None,
    "post_y1": 12,
    "post_y1_2": 24,
}

EXPLICIT_SHOCK_COLS = ["covid_shock", "war_shock"]


def find_thesis_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


THESIS_ROOT = find_thesis_root()
KL_PANEL_PATH = THESIS_ROOT / "Data" / "KL" / "KL_Panel_monthly.tsv"
OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1B"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)


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
# Series-id integration with the new K build
# ----------------------------------------------------------------------

# The new K pipeline produces multiple Haifa port-series variants by entity and
# depreciation scenario. For the econometric layer we need one canonical
# scenario to stand in for the Haifa aggregate / port object. Until the K/L
# build is expanded further, we use the central cluster series by default.
CANONICAL_K_SCENARIO = "central"
CANONICAL_HAIFA_PORT_SERIES_ID = f"Haifa_port_KL_cluster_{CANONICAL_K_SCENARIO}"
CANONICAL_HAIFA_AGGREGATE_SERIES_ID = CANONICAL_HAIFA_PORT_SERIES_ID
CANONICAL_HAIFA_BAYPORT_SERIES_ID = f"Haifa_port_KL_SIPG_{CANONICAL_K_SCENARIO}"

LABEL_FILTERS: Dict[str, List[Dict[str, str]]] = {
    "Haifa port": [
        {"series_id": CANONICAL_HAIFA_PORT_SERIES_ID},
        {"series_id": "haifa_port_kl_cluster_central"},
        {"series_id": "haifa_aggregate_port"},
        {"series_id": "haifa_aggregate"},
        {"series_id": "Haifa aggregate"},
        {"series_id": "haifa_port"},
        {"series_id": "Haifa port"},
        {"series_id": "haifa_port_kl"},
        {"series_id": "Haifa_port_KL"},
    ],
    "Ashdod port": [
        {"series_id": "ashdod_port_kl_cluster_central"},
        {"series_id": "ashdod_aggregate_port"},
        {"series_id": "ashdod_aggregate"},
        {"series_id": "Ashdod aggregate"},
        {"series_id": "ashdod_port"},
        {"series_id": "Ashdod port"},
        {"series_id": "ashdod_port_kl"},
        {"series_id": "Ashdod_port_KL"},
    ],
    "Haifa-Legacy": [
        {"series_id": "Haifa_Legacy_KL"},
        {"level": "terminal", "port": "Haifa", "terminal": "Haifa-Legacy"},
        {"series_id": "haifa_legacy"},
        {"series_id": "Haifa-Legacy"},
    ],
    "Haifa-Bayport": [
        {"series_id": CANONICAL_HAIFA_BAYPORT_SERIES_ID},
        {"series_id": "Haifa_SIPG_KL"},
        {"level": "terminal", "port": "Haifa", "terminal": "Haifa-Bayport"},
        {"level": "terminal", "port": "Haifa", "terminal": "Haifa-SIPG"},
        {"series_id": "haifa_bayport"},
        {"series_id": "Haifa-Bayport"},
    ],
    "Ashdod-Legacy": [
        {"series_id": "Ashdod_Legacy_KL"},
        {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-Legacy"},
        {"series_id": "ashdod_legacy"},
        {"series_id": "Ashdod-Legacy"},
    ],
    "Ashdod-HCT": [
        {"series_id": "Ashdod_HCT_KL"},
        {"level": "terminal", "port": "Ashdod", "terminal": "Ashdod-HCT"},
        {"series_id": "ashdod_hct"},
        {"series_id": "Ashdod-HCT"},
    ],
    "Haifa aggregate": [
        {"series_id": CANONICAL_HAIFA_AGGREGATE_SERIES_ID},
        {"series_id": "haifa_aggregate_port"},
        {"series_id": "haifa_aggregate"},
        {"series_id": "Haifa aggregate"},
    ],
    "Ashdod aggregate": [
        {"series_id": "ashdod_aggregate_port"},
        {"series_id": "ashdod_aggregate"},
        {"series_id": "Ashdod aggregate"},
    ],
}


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
    if port_name == "Haifa":
        candidates = {
            CANONICAL_HAIFA_AGGREGATE_SERIES_ID,
            "haifa_aggregate_port",
            "haifa_aggregate",
            "Haifa aggregate",
        }
    else:
        candidates = {
            "ashdod_aggregate_port",
            "ashdod_aggregate",
            "Ashdod aggregate",
        }
    sids = set(df["series_id"].dropna().astype(str).unique())
    return len(candidates.intersection(sids)) > 0


def try_get_series_id_for_label(df: pd.DataFrame, label: str) -> Optional[str]:
    if label not in LABEL_FILTERS:
        return None

    ambiguous_matches: List[Tuple[Dict[str, str], List[str]]] = []
    for candidate in LABEL_FILTERS[label]:
        sids = _match_one_candidate(df, candidate)
        if len(sids) == 1:
            return sids[0]
        if len(sids) > 1:
            ambiguous_matches.append((candidate, sids))

    if ambiguous_matches:
        candidate, sids = ambiguous_matches[0]
        raise ValueError(f"Label {label!r} matched multiple series_ids under candidate {candidate}: {sids}")
    return None


def label_available(df: pd.DataFrame, label: str) -> bool:
    try:
        return try_get_series_id_for_label(df, label) is not None
    except Exception:
        return False


def filter_available_control_windows(df: pd.DataFrame, windows: List[Window]) -> List[Window]:
    out: List[Window] = []
    for w in windows:
        if label_available(df, w.label):
            out.append(w)
    return out


def build_nyt_specs(df: pd.DataFrame) -> List[Spec]:
    specs: List[Spec] = []

    # --------------------------------------------------
    # Competition (strict NYT: Haifa only), but only if the
    # current KL panel actually contains the required Ashdod controls.
    # --------------------------------------------------
    if all(label_available(df, lbl) for lbl in ["Haifa port", "Haifa-Legacy", "Ashdod port", "Ashdod-Legacy"]):
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
    else:
        print("[INFO] Skipping NYT Haifa-competition K/L spec because the current KL panel does not contain the full Ashdod control path.")

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
    # Privatization
    # Keep Bayport as the key placebo/control path. Add Ashdod controls only if
    # they exist in the current KL panel.
    # --------------------------------------------------
    priv_controls = filter_available_control_windows(
        df,
        [
            Window("Haifa-Bayport", (2022, 1), (2023, 9), "haifa_bayport"),
            Window("Ashdod-Legacy", (2022, 1), (2023, 9), "ashdod_legacy"),
            Window("Ashdod-HCT", (2022, 1), (2023, 9), "ashdod_hct"),
        ],
    )
    if label_available(df, "Haifa-Legacy") and len(priv_controls) > 0:
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
                control_windows = priv_controls,
            )
        )

    bayport_controls = filter_available_control_windows(
        df,
        [
            Window("Haifa-Legacy", (2022, 1), (2023, 9), "haifa_legacy"),
            Window("Ashdod-Legacy", (2022, 1), (2023, 9), "ashdod_legacy"),
            Window("Ashdod-HCT", (2022, 1), (2023, 9), "ashdod_hct"),
        ],
    )
    if label_available(df, "Haifa-Bayport") and len(bayport_controls) > 0:
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
                control_windows = bayport_controls,
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
    if all(label_available(df, lbl) for lbl in ["Haifa port", "Haifa-Legacy", "Ashdod port", "Ashdod-Legacy"]):
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
    else:
        print("[INFO] Skipping TWFE Haifa-competition K/L spec because the current KL panel does not contain the full Ashdod control path.")

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
    if all(label_available(df, lbl) for lbl in ["Ashdod port", "Ashdod-Legacy", "Haifa port", "Haifa-Legacy"]):
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
    else:
        print("[INFO] Skipping TWFE Ashdod-competition K/L spec because the current KL panel does not contain the Ashdod K/L series.")

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
    priv_controls = filter_available_control_windows(
        df,
        [
            Window("Haifa-Bayport", (2022, 1), (2099, 12), "haifa_bayport"),
            Window("Ashdod-Legacy", (2022, 1), (2099, 12), "ashdod_legacy"),
            Window("Ashdod-HCT", (2022, 1), (2099, 12), "ashdod_hct"),
        ],
    )
    if label_available(df, "Haifa-Legacy") and len(priv_controls) > 0:
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
                control_windows = priv_controls,
            )
        )

    bayport_controls = filter_available_control_windows(
        df,
        [
            Window("Haifa-Legacy", (2022, 1), (2099, 12), "haifa_legacy"),
            Window("Ashdod-Legacy", (2022, 1), (2099, 12), "ashdod_legacy"),
            Window("Ashdod-HCT", (2022, 1), (2099, 12), "ashdod_hct"),
        ],
    )
    if label_available(df, "Haifa-Bayport") and len(bayport_controls) > 0:
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
                control_windows = bayport_controls,
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


def load_kl_panel(path: Path) -> pd.DataFrame:
    print(f"Reading monthly KL panel from: {path}")
    df = pd.read_csv(path, sep = "\t")
    print(f"Loaded {len(df)} rows from KL_Panel_monthly.tsv.")

    required_cols = {"year", "month", "series_id"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"KL_Panel_monthly.tsv is missing required columns: {sorted(missing)}")

    for col in ["level", "port", "terminal"]:
        if col not in df.columns:
            df[col] = ""
    df["terminal"] = df["terminal"].fillna("")

    if "log_KL" not in df.columns:
        if "KL" not in df.columns:
            raise ValueError("KL_Panel_monthly.tsv must contain 'KL' or 'log_KL'.")
        if (df["KL"] <= 0).any():
            bad = df.loc[df["KL"] <= 0, ["series_id", "year", "month", "KL"]].head(10)
            raise ValueError(f"KL must be positive to take logs. Example bad rows:\n{bad}")
        df["log_KL"] = np.log(df["KL"])

    covid_mask = df["year"].between(2020, 2021, inclusive = "both").fillna(False)
    df["covid_shock"] = covid_mask.astype(int)
    war_mask = ((df["year"] > 2023) | ((df["year"] == 2023) & (df["month"] >= 10))).fillna(False)
    df["war_shock"] = war_mask.astype(int)

    df["ym_tuple"] = list(zip(df["year"], df["month"]))
    ym_sorted = sorted(df["ym_tuple"].unique())
    ym_to_idx = {ym: i + 1 for i, ym in enumerate(ym_sorted)}
    df["month_index"] = df["ym_tuple"].map(ym_to_idx)

    df["unit_id"] = df["series_id"].astype(str)
    df["time_id"] = df["month_index"]
    if "port" in df.columns and df["port"].notna().any():
        df["cluster_id"] = df["port"].astype(str)
    else:
        df["cluster_id"] = df["series_id"].astype(str)

    print(f"Built (year, month) -> month_index mapping for {len(ym_sorted)} months.")
    return df


def build_year_month_to_index(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    grouped = df.groupby(["year", "month"])["month_index"].unique()
    mapping: Dict[Tuple[int, int], int] = {}
    for (y, m), vals in grouped.items():
        vals = [int(v) for v in vals if pd.notna(v)]
        if len(vals) != 1:
            raise ValueError(f"(year={y}, month={m}) has multiple month_index values: {vals}")
        mapping[(int(y), int(m))] = int(vals[0])
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
        raise KeyError(f"No available month <= {ym} in mapping.")
    best_serial = serials[ok].max()
    best_key = keys[int(np.where(serials == best_serial)[0][0])]
    return mapping[best_key]
def _match_one_candidate(df: pd.DataFrame, candidate: Dict[str, str]) -> List[str]:
    mask = pd.Series(True, index = df.index)
    for col, val in candidate.items():
        if col not in df.columns:
            return []
        mask &= (df[col].astype(str) == str(val))
    return sorted(df.loc[mask, "series_id"].dropna().astype(str).unique().tolist())


def get_series_id_for_label(df: pd.DataFrame, label: str) -> str:
    sid = try_get_series_id_for_label(df, label)
    if sid is not None:
        return sid
    unique = df[["series_id", "level", "port", "terminal"]].drop_duplicates().head(40)
    raise ValueError(f"No series_id found for label {label!r}.\nFirst unique series rows:\n{unique}")


def build_es_sample(df: pd.DataFrame, spec: Spec, ym_to_idx: Dict[Tuple[int, int], int], clamp_windows: bool) -> pd.DataFrame:
    df_es = df.copy()
    df_es["in_sample"] = False
    df_es["treated"] = False
    df_es["analysis_unit_id"] = df_es["series_id"].astype(str)

    ym_to_index = year_month_to_index_clamped if clamp_windows else year_month_to_index_strict

    matched_treat = 0
    matched_controls = 0

    for w in spec.treat_windows:
        sid = get_series_id_for_label(df_es, w.label)
        start_idx = ym_to_index(ym_to_idx, w.start)
        end_idx = ym_to_index(ym_to_idx, w.end)
        mask = (df_es["series_id"] == sid) & df_es["month_index"].between(start_idx, end_idx)
        if mask.any():
            matched_treat += 1
        df_es.loc[mask, "in_sample"] = True
        df_es.loc[mask, "treated"] = True
        if w.analysis_unit_label is not None:
            df_es.loc[mask, "analysis_unit_id"] = w.analysis_unit_label

    for w in spec.control_windows:
        sid = try_get_series_id_for_label(df_es, w.label)
        if sid is None:
            continue
        start_idx = ym_to_index(ym_to_idx, w.start)
        end_idx = ym_to_index(ym_to_idx, w.end)
        mask = (df_es["series_id"] == sid) & df_es["month_index"].between(start_idx, end_idx)
        if mask.any():
            matched_controls += 1
        df_es.loc[mask, "in_sample"] = True
        if w.analysis_unit_label is not None:
            df_es.loc[mask, "analysis_unit_id"] = w.analysis_unit_label

    if matched_treat == 0:
        raise ValueError(f"No treated windows could be matched for spec {spec.reform} / {spec.target}.")
    if matched_controls == 0:
        raise ValueError(
            f"No control windows could be matched for spec {spec.reform} / {spec.target}. "
            "This usually means the current KL panel does not yet contain the comparator units required by the design."
        )

    df_es = df_es[df_es["in_sample"]].copy().reset_index(drop = True)
    if df_es.empty:
        return df_es

    event_index = ym_to_index(ym_to_idx, (spec.event_year, spec.event_month))
    df_es["event_time"] = -1
    treated_mask = df_es["treated"]
    df_es.loc[treated_mask, "event_time"] = df_es.loc[treated_mask, "month_index"] - event_index

    df_es["time_index"] = df_es["time_id"]
    df_es["port_trend"] = 0.0
    for port_name in sorted(df_es["port"].dropna().astype(str).unique()):
        if port_name == "":
            continue
        mask_port = df_es["port"].astype(str) == port_name
        slope_sign = 1.0 if port_name == "Haifa" else -1.0
        df_es.loc[mask_port, "port_trend"] = df_es.loc[mask_port, "time_index"] * slope_sign

    df_es["unit_id"] = df_es["analysis_unit_id"].astype(str)
    df_es["m"] = df_es["event_time"]
    df_es["j"] = df_es["event_time"]
    return df_es


def _parse_event_time_from_param_name(name: str) -> Optional[int]:
    prefix = "C(event_time, Treatment(reference=-1))[T."
    if not str(name).startswith(prefix):
        return None
    tail = str(name)[len(prefix):].rstrip("]")
    try:
        return int(float(tail))
    except Exception:
        return None


def _coerce_for_patsy(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors = "coerce")
            # Patsy / statsmodels do not reliably accept pandas nullable Int64 dtypes.
            # Use plain NumPy-backed dtypes instead.
            if s.isna().any():
                out[c] = s.astype(float)
            else:
                out[c] = s.astype(np.int64)
    return out


def fit_clustered_ols(formula: str, data: pd.DataFrame):
    model = smf.ols(formula = formula, data = data, missing = "drop")
    used_idx = pd.Index(model.data.row_labels)
    groups = data.loc[used_idx, "cluster_id"].to_numpy() if "cluster_id" in data.columns else None
    if groups is None or len(np.unique(groups)) < 2:
        return model.fit(cov_type = "HC1")
    try:
        return model.fit(cov_type = "cluster", cov_kwds = {"groups": groups})
    except Exception:
        return model.fit(cov_type = "HC1")


def fit_hc1_ols(formula: str, data: pd.DataFrame):
    model = smf.ols(formula = formula, data = data, missing = "drop")
    return model.fit(cov_type = "HC1")


def run_event_study(df_es: pd.DataFrame, spec_with_fe: SpecWithFE, shock_cols: List[str]):
    df_es = _coerce_for_patsy(df_es.copy(), ["event_time", "time_id"])
    formula = "log_KL ~ C(event_time, Treatment(reference=-1)) + C(unit_id) + C(time_id)"
    if spec_with_fe.include_port_trends:
        formula += " + port_trend"
    if spec_with_fe.include_shocks and shock_cols:
        used_shocks = [c for c in shock_cols if c in df_es.columns]
        if used_shocks:
            formula += " + " + " + ".join(used_shocks)
    n_by_event_time = df_es.groupby("event_time")["unit_id"].size().to_dict()
    result = fit_clustered_ols(formula = formula, data = df_es)
    return result, n_by_event_time
def subset_for_static_horizon(df_es: pd.DataFrame, event_index: int, horizon_end: Optional[int]) -> pd.DataFrame:
    df_out = df_es.copy()
    if horizon_end is None:
        return df_out
    max_month = int(event_index + horizon_end)
    return df_out[df_out["month_index"] <= max_month].copy()


def run_static_did(df_es: pd.DataFrame, spec_with_fe: SpecWithFE, shock_cols: List[str], event_index: int, horizon_name: str, horizon_end: Optional[int]):
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
    formula = "log_KL ~ treated_post + C(unit_id) + C(time_id)"
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
    se_type = str(getattr(res, "cov_type", "cluster"))
    if not np.isfinite(se):
        res_hc1 = fit_hc1_ols(formula = formula, data = df_did)
        beta = float(res_hc1.params.get("treated_post", np.nan))
        se = float(res_hc1.bse.get("treated_post", np.nan))
        pval = float(res_hc1.pvalues.get("treated_post", np.nan))
        res = res_hc1
        se_type = "HC1_fallback"
    treated_post_ms = df_did.loc[(df_did["treated"]) & (df_did["month_index"] >= (event_index + 1)), "month_index"] - event_index
    if len(treated_post_ms) == 0:
        return None
    max_post_supported = int(treated_post_ms.max())
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
        m = _parse_event_time_from_param_name(name)
        if m is None:
            continue
        se = float(result.bse.get(name, np.nan))
        t_stat = float(beta / se) if (np.isfinite(se) and se != 0) else np.nan
        pval = float(result.pvalues.get(name, np.nan))
        rows.append({
            "design": design_name,
            "table_group": spec.table_group,
            "reform": spec.reform,
            "target": spec.target,
            "target_key": spec.target_key,
            "spec_name": spec_with_fe.spec_name,
            "event_time": int(m),
            "m": int(m),
            "j": int(m),
            "beta": float(beta),
            "se": se,
            "t": t_stat,
            "pvalue": pval,
            "n_event_obs": float(n_by_event_time.get(int(m), np.nan)),
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        })
    return pd.DataFrame(rows)


def compute_window_averages(result, spec_with_fe: SpecWithFE, design_name: str) -> pd.DataFrame:
    spec = spec_with_fe.spec
    params = result.params
    cov = result.cov_params()
    m_to_name: Dict[int, str] = {}
    for name in params.index:
        m = _parse_event_time_from_param_name(name)
        if m is not None:
            m_to_name[int(m)] = name
    if not m_to_name:
        return pd.DataFrame([])
    available_ms = sorted(m_to_name.keys())
    max_post_m = max((m for m in available_ms if m > 0), default = None)
    rows = []
    for wname, (a, b) in WINDOWS.items():
        b_eff = b
        if max_post_m is not None:
            if b == 999:
                b_eff = int(max_post_m)
            elif a >= 1:
                b_eff = int(min(b, max_post_m))
        ms = [m for m in available_ms if (m >= a and m <= b_eff)]
        if not ms:
            continue
        w = pd.Series(0.0, index = params.index)
        weight = 1.0 / len(ms)
        for m in ms:
            w[m_to_name[m]] = weight
        beta_w = float(np.dot(w.values, params.values))
        var_w = float(np.dot(w.values, np.dot(cov.values, w.values)))
        se_w = float(np.sqrt(var_w)) if var_w >= 0 else np.nan
        t_w = beta_w / se_w if np.isfinite(se_w) and se_w > 0 else np.nan
        p_w = np.nan
        if np.isfinite(t_w):
            z = abs(float(t_w))
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            p_w = 2.0 * (1.0 - cdf)
        rows.append({
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
            "pvalue": p_w,
            "n_obs": int(result.nobs),
            "r2": float(result.rsquared),
        })
    return pd.DataFrame(rows)


def compute_pretrend_f_test(result, spec_with_fe: SpecWithFE, design_name: str) -> pd.DataFrame:
    params = result.params
    names = list(params.index)
    k = len(names)
    event_param_info: List[Tuple[int, int]] = []
    for idx, name in enumerate(names):
        m = _parse_event_time_from_param_name(name)
        if m is not None and PRETREND_MIN <= m <= PRETREND_MAX:
            event_param_info.append((int(m), idx))
    if not event_param_info:
        return pd.DataFrame([])
    R_rows: List[np.ndarray] = []
    bins_used: List[Tuple[int, int]] = []
    for (a, b) in PRETREND_BINS:
        idxs = [idx for (m, idx) in event_param_info if a <= m <= b]
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
    return pd.DataFrame([{
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
    }])


def expand_specs_with_fe(base_specs: List[Spec], shock_cols: List[str]) -> List[SpecWithFE]:
    return [SpecWithFE(spec = s, spec_name = "baseline", include_port_trends = False, include_shocks = False) for s in base_specs]


def run_design(df: pd.DataFrame, ym_to_idx: Dict[Tuple[int, int], int], shock_cols: List[str], base_specs: List[Spec], design_name: str, clamp_windows: bool, suffix: str, run_static: bool) -> None:
    dynamic_by_spec: Dict[str, List[pd.DataFrame]] = {}
    window_by_spec: Dict[str, List[pd.DataFrame]] = {}
    pretrend_by_spec: Dict[str, List[pd.DataFrame]] = {}
    static_by_spec: Dict[str, List[pd.DataFrame]] = {}
    specs_with_fe = expand_specs_with_fe(base_specs, shock_cols)
    for spec_with_fe in specs_with_fe:
        spec = spec_with_fe.spec
        print(f"\n=== [{design_name}] table={spec.table_group}, reform={spec.reform}, target={spec.target}, spec={spec_with_fe.spec_name} ===")
        try:
            df_es = build_es_sample(df = df, spec = spec, ym_to_idx = ym_to_idx, clamp_windows = clamp_windows)
        except Exception as e:
            print(f"[WARN] Failed to build ES sample for {spec.reform} / {spec.target}: {e}")
            continue
        if df_es.empty:
            print("[WARN] Empty estimation sample; skipping.")
            continue
        n_treated = int(df_es["treated"].sum())
        n_control = int((~df_es["treated"]).sum())
        print(f"Sample size: {len(df_es)} rows ({n_treated} treated, {n_control} controls).")
        try:
            es_res, n_by_event_time = run_event_study(df_es, spec_with_fe, shock_cols)
        except Exception as e:
            print(f"[WARN] Event-study regression failed for {spec.reform} / {spec.target}: {e}")
            continue
        dynamic = extract_dynamic_betas(es_res, spec_with_fe, n_by_event_time, design_name)
        windows = compute_window_averages(es_res, spec_with_fe, design_name)
        pre = compute_pretrend_f_test(es_res, spec_with_fe, design_name)
        dynamic_by_spec.setdefault(spec_with_fe.spec_name, []).append(dynamic)
        window_by_spec.setdefault(spec_with_fe.spec_name, []).append(windows)
        if not pre.empty:
            pretrend_by_spec.setdefault(spec_with_fe.spec_name, []).append(pre)
        if run_static:
            event_index = year_month_to_index_clamped(ym_to_idx, (spec.event_year, spec.event_month))
            static_rows = []
            for hname, hend in STATIC_HORIZONS.items():
                out = run_static_did(df_es, spec_with_fe, shock_cols, event_index, hname, hend)
                if out is None:
                    continue
                static_rows.append({
                    "design": design_name,
                    "table_group": spec.table_group,
                    "reform": spec.reform,
                    "target": spec.target,
                    "target_key": spec.target_key,
                    "spec_name": spec_with_fe.spec_name,
                    **out,
                })
            if static_rows:
                static_by_spec.setdefault(spec_with_fe.spec_name, []).append(pd.DataFrame(static_rows))
    base_name = "model1b_kl"
    pooled = {"dynamic": [], "window": [], "pretrend": [], "static": []}
    for spec_name, frames in dynamic_by_spec.items():
        pdf = pd.concat(frames, ignore_index = True)
        pooled["dynamic"].append(pdf)
        path = OUTPUT_DIR / f"{base_name}_dynamic_betas_{spec_name}{suffix}.tsv"
        pdf.to_csv(path, sep = "\t", index = False)
        print(f"Saved dynamic betas ({design_name}, spec={spec_name}) to: {path}")
    for spec_name, frames in window_by_spec.items():
        pdf = pd.concat(frames, ignore_index = True)
        pooled["window"].append(pdf)
        path = OUTPUT_DIR / f"{base_name}_window_betas_{spec_name}{suffix}.tsv"
        pdf.to_csv(path, sep = "\t", index = False)
        print(f"Saved window betas ({design_name}, spec={spec_name}) to: {path}")
    for spec_name, frames in pretrend_by_spec.items():
        pdf = pd.concat(frames, ignore_index = True)
        pooled["pretrend"].append(pdf)
        path = OUTPUT_DIR / f"{base_name}_pretrend_tests_{spec_name}{suffix}.tsv"
        pdf.to_csv(path, sep = "\t", index = False)
        print(f"Saved pretrend tests ({design_name}, spec={spec_name}) to: {path}")
    if run_static:
        for spec_name, frames in static_by_spec.items():
            pdf = pd.concat(frames, ignore_index = True)
            pooled["static"].append(pdf)
            path = OUTPUT_DIR / f"{base_name}_static_betas_{spec_name}{suffix}.tsv"
            pdf.to_csv(path, sep = "\t", index = False)
            print(f"Saved static betas ({design_name}, spec={spec_name}) to: {path}")
    if pooled["dynamic"]:
        dyn_all = pd.concat(pooled["dynamic"], ignore_index = True)
        path = OUTPUT_DIR / f"{base_name}_dynamic_betas_all{suffix}.tsv"
        dyn_all.to_csv(path, sep = "\t", index = False)
        print(f"Saved pooled dynamic betas ({design_name}) to: {path}")
    if pooled["window"]:
        win_all = pd.concat(pooled["window"], ignore_index = True)
        path = OUTPUT_DIR / f"{base_name}_window_betas_all{suffix}.tsv"
        win_all.to_csv(path, sep = "\t", index = False)
        print(f"Saved pooled window betas ({design_name}) to: {path}")
    if pooled["pretrend"]:
        pre_all = pd.concat(pooled["pretrend"], ignore_index = True)
        path = OUTPUT_DIR / f"{base_name}_pretrend_tests_all{suffix}.tsv"
        pre_all.to_csv(path, sep = "\t", index = False)
        print(f"Saved pooled pretrend tests ({design_name}) to: {path}")
    if run_static and pooled["static"]:
        static_all = pd.concat(pooled["static"], ignore_index = True)
        path = OUTPUT_DIR / f"{base_name}_static_betas_all{suffix}.tsv"
        static_all.to_csv(path, sep = "\t", index = False)
        print(f"Saved pooled static betas ({design_name}) to: {path}")


def clear_model1b_outputs(output_dir: Path) -> None:
    patterns = [
        "model1b_kl_dynamic_betas_*.tsv",
        "model1b_kl_window_betas_*.tsv",
        "model1b_kl_pretrend_tests_*.tsv",
        "model1b_kl_static_betas_*.tsv",
    ]
    removed = 0
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok = True)
            removed += 1
    print(f"Cleared {removed} old Model 1B output files from: {output_dir}")


def main() -> None:
    clear_model1b_outputs(OUTPUT_DIR)
    df = load_kl_panel(KL_PANEL_PATH)
    ym_to_idx = build_year_month_to_index(df)
    shock_cols = get_shock_control_cols(df)
    print(f"Detected shock-control columns: {shock_cols}")
    if not ENABLE_AGGREGATE_SPECS:
        print("Aggregate specs disabled by configuration. Aggregate hooks remain scaffolded in the code.")
    print("\n==================== NYT run ====================")
    nyt_specs = build_nyt_specs(df)
    run_design(df = df, ym_to_idx = ym_to_idx, shock_cols = shock_cols, base_specs = nyt_specs, design_name = "NYT", clamp_windows = True, suffix = "", run_static = WRITE_STATIC_FOR_NYT)
    print("\n==================== TWFE run ====================")
    twfe_specs = build_twfe_specs(df)
    run_design(df = df, ym_to_idx = ym_to_idx, shock_cols = shock_cols, base_specs = twfe_specs, design_name = "TWFE", clamp_windows = True, suffix = "_twfe", run_static = WRITE_STATIC_FOR_TWFE)


if __name__ == "__main__":
    main()



# =============================================================================
# EVALUATION NOTE AFTER FIRST RUN OF Model_1B(v4)
#
# Summary:
# The first run of Model_1B(v4) did NOT succeed. This was not due to ordinary
# collinearity / small-sample issues. Instead, the run failed earlier at the
# sample-construction stage because the label-to-series mapping in LABEL_FILTERS
# does not match the actual structure of KL_Panel_monthly.tsv.
#
# What happened:
# 1. No estimation sample was successfully built for any Model 1B spec.
#    Therefore, no dynamic/window/pretrend/static outputs were produced.
#
# 2. The main issue is a schema mismatch between the code's assumed series
#    labels and the actual K/L panel series IDs.
#
# Specific failures:
# 3. "Haifa port" matched too many series IDs under the generic candidate
#    {"level": "port", "port": "Haifa"}.
#    The K/L panel appears to contain many Haifa port series, including
#    variants such as:
#       - HPC_central / high / low
#       - IPC_central / high / low
#       - SIPG_central / high / low
#       - cluster_central / high / low
#    Therefore the current "Haifa port" mapping is too broad.
#
# 4. "Haifa-Bayport" was not found at all.
#    This suggests that the K/L panel does not use "Haifa-Bayport" as a direct
#    terminal label, and likely uses another naming convention (for example,
#    SIPG-related series).
#
# 5. "Ashdod port" was not found at all under the current mapping.
#    This suggests that the Ashdod series naming convention in the K/L panel
#    also differs from what the code currently assumes.
#
# Interpretation:
# 6. This is a real code/data-integration issue, not a substantive regression
#    problem and not merely a consequence of temporary LP/KL measurement noise.
#
# 7. The estimator logic itself has not yet been tested, because the script
#    never advanced beyond series selection and sample construction.
#
# Practical next step:
# 8. Before Model_1B(v4) can be evaluated econometrically, LABEL_FILTERS must
#    be rewritten to match the actual KL_Panel_monthly.tsv series universe.
#    In practice this likely means:
#       - choosing one canonical scenario (probably "central")
#       - mapping aggregate objects to cluster_central-style series
#       - mapping Bayport/placebo to the actual SIPG-style series name
#       - mapping Ashdod port / Ashdod legacy to the actual Ashdod K/L names
#
# Bottom line:
# Model_1B(v4) is NOT yet functioning. The current failure is due to incorrect
# label mapping, so the next task is to fix LABEL_FILTERS using the real K/L
# panel schema before rerunning the estimator.
# =============================================================================

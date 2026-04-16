#!/usr/bin/env python
"""Model_1A_to_tables(v4).py

Post-processing / table-builder script for Model 1A (ln LP).

Purpose:
    Convert the v4 Model_1A outputs into table-ready helper files that match
    the revised thesis architecture:

    - Competition table:
        Panel A = TWFE static DiD rows
        Panel B = NYT event-study summary rows

    - Privatization table:
        Panel A = TWFE static DiD rows
        Panel B = NYT event-study summary rows

Inputs (from Model_1A(v4).py):
    Required:
      - model1a_lp_dynamic_betas_all.tsv
      - model1a_lp_window_betas_all.tsv
      - model1a_lp_pretrend_tests_all.tsv
      - model1a_lp_static_betas_all_twfe.tsv

    Optional:
      - model1a_lp_dynamic_betas_all_twfe.tsv
      - model1a_lp_window_betas_all_twfe.tsv
      - model1a_lp_pretrend_tests_all_twfe.tsv

Outputs:
    Main long-form helper:
      - Tables/model1a_lp_table_cells_v4.tsv

    Main table wide helpers:
      - Tables/model1a_competition_panelA_v4.tsv
      - Tables/model1a_competition_panelB_v4.tsv
      - Tables/model1a_privatization_panelA_v4.tsv
      - Tables/model1a_privatization_panelB_v4.tsv

    Appendix / diagnostics:
      - Tables/model1a_lp_dynamic_for_appendix_v4.tsv
      - Tables/model1a_lp_pretrend_for_appendix_v4.tsv

Notes:
    1. Panel A summary rows (Observations, Within R^2) are taken from the
       "full_post" static regression for each column. This is intentional:
       Panel A rows are separate static regressions, so n_obs / r2 are not
       identical across horizons. Using the full-post regression gives one
       canonical column summary.
    2. Panel B summary rows come from the single NYT event-study regression,
       so n_obs / r2 are naturally constant across windows for a given column.
    3. Aggregate columns are included only if they actually exist in the input
       outputs. This keeps the script compatible with aggregate hooks being
       disabled upstream.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]

MODEL1A_OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1A"
TABLES_OUTPUT_DIR = MODEL1A_OUTPUT_DIR / "Tables"
TABLES_OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

DYNAMIC_NYT_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_dynamic_betas_all.tsv"
WINDOWS_NYT_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_window_betas_all.tsv"
PRETREND_NYT_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_pretrend_tests_all.tsv"
STATIC_TWFE_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_static_betas_all_twfe.tsv"

DYNAMIC_TWFE_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_dynamic_betas_all_twfe.tsv"
WINDOWS_TWFE_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_window_betas_all_twfe.tsv"
PRETREND_TWFE_ALL_PATH = MODEL1A_OUTPUT_DIR / "model1a_lp_pretrend_tests_all_twfe.tsv"


# ---------------------------------------------------------------------
# Display / ordering metadata
# ---------------------------------------------------------------------

SPEC_DISPLAY_BY_PANEL = {
    "panelA": {
        "baseline": "TWFE",
        "porttr": "TWFE+Tr",
    },
    "panelB": {
        "baseline": "NYT",
        "porttr": "NYT+Tr",
    },
}

SPEC_ORDER = {
    "baseline": 1,
    "porttr": 2,
}

PREFERRED_TARGET_ORDER = {
    "competition": {
        "panelA": [
            "Haifa-Legacy",
            "Haifa aggregate",
            "Ashdod-Legacy",
            "Ashdod aggregate",
        ],
        "panelB": [
            "Haifa-Legacy",
            "Haifa aggregate",
        ],
    },
    "privatization": {
        "panelA": [
            "Haifa-Legacy",
            "Haifa-Bayport",
            "Haifa aggregate",
        ],
        "panelB": [
            "Haifa-Legacy",
            "Haifa-Bayport",
            "Haifa aggregate",
        ],
    },
}

PANEL_A_HORIZON_LABELS = {
    "full_post": "Static DiD: full post",
    "post_y1": "Static DiD: post year 1",
    "post_y1_2": "Static DiD: post years 1-2",
}

PANEL_A_ROW_ORDER = {
    "full_post": 1,
    "post_y1": 2,
    "post_y1_2": 3,
    "n_obs": 4,
    "r2": 5,
}

PANEL_B_WINDOW_LABELS = {
    "full_post": "Full post",
    "post_y1": "Post year 1",
    "avg_pre": "Average pre",
}

PANEL_B_ROW_ORDER = {
    "full_post": 1,
    "post_y1": 2,
    "avg_pre": 3,
    "pretrend_p": 4,
    "n_obs": 5,
    "r2": 6,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_tsv(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    df = pd.read_csv(path, sep = "\t")
    missing = set(required_cols).difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    return df


def read_tsv_optional(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_tsv(path, required_cols)


def stars_from_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p <= 0.01:
        return "***"
    if p <= 0.05:
        return "**"
    if p <= 0.10:
        return "*"
    return ""


def fmt_num(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_intlike(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return str(int(round(float(x))))


def fmt_estimate(beta: float, se: float, stars: str) -> str:
    if not np.isfinite(beta) and not np.isfinite(se):
        return ""
    if np.isfinite(beta) and np.isfinite(se):
        return f"{beta:.3f}{stars} ({se:.3f})"
    if np.isfinite(beta):
        return f"{beta:.3f}{stars}"
    return ""


def available_targets(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    present = set(df["target"].dropna().unique())
    return [t for t in preferred if t in present]


def column_order_map(
    table_group: str,
    panel: str,
    df: pd.DataFrame,
) -> Dict[str, int]:
    targets = available_targets(df, PREFERRED_TARGET_ORDER[table_group][panel])
    order: Dict[str, int] = {}

    k = 1
    for target in targets:
        for spec_name in ["baseline", "porttr"]:
            mask = (df["target"] == target) & (df["spec_name"] == spec_name)
            if mask.any():
                col_key = f"{target}__{spec_name}"
                order[col_key] = k
                k += 1

    return order


def make_column_label(target: str, spec_display: str) -> str:
    return f"{target} — {spec_display}"


def sort_table_cells(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["table_group", "panel", "row_order", "column_order"]
    existing = [c for c in sort_cols if c in df.columns]
    return df.sort_values(existing).reset_index(drop = True)


# ---------------------------------------------------------------------
# Build Panel A from TWFE static outputs
# ---------------------------------------------------------------------

def build_panel_a_cells(static_twfe: pd.DataFrame, table_group: str) -> pd.DataFrame:
    df = static_twfe.copy()
    df = df[df["table_group"] == table_group].copy()
    df = df[df["horizon"].isin(PANEL_A_HORIZON_LABELS.keys())].copy()

    if df.empty:
        return pd.DataFrame()

    col_order = column_order_map(table_group = table_group, panel = "panelA", df = df)
    rows: List[dict] = []

    for _, r in df.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue

        spec_display = SPEC_DISPLAY_BY_PANEL["panelA"].get(r["spec_name"], r["spec_name"])
        stars = stars_from_p(r["pvalue"])

        rows.append(
            {
                "table_group": table_group,
                "panel": "panelA",
                "row_key": r["horizon"],
                "row_label": PANEL_A_HORIZON_LABELS[r["horizon"]],
                "row_order": PANEL_A_ROW_ORDER[r["horizon"]],
                "row_type": "estimate",
                "target": r["target"],
                "target_key": r["target_key"],
                "spec_name": r["spec_name"],
                "spec_display": spec_display,
                "column_key": col_key,
                "column_label": make_column_label(r["target"], spec_display),
                "column_order": col_order[col_key],
                "design": r["design"],
                "reform": r["reform"],
                "beta": r["beta"],
                "se": r["se"],
                "pvalue": r["pvalue"],
                "stars": stars,
                "n_obs": r["n_obs"],
                "r2": r["r2"],
                "value_num": r["beta"],
                "value_display": fmt_estimate(r["beta"], r["se"], stars),
            }
        )

    # Summary rows: use full_post regression as the canonical column summary.
    full_post = df[df["horizon"] == "full_post"].copy()
    for _, r in full_post.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue

        spec_display = SPEC_DISPLAY_BY_PANEL["panelA"].get(r["spec_name"], r["spec_name"])

        rows.append(
            {
                "table_group": table_group,
                "panel": "panelA",
                "row_key": "n_obs",
                "row_label": "Observations",
                "row_order": PANEL_A_ROW_ORDER["n_obs"],
                "row_type": "summary",
                "target": r["target"],
                "target_key": r["target_key"],
                "spec_name": r["spec_name"],
                "spec_display": spec_display,
                "column_key": col_key,
                "column_label": make_column_label(r["target"], spec_display),
                "column_order": col_order[col_key],
                "design": r["design"],
                "reform": r["reform"],
                "beta": np.nan,
                "se": np.nan,
                "pvalue": np.nan,
                "stars": "",
                "n_obs": r["n_obs"],
                "r2": r["r2"],
                "value_num": r["n_obs"],
                "value_display": fmt_intlike(r["n_obs"]),
            }
        )

        rows.append(
            {
                "table_group": table_group,
                "panel": "panelA",
                "row_key": "r2",
                "row_label": "Within R^2",
                "row_order": PANEL_A_ROW_ORDER["r2"],
                "row_type": "summary",
                "target": r["target"],
                "target_key": r["target_key"],
                "spec_name": r["spec_name"],
                "spec_display": spec_display,
                "column_key": col_key,
                "column_label": make_column_label(r["target"], spec_display),
                "column_order": col_order[col_key],
                "design": r["design"],
                "reform": r["reform"],
                "beta": np.nan,
                "se": np.nan,
                "pvalue": np.nan,
                "stars": "",
                "n_obs": r["n_obs"],
                "r2": r["r2"],
                "value_num": r["r2"],
                "value_display": fmt_num(r["r2"], digits = 3),
            }
        )

    out = pd.DataFrame(rows)
    return sort_table_cells(out)


# ---------------------------------------------------------------------
# Build Panel B from NYT windows + pretrend
# ---------------------------------------------------------------------

def build_panel_b_cells(
    windows_nyt: pd.DataFrame,
    pretrend_nyt: pd.DataFrame,
    table_group: str,
) -> pd.DataFrame:
    win = windows_nyt.copy()
    pre = pretrend_nyt.copy()

    win = win[win["table_group"] == table_group].copy()
    pre = pre[pre["table_group"] == table_group].copy()

    win = win[win["window"].isin(PANEL_B_WINDOW_LABELS.keys())].copy()

    if win.empty:
        return pd.DataFrame()

    col_order = column_order_map(table_group = table_group, panel = "panelB", df = win)
    rows: List[dict] = []

    # Main NYT summary rows
    for _, r in win.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue

        spec_display = SPEC_DISPLAY_BY_PANEL["panelB"].get(r["spec_name"], r["spec_name"])
        t_stat = r["beta"] / r["se"] if np.isfinite(r["beta"]) and np.isfinite(r["se"]) and r["se"] != 0 else np.nan
        p_norm = np.nan
        if np.isfinite(t_stat):
            z = abs(float(t_stat))
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            p_norm = 2.0 * (1.0 - cdf)
        stars = stars_from_p(p_norm)

        rows.append(
            {
                "table_group": table_group,
                "panel": "panelB",
                "row_key": r["window"],
                "row_label": PANEL_B_WINDOW_LABELS[r["window"]],
                "row_order": PANEL_B_ROW_ORDER[r["window"]],
                "row_type": "estimate",
                "target": r["target"],
                "target_key": r["target_key"],
                "spec_name": r["spec_name"],
                "spec_display": spec_display,
                "column_key": col_key,
                "column_label": make_column_label(r["target"], spec_display),
                "column_order": col_order[col_key],
                "design": r["design"],
                "reform": r["reform"],
                "beta": r["beta"],
                "se": r["se"],
                "pvalue": p_norm,
                "stars": stars,
                "n_obs": r["n_obs"],
                "r2": r["r2"],
                "value_num": r["beta"],
                "value_display": fmt_estimate(r["beta"], r["se"], stars),
            }
        )

    # Pretrend p-value row
    for _, r in pre.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue

        spec_display = SPEC_DISPLAY_BY_PANEL["panelB"].get(r["spec_name"], r["spec_name"])

        rows.append(
            {
                "table_group": table_group,
                "panel": "panelB",
                "row_key": "pretrend_p",
                "row_label": "Pre-trends F-test p-value",
                "row_order": PANEL_B_ROW_ORDER["pretrend_p"],
                "row_type": "summary",
                "target": r["target"],
                "target_key": r["target_key"],
                "spec_name": r["spec_name"],
                "spec_display": spec_display,
                "column_key": col_key,
                "column_label": make_column_label(r["target"], spec_display),
                "column_order": col_order[col_key],
                "design": r["design"],
                "reform": r["reform"],
                "beta": np.nan,
                "se": np.nan,
                "pvalue": r["pvalue"],
                "stars": "",
                "n_obs": r["n_obs"],
                "r2": r["r2"],
                "value_num": r["pvalue"],
                "value_display": fmt_num(r["pvalue"], digits = 3),
            }
        )

    # Observations / R^2 summary rows from the NYT event-study regression
    # Use the full_post row as the canonical summary row when available.
    full_post = win[win["window"] == "full_post"].copy()
    if full_post.empty:
        full_post = (
            win.sort_values(["target", "spec_name", "window"])
               .groupby(["target", "spec_name"], as_index = False)
               .first()
        )

    for _, r in full_post.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue

        spec_display = SPEC_DISPLAY_BY_PANEL["panelB"].get(r["spec_name"], r["spec_name"])

        rows.append(
            {
                "table_group": table_group,
                "panel": "panelB",
                "row_key": "n_obs",
                "row_label": "Observations",
                "row_order": PANEL_B_ROW_ORDER["n_obs"],
                "row_type": "summary",
                "target": r["target"],
                "target_key": r["target_key"],
                "spec_name": r["spec_name"],
                "spec_display": spec_display,
                "column_key": col_key,
                "column_label": make_column_label(r["target"], spec_display),
                "column_order": col_order[col_key],
                "design": r["design"],
                "reform": r["reform"],
                "beta": np.nan,
                "se": np.nan,
                "pvalue": np.nan,
                "stars": "",
                "n_obs": r["n_obs"],
                "r2": r["r2"],
                "value_num": r["n_obs"],
                "value_display": fmt_intlike(r["n_obs"]),
            }
        )

        rows.append(
            {
                "table_group": table_group,
                "panel": "panelB",
                "row_key": "r2",
                "row_label": "Within R^2",
                "row_order": PANEL_B_ROW_ORDER["r2"],
                "row_type": "summary",
                "target": r["target"],
                "target_key": r["target_key"],
                "spec_name": r["spec_name"],
                "spec_display": spec_display,
                "column_key": col_key,
                "column_label": make_column_label(r["target"], spec_display),
                "column_order": col_order[col_key],
                "design": r["design"],
                "reform": r["reform"],
                "beta": np.nan,
                "se": np.nan,
                "pvalue": np.nan,
                "stars": "",
                "n_obs": r["n_obs"],
                "r2": r["r2"],
                "value_num": r["r2"],
                "value_display": fmt_num(r["r2"], digits = 3),
            }
        )

    out = pd.DataFrame(rows)
    return sort_table_cells(out)


# ---------------------------------------------------------------------
# Wide-file writer
# ---------------------------------------------------------------------

def write_wide_panel(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(out_path, sep = "\t", index = False)
        print(f"Saved empty wide panel helper to: {out_path}")
        return

    wide = (
        df.pivot(index = "row_label", columns = "column_label", values = "value_display")
          .reset_index()
    )

    row_order_map = (
        df[["row_label", "row_order"]]
        .drop_duplicates()
        .set_index("row_label")["row_order"]
        .to_dict()
    )

    wide["__row_order"] = wide["row_label"].map(row_order_map)
    wide = wide.sort_values("__row_order").drop(columns = "__row_order")

    out_path.parent.mkdir(parents = True, exist_ok = True)
    wide.to_csv(out_path, sep = "\t", index = False)
    print(f"Saved wide panel helper to: {out_path}")


# ---------------------------------------------------------------------
# Appendix helpers
# ---------------------------------------------------------------------

def build_dynamic_for_appendix() -> pd.DataFrame:
    required = {
        "design",
        "table_group",
        "reform",
        "target",
        "target_key",
        "spec_name",
        "event_time",
        "beta",
        "se",
        "pvalue",
        "n_event_obs",
        "n_obs",
        "r2",
    }

    frames: List[pd.DataFrame] = []

    dyn_nyt = read_tsv(DYNAMIC_NYT_ALL_PATH, required)
    frames.append(dyn_nyt)

    dyn_twfe = read_tsv_optional(DYNAMIC_TWFE_ALL_PATH, required)
    if not dyn_twfe.empty:
        frames.append(dyn_twfe)

    dyn = pd.concat(frames, ignore_index = True)

    if "tvalue" not in dyn.columns and "t" in dyn.columns:
        dyn = dyn.rename(columns = {"t": "tvalue"})

    dyn["stars"] = dyn["pvalue"].apply(stars_from_p)
    dyn["N_event"] = dyn["n_event_obs"]
    dyn["spec_display"] = np.where(
        dyn["design"].eq("TWFE"),
        dyn["spec_name"].map(SPEC_DISPLAY_BY_PANEL["panelA"]).fillna(dyn["spec_name"]),
        dyn["spec_name"].map(SPEC_DISPLAY_BY_PANEL["panelB"]).fillna(dyn["spec_name"]),
    )

    sort_cols = ["design", "table_group", "reform", "target", "spec_name", "event_time"]
    dyn = dyn.sort_values(sort_cols).reset_index(drop = True)

    out_path = TABLES_OUTPUT_DIR / "model1a_lp_dynamic_for_appendix_v4.tsv"
    dyn.to_csv(out_path, sep = "\t", index = False)
    print(f"Saved dynamic appendix helper to: {out_path}")

    return dyn


def build_pretrend_for_appendix() -> pd.DataFrame:
    required = {
        "design",
        "table_group",
        "reform",
        "target",
        "target_key",
        "spec_name",
        "pre_min",
        "pre_max",
        "n_leads_total",
        "n_bins_defined",
        "n_bins_used",
        "f_stat",
        "pvalue",
        "df_num",
        "df_denom",
        "n_obs",
        "r2",
    }

    frames: List[pd.DataFrame] = []

    pre_nyt = read_tsv(PRETREND_NYT_ALL_PATH, required)
    frames.append(pre_nyt)

    pre_twfe = read_tsv_optional(PRETREND_TWFE_ALL_PATH, required)
    if not pre_twfe.empty:
        frames.append(pre_twfe)

    pre = pd.concat(frames, ignore_index = True)

    pre["spec_display"] = np.where(
        pre["design"].eq("TWFE"),
        pre["spec_name"].map(SPEC_DISPLAY_BY_PANEL["panelA"]).fillna(pre["spec_name"]),
        pre["spec_name"].map(SPEC_DISPLAY_BY_PANEL["panelB"]).fillna(pre["spec_name"]),
    )

    sort_cols = ["design", "table_group", "reform", "target", "spec_name"]
    pre = pre.sort_values(sort_cols).reset_index(drop = True)

    out_path = TABLES_OUTPUT_DIR / "model1a_lp_pretrend_for_appendix_v4.tsv"
    pre.to_csv(out_path, sep = "\t", index = False)
    print(f"Saved pretrend appendix helper to: {out_path}")

    return pre


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Model_1A_to_tables(v4): starting ===")
    print(f"THESIS_ROOT: {THESIS_ROOT}")
    print(f"Model 1A output dir: {MODEL1A_OUTPUT_DIR}")
    print(f"Tables output dir: {TABLES_OUTPUT_DIR}")

    static_twfe = read_tsv(
        STATIC_TWFE_ALL_PATH,
        required_cols = {
            "design",
            "table_group",
            "reform",
            "target",
            "target_key",
            "spec_name",
            "horizon",
            "a",
            "b",
            "beta",
            "se",
            "pvalue",
            "n_obs",
            "r2",
        },
    )

    windows_nyt = read_tsv(
        WINDOWS_NYT_ALL_PATH,
        required_cols = {
            "design",
            "table_group",
            "reform",
            "target",
            "target_key",
            "spec_name",
            "window",
            "a",
            "b",
            "beta",
            "se",
            "n_obs",
            "r2",
        },
    )

    pretrend_nyt = read_tsv(
        PRETREND_NYT_ALL_PATH,
        required_cols = {
            "design",
            "table_group",
            "reform",
            "target",
            "target_key",
            "spec_name",
            "pre_min",
            "pre_max",
            "n_leads_total",
            "n_bins_defined",
            "n_bins_used",
            "f_stat",
            "pvalue",
            "df_num",
            "df_denom",
            "n_obs",
            "r2",
        },
    )

    comp_a = build_panel_a_cells(static_twfe = static_twfe, table_group = "competition")
    comp_b = build_panel_b_cells(
        windows_nyt = windows_nyt,
        pretrend_nyt = pretrend_nyt,
        table_group = "competition",
    )

    priv_a = build_panel_a_cells(static_twfe = static_twfe, table_group = "privatization")
    priv_b = build_panel_b_cells(
        windows_nyt = windows_nyt,
        pretrend_nyt = pretrend_nyt,
        table_group = "privatization",
    )

    table_cells = pd.concat([comp_a, comp_b, priv_a, priv_b], ignore_index = True)
    table_cells = sort_table_cells(table_cells)

    long_out = TABLES_OUTPUT_DIR / "model1a_lp_table_cells_v4.tsv"
    table_cells.to_csv(long_out, sep = "\t", index = False)
    print(f"Saved main long-form table helper to: {long_out}")

    write_wide_panel(comp_a, TABLES_OUTPUT_DIR / "model1a_competition_panelA_v4.tsv")
    write_wide_panel(comp_b, TABLES_OUTPUT_DIR / "model1a_competition_panelB_v4.tsv")
    write_wide_panel(priv_a, TABLES_OUTPUT_DIR / "model1a_privatization_panelA_v4.tsv")
    write_wide_panel(priv_b, TABLES_OUTPUT_DIR / "model1a_privatization_panelB_v4.tsv")

    dyn = build_dynamic_for_appendix()
    pre = build_pretrend_for_appendix()

    print("\nSummary:")
    print(f"  Main table cells: {len(table_cells)}")
    print(f"  Competition Panel A cells: {len(comp_a)}")
    print(f"  Competition Panel B cells: {len(comp_b)}")
    print(f"  Privatization Panel A cells: {len(priv_a)}")
    print(f"  Privatization Panel B cells: {len(priv_b)}")
    print(f"  Dynamic appendix rows: {len(dyn)}")
    print(f"  Pretrend appendix rows: {len(pre)}")
    print(f"Table helper files written to: {TABLES_OUTPUT_DIR}")
    print("=== Model_1A_to_tables(v4): done ===")


if __name__ == "__main__":
    main()


# =============================================================================
# EVALUATION NOTE AFTER FIRST RUN OF Model_1A_to_tables(v4)
#
# Summary:
# The v4 table-builder ran successfully and produced the intended helper files
# for the revised Model 1A architecture. The long-form table-cells output is
# behaving as intended. The main remaining issue is only a presentation-order
# problem in the wide helper files, not a regression or mapping bug.
#
# What worked:
# 1. The script ran cleanly and wrote all expected v4 outputs:
#    - model1a_lp_table_cells_v4.tsv
#    - model1a_competition_panelA_v4.tsv
#    - model1a_competition_panelB_v4.tsv
#    - model1a_privatization_panelA_v4.tsv
#    - model1a_privatization_panelB_v4.tsv
#    - model1a_lp_dynamic_for_appendix_v4.tsv
#    - model1a_lp_pretrend_for_appendix_v4.tsv
#
# 2. Output counts are internally consistent with the current upstream Model 1A
#    estimation outputs:
#    - Competition Panel A: 20 cells = 5 rows x 4 columns
#    - Competition Panel B: 12 cells = 6 rows x 2 columns
#    - Privatization Panel A: 20 cells = 5 rows x 4 columns
#    - Privatization Panel B: 24 cells = 6 rows x 4 columns
#
# 3. These counts are exactly what should occur when aggregate specs remain
#    disabled upstream. Therefore the absence of aggregate columns in the
#    current table outputs is expected and is not a bug in this script.
#
# 4. The long-form output file (model1a_lp_table_cells_v4.tsv) is the canonical
#    and most reliable table object. It correctly preserves panel identity,
#    row identity, target identity, spec identity, and display strings.
#
# 5. The appendix helpers also look structurally correct:
#    - dynamic appendix file combines NYT and optional TWFE dynamic rows
#    - pretrend appendix file combines NYT and optional TWFE pretrend rows
#
# Non-bug results that are expected under current interim LP data:
# 6. Competition Panel B only includes Haifa-Legacy NYT columns, with no
#    aggregate NYT columns. This is expected because aggregate specs are
#    currently disabled upstream in Model_1A(v4).
#
# 7. Privatization NYT estimates for Haifa-Legacy and Haifa-Bayport remain
#    numerically identical. This was previously debugged and reflects the
#    current interim LP data structure, not a table-builder bug.
#
# 8. NaN pretrend p-values for Haifa competition are also expected given the
#    saturated NYT competition design in the current interim LP sample.
#
# Remaining issue:
# 9. The wide helper files produced via pivot() do not preserve the preferred
#    manuscript column order. Instead, columns are alphabetically ordered in
#    the wide TSVs (e.g. Ashdod may appear before Haifa). This is a small
#    presentation-order issue in the table-builder and does not affect the
#    correctness of the long-form table-cells file.
#
# Practical implication:
# 10. The current outputs are sufficient for downstream table construction, but
#     the exact LaTeX tables that include aggregate columns cannot yet be fully
#     populated from these files unless aggregate specs are first enabled and
#     rerun upstream, or unless a temporary no-aggregate table version is used.
#
# Bottom line:
# Model_1A_to_tables(v4) is functioning correctly for the current v4 pipeline.
# No major code bug is evident. The only real remaining issue is column-order
# preservation in the wide panel helpers; otherwise the script is ready to use.
# =============================================================================
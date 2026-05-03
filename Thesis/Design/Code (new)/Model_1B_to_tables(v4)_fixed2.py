from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

# =============================================================================
# Model_1B_to_tables(v4).py
#
# Purpose:
#   Merge baseline Model_1B(v4) outputs with the relaxed+trend alternative
#   from Model_1B_relaxed(v4), then build table-ready helper files matching
#   the revised thesis architecture.
#
# Main tables built here:
#   - competition Panel A / Panel B
#   - privatization Panel A / Panel B
#
# Appendix helpers built here:
#   - combined dynamic rows (baseline + relaxed, NYT + TWFE)
#   - combined pretrend rows (baseline + relaxed, NYT + TWFE)
# =============================================================================

THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]

BASE_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1B"
RELAXED_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1B_relaxed"
TABLES_OUTPUT_DIR = BASE_DIR / "Tables"
TABLES_OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

# Lean-output options. The long-form table-cells file is canonical; the wide-panel and
# appendix helper files can be suppressed to reduce file clutter.
WRITE_WIDE_PANEL_HELPERS = False
WRITE_APPENDIX_HELPERS = False

SPEC_DISPLAY_BY_PANEL = {
    "panelA": {
        "baseline": "Baseline",
        "relaxed_tr": "Relaxed+Tr",
    },
    "panelB": {
        "baseline": "Baseline",
        "relaxed_tr": "Relaxed+Tr",
    },
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
PANEL_A_ROW_ORDER = {"full_post": 1, "post_y1": 2, "post_y1_2": 3, "n_obs": 4, "r2": 5}

PANEL_B_WINDOW_LABELS = {
    "full_post": "Full post",
    "post_y1": "Post year 1",
    "avg_pre": "Average pre",
}
PANEL_B_ROW_ORDER = {"full_post": 1, "post_y1": 2, "avg_pre": 3, "pretrend_p": 4, "n_obs": 5, "r2": 6}


BASE_REQUIRED_STATIC = {
    "design", "table_group", "reform", "target", "target_key", "spec_name",
    "horizon", "a", "b", "beta", "se", "pvalue", "n_obs", "r2",
}
BASE_REQUIRED_WINDOWS = {
    "design", "table_group", "reform", "target", "target_key", "spec_name",
    "window", "a", "b", "beta", "se", "pvalue", "n_obs", "r2",
}
BASE_REQUIRED_PRE = {
    "design", "table_group", "reform", "target", "target_key", "spec_name",
    "pre_min", "pre_max", "n_leads_total", "n_bins_defined", "n_bins_used",
    "f_stat", "pvalue", "df_num", "df_denom", "n_obs", "r2",
}
BASE_REQUIRED_DYNAMIC = {
    "design", "table_group", "reform", "target", "target_key", "spec_name",
    "event_time", "beta", "se", "pvalue", "n_event_obs", "n_obs", "r2",
}


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
        return pd.DataFrame(columns = list(required_cols))
    return read_tsv(path, required_cols)


def empty_like(required_cols: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns = list(required_cols))


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


def column_order_map(table_group: str, panel: str, df: pd.DataFrame):
    targets = available_targets(df, PREFERRED_TARGET_ORDER[table_group][panel])
    order = {}
    k = 1
    for target in targets:
        for spec_name in ["baseline", "relaxed_tr"]:
            mask = (df["target"] == target) & (df["spec_name"] == spec_name)
            if mask.any():
                order[f"{target}__{spec_name}"] = k
                k += 1
    return order


def make_column_label(target: str, spec_display: str) -> str:
    return f"{target} — {spec_display}"


def sort_table_cells(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["table_group", "panel", "row_order", "column_order"] if c in df.columns]
    return df.sort_values(cols).reset_index(drop = True)


def build_panel_a_cells(static_all: pd.DataFrame, table_group: str) -> pd.DataFrame:
    df = static_all.copy()
    df = df[(df["table_group"] == table_group) & (df["horizon"].isin(PANEL_A_HORIZON_LABELS.keys()))].copy()
    if df.empty:
        return pd.DataFrame()

    col_order = column_order_map(table_group, "panelA", df)
    rows = []

    for _, r in df.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue
        spec_display = SPEC_DISPLAY_BY_PANEL["panelA"].get(r["spec_name"], r["spec_name"])
        stars = stars_from_p(r["pvalue"])
        rows.append({
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
        })

    full_post = df[df["horizon"] == "full_post"].copy()
    for _, r in full_post.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue
        spec_display = SPEC_DISPLAY_BY_PANEL["panelA"].get(r["spec_name"], r["spec_name"])
        rows.append({
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
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "value_num": r["n_obs"],
            "value_display": fmt_intlike(r["n_obs"]),
        })
        rows.append({
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
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "value_num": r["r2"],
            "value_display": fmt_num(r["r2"], 3),
        })

    return sort_table_cells(pd.DataFrame(rows))
def build_panel_b_cells(windows_all: pd.DataFrame, pretrend_all: pd.DataFrame, table_group: str) -> pd.DataFrame:
    win = windows_all.copy()
    pre = pretrend_all.copy()
    win = win[(win["table_group"] == table_group) & (win["window"].isin(PANEL_B_WINDOW_LABELS.keys()))].copy()
    pre = pre[pre["table_group"] == table_group].copy()
    if win.empty:
        return pd.DataFrame()

    col_order = column_order_map(table_group, "panelB", win)
    rows = []

    for _, r in win.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue
        spec_display = SPEC_DISPLAY_BY_PANEL["panelB"].get(r["spec_name"], r["spec_name"])
        stars = stars_from_p(r["pvalue"])
        rows.append({
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
            "pvalue": r["pvalue"],
            "stars": stars,
            "n_obs": r["n_obs"],
            "r2": r["r2"],
            "value_num": r["beta"],
            "value_display": fmt_estimate(r["beta"], r["se"], stars),
        })

    for _, r in pre.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue
        spec_display = SPEC_DISPLAY_BY_PANEL["panelB"].get(r["spec_name"], r["spec_name"])
        rows.append({
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
            "value_num": r["pvalue"],
            "value_display": fmt_num(r["pvalue"], 3),
            "n_obs": r["n_obs"],
            "r2": r["r2"],
        })

    full_post = win[win["window"] == "full_post"].copy()
    if full_post.empty:
        full_post = win.sort_values(["target", "spec_name", "window"]).groupby(["target", "spec_name"], as_index = False).first()

    for _, r in full_post.iterrows():
        col_key = f"{r['target']}__{r['spec_name']}"
        if col_key not in col_order:
            continue
        spec_display = SPEC_DISPLAY_BY_PANEL["panelB"].get(r["spec_name"], r["spec_name"])
        rows.append({
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
            "value_num": r["n_obs"],
            "value_display": fmt_intlike(r["n_obs"]),
            "n_obs": r["n_obs"],
            "r2": r["r2"],
        })
        rows.append({
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
            "value_num": r["r2"],
            "value_display": fmt_num(r["r2"], 3),
            "n_obs": r["n_obs"],
            "r2": r["r2"],
        })

    return sort_table_cells(pd.DataFrame(rows))


def write_wide_panel(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(out_path, sep = "\t", index = False)
        print(f"Saved empty wide panel helper to: {out_path}")
        return
    wide = df.pivot(index = "row_label", columns = "column_label", values = "value_display").reset_index()
    row_order_map = df[["row_label", "row_order"]].drop_duplicates().set_index("row_label")["row_order"].to_dict()
    wide["__row_order"] = wide["row_label"].map(row_order_map)
    wide = wide.sort_values("__row_order").drop(columns = "__row_order")
    wide.to_csv(out_path, sep = "\t", index = False)
    print(f"Saved wide panel helper to: {out_path}")


def load_baseline_and_relaxed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static_base = read_tsv_optional(BASE_DIR / "model1b_kl_static_betas_all_twfe.tsv", BASE_REQUIRED_STATIC)
    win_base = read_tsv_optional(BASE_DIR / "model1b_kl_window_betas_all.tsv", BASE_REQUIRED_WINDOWS)
    pre_base = read_tsv_optional(BASE_DIR / "model1b_kl_pretrend_tests_all.tsv", BASE_REQUIRED_PRE)

    static_rel = read_tsv_optional(RELAXED_DIR / "model1b_kl_static_betas_all_relaxed_twfe.tsv", BASE_REQUIRED_STATIC)
    win_rel = read_tsv_optional(RELAXED_DIR / "model1b_kl_window_betas_all_relaxed.tsv", BASE_REQUIRED_WINDOWS)
    pre_rel = read_tsv_optional(RELAXED_DIR / "model1b_kl_pretrend_tests_all_relaxed.tsv", BASE_REQUIRED_PRE)

    static_frames = [df for df in [static_base, static_rel] if not df.empty]
    win_frames = [df for df in [win_base, win_rel] if not df.empty]
    pre_frames = [df for df in [pre_base, pre_rel] if not df.empty]

    static_all = pd.concat(static_frames, ignore_index = True) if static_frames else empty_like(BASE_REQUIRED_STATIC)
    win_all = pd.concat(win_frames, ignore_index = True) if win_frames else empty_like(BASE_REQUIRED_WINDOWS)
    pre_all = pd.concat(pre_frames, ignore_index = True) if pre_frames else empty_like(BASE_REQUIRED_PRE)
    return static_all, win_all, pre_all, static_base, win_base, pre_base


def build_dynamic_for_appendix() -> pd.DataFrame:
    frames = []
    for path in [
        BASE_DIR / "model1b_kl_dynamic_betas_all.tsv",
        BASE_DIR / "model1b_kl_dynamic_betas_all_twfe.tsv",
        RELAXED_DIR / "model1b_kl_dynamic_betas_all_relaxed.tsv",
        RELAXED_DIR / "model1b_kl_dynamic_betas_all_relaxed_twfe.tsv",
    ]:
        df = read_tsv_optional(path, BASE_REQUIRED_DYNAMIC)
        if not df.empty:
            frames.append(df)
    dyn = pd.concat(frames, ignore_index = True) if frames else pd.DataFrame()
    if not dyn.empty:
        dyn["spec_display"] = dyn["spec_name"].map({"baseline": "Baseline", "relaxed_tr": "Relaxed+Tr"}).fillna(dyn["spec_name"])
        dyn = dyn.sort_values(["design", "table_group", "reform", "target", "spec_name", "event_time"]).reset_index(drop = True)
    out_path = TABLES_OUTPUT_DIR / "model1b_kl_dynamic_for_appendix_v4.tsv"
    dyn.to_csv(out_path, sep = "\t", index = False)
    print(f"Saved dynamic appendix helper to: {out_path}")
    return dyn


def build_pretrend_for_appendix() -> pd.DataFrame:
    frames = []
    for path in [
        BASE_DIR / "model1b_kl_pretrend_tests_all.tsv",
        BASE_DIR / "model1b_kl_pretrend_tests_all_twfe.tsv",
        RELAXED_DIR / "model1b_kl_pretrend_tests_all_relaxed.tsv",
        RELAXED_DIR / "model1b_kl_pretrend_tests_all_relaxed_twfe.tsv",
    ]:
        df = read_tsv_optional(path, BASE_REQUIRED_PRE)
        if not df.empty:
            frames.append(df)
    pre = pd.concat(frames, ignore_index = True) if frames else pd.DataFrame()
    if not pre.empty:
        pre["spec_display"] = pre["spec_name"].map({"baseline": "Baseline", "relaxed_tr": "Relaxed+Tr"}).fillna(pre["spec_name"])
        pre = pre.sort_values(["design", "table_group", "reform", "target", "spec_name"]).reset_index(drop = True)
    out_path = TABLES_OUTPUT_DIR / "model1b_kl_pretrend_for_appendix_v4.tsv"
    pre.to_csv(out_path, sep = "\t", index = False)
    print(f"Saved pretrend appendix helper to: {out_path}")
    return pre


def main() -> None:
    print("=== Model_1B_to_tables(v4): starting ===")
    print(f"THESIS_ROOT: {THESIS_ROOT}")
    print(f"Baseline output dir: {BASE_DIR}")
    print(f"Relaxed output dir: {RELAXED_DIR}")
    print(f"Tables output dir: {TABLES_OUTPUT_DIR}")

    static_all, win_all, pre_all, _, _, _ = load_baseline_and_relaxed()

    comp_a = build_panel_a_cells(static_all, "competition")
    comp_b = build_panel_b_cells(win_all, pre_all, "competition")
    priv_a = build_panel_a_cells(static_all, "privatization")
    priv_b = build_panel_b_cells(win_all, pre_all, "privatization")

    table_cells = pd.concat([comp_a, comp_b, priv_a, priv_b], ignore_index = True)
    table_cells = sort_table_cells(table_cells)
    long_out = TABLES_OUTPUT_DIR / "model1b_kl_table_cells_v4.tsv"
    table_cells.to_csv(long_out, sep = "\t", index = False)
    print(f"Saved main long-form table helper to: {long_out}")

    if WRITE_WIDE_PANEL_HELPERS:
        write_wide_panel(comp_a, TABLES_OUTPUT_DIR / "model1b_competition_panelA_v4.tsv")
        write_wide_panel(comp_b, TABLES_OUTPUT_DIR / "model1b_competition_panelB_v4.tsv")
        write_wide_panel(priv_a, TABLES_OUTPUT_DIR / "model1b_privatization_panelA_v4.tsv")
        write_wide_panel(priv_b, TABLES_OUTPUT_DIR / "model1b_privatization_panelB_v4.tsv")

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
    print("=== Model_1B_to_tables(v4): done ===")


if __name__ == "__main__":
    main()

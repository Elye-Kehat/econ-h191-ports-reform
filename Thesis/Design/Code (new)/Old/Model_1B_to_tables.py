from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import Model_1B as m1b


def find_thesis_root() -> Path:
    """
    Mirror the logic in Model_1B.py: this file lives in THESIS/Design/Code (new)/.
    Thesis root is two parents up from this file.
    """
    here = Path(__file__).resolve()
    return here.parents[2]


def build_ym_mapping(df_kl: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    df_kl = df_kl.copy()
    df_kl["ym_tuple"] = list(zip(df_kl["year"], df_kl["month"]))
    ym_sorted = sorted(df_kl["ym_tuple"].unique())
    ym_to_idx: Dict[Tuple[int, int], int] = {ym: i + 1 for i, ym in enumerate(ym_sorted)}
    df_kl["month_index"] = df_kl["ym_tuple"].map(ym_to_idx)
    return ym_to_idx


def build_Nm_counts(thesis_root: Path) -> pd.DataFrame:
    """
    Rebuild N(m) counts for Model 1B using the same ES-sample construction as in Model_1B.py.

    For each base spec (haifa_comp, haifa_priv) we:
      * reconstruct the ES sample with m1b.build_es_sample_kl
      * count the number of terminal×month observations at each event-time m
      * then merge those counts back to dynamic betas by (reform, target, event_time)
    """
    kl_panel_path = thesis_root / "Data" / "KL" / "KL_Panel_monthly.tsv"
    df_kl = pd.read_csv(kl_panel_path, sep="\t")

    # Ensure month_index is present (same convention as in Model_1B.py)
    ym_to_idx = build_ym_mapping(df_kl)

    Nm_rows: List[dict] = []

    base_specs = m1b.build_base_specs_kl()
    for base in base_specs:
        df_es = m1b.build_es_sample_kl(df_kl, base, ym_to_idx)
        if df_es is None:
            continue

        # For N(m), we care about event_time (not event_time_treat)
        for m in range(m1b.MIN_EVENT_TIME, m1b.MAX_EVENT_TIME + 1):
            if m == -1:
                continue
            count_m = int((df_es["event_time"] == m).sum())
            Nm_rows.append(
                dict(
                    reform=base.reform,
                    target=base.target,
                    event_time=m,
                    N_m=count_m,
                )
            )

    if not Nm_rows:
        return pd.DataFrame(columns=["reform", "target", "event_time", "N_m"])

    Nm = (
        pd.DataFrame(Nm_rows)
        .drop_duplicates(subset=["reform", "target", "event_time"])
        .reset_index(drop=True)
    )
    return Nm


def build_windows_for_tables(win_all: pd.DataFrame) -> pd.DataFrame:
    """
    Slim down the window-level results for easy LaTeX table use.

    We keep:
      * reform (clock)
      * target (treated unit)
      * fe_type (spec variant)
      * window_name in {post_all, post_y1, post_y2, pre_all}
      * beta_hat, se, pvalue, n_obs
    """
    keep_windows = ["post_all", "post_y1", "post_y2", "pre_all"]
    df = win_all[win_all["window_name"].isin(keep_windows)].copy()

    col_order = [
        "model",
        "reform",
        "target",
        "fe_type",
        "window_name",
        "m_start",
        "m_end",
        "beta_hat",
        "se",
        "tvalue",
        "pvalue",
        "n_obs",
        "treated_n",
        "control_n",
        "cov_type",
        "cluster_by",
        "r2",
    ]
    df = df[col_order]

    return df


def build_dynamic_for_appendix(dyn_all: pd.DataFrame, Nm: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the long event-time table for the appendix.

    We join in N(m) counts so that each row (m, reform, target, fe_type) has
    the corresponding number of obs supporting that event-month.
    """
    df = dyn_all.copy()

    # Merge N(m) by (reform, target, event_time)
    df = df.merge(
        Nm,
        how="left",
        left_on=["reform", "target", "event_time"],
        right_on=["reform", "target", "event_time"],
    )

    # Sort for nice printing
    df = df.sort_values(
        by=["reform", "target", "fe_type", "event_time"]
    ).reset_index(drop=True)

    col_order = [
        "model",
        "reform",
        "target",
        "fe_type",
        "event_time",
        "beta_hat",
        "se",
        "tvalue",
        "pvalue",
        "N_m",
        "n_obs",
        "treated_n",
        "control_n",
        "cov_type",
        "cluster_by",
        "r2",
    ]
    df = df[col_order]

    return df


def build_pretrend_for_tables(pre_all: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a compact pretrend helper file.

    One row per (reform, target, fe_type), with:
      * F_stat, p_value, df_num, df_denom
    """
    df = pre_all.copy()

    # Keep only the standard test
    df = df[df["test_name"] == "pretrend_all_m_le_-2"].copy()

    df["F_stat"] = df["stat"]
    df["p_value"] = df["pvalue"]

    col_order = [
        "model",
        "reform",
        "target",
        "fe_type",
        "test_name",
        "F_stat",
        "p_value",
        "df_num",
        "df_denom",
        "cov_type",
        "cluster_by",
        "r2",
    ]
    df = df[col_order]

    return df


def main() -> None:
    print("=== Model_1B_to_tables: starting ===")

    thesis_root = find_thesis_root()
    print(f"THESIS_ROOT: {thesis_root}")

    out_dir_m1b = thesis_root / "Design" / "Output (new)" / "Model_1B"
    tables_dir = out_dir_m1b / "Tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model 1B output dir: {out_dir_m1b}")
    print(f"Tables output dir: {tables_dir}")

    dyn_path = out_dir_m1b / "model1b_kl_dynamic_betas_all.tsv"
    win_path = out_dir_m1b / "model1b_kl_window_betas_all.tsv"
    pre_path = out_dir_m1b / "model1b_kl_pretrend_tests_all.tsv"

    dyn_all = pd.read_csv(dyn_path, sep="\t")
    win_all = pd.read_csv(win_path, sep="\t")
    pre_all = pd.read_csv(pre_path, sep="\t")

    # Build N(m) counts from KL panel using the same ES sample logic
    Nm = build_Nm_counts(thesis_root)

    # Windows helper
    windows_for_tables = build_windows_for_tables(win_all)
    windows_out_path = tables_dir / "model1b_kl_windows_for_tables.tsv"
    windows_for_tables.to_csv(windows_out_path, sep="\t", index=False)
    print(f"Saved window-level helper file to: {windows_out_path}")

    # Dynamic (appendix) helper
    dynamic_for_appendix = build_dynamic_for_appendix(dyn_all, Nm)
    dynamic_out_path = tables_dir / "model1b_kl_dynamic_for_appendix.tsv"
    dynamic_for_appendix.to_csv(dynamic_out_path, sep="\t", index=False)
    print(f"Saved dynamic-event-time helper file to: {dynamic_out_path}")

    # Pretrend helper
    pretrend_for_tables = build_pretrend_for_tables(pre_all)
    pretrend_out_path = tables_dir / "model1b_kl_pretrend_for_tables.tsv"
    pretrend_for_tables.to_csv(pretrend_out_path, sep="\t", index=False)
    print(f"Saved pretrend helper file to: {pretrend_out_path}")

    print(
        "\nSummary:\n"
        f"  Windows rows: {len(windows_for_tables)}\n"
        f"  Dynamic rows: {len(dynamic_for_appendix)}\n"
        f"  Pretrend rows: {len(pretrend_for_tables)}\n"
        f"Table helper files written to: {tables_dir}"
    )
    print("=== Model_1B_to_tables: done ===")


if __name__ == "__main__":
    main()

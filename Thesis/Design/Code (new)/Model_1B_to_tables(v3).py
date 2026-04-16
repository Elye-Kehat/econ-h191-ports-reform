from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load Model_1B(v3) safely (supports filenames with parentheses)
# ---------------------------------------------------------------------------

MODEL_1B_FILENAME = "Model_1B(v3).py"

MODEL_1B_PATH = Path(__file__).with_name(MODEL_1B_FILENAME)
spec = importlib.util.spec_from_file_location("Model_1B_v3", MODEL_1B_PATH)
m1b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m1b
assert spec.loader is not None
spec.loader.exec_module(m1b)


def find_thesis_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


def rebuild_month_index_inplace(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    """
    Rebuild month_index exactly like Model_1B(v3) main, and return ym_to_idx.
    This is CRITICAL for N(m) counts: month_index and ym_to_idx must be consistent.
    """
    df["ym_tuple"] = list(zip(df["year"], df["month"]))
    ym_sorted = sorted(df["ym_tuple"].unique())
    ym_to_idx: Dict[Tuple[int, int], int] = {ym: i + 1 for i, ym in enumerate(ym_sorted)}
    df["month_index"] = df["ym_tuple"].map(ym_to_idx)
    return ym_to_idx


def build_Nm_counts(thesis_root: Path, design: str) -> pd.DataFrame:
    kl_panel_path = thesis_root / "Data" / "KL" / "KL_Panel_monthly.tsv"
    df_kl = pd.read_csv(kl_panel_path, sep="\t")

    # Ensure log_KL exists (build_es_sample_kl checks notna on log_KL)
    if "log_KL" not in df_kl.columns:
        if "KL" not in df_kl.columns:
            raise ValueError("KL_Panel_monthly.tsv must contain 'KL' or 'log_KL'.")
        df_kl["log_KL"] = np.log(df_kl["KL"])

    ym_to_idx = rebuild_month_index_inplace(df_kl)

    Nm_rows: List[dict] = []
    base_specs = m1b.build_base_specs_kl()

    for base in base_specs:
        df_es = m1b.build_es_sample_kl(df_kl, base, ym_to_idx, design=design)
        if df_es is None or df_es.empty:
            continue

        for m in range(m1b.MIN_EVENT_TIME, m1b.MAX_EVENT_TIME + 1):
            if m == -1:
                continue
            # Count treated observations at true event_time m
            N_m = int(((df_es["treat"] == 1) & (df_es["event_time"] == m)).sum())
            Nm_rows.append(dict(reform=base.reform, target=base.target, event_time=m, N_m=N_m))

    if not Nm_rows:
        return pd.DataFrame(columns=["reform", "target", "event_time", "N_m"])

    Nm = (
        pd.DataFrame(Nm_rows)
        .drop_duplicates(subset=["reform", "target", "event_time"])
        .reset_index(drop=True)
    )
    return Nm


def build_windows_for_tables(win_all: pd.DataFrame) -> pd.DataFrame:
    # Now include did_post so we can populate the “conventional DiD” row/column later
    keep_windows = ["did_post", "post_all", "post_y1", "post_y2", "pre_all"]
    df = win_all[win_all["window_name"].isin(keep_windows)].copy()

    col_order = [
        "model",
        "reform",
        "target",
        "fe_type",
        "spec_name",
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
    for c in col_order:
        if c not in df.columns:
            df[c] = ""
    return df[col_order]


def build_dynamic_for_appendix(dyn_all: pd.DataFrame, Nm: pd.DataFrame) -> pd.DataFrame:
    df = dyn_all.copy()
    df = df.merge(Nm, how="left", on=["reform", "target", "event_time"])
    df = df.sort_values(by=["reform", "target", "fe_type", "event_time"]).reset_index(drop=True)

    col_order = [
        "model",
        "reform",
        "target",
        "fe_type",
        "spec_name",
        "event_time",
        "j",
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
    for c in col_order:
        if c not in df.columns:
            df[c] = np.nan if c in {"j", "beta_hat", "se", "tvalue", "pvalue", "N_m"} else ""
    return df[col_order]


def build_pretrend_for_tables(pre_all: pd.DataFrame) -> pd.DataFrame:
    df = pre_all.copy()
    df = df[df["test_name"] == "pretrend_all_m_le_-2"].copy()

    df["F_stat"] = df["stat"]
    df["p_value"] = df["pvalue"]

    col_order = [
        "model",
        "reform",
        "target",
        "fe_type",
        "spec_name",
        "test_name",
        "F_stat",
        "p_value",
        "df_num",
        "df_denom",
        "cov_type",
        "cluster_by",
        "r2",
    ]
    for c in col_order:
        if c not in df.columns:
            df[c] = ""
    return df[col_order]


def process_one_design(thesis_root: Path, suffix: str, design: str) -> None:
    out_dir = thesis_root / "Design" / "Output (new)" / "Model_1B"
    tables_dir = out_dir / "Tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    dyn_path = out_dir / f"model1b_kl_dynamic_betas_all{suffix}.tsv"
    win_path = out_dir / f"model1b_kl_window_betas_all{suffix}.tsv"
    pre_path = out_dir / f"model1b_kl_pretrend_tests_all{suffix}.tsv"

    if not (dyn_path.exists() and win_path.exists() and pre_path.exists()):
        print(f"[SKIP] Missing Model 1B outputs for design={design} suffix={suffix}")
        return

    dyn_all = pd.read_csv(dyn_path, sep="\t")
    win_all = pd.read_csv(win_path, sep="\t")
    pre_all = pd.read_csv(pre_path, sep="\t")

    Nm = build_Nm_counts(thesis_root, design=design)

    windows_for_tables = build_windows_for_tables(win_all)
    windows_out = tables_dir / f"model1b_kl_windows_for_tables{suffix}.tsv"
    windows_for_tables.to_csv(windows_out, sep="\t", index=False)
    print(f"Saved windows helper: {windows_out}")

    dyn_app = build_dynamic_for_appendix(dyn_all, Nm)
    dyn_out = tables_dir / f"model1b_kl_dynamic_for_appendix{suffix}.tsv"
    dyn_app.to_csv(dyn_out, sep="\t", index=False)
    print(f"Saved dynamic helper: {dyn_out}")

    pre_tbl = build_pretrend_for_tables(pre_all)
    pre_out = tables_dir / f"model1b_kl_pretrend_for_tables{suffix}.tsv"
    pre_tbl.to_csv(pre_out, sep="\t", index=False)
    print(f"Saved pretrend helper: {pre_out}")

    print(
        f"\nSummary ({design}):\n"
        f"  Windows rows:  {len(windows_for_tables)}\n"
        f"  Dynamic rows:  {len(dyn_app)}\n"
        f"  Pretrend rows: {len(pre_tbl)}\n"
        f"  Nm nonmissing: {int(dyn_app['N_m'].notna().sum())} / {len(dyn_app)}\n"
    )


def main() -> None:
    print("=== Model_1B_to_tables(v3): starting ===")
    thesis_root = find_thesis_root()
    print(f"THESIS_ROOT: {thesis_root}")

    process_one_design(thesis_root, suffix="", design="nyt")
    process_one_design(thesis_root, suffix="_twfe", design="twfe")

    print("=== Model_1B_to_tables(v3): done ===")


if __name__ == "__main__":
    main()
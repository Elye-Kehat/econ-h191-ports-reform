from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


import pandas as pd

# =============================================================================
# Model_1B_relaxed(v4).py
#
# Relaxed + port-trend alternative for Model 1B.
#
# Role in v4 architecture:
#   - supplies the "Relaxed+Tr" column family in the main Model 1B tables
#   - uses the same v4 sample construction / target universe as Model_1B(v4)
#   - replaces calendar-month FE with port-specific linear trends to reduce
#     saturation in the very small K/L event-study samples
#
# Outputs written here:
#   NYT:
#       model1b_kl_dynamic_betas_all_relaxed.tsv
#       model1b_kl_window_betas_all_relaxed.tsv
#       model1b_kl_pretrend_tests_all_relaxed.tsv
#
#   TWFE:
#       model1b_kl_dynamic_betas_all_relaxed_twfe.tsv
#       model1b_kl_window_betas_all_relaxed_twfe.tsv
#       model1b_kl_pretrend_tests_all_relaxed_twfe.tsv
#       model1b_kl_static_betas_all_relaxed_twfe.tsv
# =============================================================================


THIS_FILE = Path(__file__).resolve()
THESIS_ROOT = THIS_FILE.parents[2]
OUTPUT_DIR = THESIS_ROOT / "Design" / "Output (new)" / "Model_1B_relaxed"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

MODEL_1B_FILENAME = "Model_1B(v4).py"
MODEL_1B_PATH = Path(__file__).with_name(MODEL_1B_FILENAME)
spec = importlib.util.spec_from_file_location("Model_1B_v4_relaxed_import", MODEL_1B_PATH)
m1b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m1b
assert spec.loader is not None
spec.loader.exec_module(m1b)


def run_relaxed_event_study(df_es: pd.DataFrame, spec_with_fe: Any, shock_cols: List[str]):

    df_es = m1b._coerce_for_patsy(df_es.copy(), ["event_time", "time_id"])

    # Relaxed+Tr estimator: unit FE + port-specific linear trends + event-time dummies
    formula = "log_KL ~ C(event_time, Treatment(reference=-1)) + C(unit_id) + port_trend"
    n_by_event_time = df_es.groupby("event_time")["unit_id"].size().to_dict()
    result = m1b.fit_hc1_ols(formula = formula, data = df_es)
    return result, n_by_event_time


def run_relaxed_static_did(
    df_es: pd.DataFrame,
    spec_with_fe: Any,
    shock_cols: List[str],
    event_index: int,
    horizon_name: str,
    horizon_end,
):

    df_did = m1b.subset_for_static_horizon(df_es, event_index, horizon_end)
    df_did = m1b._coerce_for_patsy(df_did, ["time_id"])
    if df_did.empty:
        return None

    n_post_treated = int(((df_did["treated"]) & (df_did["month_index"] >= (event_index + 1))).sum())
    if n_post_treated == 0:
        return None

    df_did["treated_int"] = df_did["treated"].astype(int)
    df_did["post_in_horizon"] = ((df_did["month_index"] >= (event_index + 1))).astype(int)
    df_did["treated_post"] = df_did["treated_int"] * df_did["post_in_horizon"]

    formula = "log_KL ~ treated_post + C(unit_id) + port_trend"
    res = m1b.fit_hc1_ols(formula = formula, data = df_did)

    treated_post_ms = df_did.loc[
        (df_did["treated"]) & (df_did["month_index"] >= (event_index + 1)),
        "month_index"
    ] - event_index
    if len(treated_post_ms) == 0:
        return None

    max_post_supported = int(treated_post_ms.max())
    if horizon_end is not None:
        max_post_supported = min(max_post_supported, int(horizon_end))

    return {
        "horizon": horizon_name,
        "a": 1,
        "b": max_post_supported,
        "beta": float(res.params.get("treated_post", float("nan"))),
        "se": float(res.bse.get("treated_post", float("nan"))),
        "pvalue": float(res.pvalues.get("treated_post", float("nan"))),
        "n_obs": int(res.nobs),
        "r2": float(res.rsquared),
        "n_treated": int(df_did["treated"].sum()),
        "n_control": int((~df_did["treated"]).sum()),
        "n_post_treated": n_post_treated,
        "se_type": str(getattr(res, "cov_type", "HC1")),
    }


def expand_relaxed_specs(base_specs: List[Any], shock_cols: List[str]) -> List[Any]:
    return [
        m1b.SpecWithFE(spec = s, spec_name = "relaxed_tr", include_port_trends = True, include_shocks = False)
        for s in base_specs
    ]


def run_design(
    df: pd.DataFrame,
    ym_to_idx: Dict[Tuple[int, int], int],
    shock_cols: List[str],
    base_specs: List[Any],
    design_name: str,
    clamp_windows: bool,
    suffix: str,
    run_static: bool,
) -> None:

    dynamic_by_spec = {}
    window_by_spec = {}
    pretrend_by_spec = {}
    static_by_spec = {}

    specs_with_fe = expand_relaxed_specs(base_specs, shock_cols)

    for spec_with_fe in specs_with_fe:
        spec = spec_with_fe.spec
        print(f"\n=== [RELAXED {design_name}] table={spec.table_group}, reform={spec.reform}, target={spec.target}, spec={spec_with_fe.spec_name} ===")
        try:
            df_es = m1b.build_es_sample(df = df, spec = spec, ym_to_idx = ym_to_idx, clamp_windows = clamp_windows)
        except Exception as e:
            print(f"[WARN] Failed to build relaxed ES sample for {spec.reform} / {spec.target}: {e}")
            continue

        if df_es.empty:
            print("[WARN] Empty estimation sample; skipping.")
            continue

        n_treated = int(df_es["treated"].sum())
        n_control = int((~df_es["treated"]).sum())
        print(f"Sample size: {len(df_es)} rows ({n_treated} treated, {n_control} controls).")

        try:
            es_res, n_by_event_time = run_relaxed_event_study(df_es, spec_with_fe, shock_cols)
        except Exception as e:
            print(f"[WARN] Relaxed regression failed for {spec.reform} / {spec.target}: {e}")
            continue

        dynamic = m1b.extract_dynamic_betas(es_res, spec_with_fe, n_by_event_time, design_name)
        windows = m1b.compute_window_averages(es_res, spec_with_fe, design_name)
        pre = m1b.compute_pretrend_f_test(es_res, spec_with_fe, design_name)

        dynamic_by_spec.setdefault(spec_with_fe.spec_name, []).append(dynamic)
        window_by_spec.setdefault(spec_with_fe.spec_name, []).append(windows)
        if not pre.empty:
            pretrend_by_spec.setdefault(spec_with_fe.spec_name, []).append(pre)

        if run_static:
            event_index = m1b.year_month_to_index_clamped(ym_to_idx, (spec.event_year, spec.event_month))
            static_rows = []
            for hname, hend in m1b.STATIC_HORIZONS.items():
                out = run_relaxed_static_did(df_es, spec_with_fe, shock_cols, event_index, hname, hend)
                if out is None:
                    continue
                static_rows.append(
                    {
                        "design": design_name,
                        "table_group": spec.table_group,
                        "reform": spec.reform,
                        "target": spec.target,
                        "target_key": spec.target_key,
                        "spec_name": spec_with_fe.spec_name,
                        **out,
                    }
                )
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
        pd.concat(pooled["dynamic"], ignore_index = True).to_csv(
            OUTPUT_DIR / f"{base_name}_dynamic_betas_all{suffix}.tsv", sep = "\t", index = False
        )
    if pooled["window"]:
        pd.concat(pooled["window"], ignore_index = True).to_csv(
            OUTPUT_DIR / f"{base_name}_window_betas_all{suffix}.tsv", sep = "\t", index = False
        )
    if pooled["pretrend"]:
        pd.concat(pooled["pretrend"], ignore_index = True).to_csv(
            OUTPUT_DIR / f"{base_name}_pretrend_tests_all{suffix}.tsv", sep = "\t", index = False
        )
    if run_static and pooled["static"]:
        pd.concat(pooled["static"], ignore_index = True).to_csv(
            OUTPUT_DIR / f"{base_name}_static_betas_all{suffix}.tsv", sep = "\t", index = False
        )
def clear_outputs(output_dir: Path) -> None:
    patterns = [
        "model1b_kl_dynamic_betas_*relaxed*.tsv",
        "model1b_kl_window_betas_*relaxed*.tsv",
        "model1b_kl_pretrend_tests_*relaxed*.tsv",
        "model1b_kl_static_betas_*relaxed*.tsv",
    ]
    removed = 0
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok = True)
            removed += 1
    print(f"Cleared {removed} old relaxed Model 1B output files from: {output_dir}")


def main() -> None:
    clear_outputs(OUTPUT_DIR)
    df = m1b.load_kl_panel(m1b.KL_PANEL_PATH)
    ym_to_idx = m1b.build_year_month_to_index(df)
    shock_cols = m1b.get_shock_control_cols(df)

    print("\n==================== RELAXED NYT run ====================")
    nyt_specs = m1b.build_nyt_specs(df)
    run_design(
        df = df,
        ym_to_idx = ym_to_idx,
        shock_cols = shock_cols,
        base_specs = nyt_specs,
        design_name = "NYT",
        clamp_windows = True,
        suffix = "_relaxed",
        run_static = False,
    )

    print("\n==================== RELAXED TWFE run ====================")
    twfe_specs = m1b.build_twfe_specs(df)
    run_design(
        df = df,
        ym_to_idx = ym_to_idx,
        shock_cols = shock_cols,
        base_specs = twfe_specs,
        design_name = "TWFE",
        clamp_windows = True,
        suffix = "_relaxed_twfe",
        run_static = True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 06_run_es.py — Unified runner
# Runs ALL event-study regressions (Tables 1–4 + appendix grids) in one pass,
# using a single YAML. Outputs tidy CSVs for 08_make_tables.py to consume.
#
# Key robustness vs prior version:
# - Accepts MODEL1_YAML env var or --yaml path (fallbacks provided).
# - Tolerates both list-style and mapping-style "specs" in YAML.
# - Does NOT require a 'data' block; will use sensible defaults if missing.
# - Avoids backslash escapes in docstrings to fix warnings.
# - Clear errors when required columns are missing.
# - Computes time index and ln_lp if absent in the input panel.
# - Writes both full k-grids (appendix) and compact subsets (tables 3–4).
#
# This script intentionally keeps dependencies modest (pandas, numpy, statsmodels).
# If statsmodels is missing, it will explain what to pip install.

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

# Try to import statsmodels; guide the user if unavailable
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception as e:
    msg = (
        "statsmodels is required for 06_run_es.py.\n"
        "Try: pip install statsmodels\n\n"
        f"Import error: {e}"
    )
    print(msg, file=sys.stderr)
    raise

# ------------------------------
# Utilities
# ------------------------------

def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as e:
        print("PyYAML is required. Try: pip install pyyaml", file=sys.stderr)
        raise
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _qstr_to_tuple(qstr: str) -> Tuple[int, int]:
    # Parse 'YYYYQ#' into (year, quarter).
    qstr = qstr.strip().upper()
    if "Q" not in qstr:
        raise ValueError(f"Bad quarter string '{qstr}'. Expected 'YYYYQ#'.")
    y, q = qstr.split("Q")
    return int(y), int(q)

def _to_t_index(year: int, quarter: int, base_year: int) -> int:
    # Convert (year, quarter) to an integer time index anchored at base_year.
    return 4 * (year - base_year) + (quarter - 1)

def _specs_to_flags(specs_cfg) -> Dict[str, Dict[str, bool]]:
    # Accept two forms:
    #   - ['baseline','port_tr','tr_shocks']
    #   - {'baseline': {...}, 'port_tr': {...}, ...}
    # Return {spec_name: {'add_port_trend': bool, 'add_shocks': bool}}.
    mapping = {}
    if isinstance(specs_cfg, list):
        for s in specs_cfg:
            sname = str(s)
            if sname.lower() == "baseline":
                mapping["baseline"] = {"add_port_trend": False, "add_shocks": False}
            elif sname.lower() in ("porttr", "port_tr", "port-tr"):
                mapping["port_tr"] = {"add_port_trend": True, "add_shocks": False}
            elif sname.lower() in ("tr&shocks", "tr_shocks", "tr-shocks", "shocks"):
                mapping["tr_shocks"] = {"add_port_trend": True, "add_shocks": True}
            else:
                mapping[sname] = {"add_port_trend": False, "add_shocks": False}
    elif isinstance(specs_cfg, dict):
        for k, v in specs_cfg.items():
            if not isinstance(v, dict):
                mapping[str(k)] = {"add_port_trend": False, "add_shocks": False}
            else:
                mapping[str(k)] = {
                    "add_port_trend": bool(v.get("add_port_trend", False)),
                    "add_shocks": bool(v.get("add_shocks", False)),
                }
    else:
        mapping["baseline"] = {"add_port_trend": False, "add_shocks": False}
    return mapping

def _ensure_cols(df: pd.DataFrame, cols: List[str], context: str = ""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} required {context}")

def _clustered_results(model, df, cluster_by: str = "port"):
    # Fit OLS with cluster-robust SE by cluster_by (default: 'port').
    res = model.fit()
    if cluster_by in df.columns:
        try:
            robust = res.get_robustcov_results(cov_type="cluster", groups=df[cluster_by], use_correction=True)
            return robust
        except Exception:
            robust = res.get_robustcov_results(cov_type="cluster", groups=df[cluster_by])
            return robust
    else:
        return res.get_robustcov_results(cov_type="HC1")

def _wald_test_on_leads(res, lead_names: List[str]) -> float:
    # Wald test that all 'lead' coeffs == 0. Returns p-value (float) or np.nan if none exist.
    if not lead_names:
        return float("nan")
    name_to_idx = {name: i for i, name in enumerate(res.model.exog_names)}
    present = [name for name in lead_names if name in name_to_idx]
    if not present:
        return float("nan")
    import numpy as np
    p = len(res.params)
    R = np.zeros((len(present), p))
    for r, nm in enumerate(present):
        R[r, name_to_idx[nm]] = 1.0
    rtest = res.wald_test(R)
    try:
        return float(rtest.pvalue)
    except Exception:
        return float("nan")

# ------------------------------
# Main runner
# ------------------------------

def load_config(yaml_path: Path) -> dict:
    # Load YAML and provide safe defaults for keys 06 needs,
    # without breaking older steps that ignore these keys.
    cfg = _read_yaml(yaml_path)

    cfg.setdefault("events", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("data", {})
    cfg.setdefault("k_grid", {})
    cfg.setdefault("groups", [])
    cfg.setdefault("specs", ["baseline", "port_tr", "tr_shocks"])
    cfg.setdefault("shocks", {})

    data_block = cfg["data"]
    if "panel_path" not in data_block:
        data_block["panel_path"] = "Design/Output Data/05_panel_terminal_sharedpre_model1.csv"
    if "output_dir" not in data_block:
        data_block["output_dir"] = "Design/Output Data/ES_All"

    kg = cfg["k_grid"]
    kg.setdefault("min_k", -8)
    kg.setdefault("max_k", 4)
    kg.setdefault("omit_k", -1)

    return cfg

def prepare_panel(panel_csv: Path) -> pd.DataFrame:
    # Load step-05 panel (or equivalent), and ensure we have basics
    df = pd.read_csv(panel_csv)
    required_basic = ["port", "terminal", "year", "quarter"]
    _ensure_cols(df, required_basic, "in the input panel")

    # ln_lp
    if "ln_lp" not in df.columns:
        cand = None
        for c in ["lp", "LP", "Lp", "labor_productivity", "ln(LP)"]:
            if c in df.columns:
                cand = c
                break
        if cand is None:
            raise KeyError("Could not find 'ln_lp' nor a raw 'lp' column to log-transform.")
        df["ln_lp"] = np.log(df[cand].astype(float))

    # time index
    if "t_index" not in df.columns:
        base_year = int(df["year"].min())
        df["t_index"] = (df["year"].astype(int) - base_year) * 4 + (df["quarter"].astype(int) - 1)

    df["t_key"] = df["t_index"].astype(int)
    return df

def build_event_index(cfg_events: dict, port: str, clock: str, base_year: int) -> int:
    # Get the event t_index for a given port & clock label ('competition' or 'privatization').
    if clock not in cfg_events or cfg_events[clock] is None:
        raise KeyError(f"Clock '{clock}' not found in YAML under 'events'.")
    port_map = cfg_events[clock]
    if port not in port_map or port_map[port] is None:
        raise KeyError(f"No event quarter for port '{port}' under clock '{clock}'.")
    y, q = _qstr_to_tuple(str(port_map[port]))
    return _to_t_index(y, q, base_year)

def add_shock_interactions(df: pd.DataFrame, shocks_cfg: dict) -> pd.DataFrame:
    # For each named window, create a binary 'shock_{name}' and later interact with C(port).
    if not shocks_cfg:
        return df
    base_year = int(df["year"].min())
    def in_window(row, start_str, end_str, base_year):
        y1, q1 = _qstr_to_tuple(start_str)
        y2, q2 = _qstr_to_tuple(end_str)
        t1 = _to_t_index(y1, q1, base_year)
        t2 = _to_t_index(y2, q2, base_year)
        return int(t1 <= row["t_index"] <= t2)
    for name, win in shocks_cfg.items():
        s = str(win.get("start", "")).strip()
        e = str(win.get("end", "")).strip()
        if not s or not e:
            continue
        col = f"shock_{name}"
        df[col] = df.apply(lambda r: in_window(r, s, e, base_year), axis=1)
    return df

def run_group_spec(df_panel: pd.DataFrame,
                   group: dict,
                   spec_name: str,
                   spec_flags: dict,
                   cfg_events: dict,
                   k_grid: dict) -> tuple[pd.DataFrame, float, int]:
    # Estimate the event-study for one (group, spec).
    g_port = group["port"]
    g_term = group["terminal"]
    g_clock = group.get("clock", "competition")

    df = df_panel.copy()

    # Determine event index (anchor at panel's base year)
    base_year = int(df["year"].min())
    t_event = build_event_index(cfg_events, g_port, g_clock, base_year)

    # Indicator for focal terminal
    df["is_g"] = ((df["port"].astype(str) == str(g_port)) &
                  (df["terminal"].astype(str) == str(g_term))).astype(int)

    # Event-time range and omit bin
    min_k, max_k, omit_k = int(k_grid["min_k"]), int(k_grid["max_k"]), int(k_grid["omit_k"])
    k_values = [k for k in range(min_k, max_k + 1) if k != omit_k]

    # Compute k for every row relative to the port's event
    df["k"] = df["t_index"] - t_event

    # Create named columns like D_km4_g, D_k0_g, D_k1_g, etc.
    def kname(k):
        return f"D_k{str(k).replace('-', 'm')}_g"

    for k in k_values:
        col = kname(k)
        df[col] = ((df["k"] == k) & (df["is_g"] == 1)).astype(int)

    # Build formula
    rhs = []
    rhs += [kname(k) for k in k_values]       # event-time terms
    rhs.append("C(terminal)")                 # terminal FE
    rhs.append("C(t_key)")                    # quarter FE

    if spec_flags.get("add_port_trend", False):
        rhs.append("t_index:C(port)")         # port-specific linear trends

    if spec_flags.get("add_shocks", False):
        shock_cols = [c for c in df.columns if c.startswith("shock_")]
        rhs += [f"C(port):{c}" for c in shock_cols]   # shocks × port

    formula = "ln_lp ~ " + " + ".join(rhs)

    # Fit OLS with cluster-robust SEs by port
    model = smf.ols(formula, data=df)
    res = model.fit()
    try:
        res = res.get_robustcov_results(cov_type="cluster", groups=df["port"], use_correction=True)
    except Exception:
        res = res.get_robustcov_results(cov_type="cluster", groups=df["port"])

    n_obs = int(res.nobs)

    # Collect coefficients & SE for event-time terms
    rows = []
    for k in k_values:
        pname = kname(k)
        beta = float(res.params.get(pname, np.nan))
        se = float(res.bse.get(pname, np.nan))
        rows.append({"k": k, "beta": beta, "se": se, "param": pname})

    dyn = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # Wald test on leads k in {-4,-3,-2} (if present)
    leads = [-4, -3, -2]
    lead_names = [kname(k) for k in leads if k in k_values]
    # Build R matrix if all are present; else return nan
    def wald_p(res, names):
        if not names:
            return float("nan")
        name_to_idx = {name: i for i, name in enumerate(res.model.exog_names)}
        present = [name for name in names if name in name_to_idx]
        if not present:
            return float("nan")
        p = len(res.params)
        R = np.zeros((len(present), p))
        for r, nm in enumerate(present):
            R[r, name_to_idx[nm]] = 1.0
        try:
            return float(res.wald_test(R).pvalue)
        except Exception:
            return float("nan")
    p_leads = wald_p(res, lead_names)

    return dyn, p_leads, n_obs

def orchestrate_all(yaml_path: Optional[str] = None):
    # Resolve YAML path
    if yaml_path is None:
        yaml_path = os.environ.get("MODEL1_YAML", "model1_params.yaml")
    ypath = Path(yaml_path)
    if not ypath.exists():
        alt = Path(__file__).parent / "model1_params.yaml"
        if alt.exists():
            ypath = alt
        else:
            raise FileNotFoundError(f"YAML not found at {yaml_path} or {alt}")

    cfg = load_config(ypath)

    # Paths & k-grid
    panel_csv = Path(cfg["data"]["panel_path"])
    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load panel and add shocks
    df = prepare_panel(panel_csv)
    df = add_shock_interactions(df, cfg.get("shocks", {}))

    # Specs -> flags
    spec_flags_map = _specs_to_flags(cfg.get("specs", ["baseline", "port_tr", "tr_shocks"]))

    # Run all groups × specs
    results_index = []
    full_dyn_frames = []

    for group in cfg["groups"]:
        gname = str(group.get("name", f"{group.get('port','?')}_{group.get('terminal','?')}"))
        for sname, sflags in spec_flags_map.items():
            dyn, p_leads, n_obs = run_group_spec(
                df_panel=df,
                group=group,
                spec_name=sname,
                spec_flags=sflags,
                cfg_events=cfg["events"],
                k_grid=cfg["k_grid"],
            )
            # Save per (group,spec) full k-grid
            fname = f"es_dynamic_{gname}_{sname}.csv"
            fpath = out_dir / fname
            dyn.assign(group=gname, spec=sname, p_leads=p_leads, n_obs=n_obs).to_csv(fpath, index=False)
            results_index.append({"artifact": str(fpath.relative_to(out_dir)), "group": gname, "spec": sname})

            # Build compact subset for tables 3–4 (lead avg, k=0..4)
            disp_rows = []
            leads = dyn[dyn["k"].isin([-4, -3, -2])].copy()
            if not leads.empty:
                beta_avg = leads["beta"].mean(skipna=True)
                se_avg = leads["se"].mean(skipna=True)
                disp_rows.append({"row": "(-4..-2) avg", "beta": beta_avg, "se": se_avg})
            for kk in [0, 1, 2, 3, 4]:
                row = dyn.loc[dyn["k"] == kk]
                if row.empty:
                    disp_rows.append({"row": str(kk), "beta": np.nan, "se": np.nan})
                else:
                    disp_rows.append({"row": str(kk), "beta": float(row["beta"].iloc[0]), "se": float(row["se"].iloc[0])})
            disp = pd.DataFrame(disp_rows)
            disp["group"] = gname
            disp["spec"] = sname
            disp["p_leads"] = p_leads
            full_dyn_frames.append(disp)

    # Aggregate compacts into two CSVs: one Haifa, one Ashdod (for tables 3 and 4)
    if full_dyn_frames:
        compact = pd.concat(full_dyn_frames, ignore_index=True)

        def port_of(g):
            if g.startswith("Haifa_"):
                return "Haifa"
            if g.startswith("Ashdod_"):
                return "Ashdod"
            return "Other"

        compact["port"] = compact["group"].map(port_of)

        haifa = compact[compact["port"] == "Haifa"].copy()
        ashdod = compact[compact["port"] == "Ashdod"].copy()

        haifa_path = out_dir / "table3_dyn_haifa_compact.csv"
        ashdod_path = out_dir / "table4_dyn_ashdod_compact.csv"
        haifa.to_csv(haifa_path, index=False)
        ashdod.to_csv(ashdod_path, index=False)
        results_index.append({"artifact": str(haifa_path.relative_to(out_dir)), "group": "ALL_Haifa", "spec": "ALL"})
        results_index.append({"artifact": str(ashdod_path.relative_to(out_dir)), "group": "ALL_Ashdod", "spec": "ALL"})

    (out_dir / "index.json").write_text(json.dumps(results_index, indent=2))

    print(f"[06] Done. Outputs written under: {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=str, default=os.environ.get("MODEL1_YAML", "model1_params.yaml"))
    args = ap.parse_args()
    orchestrate_all(args.yaml)

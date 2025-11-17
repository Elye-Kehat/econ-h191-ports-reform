#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_run_es_dyn_tables.py
---------------------------------
Goal: Run *pooled* event-study regressions needed for Tables 3 & 4 (Haifa & Ashdod)
once, and print tidy results for both ports and both terminals (entrant vs legacy)
under three specifications: Baseline, +PortTr, +Tr&Shocks.

This prints to stdout and also writes compact CSVs you can reuse later.
You can expand this later to feed LaTeX, but for now it's focused on
"run once → see all coefficients".

Key design choices (kept simple & explicit):
- One pooled regression per SPEC (so clusters by port = 2 are valid).
- Dynamic effects estimated separately for each (port × terminal) by interacting
  event-time dummies with both port and terminal indicators. That means a *single*
  regression produces *four* series: Haifa-SIPG, Haifa-Legacy, Ashdod-HCT, Ashdod-Legacy.
- Terminal FE and time FE (quarter index FE) are always included.
- +PortTr adds a linear time trend interacted with port (two parameters).
- +Tr&Shocks additionally includes dummies for COVID (2020–21) and late-2023/24 windows.

Inputs:
- A terminal-quarter panel with at least: port, terminal, year, quarter, and LP (level) or ln_lp.
  Default location (override with --panel or $PANEL_PATH):
    "Design/Output Data/05_es_input_terminal_sharedpre_nyt.csv"
  but you can pass any CSV/TSV that has the required columns.
- A YAML config (optional) for event clocks and shock windows (override with --yaml or $MODEL1_YAML).
  If missing, the script falls back to simple built-in placeholders that you can edit quickly.

Outputs:
- Prints compact tables for k ∈ {(-4..-2) avg, 0, 1, 2, 3, 4} and the joint F-test p-value for leads.
- Writes tidy CSVs under "Design/Output Data/Tables3_4/":
  * dyn_full_coeffs_<spec>.csv         (all k’s, all groups from that spec)
  * dyn_compact_display_<spec>.csv     (only the 6 display rows per group + leads F p)
  * run_summary.txt                    (paths and quick diagnostics)

Run:
  python 06_run_es_dyn_tables.py \
       --panel "Design/Output Data/05_es_input_terminal_sharedpre_nyt.csv" \
       --yaml  "Design/Code/Econometrics/model1_params.yaml"

Notes:
- Cluster robust SEs use statsmodels 'cluster' covariance with groups = port.
  This is CR1-like; switching to CR2 can be added later if needed.
- If your panel already has ln_lp, we will use it; otherwise we compute ln(LP)
  from 'LP' or 'lp' column.
- Time FE uses an integer t_index constructed from (year, quarter).

Author: ChatGPT (Econ H191 helper)
Date: 2025-11-12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yaml

import statsmodels.api as sm
from statsmodels.stats.contrast import ContrastResults

# ------------------------------
# Utilities
# ------------------------------

def log(msg: str):
    print(f"[06] {msg}", flush=True)

def infer_sep(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    return ","  # default CSV

def coalesce_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(f"None of the expected columns present: {candidates}")

def ensure_ln_lp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a column named 'ln_lp' exists.
    Accept common aliases from the earlier pipeline and, if only levels exist,
    compute ln() safely (dropping nonpositive levels).
    """
    import numpy as np

    # Common names used across 03/04/05
    LP_LOG_CANDS   = ["ln_lp", "Y", "ln(LP)", "log_lp", "lnlp", "ln_lp_model1"]
    LP_LEVEL_CANDS = ["LP", "lp", "LP_level", "lp_level", "LP_quarter", "lp_quarter"]

    # If any known log column exists, standardize to ln_lp
    for c in LP_LOG_CANDS:
        if c in df.columns:
            if c != "ln_lp":
                df = df.rename(columns={c: "ln_lp"})
            return df

    # Else try to construct from a level column
    for c in LP_LEVEL_CANDS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            # guard against nonpositive values
            bad = (s <= 0) | ~np.isfinite(s)
            if bad.any():
                print(f"[06] WARNING: {bad.sum()} nonpositive/invalid LP levels in '{c}' will be dropped before log.")
            df = df.loc[~bad].copy()
            df["ln_lp"] = np.log(s.loc[~bad])
            return df

    raise KeyError("Missing ln_lp and no LP level column found among "
                   f"{LP_LEVEL_CANDS}. Add/rename a column or precompute ln_lp.")


def ensure_t_index(df: pd.DataFrame) -> pd.DataFrame:
    """Construct integer time index t_index from year & quarter (q ∈ {1,2,3,4})."""
    if "t_index" in df.columns:
        return df
    if not {"year", "quarter"} <= set(df.columns):
        raise KeyError("Need 'year' and 'quarter' to build t_index.")
    df = df.copy()
    # Normalize quarter to 1..4
    df["quarter"] = df["quarter"].astype(int)
    df["t_index"] = df["year"].astype(int) * 4 + (df["quarter"] - 1)
    return df

def build_event_time(df: pd.DataFrame, clocks: dict[tuple[str,str], int],
                     min_k: int, max_k: int) -> pd.DataFrame:
    """
    Add event-time k for each row using (port, terminal) → entry_t_index.
    k = t_index - entry_t_index.
    """
    df = df.copy()
    def key(row):
        return (str(row["port"]), str(row["terminal"]))
    # Default to NaN for units without a clock; they will receive NaN k and no dummies.
    entry_map = { (str(p),str(t)): int(v) for (p,t), v in clocks.items() }
    ks = []
    for _, r in df.iterrows():
        k = np.nan
        kt = key(r)
        if kt in entry_map:
            k = int(r["t_index"]) - entry_map[kt]
        ks.append(k)
    df["k"] = ks
    # Cap/trim to [min_k, max_k] to limit the dummy explosion later
    df["k_trim"] = df["k"].where(df["k"].between(min_k, max_k), other=np.nan)
    return df

def add_shock_dummies(df: pd.DataFrame, windows: dict[str, dict[str,int]]) -> pd.DataFrame:
    """
    windows: mapping name -> {'start_t': int, 'end_t': int} inclusive
    Adds 0/1 columns 'shock_<name>'.
    """
    df = df.copy()
    for name, w in windows.items():
        col = f"shock_{name}"
        st = int(w["start_t"]); en = int(w["end_t"])
        df[col] = ((df["t_index"] >= st) & (df["t_index"] <= en)).astype(int)
    return df

def make_port_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add linear time trend interacted with port. (Common within a port)
    trend_Haifa = (t_index - min_t_overall) * 1{port=Haifa}, similar for Ashdod.
    """
    df = df.copy()
    t0 = int(df["t_index"].min())
    for p in df["port"].unique():
        col = f"trend_{p}"
        df[col] = (df["t_index"] - t0) * (df["port"] == p).astype(int)
    return df

def wald_pvalue(model_res, param_names: list[str]) -> float:
    """
    Wald test that the listed parameters are jointly zero, using the model's robust covariance.
    Returns np.nan if any param is missing.
    """
    # build restriction matrix R * beta = 0
    idxmap = {p:i for i,p in enumerate(model_res.params.index)}
    cols = len(model_res.params)
    rows = len(param_names)
    import numpy as np
    R = np.zeros((rows, cols))
    ok = True
    for r, pn in enumerate(param_names):
        if pn not in idxmap:
            ok = False
            break
        R[r, idxmap[pn]] = 1.0
    if not ok or rows == 0:
        return np.nan
    try:
        wt = model_res.wald_test(R)
        return float(wt.pvalue)
    except Exception:
        return np.nan

def avg_of_coeffs(model_res, names: list[str]) -> tuple[float,float]:
    """
    Average a set of coefficients and delta-method SE using the robust covariance.
    Returns (avg_beta, se_avg). If any name missing, returns (np.nan, np.nan).
    """
    import numpy as np
    idxmap = {p:i for i,p in enumerate(model_res.params.index)}
    if any(n not in idxmap for n in names) or len(names) == 0:
        return (np.nan, np.nan)
    b = model_res.params.values
    V = model_res.cov_params()
    w = np.ones(len(names)) / len(names)
    sel = np.array([idxmap[n] for n in names], dtype=int)
    # beta_avg = w' * b_sel
    beta_avg = float((w @ b[sel]))
    # var_avg = w' * V_sel * w
    V_sel = V.iloc[sel, sel].values
    var_avg = float(w @ V_sel @ w)
    se_avg = float(np.sqrt(max(var_avg, 0.0)))
    return (beta_avg, se_avg)

# ------------------------------
# Robust YAML loader with forgiving schema
# ------------------------------

def load_yaml_config(yaml_path: Path) -> dict:
    """
    We accept multiple schemas to avoid breaking older 01-05:
    Expect at least:
      clocks:
        Haifa:
          SIPG:  <t_index>
          Legacy:<t_index (same reference clock)>
        Ashdod:
          HCT:   <t_index>
          Legacy:<t_index>
      k_grid: {min_k: -8, max_k: 4, omit_k: -1}
      shocks:
        covid:   {start_t: 2020*4+0, end_t: 2021*4+3}
        late2324:{start_t: 2023*4+?, end_t: 2024*4+?}
    Also allow flat mappings, e.g. clocks_flat: {("Haifa","SIPG"): t, ...}
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Flexible resolution of panel path if provided
    data = cfg.get("data", {})
    panel_path = None
    if isinstance(data, dict):
        panel_path = data.get("panel_path")

    # Flexible clocks parsing
    clocks = {}
    if "clocks" in cfg and isinstance(cfg["clocks"], dict):
        for port, sub in cfg["clocks"].items():
            if isinstance(sub, dict):
                for term, ti in sub.items():
                    clocks[(str(port), str(term))] = int(ti)
    elif "clocks_flat" in cfg and isinstance(cfg["clocks_flat"], dict):
        for key, ti in cfg["clocks_flat"].items():
            if isinstance(key, (list, tuple)) and len(key) == 2:
                p, t = key
                clocks[(str(p), str(t))] = int(ti)

    # k-grid defaults
    kg = cfg.get("k_grid", {})
    min_k = int(kg.get("min_k", -8))
    max_k = int(kg.get("max_k", 4))
    omit_k = int(kg.get("omit_k", -1))

    # shocks
    shocks_cfg = cfg.get("shocks", {})
    shocks = {}
    for name, w in shocks_cfg.items():
        if isinstance(w, dict) and {"start_t", "end_t"} <= set(w.keys()):
            shocks[name] = {"start_t": int(w["start_t"]), "end_t": int(w["end_t"])}

    return {
        "panel_path": panel_path,  # may be None
        "clocks": clocks,          # dict[(port,terminal)] -> entry_t_index
        "min_k": min_k,
        "max_k": max_k,
        "omit_k": omit_k,
        "shocks": shocks,
    }

# ------------------------------
# Core estimation routine
# ------------------------------

def run_eventstudy(panel: pd.DataFrame,
                   clocks: dict[tuple[str,str], int],
                   min_k: int, max_k: int, omit_k: int,
                   add_port_trend: bool = False,
                   add_shocks: dict[str, dict[str,int]] | None = None):
    """
    Returns:
      model_res: statsmodels result object with cluster-robust cov by port
      design_info: dict with parameter name builders for later extraction
    """
    # Ensure required basics
    panel = ensure_ln_lp(panel)
    panel = ensure_t_index(panel)

    # Basic normalizations
    needed = ["port", "terminal"]
    for c in needed:
        if c not in panel.columns:
            raise KeyError(f"Missing column '{c}' in panel.")
    panel["port"] = panel["port"].astype(str)
    panel["terminal"] = panel["terminal"].astype(str)

    # Derived keys
    panel["port_terminal"] = panel["port"] + " — " + panel["terminal"]

    # Build event time
    panel = build_event_time(panel, clocks=clocks, min_k=min_k, max_k=max_k)

    # Build FE components: terminal FE (always) and time FE (t_index FE)
    # We'll use explicit dummy construction for clarity/speed
    # Terminal FE
    term_dummies = pd.get_dummies(panel["port_terminal"], prefix="FE_term", drop_first=False)
    # Time FE (quarter index FE): FE_t_<index>
    time_dummies = pd.get_dummies(panel["t_index"], prefix="FE_t", drop_first=False)

    # Optionally: port trends
    if add_port_trend:
        panel = make_port_trend(panel)

    # Optionally: shocks
    if add_shocks:
        panel = add_shock_dummies(panel, add_shocks)

    # Dynamic dummies: for each (port, terminal, k) create D_<port>_<term>_kXX
    # (omit k = omit_k), and only where k_trim equals that k (NaN otherwise).
    ks = [k for k in range(min_k, max_k + 1) if k != omit_k]
    # Identify which (port,terminal) have a clock; only those get dynamic dummies
    treated_keys = set(clocks.keys())

    def pname(p, t, k):
        # Safe parameter name; avoid spaces
        return f"D_{p.replace(' ','_')}_{t.replace(' ','_')}_k{k}"

    dyn_cols = []
    for (p, t) in treated_keys:
        mask_unit = (panel["port"] == p) & (panel["terminal"] == t)
        for k in ks:
            col = pname(p, t, k)
            panel[col] = ((mask_unit) & (panel["k_trim"] == k)).astype(int)
            dyn_cols.append(col)

    # Assemble X matrix
    X_blocks = []
    X_blocks.append(pd.DataFrame({c: panel[c].values for c in dyn_cols}))
    X_blocks.append(term_dummies)  # terminal FE
    X_blocks.append(time_dummies)  # time FE
    if add_port_trend:
        trend_cols = [c for c in panel.columns if c.startswith("trend_")]
        X_blocks.append(panel[trend_cols])
    if add_shocks:
        shock_cols = [c for c in panel.columns if c.startswith("shock_")]
        X_blocks.append(panel[shock_cols])

    X = pd.concat(X_blocks, axis=1)
    y = panel["ln_lp"].astype(float).values

    # Cluster by port (2 clusters: Haifa, Ashdod). Provide group labels vector.
    groups = panel["port"].values

    # Fit OLS with no explicit intercept since FE cover means thoroughly
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)

    design_info = {
        "ks": ks,
        "param_name": pname,
        "treated_keys": list(treated_keys),
    }
    return res, design_info

def extract_for_display(res, design_info, lead_range=(-4,-3,-2)):
    """
    Build a compact display DataFrame with rows:
      (-4..-2) avg, 0,1,2,3,4
    for each (port, terminal), and compute a Leads F-test p-value.

    Returns a tidy long DF with columns:
      port, terminal, k_label, beta, se, t, p, leads_F_p
    """
    rows = []
    ks = design_info["ks"]
    pname = design_info["param_name"]
    treated_keys = design_info["treated_keys"]

    # Helper to get beta/se
    params = res.params
    cov = res.cov_params()

    def get_beta_se(name):
        if name not in params.index:
            return (np.nan, np.nan, np.nan, np.nan)
        b = float(params[name])
        se = float(np.sqrt(max(cov.loc[name, name], 0.0)))
        t = float(b / se) if se > 0 else np.nan
        p = float(res.pvalues[name]) if name in res.pvalues.index else np.nan
        return (b, se, t, p)

    for (p, t) in treated_keys:
        # Build list of param names for each display k
        lead_names = [pname(p, t, k) for k in lead_range if k in ks]
        b_avg, se_avg = avg_of_coeffs(res, lead_names)
        t_avg = (b_avg / se_avg) if se_avg and se_avg > 0 else np.nan
        p_avg = np.nan  # we report leads joint p separately

        # Joint F-test (wald) that all lead coefficients = 0
        Fp = wald_pvalue(res, lead_names)

        rows.append({
            "port": p, "terminal": t, "k_label": "(-4..-2) avg",
            "beta": b_avg, "se": se_avg, "t": t_avg, "p": p_avg,
            "leads_F_p": Fp
        })

        for k_disp in [0,1,2,3,4]:
            if k_disp not in ks:
                rows.append({"port": p, "terminal": t, "k_label": str(k_disp),
                             "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                             "leads_F_p": Fp})
                continue
            name = pname(p, t, k_disp)
            b, se, tt, pp = get_beta_se(name)
            rows.append({"port": p, "terminal": t, "k_label": str(k_disp),
                         "beta": b, "se": se, "t": tt, "p": pp,
                         "leads_F_p": Fp})

    disp = pd.DataFrame(rows)
    return disp

# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run pooled event-study regressions for Tables 3 & 4 and print results.")
    ap.add_argument("--panel", type=str, default=None,
                    help="Path to terminal-quarter panel CSV/TSV with port, terminal, year, quarter, and LP/ln_lp.")
    ap.add_argument("--yaml", type=str, default=None,
                    help="Optional YAML config for clocks/k-grid/shocks.")
    ap.add_argument("--outdir", type=str, default="Design/Output Data/Tables3_4",
                    help="Where to save tidy outputs.")
    args = ap.parse_args()

    # Defaults (for safety if YAML not provided)
    default_cfg = {
        # Fallback panel (override this in --panel or YAML:data.panel_path)
        "panel_path": "Design/Output Data/05_es_input_terminal_sharedpre_nyt.csv",
        # Example clocks (fill with your actual go-live quarters as t_index = year*4 + (q-1))
        # You should override via YAML once ready.
        "clocks": {
            ("Haifa", "SIPG"):   2021*4 + 3,  # 2021Q4 example
            ("Haifa", "Legacy"): 2021*4 + 3,  # same clock reference
            ("Ashdod","HCT"):    2021*4 + 2,  # 2021Q3 example
            ("Ashdod","Legacy"): 2021*4 + 2,
        },
        "min_k": -8, "max_k": 4, "omit_k": -1,
        # Example shock windows; override in YAML
        "shocks": {
            "covid":   {"start_t": 2020*4 + 0, "end_t": 2021*4 + 3},
            "late2324":{"start_t": 2023*4 + 2, "end_t": 2024*4 + 1},
        }
    }

    # Load YAML if present
    cfg_loaded = {}
    if args.yaml:
        ypath = Path(args.yaml)
        cfg_loaded = load_yaml_config(ypath)
        log(f"Loaded YAML: {ypath}")

    # Merge defaults with loaded
    cfg = default_cfg.copy()
    cfg.update({k:v for k,v in cfg_loaded.items() if v is not None})

    # Panel path resolution
    panel_path = Path(args.panel) if args.panel else Path(cfg.get("panel_path") or default_cfg["panel_path"])
    if not panel_path.exists():
        log(f"WARNING: Panel not found at {panel_path}. If you have a different output from 05_prep_model1_terminal.py,"
            " pass it via --panel. Proceeding anyway may fail.")
    sep = infer_sep(panel_path)
    panel = pd.read_csv(panel_path, sep=sep)

    # Set up spec toggles
    specs = [
        ("baseline",     False, False),  # (name, add_port_trend, add_shocks)
        ("port_tr",      True,  False),
        ("tr_shocks",    True,  True),
    ]

    # Ensure output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run one pooled regression per spec
    for spec_name, add_tr, add_sh in specs:
        log(f"=== SPEC: {spec_name} ===")
        shocks = cfg["shocks"] if add_sh else None
        res, dsg = run_eventstudy(panel=panel,
                                  clocks=cfg["clocks"],
                                  min_k=cfg["min_k"], max_k=cfg["max_k"], omit_k=cfg["omit_k"],
                                  add_port_trend=add_tr, add_shocks=shocks)

        # Save full coefficient table for this spec
        full = res.params.rename("beta").to_frame()
        full["se"] = np.sqrt(np.diag(res.cov_params()))
        full["p"]  = res.pvalues
        full.to_csv(outdir / f"dyn_full_coeffs_{spec_name}.csv", index=True)

        # Prepare compact display (the 6 rows) for each group
        compact = extract_for_display(res, dsg)
        compact.to_csv(outdir / f"dyn_compact_display_{spec_name}.csv", index=False)

        # Pretty print to stdout, grouped by port/terminal
        for (port, term), group in compact.groupby(["port","terminal"]):
            log(f"-- {port} — {term} ({spec_name}) --")
            # Format columns
            sub = group[["k_label","beta","se","t","p","leads_F_p"]].copy()
            # Round a bit for display
            sub["beta"] = sub["beta"].astype(float).round(4)
            sub["se"]   = sub["se"].astype(float).round(4)
            sub["t"]    = sub["t"].astype(float).round(2)
            sub["p"]    = sub["p"].astype(float).round(3)
            sub["leads_F_p"] = sub["leads_F_p"].astype(float).round(3)
            print(sub.to_string(index=False))

    # Minimal run summary
    with open(outdir / "run_summary.txt", "w") as f:
        f.write("Wrote:\n")
        for spec_name, _, _ in specs:
            f.write(f"- dyn_full_coeffs_{spec_name}.csv\n")
            f.write(f"- dyn_compact_display_{spec_name}.csv\n")
        f.write("\nTip: Use these compact CSVs to feed your LaTeX Table 3 (Haifa) and Table 4 (Ashdod).\n")

if __name__ == "__main__":
    # Silence a few benign pandas warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

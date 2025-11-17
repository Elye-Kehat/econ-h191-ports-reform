#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare model-ready CSV for terminal-level ES regressions (NYT design)."""
import argparse, json, sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
PANEL_CANDIDATES = [
    Path("Design/Output Data/04_panel_terminal_sharedpre_log.csv"),
    Path("/Users/elyekehat/Downloads/Fall 2025/Econ H191/Thesis/Design/Output Data/04_panel_terminal_sharedpre_log.csv"),
    Path("Design/Output Data/panel_terminal_sharedpre_log.csv"),
    Path("/mnt/data/panel_terminal_sharedpre_log.csv"),
]

def find_input(cands):
    for p in cands:
        if p.exists():
            return p, p.parent
    sys.exit("Could not find panel_terminal_sharedpre_log.csv in expected locations. Aborting.")
def meta_counts(df: pd.DataFrame) -> Dict:
    return {"n_rows": int(len(df)), "n_ports": int(df["port"].nunique()) if "port" in df.columns else 0,
            "n_terminals": int(df["terminal"].nunique()) if "terminal" in df.columns else 0,
            "terminals": {t: int((df["terminal"]==t).sum()) for t in sorted(df["terminal"].unique().tolist())} if "terminal" in df.columns else {},
            "years": {"min": int(pd.to_numeric(df["year"], errors="coerce").dropna().min()) if ("year" in df.columns and df["year"].notna().any()) else None,
                      "max": int(pd.to_numeric(df["year"], errors="coerce").dropna().max()) if ("year" in df.columns and df["year"].notna().any()) else None},
            "has_tau_comp": bool(df["tau_comp"].notna().any()) if "tau_comp" in df.columns else False,
            "has_nyt_ok_comp": bool(df["nyt_ok_comp"].notna().any()) if "nyt_ok_comp" in df.columns else False}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, default=8)
    ap.add_argument("--lags", type=int, default=8)
    ap.add_argument("--nyt-only", type=int, default=1)
    args = ap.parse_args()
    inp, out_dir = find_input(PANEL_CANDIDATES)
    df = pd.read_csv(inp)
    need = {"port","terminal","qtr","t_index","Y","tau_comp"}
    miss = need - set(df.columns)
    if miss:
        sys.exit(f"Missing required columns in panel: {sorted(miss)}")
    c = df.copy(); c = c[c["Y"].notna() & np.isfinite(c["Y"])]; c = c[c["t_index"].notna()]; c = c[c["tau_comp"].notna()]
    leads = int(args.leads); lags = int(args.lags)
    c["tau_bin"] = c["tau_comp"].astype(int).clip(lower=-leads, upper=lags)
    c = c[c["tau_bin"] != -1]
    if "nyt_ok_comp" in c.columns and int(args.nyt_only) == 1:
        c = c[(c["tau_bin"] < 0) | (c["nyt_ok_comp"] == 1)]
    c = c.sort_values(["port","terminal","t_index","tau_bin"], kind="mergesort")
    out_csv = out_dir / "05_panel_terminal_sharedpre_model1.csv"
    c.to_csv(out_csv, index=False)
    meta = meta_counts(c)
    (out_dir / "05_meta_panel_terminal_sharedpre_model1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[write] {out_csv}")
    print(f"[write] {out_dir / "05_meta_panel_terminal_sharedpre_model1.json"}")
if __name__ == "__main__":
    main()

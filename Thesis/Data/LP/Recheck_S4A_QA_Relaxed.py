#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recheck_S4A_QA_Relaxed.py
---------------------------------
Purpose:
  Re-check S4A "annual preservation" for monthly port LP using relaxed tolerances.

Checks (per (port,year)):
  - rel_diff = |E[LP_mix] - E[Pi_port_q]| / E[Pi_port_q]  <= rel_tol  (default 0.02)
  OR
  - abs_diff = |E[LP_mix] - E[Pi_port_q]|                 <= abs_tol  (default 0.05)

Outputs:
  - S4A_qa_relaxed.tsv     (per-port-year results + summary rows)
  - _meta_recheck.json     (tolerances, input paths)

Usage (from project root):
  python "Data/LP/Recheck_S4A_QA_Relaxed.py"     --haifa "Data/LP/LP_Haifa_port_month.tsv"     --ashdod "Data/LP/LP_Ashdod_port_month.tsv"     --out_dir "Data/LP"     --rel_tol 0.02 --abs_tol 0.05
"""
import argparse, os, json
import numpy as np
import pandas as pd

TAB = '\t'

def _read(path):
    return pd.read_csv(path, sep=TAB)

def _write(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=TAB, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--haifa', default='Data/LP/LP_Haifa_port_month.tsv')
    ap.add_argument('--ashdod', default='Data/LP/LP_Ashdod_port_month.tsv')
    ap.add_argument('--out_dir', default='Data/LP')
    ap.add_argument('--rel_tol', type=float, default=0.02)
    ap.add_argument('--abs_tol', type=float, default=0.05)
    args = ap.parse_args()

    haifa = _read(args.haifa)
    ashdod = _read(args.ashdod)

    df = pd.concat([haifa, ashdod], ignore_index=True)

    # Required columns
    needed = {'port','year','LP_mix','Pi_port_q'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in monthly LP: {missing}")

    grp = df.groupby(['port','year'], as_index=False).agg(
        mean_LP=('LP_mix','mean'),
        mean_Pi=('Pi_port_q','mean')
    )
    grp['abs_diff'] = (grp['mean_LP'] - grp['mean_Pi']).abs()
    grp['rel_diff'] = grp['abs_diff'] / grp['mean_Pi'].replace(0, np.nan)
    grp['pass_rel'] = (grp['rel_diff'] <= args.rel_tol) | (grp['abs_diff'] <= args.abs_tol)

    # Summary rows
    summary = {
        'max_abs_diff': float(grp['abs_diff'].max() if len(grp) else np.nan),
        'max_rel_diff': float(grp['rel_diff'].max() if len(grp) else np.nan),
        'all_pass': bool(grp['pass_rel'].all()) if len(grp) else True,
        'rel_tol': args.rel_tol,
        'abs_tol': args.abs_tol
    }

    # Append a final summary row (with port='ALL', year=-1)
    tail = pd.DataFrame([{
        'port':'ALL','year':-1,
        'mean_LP':np.nan,'mean_Pi':np.nan,
        'abs_diff':summary['max_abs_diff'],
        'rel_diff':summary['max_rel_diff'],
        'pass_rel':summary['all_pass']
    }])

    out = pd.concat([grp, tail], ignore_index=True)

    qa_path = os.path.join(args.out_dir, 'S4A_qa_relaxed.tsv')
    _write(out, qa_path)

    meta = {
        'inputs': {'haifa': args.haifa, 'ashdod': args.ashdod},
        'tolerances': {'rel_tol': args.rel_tol, 'abs_tol': args.abs_tol},
        'summary': summary
    }
    with open(os.path.join(args.out_dir, '_meta_recheck.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"[Recheck] Wrote relaxed QA to {qa_path}")

if __name__ == '__main__':
    main()

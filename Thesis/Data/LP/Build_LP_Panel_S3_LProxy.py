#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build_LP_Panel_S3_LProxy.py — Stage 3: load & harmonize L_Proxy (terminal×month labor & Π)

Inputs:
  --lproxy           Data/L_proxy/L_Proxy.tsv
  --s2_term_quarter  Data/LP/S2_terminal_quarter_teu.tsv   (for coverage checks)
  --out              Output directory (e.g., Data/LP)

Writes:
  - S3_lproxy_clean.tsv         (terminal×month; canonical names & strict dtypes)
  - S3_port_month_labor.tsv     (port×month labor sums for LP_id diagnostic)
  - S3_terminal_year_pi.tsv     (terminal×year Π, unique and de-duplicated)
  - S3_coverage_gaps.tsv        (terminal-years used in S2 but missing Π and/or labor months)
  - S3_qa.tsv                   (checks & notes)
  - _meta_s3.json               (row counts, duplicates, variance stats)

Notes:
  * Canonical ports:   { 'Haifa', 'Ashdod' }
  * Canonical terminals: { 'Haifa-Bayport', 'Haifa-Legacy', 'Ashdod-HCT', 'Ashdod-Legacy' }
  * We normalize common variants (e.g., 'Haifa SIPG' → 'Haifa-Bayport', 'Ashdod HCT' → 'Ashdod-HCT').
  * month_index = year*12 + month ; quarter in {Q1..Q4}
  * Duplicates at (port,terminal,year,month) are aggregated: L_hours_i_m & TEU_i_m summed; share_i_p_q averaged.
  * Π per (terminal,year) is made unique (median if multiple distinct values); any variance is recorded in QA.
"""

import argparse, os, json, re
from typing import Dict, Tuple
import numpy as np
import pandas as pd

TAB = '\t'

# -------------------------- helpers --------------------------

def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=TAB, engine='python')

def _write_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=TAB, index=False)

def _to_int64(x):
    return pd.to_numeric(x, errors='coerce').astype('Int64')

def _quarter_from_month(m: int) -> str:
    q = (int(m) - 1) // 3 + 1
    return f"Q{q}"

# Terminal synonyms → canonical
TERMINAL_MAP = {
    'Haifa SIPG': 'Haifa-Bayport',
    'Haifa-Bayport': 'Haifa-Bayport',
    'Haifa Bayport': 'Haifa-Bayport',
    'Haifa-SIPG': 'Haifa-Bayport',
    'Ashdod HCT': 'Ashdod-HCT',
    'Ashdod-HCT': 'Ashdod-HCT',
    'Ashdod Hct': 'Ashdod-HCT',
    'Ashdod-legacy': 'Ashdod-Legacy',
    'Ashdod Legacy': 'Ashdod-Legacy',
    'Ashdod-Legacy': 'Ashdod-Legacy',
    'Haifa-legacy': 'Haifa-Legacy',
    'Haifa Legacy': 'Haifa-Legacy',
    'Haifa-Legacy': 'Haifa-Legacy',
}

TERMINAL_TO_PORT = {
    'Haifa-Bayport': 'Haifa',
    'Haifa-Legacy': 'Haifa',
    'Ashdod-HCT': 'Ashdod',
    'Ashdod-Legacy': 'Ashdod',
}

CANON_PORTS = {'Haifa','Ashdod'}
CANON_TERMS = set(TERMINAL_TO_PORT.keys())

# -------------------------- core -----------------------------

def load_lproxy_clean(lproxy_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = _read_tsv(lproxy_path).copy()

    # Required columns (minimum set)
    required = {'port','terminal','year','month','L_hours_i_m','Pi_teu_per_hour_i_y'}
    missing = required - set(raw.columns)
    if missing:
        raise SystemExit(f"[FATAL] L_Proxy missing columns: {sorted(missing)}")

    # Trim & normalize names
    for c in ['port','terminal']:
        raw[c] = raw[c].astype(str).str.strip()

    # Drop Eilat / others outside canon ports, then map terminals
    raw = raw[raw['port'].isin(CANON_PORTS)].copy()
    raw['terminal'] = raw['terminal'].map(lambda s: TERMINAL_MAP.get(s, s))
    # Infer port from terminal if inconsistent
    raw['port'] = np.where(raw['terminal'].isin(CANON_TERMS), raw['terminal'].map(TERMINAL_TO_PORT), raw['port'])

    # DTypes & dates
    raw['year'] = _to_int64(raw['year'])
    raw['month'] = _to_int64(raw['month'])
    raw = raw.dropna(subset=['year','month'])
    raw['month_index'] = _to_int64(raw['year']*12 + raw['month'])

    # Quarter handling (create if missing; coerce numeric → Qk if present)
    if 'quarter' in raw.columns:
        # If numeric-like (e.g., 2.0) or strings like '2.0'/'2'
        def _qnorm(v):
            try:
                f = float(v)
                i = int(f)
                if 1 <= i <= 4:
                    return f"Q{i}"
            except Exception:
                pass
            s = str(v).strip()
            m = re.match(r"^Q([1-4])$", s)
            return f"Q{m.group(1)}" if m else None
        raw['quarter'] = raw['quarter'].apply(_qnorm)
    else:
        raw['quarter'] = None
    raw.loc[raw['quarter'].isna(), 'quarter'] = raw.loc[raw['quarter'].isna(), 'month'].apply(_quarter_from_month)

    # Numeric columns
    num_cols = ['L_hours_i_m','Pi_teu_per_hour_i_y']
    if 'TEU_i_m' in raw.columns:
        num_cols.append('TEU_i_m')
    if 'share_i_p_q' in raw.columns:
        num_cols.append('share_i_p_q')
    for c in num_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')

    # Aggregate duplicates at (port,terminal,year,month)
    agg_dict = {'L_hours_i_m':'sum'}
    if 'TEU_i_m' in raw.columns:
        agg_dict['TEU_i_m'] = 'sum'
    if 'share_i_p_q' in raw.columns:
        agg_dict['share_i_p_q'] = 'mean'
    # Keep one representative Pi for monthly rows by later merge from terminal-year table
    base = raw.groupby(['port','terminal','year','month','month_index','quarter'], as_index=False).agg(agg_dict)

    # Build terminal-year Pi table and resolve variance
    pi_raw = raw[['terminal','year','Pi_teu_per_hour_i_y']].copy()
    pi_raw = pi_raw.dropna(subset=['terminal','year'])
    grp = pi_raw.groupby(['terminal','year'])
    # collect unique non-null values per group
    def _collapse_pi(g):
        vals = sorted(set([float(x) for x in g['Pi_teu_per_hour_i_y'].dropna().tolist()]))
        representative = float(np.median(vals)) if len(vals) else np.nan
        n_unique = len(vals)
        return pd.Series({'Pi_teu_per_hour_i_y': representative, 'Pi_unique_count': n_unique})
    pi_tbl = grp.apply(_collapse_pi).reset_index()

    # Attach Pi to monthly base rows
    lproxy_clean = base.merge(pi_tbl[['terminal','year','Pi_teu_per_hour_i_y']], on=['terminal','year'], how='left')

    # Port×month labor sums
    port_month_labor = (lproxy_clean.groupby(['port','year','month','month_index'], as_index=False)
                                      ['L_hours_i_m'].sum(min_count=1)
                                      .rename(columns={'L_hours_i_m':'L_hours_port_m'}))

    # QA & meta
    qa_rows = []
    def add(check, ok, note):
        qa_rows.append({'check':check, 'ok':bool(ok), 'note':note})

    # Uniqueness after aggregation
    add('unique_terminal_month', not lproxy_clean.duplicated(['port','terminal','year','month']).any(), 'terminal×month unique')
    add('unique_terminal_year_pi', not pi_tbl.duplicated(['terminal','year']).any(), 'terminal×year Π unique')

    # Variance counts in Pi within terminal-year groups
    pi_variance = int((pi_tbl['Pi_unique_count'] > 1).sum())
    add('pi_variance_groups', True, f"{pi_variance} terminal-year groups had >1 distinct Π; median used")

    # Basic coverage per terminal by year
    for t in sorted(lproxy_clean['terminal'].dropna().unique().tolist()):
        yrs = sorted([int(y) for y in lproxy_clean.loc[lproxy_clean['terminal']==t, 'year'].dropna().unique()])
        add('coverage_terminal_years', True, f"{t}: {yrs}")

    meta = {
        'rows': {
            'lproxy_clean': int(len(lproxy_clean)),
            'port_month_labor': int(len(port_month_labor)),
            'terminal_year_pi': int(len(pi_tbl)),
        },
        'pi_variance_groups': pi_variance,
    }

    return lproxy_clean, port_month_labor, pi_tbl, qa_rows, meta


def coverage_against_s2(lproxy_clean: pd.DataFrame, pi_tbl: pd.DataFrame, s2_term_quarter_path: str) -> pd.DataFrame:
    s2 = _read_tsv(s2_term_quarter_path).copy()
    req = {'port','terminal','year','quarter','TEU_i_q'}
    miss = req - set(s2.columns)
    if miss:
        raise SystemExit(f"[FATAL] S2_terminal_quarter_teu.tsv missing columns: {sorted(miss)}")

    # canonicalize terminal names if needed
    s2['terminal'] = s2['terminal'].map(lambda s: TERMINAL_MAP.get(s, s))

    # Unique terminal-years used in S2
    need = (s2.groupby(['port','terminal','year'], as_index=False)
              .agg(quarters_in_s2=('quarter','nunique')))

    # Months with labor observed in S3 per terminal-year
    labor_months = (lproxy_clean.groupby(['port','terminal','year'], as_index=False)
                                  .agg(months_with_labor=('month','nunique')))

    # Π availability per terminal-year
    has_pi = pi_tbl[['terminal','year']].copy()
    has_pi['has_pi'] = True

    cov = (need.merge(labor_months, on=['port','terminal','year'], how='left')
               .merge(has_pi, on=['terminal','year'], how='left'))
    cov['months_with_labor'] = cov['months_with_labor'].fillna(0).astype(int)
    cov['has_pi'] = cov['has_pi'].fillna(False)

    # Gaps: missing Π or no labor months for a terminal-year used in S2
    gaps = cov[(~cov['has_pi']) | (cov['months_with_labor'] == 0)].copy()

    return cov, gaps

# -------------------------- CLI -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lproxy', required=True, help='Path to Data/L_proxy/L_Proxy.tsv')
    ap.add_argument('--s2_term_quarter', required=True, help='Path to Data/LP/S2_terminal_quarter_teu.tsv')
    ap.add_argument('--out', required=True, help='Output directory (e.g., Data/LP)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    lproxy_clean, port_month_labor, pi_tbl, qa_rows, meta = load_lproxy_clean(args.lproxy)

    # Coverage vs S2 terminal quarters
    cov, gaps = coverage_against_s2(lproxy_clean, pi_tbl, args.s2_term_quarter)

    # Extend QA with coverage notes
    def add(check, ok, note):
        qa_rows.append({'check':check, 'ok':bool(ok), 'note':note})

    add('coverage_rows', True, f"{int(len(cov))} terminal-years in S2")
    add('coverage_gaps', len(gaps)==0, f"{int(len(gaps))} gaps (missing Π and/or labor months)")

    qa = pd.DataFrame(qa_rows)

    # Write artifacts
    _write_tsv(lproxy_clean,     os.path.join(args.out, 'S3_lproxy_clean.tsv'))
    _write_tsv(port_month_labor, os.path.join(args.out, 'S3_port_month_labor.tsv'))
    _write_tsv(pi_tbl,           os.path.join(args.out, 'S3_terminal_year_pi.tsv'))
    _write_tsv(cov,              os.path.join(args.out, 'S3_coverage_vs_s2.tsv'))
    _write_tsv(gaps,             os.path.join(args.out, 'S3_coverage_gaps.tsv'))
    _write_tsv(qa,               os.path.join(args.out, 'S3_qa.tsv'))

    meta.update({
        'rows': {
            **meta.get('rows', {}),
            'coverage_vs_s2': int(len(cov)),
            'coverage_gaps': int(len(gaps)),
        },
        'terminals_in_s3': sorted(lproxy_clean['terminal'].dropna().unique().tolist()),
    })

    with open(os.path.join(args.out, '_meta_s3.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('[S3] Wrote Stage 3 L_Proxy artifacts to', args.out)

if __name__ == '__main__':
    main()
    
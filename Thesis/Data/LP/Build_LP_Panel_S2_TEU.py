#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build_LP_Panel_S2_TEU.py — Stage 2: build TEU tables directly from the mixed‑frequency TEU file

Inputs:
  --teu   Data/Output/teu_monthly_plus_quarterly_by_port.tsv
  --out   Output directory (e.g., Data/LP). Stage files will be written as S2_*.tsv + _meta_s2.json

What this script writes:
  - S2_port_month_teu.tsv        (port, year, month, month_index, TEU_port_m, is_pre_reform)
  - S2_terminal_quarter_teu.tsv  (port, terminal, year, quarter, TEU_i_q)
  - S2_port_quarter_teu.tsv      (port, year, quarter, TEU_port_q)
  - S2_qa.tsv                    (checks & notes)
  - _meta_s2.json                (row counts, basic coverage)

Notes:
  * Uses TEU when present; falls back to TEU_thousands * 1000 if TEU is NA.
  * Drops AllPorts / All Ports and Eilat.
  * Quarterly TEU rows encode terminals using the 'Port' field; we map them to canonical terminal names.
  * We DO NOT infer quarters from monthlies. We parse exactly by Freq.
"""
import argparse, os, json, re
import numpy as np
import pandas as pd

# -------------------------- helpers --------------------------

def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', engine='python')

def _write_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)

def _to_int64(x):
    return pd.to_numeric(x, errors='coerce').astype('Int64')

def _month_from_period(period: str):
    """Parse 'MM-YYYY' -> (year, month). Returns (None, None) if fails."""
    try:
        dt = pd.to_datetime(period, format='%m-%Y', errors='coerce')
        if pd.isna(dt):
            return (None, None)
        return (int(dt.year), int(dt.month))
    except Exception:
        return (None, None)

def _ym_from_monthindex(mi):
    """Parse MonthIndex (YYYYMM int/str) -> (year, month). Returns (None, None) on failure."""
    try:
        s = str(int(mi))
        if len(s) != 6:
            return (None, None)
        y = int(s[:4]); m = int(s[4:])
        if m < 1 or m > 12:
            return (None, None)
        return (y, m)
    except Exception:
        return (None, None)

def _quarter_from_period(period: str):
    """Parse 'Qk-YYYY' into ('Qk', year). Accepts 'Qk YYYY' too."""
    m = re.match(r"^Q([1-4])[-\s]?([0-9]{4})$", str(period).strip())
    if not m:
        return (None, None)
    q = f"Q{m.group(1)}"; y = int(m.group(2))
    return (q, y)

# terminal mapping from TEU 'Port' to canonical terminal name
TEU_PORT_TO_TERMINAL = {
    'Haifa SIPG': 'Haifa-Bayport',
    'Ashdod HCT': 'Ashdod-HCT',
    'Haifa': 'Haifa-Legacy',
    'Ashdod': 'Ashdod-Legacy',
}

TERMINAL_TO_PORT = {
    'Haifa-Bayport': 'Haifa',
    'Ashdod-HCT': 'Ashdod',
    'Haifa-Legacy': 'Haifa',
    'Ashdod-Legacy': 'Ashdod',
}

# -------------------------- core -----------------------------

def build_teu_tables(teu_path: str):
    df = _read_tsv(teu_path)

    required = {'Port','Period','Freq','Year','MonthIndex','TEU_thousands','TEU'}
    miss = required - set(df.columns)
    if miss:
        raise SystemExit(f"[FATAL] TEU file missing columns: {sorted(miss)}")

    # Drop unwanted ports
    df['Port'] = df['Port'].astype(str).str.strip()
    df = df[~df['Port'].isin(['AllPorts','All Ports','Eilat'])].copy()

    # TEU numeric value
    df['TEU'] = pd.to_numeric(df['TEU'], errors='coerce')
    df['TEU_thousands'] = pd.to_numeric(df['TEU_thousands'], errors='coerce')
    df['teu_val'] = df['TEU']
    fallback = df['teu_val'].isna() & df['TEU_thousands'].notna()
    df.loc[fallback, 'teu_val'] = df.loc[fallback, 'TEU_thousands'] * 1000.0

    # Split monthly vs quarterly by Freq
    df['Freq'] = df['Freq'].astype(str).str.strip()

    # ---------------- monthly: ports only (Ashdod, Haifa) ----------------
    m = df[(df['Freq'].str.lower() == 'monthly') & (df['Port'].isin(['Ashdod','Haifa']))].copy()

    # Parse months: prefer Period; fallback to MonthIndex if Period fails
    ym = m['Period'].apply(_month_from_period)
    m['year'] = [y for (y, mm) in ym]
    m['month'] = [mm for (y, mm) in ym]

    # Fallback via MonthIndex for any unresolved
    unresolved = m['year'].isna() | m['month'].isna()
    if unresolved.any():
        ym2 = [ _ym_from_monthindex(v) for v in m.loc[unresolved, 'MonthIndex'] ]
        m.loc[unresolved, 'year']  = [y for (y, mm) in ym2]
        m.loc[unresolved, 'month'] = [mm for (y, mm) in ym2]

    # Coerce types
    m['year'] = _to_int64(m['year'])
    m['month'] = _to_int64(m['month'])
    m = m.dropna(subset=['year','month'])
    m['month_index'] = _to_int64(m['year']*12 + m['month'])

    # Group/sum in case of accidental duplicates
    port_month = (m.groupby(['Port','year','month','month_index'], as_index=False)
                    ['teu_val'].sum(min_count=1)
                    .rename(columns={'Port':'port','teu_val':'TEU_port_m'}))

    # is_pre_reform flag (<= 2021-08)
    port_month['is_pre_reform'] = ((port_month['year']*100 + port_month['month']) <= 202108)

    # ---------------- quarterly: terminals encoded in 'Port' ----------------
    q = df[(df['Freq'].str.lower() == 'quarterly') & (df['Port'].isin(list(TEU_PORT_TO_TERMINAL.keys())))].copy()

    # Map to terminals
    q['terminal'] = q['Port'].map(TEU_PORT_TO_TERMINAL)
    # Map to canonical port
    q['port'] = q['terminal'].map(TERMINAL_TO_PORT)

    # Parse quarters
    qy = q['Period'].apply(_quarter_from_period)
    q['quarter'] = [qq for (qq, yy) in qy]
    q['year']    = [yy for (qq, yy) in qy]

    # Coerce types & drop unresolved
    q['year'] = _to_int64(q['year'])
    q = q.dropna(subset=['year','quarter','terminal','port'])

    # Group to terminal-quarter
    term_quarter = (q.groupby(['port','terminal','year','quarter'], as_index=False)
                      ['teu_val'].sum(min_count=1)
                      .rename(columns={'teu_val':'TEU_i_q'}))

    # Port-quarter = sum of terminal quarterlies
    port_quarter = (term_quarter.groupby(['port','year','quarter'], as_index=False)
                                 ['TEU_i_q'].sum(min_count=1)
                                 .rename(columns={'TEU_i_q':'TEU_port_q'}))

    # ---------------- QA ----------------
    qa_rows = []
    def add(check, ok, note):
        qa_rows.append({'check':check, 'ok':bool(ok), 'note':note})

    # Key uniqueness
    add('unique_port_month', not port_month.duplicated(['port','year','month']).any(), 'port-month TEU unique')
    add('unique_term_quarter', not term_quarter.duplicated(['port','terminal','year','quarter']).any(), 'terminal-quarter TEU unique')
    add('unique_port_quarter', not port_quarter.duplicated(['port','year','quarter']).any(), 'port-quarter TEU unique')

    # Additivity check
    chk = (term_quarter.groupby(['port','year','quarter'], as_index=False)['TEU_i_q'].sum(min_count=1)
                      .merge(port_quarter, on=['port','year','quarter'], how='outer', indicator=True))
    if not chk.empty:
        chk['delta'] = (chk['TEU_i_q'] - chk['TEU_port_q']).abs()
        max_delta = float(chk['delta'].fillna(0).max())
        add('additivity_port_q=sum(term_q)', bool(max_delta < 1e-6), f'max abs diff = {max_delta:.3f}')
    else:
        add('additivity_port_q=sum(term_q)', True, 'no rows to compare')

    # Zero/negative counts (not fatal here; ratios will mask later)
    add('monthly_zero_or_neg_teu', True, f"{int((port_month['TEU_port_m']<=0).sum())} rows")
    add('quarter_zero_or_neg_teu', True, f"{int((term_quarter['TEU_i_q']<=0).sum())} term-quarters; {int((port_quarter['TEU_port_q']<=0).sum())} port-quarters")

    # Coverage snippets
    for p in sorted(port_month['port'].dropna().unique().tolist()):
        yrs = sorted([int(y) for y in port_month.loc[port_month['port']==p, 'year'].dropna().unique()])
        add('port_month_years', True, f"{p}: {yrs}")
    for t in sorted(term_quarter['terminal'].dropna().unique().tolist()):
        yrs = sorted([int(y) for y in term_quarter.loc[term_quarter['terminal']==t, 'year'].dropna().unique()])
        add('term_quarter_years', True, f"{t}: {yrs}")

    qa = pd.DataFrame(qa_rows)

    meta = {
        'rows': {
            'port_month': int(len(port_month)),
            'term_quarter': int(len(term_quarter)),
            'port_quarter': int(len(port_quarter)),
        },
        'ports_month': sorted(port_month['port'].dropna().unique().tolist()),
        'terminals_quarter': sorted(term_quarter['terminal'].dropna().unique().tolist()),
    }

    return port_month, term_quarter, port_quarter, qa, meta

# -------------------------- CLI -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teu', required=True, help='Path to teu_monthly_plus_quarterly_by_port.tsv')
    ap.add_argument('--out', required=True, help='Output directory (e.g., Data/LP)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    port_month, term_quarter, port_quarter, qa, meta = build_teu_tables(args.teu)

    _write_tsv(port_month,   os.path.join(args.out, 'S2_port_month_teu.tsv'))
    _write_tsv(term_quarter, os.path.join(args.out, 'S2_terminal_quarter_teu.tsv'))
    _write_tsv(port_quarter, os.path.join(args.out, 'S2_port_quarter_teu.tsv'))
    _write_tsv(qa,           os.path.join(args.out, 'S2_qa.tsv'))

    with open(os.path.join(args.out, '_meta_s2.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('[S2] Wrote Stage 2 TEU artifacts to', args.out)

if __name__ == '__main__':
    main()

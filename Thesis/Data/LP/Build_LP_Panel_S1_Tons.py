#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build_LP_Panel_S1_Tons.py — Stage 1: build tons tables directly from raw monthly tons file

Inputs:
  --tons  Data/Output/monthly_output_by_1000_tons_ports_and_terminals.tsv
  --out   Output directory (e.g., Data/LP). Stage files will be written as S1_*.tsv + _meta_s1.json

What this script does (stepwise):
  1) Load monthly tons (ports + terminals), drop Eilat & All Ports.
  2) Parse Month-Year -> (year, month), compute month_index, quarter labels.
  3) Build terminal-month tons for {Ashdod-HCT, Haifa-SIPG} with canonical names.
  4) Build port-month tons with precedence: sum of terminals if any exist for that port-month; else the single port row.
  5) Aggregate to port-quarter tons: sum(port-month tons) within each quarter.
  6) Write:
       - S1_terminal_month_tons.tsv      (port, terminal, year, month, month_index, tons_i_m)
       - S1_port_month_tons.tsv          (port, year, month, month_index, quarter, tons_port_m, tons_source)
       - S1_port_quarter_tons.tsv        (port, year, quarter, tons_port_q)
       - S1_examples_port_precedence.tsv (rows where both terminal-sum and port row exist; shows values)
       - S1_qa.tsv                       (QA checks)
       - _meta_s1.json                   (counts, ranges)

Notes:
  * Tons units: input is thousands of tons (tons_k). We multiply by 1000.
  * Canonical names:
      terminals: 'Ashdod-HCT' (raw 'Ashdod HCT'), 'Haifa-SIPG' (raw 'Haifa SIPG')
      ports: 'Ashdod', 'Haifa'
  * This stage does NOT touch TEU or L_Proxy. That happens in later stages.
"""
import argparse, os, json
from typing import Dict, Tuple
import numpy as np
import pandas as pd

# -------------------------- helpers --------------------------

def _read_tsv(path: str) -> pd.DataFrame:
    # IMPORTANT: use a real tab character for sep (not "\\t")
    return pd.read_csv(path, sep='\t', engine='python')

def _write_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # IMPORTANT: use a real tab character for sep (not "\\t")
    df.to_csv(path, sep='\t', index=False)

def _quarter_from_month(m: int) -> str:
    q = (int(m) - 1) // 3 + 1
    return f"Q{q}"

def _to_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').astype('Int64')

# -------------------------- main logic ------------------------

def load_tons_build_tables(tons_path: str):
    raw = _read_tsv(tons_path).copy()
    # Basic column presence check
    required_cols = {'PortOrTerminal','Month-Year','tons_k'}
    missing = required_cols - set(raw.columns)
    if missing:
        raise SystemExit(f"[FATAL] tons file missing columns: {sorted(missing)}")

    # Clean names and drop unwanted rows
    raw['PortOrTerminal'] = raw['PortOrTerminal'].astype(str).str.strip()
    raw = raw[~raw['PortOrTerminal'].isin(['All Ports','AllPorts','Eilat'])].copy()

    # Canonical mapping for terminals
    term_map = {
        'Ashdod HCT': 'Ashdod-HCT',
        'Haifa SIPG': 'Haifa-SIPG',
    }

    # Parse date
    # Input like '03-2020' -> month=3, year=2020
    dt = pd.to_datetime(raw['Month-Year'], format='%m-%Y', errors='coerce')
    if dt.isna().any():
        bad = raw.loc[dt.isna(), ['PortOrTerminal','Month-Year']].head(10)
        raise SystemExit(f"[FATAL] Failed to parse Month-Year for some rows, examples: {bad.to_dict(orient='records')}")
    raw['year'] = _to_int64(dt.dt.year)
    raw['month'] = _to_int64(dt.dt.month)
    raw['month_index'] = _to_int64(raw['year']*12 + raw['month'])
    raw['quarter'] = raw['month'].apply(_quarter_from_month)

    # Scale tons
    raw['tons'] = pd.to_numeric(raw['tons_k'], errors='coerce') * 1000.0

    # Split terminal vs port rows
    is_terminal = raw['PortOrTerminal'].isin(term_map.keys())
    term = raw.loc[is_terminal, ['PortOrTerminal','year','month','month_index','tons']].copy()
    term['terminal'] = term['PortOrTerminal'].map(term_map)
    # Map to canonical port per terminal
    term['port'] = np.where(term['terminal'].str.startswith('Ashdod'), 'Ashdod', 'Haifa')
    term = term[['port','terminal','year','month','month_index','tons']]
    term = term.rename(columns={'tons':'tons_i_m'})

    # Terminal-month: key uniqueness (aggregate if duplicates) and REBUILD month_index explicitly
    term = term.groupby(['port','terminal','year','month'], as_index=False)['tons_i_m'].sum(min_count=1)
    term['month_index'] = _to_int64(term['year']*12 + term['month'])

    # Port rows
    port_rows = raw.loc[~is_terminal & raw['PortOrTerminal'].isin(['Ashdod','Haifa']),
                        ['PortOrTerminal','year','month','month_index','quarter','tons']].copy()
    port_rows = port_rows.rename(columns={'PortOrTerminal':'port','tons':'tons_portrow_m'})

    # Sum terminals to port-month
    term_sum = term.groupby(['port','year','month'], as_index=False)['tons_i_m'] \
                   .sum(min_count=1).rename(columns={'tons_i_m':'tons_terminal_sum_m'})

    # Merge precedence (outer join, then compute month_index & quarter from keys to avoid missing)
    pm = port_rows.merge(term_sum, on=['port','year','month'], how='outer')

    # Recompute month_index/quarter from year & month (safe even if coming only from terminals)
    pm['year'] = _to_int64(pm['year'])
    pm['month'] = _to_int64(pm['month'])
    pm['month_index'] = _to_int64(pm['year']*12 + pm['month'])
    pm['quarter'] = pm['month'].apply(_quarter_from_month)

    # Decide tons_port_m and source
    pm['tons_port_m'] = np.where(pm['tons_terminal_sum_m'].notna(), pm['tons_terminal_sum_m'], pm['tons_portrow_m'])
    pm['tons_source'] = np.where(pm['tons_terminal_sum_m'].notna(), 'sum_terminals',
                                 np.where(pm['tons_portrow_m'].notna(), 'port_row', 'no_source'))

    # Examples table where both sources present (to audit differences)
    both = pm[pm[['tons_portrow_m','tons_terminal_sum_m']].notna().all(axis=1)].copy()
    if not both.empty:
        both['abs_diff'] = (both['tons_terminal_sum_m'] - both['tons_portrow_m']).abs()
        both['rel_diff'] = both['abs_diff'] / both['tons_portrow_m'].replace(0,np.nan)
        examples = both[['port','year','month','month_index','tons_portrow_m','tons_terminal_sum_m','abs_diff','rel_diff']]
    else:
        examples = pd.DataFrame(columns=['port','year','month','month_index','tons_portrow_m','tons_terminal_sum_m','abs_diff','rel_diff'])

    # Final port-month table
    port_month = pm[['port','year','month','month_index','quarter','tons_port_m','tons_source']].copy()
    port_month = port_month.sort_values(['port','year','month']).reset_index(drop=True)

    # Port-quarter aggregation
    port_quarter = (port_month.groupby(['port','year','quarter'], as_index=False)
                               ['tons_port_m'].sum(min_count=1)
                               .rename(columns={'tons_port_m':'tons_port_q'}))

    # QA info
    qa_rows = []
    def add(check, ok, note):
        qa_rows.append({'check':check, 'ok':bool(ok), 'note':note})

    # Key uniqueness checks
    add('unique_terminal_month', not term.duplicated(['port','terminal','year','month']).any(), 'terminal-month tons unique')
    add('unique_port_month', not port_month.duplicated(['port','year','month']).any(), 'port-month tons unique')
    add('unique_port_quarter', not port_quarter.duplicated(['port','year','quarter']).any(), 'port-quarter tons unique')

    # Source distribution
    src_dist = port_month['tons_source'].value_counts(dropna=False).to_dict()
    add('tons_source_dist', True, f"{src_dist}")

    # Missing counts
    add('port_month_missing_tons', True, f"{int(port_month['tons_port_m'].isna().sum())} NA months")

    # Ranges per port
    for p in sorted(port_month['port'].dropna().unique().tolist()):
        years = port_month.loc[port_month['port']==p, 'year'].dropna().unique()
        add('port_years', True, f"{p}: {sorted([int(y) for y in years])}")

    qa = pd.DataFrame(qa_rows)

    meta = {
        'rows': {
            'terminal_month': int(len(term)),
            'port_month': int(len(port_month)),
            'port_quarter': int(len(port_quarter)),
            'examples_both_sources': int(len(examples)),
        },
        'tons_source_dist': src_dist,
    }

    return term, port_month, port_quarter, examples, meta, qa

# -------------------------- CLI -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tons', required=True, help='Path to monthly_output_by_1000_tons_ports_and_terminals.tsv')
    ap.add_argument('--out', required=True, help='Output directory (e.g., Data/LP)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    term, port_month, port_quarter, examples, meta, qa = load_tons_build_tables(args.tons)

    _write_tsv(term,         os.path.join(args.out, 'S1_terminal_month_tons.tsv'))
    _write_tsv(port_month,   os.path.join(args.out, 'S1_port_month_tons.tsv'))
    _write_tsv(port_quarter, os.path.join(args.out, 'S1_port_quarter_tons.tsv'))
    _write_tsv(examples,     os.path.join(args.out, 'S1_examples_port_precedence.tsv'))
    _write_tsv(qa,           os.path.join(args.out, 'S1_qa.tsv'))

    with open(os.path.join(args.out, '_meta_s1.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('[S1] Wrote Stage 1 tons artifacts to', args.out)

if __name__ == '__main__':
    main()

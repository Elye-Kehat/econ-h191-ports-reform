#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build_LP_Panel_S1_Tons_v2.py

Why this V2 exists
------------------
The original Stage 1 code was written around a precedence rule that replaced the port-total
monthly tons row with the sum of terminal rows whenever any terminal rows were present.
That rule is reasonable only if terminal rows exist for *all* terminals in the port-month.
In this project, however, the raw monthly tons file directly names entrant terminals
(e.g. Haifa SIPG / Ashdod HCT), while the legacy terminal is not directly named as its own
terminal row in the raw tons input. As a consequence, using "sum of terminals if any exist"
can collapse port totals toward entrant-only tons after entry.

For Model 1A this is a problem because Stage 4 then builds post-reform LP dynamics from a
port-level tons/TEU ratio. If the port-quarter tons are effectively entrant-only, the shared
quarterly factor used for all terminals becomes badly distorted.

This V2 therefore changes Stage 1 in two important ways:

  1) Port totals are anchored primarily to the raw *port* rows.
     If a port row exists, it is treated as the canonical port total.
     Only if a port row is missing do we fall back to summing available terminal rows.

  2) We explicitly construct terminal tons for *all four* terminals after reform.
     Entrant terminal tons come from the observed terminal rows.
     Legacy terminal tons are constructed residually as:

         tons_legacy = tons_port_total - tons_entrant

     Before entrant activity begins, legacy tons equal the full port total.

Outputs
-------
This V2 keeps the familiar Stage 1 outputs used elsewhere in the pipeline and adds one more:

  - S1_terminal_month_tons.tsv      (ALL terminals, not entrants only)
  - S1_terminal_quarter_tons.tsv    (NEW: all terminals aggregated to quarter)
  - S1_port_month_tons.tsv
  - S1_port_quarter_tons.tsv
  - S1_examples_port_vs_entrant.tsv
  - S1_qa.tsv
  - _meta_s1_v2.json

The goal is to make post-reform terminal LP construction in Stage 4 rely on terminal-specific
quarterly tons rather than a shared port-quarter tons ratio.
"""

import argparse
import json
import os
from typing import Dict, Optional

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


def group_sum_nan(df: pd.DataFrame, group_cols, val_col: str, out_col: str) -> pd.DataFrame:
    """Grouped sum that returns NaN when all values are NaN in the group."""
    g = df.groupby(group_cols)[val_col]
    out = pd.DataFrame({'sum_val': g.sum(), 'cnt': g.count()}).reset_index()
    out[out_col] = np.where(out['cnt'] > 0, out['sum_val'], np.nan)
    return out[group_cols + [out_col]]


# Canonical naming used throughout LP pipeline
ENTRANT_RAW_TO_CANON = {
    'Haifa SIPG': 'Haifa-Bayport',
    'Ashdod HCT': 'Ashdod-HCT',
}

ENTRANT_TERM_BY_PORT = {
    'Haifa': 'Haifa-Bayport',
    'Ashdod': 'Ashdod-HCT',
}

LEGACY_TERM_BY_PORT = {
    'Haifa': 'Haifa-Legacy',
    'Ashdod': 'Ashdod-Legacy',
}

CANON_PORTS = {'Haifa', 'Ashdod'}


# -------------------------- core -----------------------------

def load_and_build_stage1_v2(tons_path: str):
    raw = _read_tsv(tons_path).copy()

    required = {'PortOrTerminal', 'Month-Year', 'tons_k'}
    missing = required - set(raw.columns)
    if missing:
        raise SystemExit(f"[FATAL] tons file missing columns: {sorted(missing)}")

    # Clean names and drop out-of-scope rows
    raw['PortOrTerminal'] = raw['PortOrTerminal'].astype(str).str.strip()
    raw = raw[~raw['PortOrTerminal'].isin(['All Ports', 'AllPorts', 'Eilat'])].copy()

    # Parse dates like 03-2020
    dt = pd.to_datetime(raw['Month-Year'], format='%m-%Y', errors='coerce')
    if dt.isna().any():
        bad = raw.loc[dt.isna(), ['PortOrTerminal', 'Month-Year']].head(10)
        raise SystemExit(f"[FATAL] Failed to parse Month-Year for some rows: {bad.to_dict(orient='records')}")

    raw['year'] = _to_int64(dt.dt.year)
    raw['month'] = _to_int64(dt.dt.month)
    raw['month_index'] = _to_int64(raw['year'] * 12 + raw['month'])
    raw['quarter'] = raw['month'].apply(_quarter_from_month)

    # Units: input is thousands of tons
    raw['tons'] = pd.to_numeric(raw['tons_k'], errors='coerce') * 1000.0

    # ---------------- entrants: observed terminal rows ----------------
    entrant_rows = raw[raw['PortOrTerminal'].isin(ENTRANT_RAW_TO_CANON.keys())].copy()
    entrant_rows['terminal'] = entrant_rows['PortOrTerminal'].map(ENTRANT_RAW_TO_CANON)
    entrant_rows['port'] = np.where(entrant_rows['terminal'].str.startswith('Haifa'), 'Haifa', 'Ashdod')

    entrant_month = (entrant_rows.groupby(['port', 'terminal', 'year', 'month', 'month_index', 'quarter'], as_index=False)
                                  ['tons'].sum())
    entrant_month = entrant_month.rename(columns={'tons': 'tons_i_m'})
    entrant_month['tons_source'] = 'observed_entrant_terminal'

    # ---------------- canonical port totals ----------------
    port_rows = raw[raw['PortOrTerminal'].isin(CANON_PORTS)].copy()
    port_rows = (port_rows.groupby(['PortOrTerminal', 'year', 'month', 'month_index', 'quarter'], as_index=False)
                          ['tons'].sum())
    port_rows = port_rows.rename(columns={'PortOrTerminal': 'port', 'tons': 'tons_portrow_m'})

    # Terminal sum is kept only as a fallback if the port row is absent.
    entrant_sum = group_sum_nan(entrant_month, ['port', 'year', 'month', 'month_index', 'quarter'], 'tons_i_m', 'tons_entrant_sum_m')

    pm = port_rows.merge(entrant_sum, on=['port', 'year', 'month', 'month_index', 'quarter'], how='outer')
    pm['tons_port_m'] = np.where(pm['tons_portrow_m'].notna(), pm['tons_portrow_m'], pm['tons_entrant_sum_m'])
    pm['tons_source'] = np.where(pm['tons_portrow_m'].notna(), 'port_row',
                          np.where(pm['tons_entrant_sum_m'].notna(), 'fallback_sum_available_terminals', 'no_source'))

    port_month = pm[['port', 'year', 'month', 'month_index', 'quarter', 'tons_port_m', 'tons_source', 'tons_portrow_m', 'tons_entrant_sum_m']].copy()
    port_month = port_month.sort_values(['port', 'year', 'month']).reset_index(drop=True)

    # ---------------- construct ALL terminal-month tons ----------------
    # Find first month with supported entrant tons for each port.
    entrant_starts: Dict[str, Optional[int]] = {}
    for p in sorted(CANON_PORTS):
        sub = entrant_month[(entrant_month['port'] == p) & entrant_month['tons_i_m'].notna() & (entrant_month['tons_i_m'] > 0)]
        entrant_starts[p] = int(sub['month_index'].min()) if not sub.empty else None

    terminal_month_parts = []
    qa_rows = []

    def add(check, ok, note):
        qa_rows.append({'check': check, 'ok': bool(ok), 'note': note})

    for p in sorted(CANON_PORTS):
        p_pm = port_month[port_month['port'] == p].copy()
        p_pm['port'] = p
        p_pm['legacy_terminal'] = LEGACY_TERM_BY_PORT[p]
        p_pm['entrant_terminal'] = ENTRANT_TERM_BY_PORT[p]
        entrant_start = entrant_starts[p]

        p_ent = entrant_month[entrant_month['port'] == p][['year', 'month', 'month_index', 'quarter', 'terminal', 'tons_i_m']].copy()
        if p_ent.empty:
            # no entrant rows at all: legacy gets full port total for all months
            legacy = p_pm[['port', 'legacy_terminal', 'year', 'month', 'month_index', 'quarter', 'tons_port_m']].copy()
            legacy = legacy.rename(columns={'legacy_terminal': 'terminal', 'tons_port_m': 'tons_i_m'})
            legacy['tons_source'] = 'full_port_pre_entry'
            terminal_month_parts.append(legacy)
            add(f'{p}_entrant_start', True, 'no entrant start observed')
            continue

        # Merge entrant tons onto the full port-month grid.
        p_ent2 = p_ent.rename(columns={'tons_i_m': 'tons_entrant_obs'})
        merged = p_pm.merge(p_ent2[['year', 'month', 'month_index', 'quarter', 'tons_entrant_obs']],
                            on=['year', 'month', 'month_index', 'quarter'], how='left')

        # Effective entrant tons:
        # - before entrant start: missing / not active
        # - from entrant start onward: observed tons if present, otherwise 0.0
        if entrant_start is None:
            merged['tons_entrant_eff'] = np.nan
            merged['entrant_active'] = False
        else:
            merged['entrant_active'] = merged['month_index'] >= entrant_start
            merged['tons_entrant_eff'] = np.where(merged['entrant_active'], merged['tons_entrant_obs'].fillna(0.0), np.nan)

        # Legacy residual tons. Before entrant activity, legacy gets full port total.
        merged['tons_legacy_raw'] = np.where(
            merged['entrant_active'].fillna(False),
            merged['tons_port_m'] - merged['tons_entrant_eff'].fillna(0.0),
            merged['tons_port_m']
        )

        # Negative residuals are impossible economically. We clip tiny negatives to 0 and flag them.
        neg_mask = merged['tons_legacy_raw'] < -1e-6
        tiny_neg_mask = (merged['tons_legacy_raw'] < 0) & ~neg_mask
        n_neg = int(neg_mask.sum())
        n_tiny_neg = int(tiny_neg_mask.sum())
        add(f'{p}_negative_legacy_residual_rows', n_neg == 0, f'n={n_neg}; rows with entrant > port total')
        add(f'{p}_tiny_negative_legacy_residual_rows', True, f'n={n_tiny_neg}')

        merged['tons_legacy_eff'] = merged['tons_legacy_raw']
        merged.loc[tiny_neg_mask | neg_mask, 'tons_legacy_eff'] = 0.0

        # Legacy terminal rows: all months
        legacy = merged[['port', 'year', 'month', 'month_index', 'quarter', 'tons_legacy_eff']].copy()
        legacy['terminal'] = LEGACY_TERM_BY_PORT[p]
        legacy['tons_i_m'] = legacy['tons_legacy_eff']
        legacy['tons_source'] = np.where(merged['entrant_active'].fillna(False), 'residual_port_minus_entrant', 'full_port_pre_entry')
        legacy = legacy[['port', 'terminal', 'year', 'month', 'month_index', 'quarter', 'tons_i_m', 'tons_source']]
        terminal_month_parts.append(legacy)

        # Entrant terminal rows: only months from entrant start onward
        if entrant_start is not None:
            entrant = merged[merged['entrant_active'].fillna(False)][['port', 'year', 'month', 'month_index', 'quarter', 'tons_entrant_eff', 'tons_entrant_obs']].copy()
            entrant['terminal'] = ENTRANT_TERM_BY_PORT[p]
            entrant['tons_i_m'] = entrant['tons_entrant_eff']
            entrant['tons_source'] = np.where(entrant['tons_entrant_obs'].notna(), 'observed_entrant_terminal', 'imputed_zero_after_entry')
            entrant = entrant[['port', 'terminal', 'year', 'month', 'month_index', 'quarter', 'tons_i_m', 'tons_source']]
            terminal_month_parts.append(entrant)
            add(f'{p}_entrant_start', True, f'month_index={entrant_start}')

    terminal_month = pd.concat(terminal_month_parts, ignore_index=True)
    terminal_month = (terminal_month.groupby(['port', 'terminal', 'year', 'month', 'month_index', 'quarter', 'tons_source'], as_index=False)
                                     ['tons_i_m'].sum())
    terminal_month = terminal_month.sort_values(['port', 'terminal', 'year', 'month']).reset_index(drop=True)

    # ---------------- quarterly aggregations ----------------
    port_quarter = group_sum_nan(port_month, ['port', 'year', 'quarter'], 'tons_port_m', 'tons_port_q')
    terminal_quarter = group_sum_nan(terminal_month, ['port', 'terminal', 'year', 'quarter'], 'tons_i_m', 'tons_i_q')

    # ---------------- audit / examples ----------------
    examples = port_month[port_month['tons_portrow_m'].notna() & port_month['tons_entrant_sum_m'].notna()].copy()
    if not examples.empty:
        examples['abs_diff_port_vs_entrant_sum'] = (examples['tons_portrow_m'] - examples['tons_entrant_sum_m']).abs()
        examples['rel_diff_port_vs_entrant_sum'] = examples['abs_diff_port_vs_entrant_sum'] / examples['tons_portrow_m'].replace(0, np.nan)
        examples = examples[['port', 'year', 'month', 'month_index', 'tons_portrow_m', 'tons_entrant_sum_m',
                             'abs_diff_port_vs_entrant_sum', 'rel_diff_port_vs_entrant_sum']]
    else:
        examples = pd.DataFrame(columns=['port', 'year', 'month', 'month_index', 'tons_portrow_m', 'tons_entrant_sum_m',
                                         'abs_diff_port_vs_entrant_sum', 'rel_diff_port_vs_entrant_sum'])

    # ---------------- QA ----------------
    add('unique_port_month', not port_month.duplicated(['port', 'year', 'month']).any(), 'port-month totals unique')
    add('unique_port_quarter', not port_quarter.duplicated(['port', 'year', 'quarter']).any(), 'port-quarter totals unique')
    add('unique_terminal_month', not terminal_month.duplicated(['port', 'terminal', 'year', 'month']).any(), 'all-terminal month totals unique')
    add('unique_terminal_quarter', not terminal_quarter.duplicated(['port', 'terminal', 'year', 'quarter']).any(), 'all-terminal quarter totals unique')
    add('port_month_missing_tons', True, f"{int(port_month['tons_port_m'].isna().sum())} NA months")

    # How well do quarterly terminal totals add back up to the port total?
    chk = (terminal_quarter.groupby(['port', 'year', 'quarter'], as_index=False)['tons_i_q'].sum()
                          .rename(columns={'tons_i_q': 'sum_terminal_q'})
                          .merge(port_quarter, on=['port', 'year', 'quarter'], how='outer'))
    if not chk.empty:
        chk['delta'] = (chk['sum_terminal_q'] - chk['tons_port_q']).abs()
        max_delta = float(chk['delta'].fillna(0).max())
        add('terminal_quarter_adds_to_port_quarter', bool(max_delta < 1e-6), f'max abs diff = {max_delta:.3f}')

    for p in sorted(CANON_PORTS):
        yrs = sorted([int(y) for y in port_month.loc[port_month['port'] == p, 'year'].dropna().unique()])
        add('port_years', True, f'{p}: {yrs}')

    qa = pd.DataFrame(qa_rows)
    meta = {
        'rows': {
            'entrant_terminal_month_observed': int(len(entrant_month)),
            'terminal_month_all': int(len(terminal_month)),
            'terminal_quarter_all': int(len(terminal_quarter)),
            'port_month': int(len(port_month)),
            'port_quarter': int(len(port_quarter)),
            'examples_port_vs_entrant': int(len(examples)),
        },
        'entrant_starts': entrant_starts,
        'port_source_dist': port_month['tons_source'].value_counts(dropna=False).to_dict(),
        'terminal_source_dist': terminal_month['tons_source'].value_counts(dropna=False).to_dict(),
    }

    return terminal_month, terminal_quarter, port_month, port_quarter, examples, qa, meta


# -------------------------- CLI ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tons', required=True, help='Path to monthly_output_by_1000_tons_ports_and_terminals.tsv')
    ap.add_argument('--out', required=True, help='Output directory (e.g. Data/LP)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    terminal_month, terminal_quarter, port_month, port_quarter, examples, qa, meta = load_and_build_stage1_v2(args.tons)

    _write_tsv(terminal_month,   os.path.join(args.out, 'S1_terminal_month_tons.tsv'))
    _write_tsv(terminal_quarter, os.path.join(args.out, 'S1_terminal_quarter_tons.tsv'))
    _write_tsv(port_month,       os.path.join(args.out, 'S1_port_month_tons.tsv'))
    _write_tsv(port_quarter,     os.path.join(args.out, 'S1_port_quarter_tons.tsv'))
    _write_tsv(examples,         os.path.join(args.out, 'S1_examples_port_vs_entrant.tsv'))
    _write_tsv(qa,               os.path.join(args.out, 'S1_qa.tsv'))

    with open(os.path.join(args.out, '_meta_s1_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('[S1_v2] Wrote Stage 1 tons artifacts to', args.out)


if __name__ == '__main__':
    main()

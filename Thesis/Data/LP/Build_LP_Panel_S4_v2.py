#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build_LP_Panel_S4_v2.py

Why this V2 exists
------------------
The original Stage 4 built post-reform terminal LP by first constructing a *port-quarter*
ratio

    r_{p,q} = tons_port_q / TEU_port_q

then winsorizing/rebasing that into a shared quarterly factor w_{p,q}, and finally setting

    LP_{i,q} = w_{p,q} * Pi_{i,y}

for every terminal i in port p.

That construction makes terminals within the same port inherit the same quarterly movement,
with only the annual terminal-level Pi separating them. For Model 1A this can create terminal
paths that are too proportional / mirror-like, which is exactly the kind of structure that
fixed-effects event studies struggle with.

This V2 keeps the pre-reform monthly port LP logic largely intact, but rewrites the post-reform
terminal LP logic so that quarterly dynamics are terminal-specific:

    r_{i,q} = tons_{i,q} / TEU_{i,q}
    w_{i,q} = winsorize_and_rebase( r_{i,q} within terminal-year )
    LP_{i,q} = w_{i,q} * Pi_{i,y}

Key changes vs the original Stage 4
-----------------------------------
1) Pre-reform monthly port LP is still built at the port-month level.
2) Post-reform terminal LP now uses terminal-quarter tons and terminal-quarter TEU.
3) The winsor/rebase step for post-reform LP is now done within (terminal, year), not (port, year).
4) If terminal-specific quarterly tons are unavailable, the code falls back to the old port-quarter
   dynamic factor *only for those affected rows* and flags that fallback in QA.

The goal is to preserve the annual Pi anchors while giving post-reform terminals genuinely
terminal-specific quarterly movement.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

TAB = '\t'

# -------------------------- helpers --------------------------

def _read(path):
    return pd.read_csv(path, sep=TAB, engine='python')


def _write(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=TAB, index=False)


def _to_int64(x):
    return pd.to_numeric(x, errors='coerce').astype('Int64')


def _q_from_m(m: int) -> str:
    return f"Q{(int(m) - 1) // 3 + 1}"


def _qcode(qstr: str) -> int:
    m = re.match(r'^\s*Q([1-4])\s*$', str(qstr))
    return int(m.group(1)) if m else np.nan


def group_sum_nan(df: pd.DataFrame, group_cols, val_col: str, out_col: str) -> pd.DataFrame:
    """Grouped sum that returns NaN when all values are NaN in the group."""
    g = df.groupby(group_cols)[val_col]
    agg = pd.DataFrame({'sum_val': g.sum(), 'cnt': g.count()}).reset_index()
    agg[out_col] = np.where(agg['cnt'] > 0, agg['sum_val'], np.nan)
    return agg[group_cols + [out_col]]


def winsor_rebase(df: pd.DataFrame, key_cols, r_col: str, low=0.01, high=0.99, w_col='w') -> pd.DataFrame:
    """
    Winsorize r_col within groups and rebase so the groupwise mean of w_col is exactly 1.
    This preserves annual Pi anchors once LP is computed as LP = w * Pi.
    """
    out = df.copy()
    qs = (out.groupby(key_cols)[r_col]
            .quantile([low, high])
            .unstack(level=-1)
            .rename(columns={low: 'q_low', high: 'q_high'}))
    out = out.merge(qs, left_on=key_cols, right_index=True, how='left')

    out['r_clip'] = out[r_col]
    mask = out['r_clip'].notna() & out['q_low'].notna() & out['q_high'].notna()
    out.loc[mask, 'r_clip'] = out.loc[mask, r_col].clip(out.loc[mask, 'q_low'], out.loc[mask, 'q_high'])

    means = out.groupby(key_cols)['r_clip'].mean().rename('r_mean')
    out = out.merge(means, left_on=key_cols, right_index=True, how='left')
    out[w_col] = np.where((out['r_clip'] > 0) & (out['r_mean'] > 0), out['r_clip'] / out['r_mean'], np.nan)

    # second-pass exact normalization
    w_mean = out.groupby(key_cols)[w_col].mean().rename('w_mean')
    out = out.merge(w_mean, left_on=key_cols, right_index=True, how='left')
    ok = out[w_col].notna() & (out['w_mean'] > 0)
    out.loc[ok, w_col] = out.loc[ok, w_col] / out.loc[ok, 'w_mean']

    out.drop(columns=['q_low', 'q_high', 'r_clip', 'r_mean', 'w_mean'], inplace=True, errors='ignore')
    return out


def build_port_quarter_shares(lproxy_clean: pd.DataFrame, s2_tq: pd.DataFrame):
    """
    This helper remains for the pre-reform monthly port LP object.
    It constructs quarter-constant port-quarter terminal shares used to aggregate annual Pi to
    a port-level Pi baseline. This is fine for pre-reform port LP because the object is a port,
    not a terminal comparison.
    """
    a = lproxy_clean.copy()
    has_share = 'share_i_p_q' in a.columns
    cols = ['port', 'terminal', 'year', 'quarter']
    pieces = []
    if has_share:
        share_mean = (a.groupby(cols, as_index=False)['share_i_p_q'].mean()
                        .rename(columns={'share_i_p_q': 'share_from_lproxy'}))
        pieces.append(share_mean)
    if 'TEU_i_m' in a.columns:
        teu_q = group_sum_nan(a, cols, 'TEU_i_m', 'teu_from_lproxy_q')
        pieces.append(teu_q)

    from_lproxy = None
    if pieces:
        from_lproxy = pieces[0]
        for p in pieces[1:]:
            from_lproxy = from_lproxy.merge(p, on=cols, how='outer')

    b = s2_tq[['port', 'terminal', 'year', 'quarter', 'TEU_i_q']].copy()
    m = from_lproxy.merge(b, on=cols, how='outer') if from_lproxy is not None else b.copy()

    def _resolve(g):
        g = g.copy()
        if 'share_from_lproxy' in g.columns and g['share_from_lproxy'].notna().any():
            s = g['share_from_lproxy'].fillna(0.0)
            total = s.sum()
            if total > 0:
                g['share'] = s / total
                return g
        if 'teu_from_lproxy_q' in g.columns and g['teu_from_lproxy_q'].notna().any():
            s = g['teu_from_lproxy_q'].fillna(0.0)
            total = s.sum()
            if total > 0:
                g['share'] = s / total
                return g
        if 'TEU_i_q' in g.columns and g['TEU_i_q'].notna().any():
            s = g['TEU_i_q'].fillna(0.0)
            total = s.sum()
            if total > 0:
                g['share'] = s / total
                return g
        g['share'] = np.nan
        return g

    shares = m.groupby(['port', 'year', 'quarter'], group_keys=False).apply(_resolve)
    shares = shares[['port', 'year', 'quarter', 'terminal', 'share']].drop_duplicates()
    return shares


# -------------------------- stage 4A -------------------------

def build_monthly_port_lp(args):
    tons_m = _read(args.s1_port_month_tons)
    teu_m = _read(args.s2_port_month_teu)
    labor_p_m = _read(args.s3_port_month_labor)
    lproxy = _read(args.s3_lproxy_clean)
    pi_tbl = _read(args.s3_term_year_pi)

    teu_m['ym'] = teu_m['year'] * 100 + teu_m['month']
    teu_m = teu_m[(teu_m['ym'] >= args.monthly_start) & (teu_m['ym'] <= args.monthly_end)].copy()

    pm = tons_m.merge(teu_m[['port', 'year', 'month', 'month_index', 'TEU_port_m']],
                      on=['port', 'year', 'month', 'month_index'], how='inner')

    pm['r'] = np.where((pm['tons_port_m'] > 0) & (pm['TEU_port_m'] > 0), pm['tons_port_m'] / pm['TEU_port_m'], np.nan)
    pm['year'] = _to_int64(pm['year'])
    pm['month'] = _to_int64(pm['month'])
    pm['month_index'] = _to_int64(pm['month_index'])
    pm['quarter'] = pm['month'].apply(_q_from_m)

    pm_w = winsor_rebase(pm, ['port', 'year'], 'r', args.winsor_low, args.winsor_high, w_col='w')
    pm_w['w_source'] = 'monthly_port_specific'

    # Pre-reform monthly port LP still uses a port-level Pi baseline.
    s2_tq = _read(args.s2_term_quarter_teu)
    shares = build_port_quarter_shares(lproxy, s2_tq)
    pi_tbl = pi_tbl[['terminal', 'year', 'Pi_teu_per_hour_i_y']].copy()
    sh_pi = shares.merge(pi_tbl, on=['terminal', 'year'], how='left')
    sh_pi['prod'] = sh_pi['share'] * sh_pi['Pi_teu_per_hour_i_y']
    pi_q = group_sum_nan(sh_pi, ['port', 'year', 'quarter'], 'prod', 'Pi_port_q')

    pm_w = pm_w.merge(pi_q, on=['port', 'year', 'quarter'], how='left')
    pm_w['LP_mix'] = pm_w['w'] * pm_w['Pi_port_q']

    # Identity diagnostic only
    pm_w = pm_w.merge(labor_p_m, on=['port', 'year', 'month', 'month_index'], how='left')
    pm_w['LP_id'] = np.where((pm_w['TEU_port_m'] > 0) & (pm_w['L_hours_port_m'] > 0), pm_w['TEU_port_m'] / pm_w['L_hours_port_m'], np.nan)

    keep = ['port', 'year', 'month', 'month_index', 'quarter', 'TEU_port_m', 'tons_port_m', 'tons_source',
            'w', 'w_source', 'Pi_port_q', 'LP_mix', 'LP_id', 'L_hours_port_m']
    out = pm_w[keep].sort_values(['port', 'year', 'month']).reset_index(drop=True)

    _write(out[out['port'] == 'Haifa'].copy(),  os.path.join(args.out, 'LP_Haifa_port_month.tsv'))
    _write(out[out['port'] == 'Ashdod'].copy(), os.path.join(args.out, 'LP_Ashdod_port_month.tsv'))

    qa_rows = []
    def add(check, ok, note):
        qa_rows.append({'check': check, 'ok': bool(ok), 'note': note})

    for p in ['Haifa', 'Ashdod']:
        sub = out[out['port'] == p].copy()
        add(f'unique_{p}_port_month', not sub.duplicated(['port', 'year', 'month']).any(), f'n={len(sub)}')
        if not sub.empty:
            dev = float((sub.groupby(['port', 'year'])['w'].mean() - 1.0).abs().max())
            add(f'mean_w_{p}_port_year≈1', bool(dev < 1e-9), f'max |mean(w)-1|={dev:.3g}')
            j = sub.groupby(['port', 'year'])[['LP_mix', 'Pi_port_q']].mean().dropna()
            if not j.empty:
                diff = float((j['LP_mix'] - j['Pi_port_q']).abs().max())
                add(f'annual_preservation_{p}', bool(diff < 1e-6), f'max |E[LP]-E[Pi]|={diff:.3g}')
        add(f'NA_LP_months_{p}', True, f"{int(sub['LP_mix'].isna().sum())} NA of {len(sub)}")

    return pd.DataFrame(qa_rows)


# -------------------------- stage 4B -------------------------

def build_quarterly_terminal_lp(args):
    """
    Main V2 change:
    Build terminal-quarter LP using terminal-quarter tons and terminal-quarter TEU.
    Fallback to the old port-quarter dynamic factor only when terminal-quarter r cannot be formed.
    """
    term_tons_q = _read(args.s1_terminal_quarter_tons)
    port_tons_q = _read(args.s1_port_quarter_tons)
    teu_tq = _read(args.s2_term_quarter_teu)
    teu_pq = _read(args.s2_port_quarter_teu)
    pi_tbl = _read(args.s3_term_year_pi)[['terminal', 'year', 'Pi_teu_per_hour_i_y']].copy()

    # Terminal-quarter panel
    tq = term_tons_q.merge(teu_tq, on=['port', 'terminal', 'year', 'quarter'], how='outer')
    tq['r_terminal_q'] = np.where((tq['tons_i_q'] > 0) & (tq['TEU_i_q'] > 0), tq['tons_i_q'] / tq['TEU_i_q'], np.nan)

    # Terminal-specific winsor/rebase by (terminal, year)
    tq['year'] = _to_int64(tq['year'])
    tq_w = winsor_rebase(tq, ['terminal', 'year'], 'r_terminal_q', args.winsor_low, args.winsor_high, w_col='w_terminal')

    # Old-style port-wide fallback dynamic factor, retained only as a safeguard.
    pq = port_tons_q.merge(teu_pq, on=['port', 'year', 'quarter'], how='inner')
    pq['r_port_q'] = np.where((pq['tons_port_q'] > 0) & (pq['TEU_port_q'] > 0), pq['tons_port_q'] / pq['TEU_port_q'], np.nan)
    pq['year'] = _to_int64(pq['year'])
    pq_w = winsor_rebase(pq, ['port', 'year'], 'r_port_q', args.winsor_low, args.winsor_high, w_col='w_port_fallback')
    pq_w = pq_w[['port', 'year', 'quarter', 'w_port_fallback']].copy()

    # Merge Pi and fallback
    out = (tq_w.merge(pi_tbl, on=['terminal', 'year'], how='left')
               .merge(pq_w, on=['port', 'year', 'quarter'], how='left'))

    # Use terminal-specific w where available; fallback only if needed.
    use_terminal = out['w_terminal'].notna()
    out['w'] = np.where(use_terminal, out['w_terminal'], out['w_port_fallback'])
    out['w_source'] = np.where(use_terminal, 'terminal_quarter_specific', 'fallback_port_quarter')
    out['LP_mix'] = out['w'] * out['Pi_teu_per_hour_i_y']

    # Window filter
    def encode_yq(y, qstr):
        return int(y) * 10 + int(_qcode(qstr))

    qs = int(args.quarterly_start[:4]) * 10 + int(args.quarterly_start[-1])
    qe = int(args.quarterly_end[:4]) * 10 + int(args.quarterly_end[-1])
    out['yq'] = [encode_yq(y, q) for y, q in zip(out['year'], out['quarter'])]
    out = out[(out['yq'] >= qs) & (out['yq'] <= qe)].copy()

    keep = ['port', 'terminal', 'year', 'quarter', 'tons_i_q', 'TEU_i_q', 'r_terminal_q',
            'w', 'w_source', 'Pi_teu_per_hour_i_y', 'LP_mix']
    out = out[keep].sort_values(['port', 'terminal', 'year', 'quarter']).reset_index(drop=True)

    # Write per-terminal files using the same filenames expected by Stage 5.
    def write_term(term_name, file_name):
        df = out[out['terminal'] == term_name].copy()
        _write(df, os.path.join(args.out, file_name))
        return df

    df_hl = write_term('Haifa-Legacy',  'LP_Haifa_Legacy_quarter.tsv')
    df_hb = write_term('Haifa-Bayport', 'LP_Haifa_SIPG_quarter.tsv')
    df_al = write_term('Ashdod-Legacy', 'LP_Ashdod_Legacy_quarter.tsv')
    df_ah = write_term('Ashdod-HCT',    'LP_Ashdod_HCT_quarter.tsv')

    # QA focused on the V2 changes.
    qa_rows = []
    def add(check, ok, note):
        qa_rows.append({'check': check, 'ok': bool(ok), 'note': note})

    add('unique_terminal_quarter', not out.duplicated(['port', 'terminal', 'year', 'quarter']).any(), f'n={len(out)}')

    # Check that terminal-specific rebase preserves annual Pi when used directly.
    terminal_only = out[out['w_source'] == 'terminal_quarter_specific'].copy()
    if not terminal_only.empty:
        dev = float((terminal_only.groupby(['terminal', 'year'])['w'].mean() - 1.0).abs().max())
        add('mean_w_terminal_year≈1', bool(dev < 1e-9), f'max |mean(w)-1|={dev:.3g}')
        j = terminal_only.groupby(['terminal', 'year'])[['LP_mix', 'Pi_teu_per_hour_i_y']].mean().dropna()
        if not j.empty:
            diff = float((j['LP_mix'] - j['Pi_teu_per_hour_i_y']).abs().max())
            add('annual_preservation_terminal_year', bool(diff < 1e-6), f'max |E[LP]-Pi|={diff:.3g}')

    fallback_n = int((out['w_source'] == 'fallback_port_quarter').sum())
    add('fallback_port_quarter_rows', True, f'n={fallback_n}')

    # Port-quarter additivity for terminal tons
    chk = (group_sum_nan(out, ['port', 'year', 'quarter'], 'tons_i_q', 'sum_terminal_q')
             .merge(port_tons_q, on=['port', 'year', 'quarter'], how='left'))
    if not chk.empty:
        chk['delta'] = (chk['sum_terminal_q'] - chk['tons_port_q']).abs()
        max_delta = float(chk['delta'].fillna(0).max())
        add('terminal_tons_add_to_port_tons', bool(max_delta < 1e-6), f'max abs diff = {max_delta:.3f}')

    for term, df in [('Haifa-Legacy', df_hl), ('Haifa-Bayport', df_hb), ('Ashdod-Legacy', df_al), ('Ashdod-HCT', df_ah)]:
        add(f'NA_LP_quarters_{term}', True, f"{int(df['LP_mix'].isna().sum())} NA of {len(df)}")
        add(f'w_source_dist_{term}', True, str(df['w_source'].value_counts(dropna=False).to_dict()))

    return pd.DataFrame(qa_rows)


# -------------------------- CLI ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s1_port_month_tons', default='Data/LP/S1_port_month_tons.tsv')
    ap.add_argument('--s1_port_quarter_tons', default='Data/LP/S1_port_quarter_tons.tsv')
    ap.add_argument('--s1_terminal_quarter_tons', default='Data/LP/S1_terminal_quarter_tons.tsv')
    ap.add_argument('--s2_port_month_teu', default='Data/LP/S2_port_month_teu.tsv')
    ap.add_argument('--s2_term_quarter_teu', default='Data/LP/S2_terminal_quarter_teu.tsv')
    ap.add_argument('--s2_port_quarter_teu', default='Data/LP/S2_port_quarter_teu.tsv')
    ap.add_argument('--s3_lproxy_clean', default='Data/LP/S3_lproxy_clean.tsv')
    ap.add_argument('--s3_port_month_labor', default='Data/LP/S3_port_month_labor.tsv')
    ap.add_argument('--s3_term_year_pi', default='Data/LP/S3_terminal_year_pi.tsv')
    ap.add_argument('--out', default='Data/LP')
    ap.add_argument('--winsor_low', type=float, default=0.01)
    ap.add_argument('--winsor_high', type=float, default=0.99)
    ap.add_argument('--monthly_start', type=int, default=201801)   # YYYYMM
    ap.add_argument('--monthly_end', type=int, default=202110)     # YYYYMM inclusive
    ap.add_argument('--quarterly_start', type=str, default='2021Q3')
    ap.add_argument('--quarterly_end', type=str, default='2024Q4')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    qaA = build_monthly_port_lp(args)
    _write(qaA, os.path.join(args.out, 'S4A_qa.tsv'))

    qaB = build_quarterly_terminal_lp(args)
    _write(qaB, os.path.join(args.out, 'S4B_qa.tsv'))

    qa = pd.concat([qaA.assign(stage='S4A'), qaB.assign(stage='S4B')], ignore_index=True)
    _write(qa, os.path.join(args.out, 'qa_lp_report.tsv'))

    meta = {
        'params': {
            'winsor_low': args.winsor_low,
            'winsor_high': args.winsor_high,
            'monthly_start': args.monthly_start,
            'monthly_end': args.monthly_end,
            'quarterly_start': args.quarterly_start,
            'quarterly_end': args.quarterly_end,
        },
        'v2_changes': [
            'Pre-reform monthly port LP kept at port-month frequency.',
            'Post-reform terminal LP rebuilt using terminal-quarter tons / terminal-quarter TEU.',
            'Quarterly winsor/rebase now done within (terminal, year), not (port, year).',
            'Port-quarter dynamic factor retained only as a fallback when terminal-specific r cannot be formed.'
        ]
    }
    with open(os.path.join(args.out, '_meta_s4_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('[S4_v2] Wrote LP series and QA to', args.out)


if __name__ == '__main__':
    main()

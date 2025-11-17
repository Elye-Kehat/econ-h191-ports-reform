#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build_LP_Panel_S4.py — Stage 4: compute LP series from S1/S2/S3 artifacts

This script produces the six LP series (two monthly ports pre‑reform, four quarterly terminals post‑reform)
plus QA tables. It follows the direct‑from‑raw build plan.

Patch notes (compat + robustness):
- Removed uses of `min_count=` in GroupBy reductions to support wider pandas versions.
- Added safe group-sum helper that returns NaN when all inputs are NaN (instead of 0).
- **NEW:** In winsor_rebase(), added a second-pass normalization to enforce groupwise mean(w)=1
  exactly within each (port,year). This fixes edge cases like 2021-Q3/Q4 where re-centering
  could be off due to upstream idiosyncrasies.
"""
import argparse, os, json, re
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
    return f"Q{(int(m)-1)//3 + 1}"

def _qcode(qstr: str) -> int:
    m = re.match(r'^\s*Q([1-4])\s*$', str(qstr))
    return int(m.group(1)) if m else np.nan

def group_sum_nan(df: pd.DataFrame, group_cols, val_col: str, out_col: str) -> pd.DataFrame:
    """
    Grouped sum that yields NaN when all values are NaN in a group,
    mimicking sum(min_count=1) but compatible with older pandas.
    """
    g = df.groupby(group_cols)[val_col]
    agg = pd.DataFrame({'sum_val': g.sum(), 'cnt': g.count()}).reset_index()
    agg[out_col] = np.where(agg['cnt'] > 0, agg['sum_val'], np.nan)
    return agg[group_cols + [out_col]]

# winsorize & rebase per (port,year)

def winsor_rebase(df: pd.DataFrame, key_cols, r_col: str, low=0.01, high=0.99, out_prefix='', enforce_unit_mean=True):
    """
    Winsorize r_col within groups, then rebase to w = r_clip / mean(r_clip).
    If enforce_unit_mean=True, perform a second-pass normalization to guarantee
    mean(w)==1 within each group (exactly), guarding against edge cases.
    """
    out = df.copy()
    # Quantiles per group
    qs = (out.groupby(key_cols)[r_col]
             .quantile([low, high])
             .unstack(level=-1)
             .rename(columns={low:'q_low', high:'q_high'}))
    out = out.merge(qs, left_on=key_cols, right_index=True, how='left')

    # Clip to quantiles
    out['r_clip'] = out[r_col]
    mask = out['r_clip'].notna() & out['q_low'].notna() & out['q_high'].notna()
    out.loc[mask, 'r_clip'] = out.loc[mask, r_col].clip(out.loc[mask, 'q_low'], out.loc[mask, 'q_high'])

    # Mean of clipped r per group
    means = (out.groupby(key_cols)['r_clip']
               .mean()
               .rename('r_mean'))
    out = out.merge(means, left_on=key_cols, right_index=True, how='left')

    # Initial rebase
    w_col = f'{out_prefix}w' if out_prefix else 'w'
    out[w_col] = np.where((out['r_clip']>0) & (out['r_mean']>0), out['r_clip']/out['r_mean'], np.nan)
    out[f'{out_prefix}r_winsor' if out_prefix else 'r_winsor'] = out['r_clip']

    # Second-pass exact normalization (guard)
    if enforce_unit_mean:
        w_mean = (out.groupby(key_cols)[w_col]
                    .mean()
                    .rename('w_mean'))
        out = out.merge(w_mean, left_on=key_cols, right_index=True, how='left')
        ok = (out['w_mean'] > 0) & out[w_col].notna()
        out.loc[ok, w_col] = out.loc[ok, w_col] / out.loc[ok, 'w_mean']
        out.drop(columns=['w_mean'], inplace=True, errors='ignore')

    out.drop(columns=['q_low','q_high','r_clip','r_mean'], inplace=True, errors='ignore')
    return out

def build_port_quarter_shares(lproxy_clean: pd.DataFrame, s2_tq: pd.DataFrame):
    """Return shares per (port,year,quarter,terminal) using precedence rules.
    Columns returned: port, year, quarter, terminal, share
    """
    a = lproxy_clean.copy()
    has_share = 'share_i_p_q' in a.columns
    cols = ['port','terminal','year','quarter']
    pieces = []
    if has_share:
        share_mean = (a.groupby(cols, as_index=False)['share_i_p_q'].mean()
                        .rename(columns={'share_i_p_q':'share_from_lproxy'}))
        pieces.append(share_mean)
    if 'TEU_i_m' in a.columns:
        teu_q = group_sum_nan(a, cols, 'TEU_i_m', 'teu_from_lproxy_q')
        pieces.append(teu_q)
    from_lproxy = None
    if pieces:
        from_lproxy = pieces[0]
        for p in pieces[1:]:
            from_lproxy = from_lproxy.merge(p, on=cols, how='outer')

    b = s2_tq[['port','terminal','year','quarter','TEU_i_q']].copy()
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
            g['share'] = np.where(total>0, s/total, np.nan)
            if total > 0:
                return g
        if 'TEU_i_q' in g.columns and g['TEU_i_q'].notna().any():
            s = g['TEU_i_q'].fillna(0.0)
            total = s.sum()
            g['share'] = np.where(total>0, s/total, np.nan)
            return g
        g['share'] = np.nan
        return g

    shares = m.groupby(['port','year','quarter'], group_keys=False).apply(_resolve)
    shares = shares[['port','year','quarter','terminal','share']].drop_duplicates()
    return shares

# -------------------------- main build ------------------------

def build_monthly_port_lp(args):
    tons_m = _read(args.s1_port_month_tons)
    teu_m = _read(args.s2_port_month_teu)
    labor_p_m = _read(args.s3_port_month_labor)
    lproxy = _read(args.s3_lproxy_clean)
    pi_tbl = _read(args.s3_term_year_pi)

    # Restrict to pre-reform monthly window
    teu_m['ym'] = teu_m['year']*100 + teu_m['month']
    teu_m = teu_m[(teu_m['ym'] >= args.monthly_start) & (teu_m['ym'] <= args.monthly_end)]

    pm = (tons_m.merge(teu_m[['port','year','month','month_index','TEU_port_m']],
                       on=['port','year','month','month_index'], how='inner'))

    # Monthly w
    pm['r'] = np.where((pm['tons_port_m']>0) & (pm['TEU_port_m']>0), pm['tons_port_m']/pm['TEU_port_m'], np.nan)
    pm['year'] = _to_int64(pm['year']); pm['month'] = _to_int64(pm['month']); pm['month_index'] = _to_int64(pm['month_index'])
    pm_w = winsor_rebase(pm, ['port','year'], 'r', args.winsor_low, args.winsor_high, enforce_unit_mean=True)
    pm_w['w_source'] = 'monthly'
    pm_w['quarter'] = pm_w['month'].apply(_q_from_m)

    # Π mix baseline (port-month via quarter-constant shares)
    s2_tq = _read(args.s2_term_quarter_teu)
    shares = build_port_quarter_shares(lproxy, s2_tq)

    pi_tbl = pi_tbl[['terminal','year','Pi_teu_per_hour_i_y']].copy()
    sh_pi = shares.merge(pi_tbl, on=['terminal','year'], how='left')
    sh_pi['prod'] = sh_pi['share'] * sh_pi['Pi_teu_per_hour_i_y']
    pi_q = group_sum_nan(sh_pi, ['port','year','quarter'], 'prod', 'Pi_port_q')

    pm_w = pm_w.merge(pi_q, on=['port','year','quarter'], how='left')
    pm_w['LP_mix'] = pm_w['w'] * pm_w['Pi_port_q']

    # Identity diagnostic
    pm_w = pm_w.merge(labor_p_m, on=['port','year','month','month_index'], how='left')
    pm_w['LP_id'] = np.where((pm_w['TEU_port_m']>0) & (pm_w['L_hours_port_m']>0), pm_w['TEU_port_m']/pm_w['L_hours_port_m'], np.nan)

    keep = ['port','year','month','month_index','quarter','TEU_port_m','tons_port_m','tons_source','w','w_source','Pi_port_q','LP_mix','LP_id']
    out = pm_w[keep].sort_values(['port','year','month'])

    out_h = out[out['port']=='Haifa'].copy()
    out_a = out[out['port']=='Ashdod'].copy()

    _write(out_h, os.path.join(args.out, 'LP_Haifa_port_month.tsv'))
    _write(out_a, os.path.join(args.out, 'LP_Ashdod_port_month.tsv'))

    # QA A
    qa_rows = []
    def add(check, ok, note): qa_rows.append({'check':check, 'ok':bool(ok), 'note':note})

    for p in ['Haifa','Ashdod']:
        tmp = out[out['port']==p]
        ok_u = not tmp.duplicated(['port','year','month']).any()
        add(f'unique_{p}_port_month', ok_u, f"n={len(tmp)}")
        g = tmp.groupby(['port','year'])['w'].mean()
        dev = (g - 1.0).abs().max() if len(g) else 0.0
        add(f'mean_w_{p}_per_year≈1', bool((dev<1e-9) or np.isnan(dev)), f'max |mean(w)-1|={float(dev):.3g}')
        gLP = tmp.groupby(['port','year'])['LP_mix'].mean()
        gPi = tmp.groupby(['port','year'])['Pi_port_q'].mean()
        j = pd.concat([gLP, gPi], axis=1).dropna()
        if not j.empty:
            diff = (j['LP_mix'] - j['Pi_port_q']).abs().max()
            add(f'annual_preservation_{p}', bool(diff < 1e-6), f'max |E[LP]-E[Pi]|={float(diff):.3g}')
        add(f'NA_LP_months_{p}', True, f"{int(tmp['LP_mix'].isna().sum())} NA of {len(tmp)}")

    qaA = pd.DataFrame(qa_rows)
    return qaA

def build_quarterly_terminal_lp(args):
    # Inputs
    tons_q = _read(args.s1_port_quarter_tons)
    teu_qp = _read(args.s2_port_quarter_teu)
    teu_tq = _read(args.s2_term_quarter_teu)
    pi_tbl = _read(args.s3_term_year_pi)

    # Port-quarter ratio and w (winsorize + rebase + enforce mean=1 by (port,year))
    pq = tons_q.merge(teu_qp, on=['port','year','quarter'], how='inner')
    pq['r'] = np.where((pq['tons_port_q']>0) & (pq['TEU_port_q']>0), pq['tons_port_q']/pq['TEU_port_q'], np.nan)
    pq['year'] = _to_int64(pq['year'])
    pq_w = winsor_rebase(pq, ['port','year'], 'r', args.winsor_low, args.winsor_high, enforce_unit_mean=True)
    pq_w['w_source'] = 'quarterly'

    # Join terminal TEU and π, compute LP
    t = teu_tq.merge(pq_w[['port','year','quarter','w']], on=['port','year','quarter'], how='left')
    t = t.merge(pi_tbl[['terminal','year','Pi_teu_per_hour_i_y']], on=['terminal','year'], how='left')
    t['LP_mix'] = t['w'] * t['Pi_teu_per_hour_i_y']
    

    # FINAL ENFORCEMENT: ensure mean(w)==1 by (port,year)
    g = t.groupby(['port','year'])['w'].transform('mean')
    t['w'] = np.where(g > 0, t['w'] / g, np.nan)   # guarantees mean(w)==1 in each (port,year)
    # Recompute LP with the final w
    t['LP_mix'] = t['w'] * t['Pi_teu_per_hour_i_y']

    # Window filter
    def encode_q(y, qstr):
        return int(y)*10 + _qcode(qstr)
    qs = int(args.quarterly_start[0:4])*10 + int(args.quarterly_start[-1])
    qe = int(args.quarterly_end[0:4])*10 + int(args.quarterly_end[-1])
    t['yq'] = [encode_q(y, q) for y, q in zip(t['year'], t['quarter'])]
    t = t[(t['yq']>=qs) & (t['yq']<=qe)]

    keep = ['port','terminal','year','quarter','TEU_i_q','w','Pi_teu_per_hour_i_y','LP_mix']
    t = t[keep].sort_values(['port','terminal','year','quarter'])

    # Write per-terminal series
    def write_term(term_name, file_name):
        df = t[t['terminal']==term_name].copy()
        _write(df, os.path.join(args.out, file_name))
        return df

    write_term('Haifa-Legacy', 'LP_Haifa_Legacy_quarter.tsv')
    write_term('Haifa-Bayport','LP_Haifa_SIPG_quarter.tsv')
    write_term('Ashdod-Legacy','LP_Ashdod_Legacy_quarter.tsv')
    write_term('Ashdod-HCT',   'LP_Ashdod_HCT_quarter.tsv')

    # QA B
    qa_rows = []
    def add(check, ok, note): qa_rows.append({'check':check, 'ok':bool(ok), 'note':note})

    ok_u = not t.duplicated(['port','terminal','year','quarter']).any()
    add('unique_terminal_quarter', ok_u, f"n={len(t)}")
    g = pq_w.groupby(['port','year'])['w'].mean()
    dev = (g - 1.0).abs().max() if len(g) else 0.0
    add('mean_w_port_year≈1', bool((dev<1e-9) or np.isnan(dev)), f'max |mean(w)-1|={float(dev):.3g}')
    for term in ['Haifa-Legacy','Haifa-Bayport','Ashdod-Legacy','Ashdod-HCT']:
        sub = t[t['terminal']==term]
        add(f'NA_LP_quarters_{term}', True, f"{int(sub['LP_mix'].isna().sum())} NA of {len(sub)}")

    qaB = pd.DataFrame(qa_rows)
    return qaB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s1_port_month_tons',   default='Data/LP/S1_port_month_tons.tsv')
    ap.add_argument('--s1_port_quarter_tons', default='Data/LP/S1_port_quarter_tons.tsv')
    ap.add_argument('--s2_port_month_teu',    default='Data/LP/S2_port_month_teu.tsv')
    ap.add_argument('--s2_term_quarter_teu',  default='Data/LP/S2_terminal_quarter_teu.tsv')
    ap.add_argument('--s2_port_quarter_teu',  default='Data/LP/S2_port_quarter_teu.tsv')
    ap.add_argument('--s3_lproxy_clean',      default='Data/LP/S3_lproxy_clean.tsv')
    ap.add_argument('--s3_port_month_labor',  default='Data/LP/S3_port_month_labor.tsv')
    ap.add_argument('--s3_term_year_pi',      default='Data/LP/S3_terminal_year_pi.tsv')
    ap.add_argument('--out',                  default='Data/LP')
    ap.add_argument('--winsor_low',  type=float, default=0.01)
    ap.add_argument('--winsor_high', type=float, default=0.99)
    ap.add_argument('--monthly_start', type=int, default=201801)  # YYYYMM
    ap.add_argument('--monthly_end',   type=int, default=202110)  # YYYYMM inclusive
    ap.add_argument('--quarterly_start', type=str, default='2021Q3')
    ap.add_argument('--quarterly_end',   type=str, default='2024Q4')

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    qaA = build_monthly_port_lp(args)
    _write(qaA, os.path.join(args.out, 'S4A_qa.tsv'))

    qaB = build_quarterly_terminal_lp(args)
    _write(qaB, os.path.join(args.out, 'S4B_qa.tsv'))

    qaA['stage'] = 'S4A'; qaB['stage'] = 'S4B'
    qa = pd.concat([qaA, qaB], ignore_index=True)
    _write(qa, os.path.join(args.out, 'qa_lp_report.tsv'))

    meta = {
        'params': {
            'winsor_low': args.winsor_low,
            'winsor_high': args.winsor_high,
            'monthly_start': args.monthly_start,
            'monthly_end': args.monthly_end,
            'quarterly_start': args.quarterly_start,
            'quarterly_end': args.quarterly_end,
        }
    }
    with open(os.path.join(args.out, '_meta_s4.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('[S4] Wrote LP series and QA to', args.out)

if __name__ == '__main__':
    main()
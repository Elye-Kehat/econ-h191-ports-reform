#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_lp_panel.py  (v2) — LP panel builder with robust column alignment & QA

Inputs (from normalize_supply_inputs):
  norm_dir/
    - teu_port_month.tsv        (port,year,month,month_index,teu_p_m)
    - teu_port_quarter.tsv      (port,year,quarter,teu_p_q)
    - tons_port_month.tsv       (port,year,month,month_index,tons_p_m,tons_source)
    - l_proxy.tsv               (port,terminal,year,month,month_index,quarter,l_hours_i_m,teu_i_m,pi_teu_per_hour_i_y,operating)

Outputs (to --out_dir):
  - LP_port_month_mixadjusted.tsv
  - LP_port_month_identity.tsv
  - LP_terminal_month_mixadjusted.tsv
  - LP_terminal_quarter_mixadjusted.tsv
  - LP_panel_mixedfreq.tsv
  - qa_lp_report.tsv
  - _meta_lp_build.json

Changes from v1:
  * Fix KeyError on 'r_winsor' by merging r_winsor into terminal panel and
    aligning final columns via reindex (double protection).
  * Remove pandas FutureWarning by replacing groupby.apply with transform/agg.
  * Add tons_source to port panel.
  * Safer merges and explicit key checks to avoid avoidable KeyErrors.
"""

import argparse, os, json
from typing import Tuple
import numpy as np
import pandas as pd

# ----------------------------- helpers -----------------------------

def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', engine='python')

def _write_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)

def _quarter_from_month(m: int) -> str:
    if pd.isna(m): return None
    q = (int(m) - 1)//3 + 1
    return f"Q{q}"

def _assert_unique(df: pd.DataFrame, keys: list, name: str):
    dupe = df.duplicated(keys)
    if dupe.any():
        raise SystemExit(f"[FATAL] {name} has duplicate keys on {keys}. Examples: " + df.loc[dupe, keys].head(5).to_json(orient='records'))

def _to_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').astype('Int64')

def _winsorize_series(x: pd.Series, low: float, high: float) -> pd.Series:
    x = x.astype('float64')
    if x.notna().sum() == 0:
        return x
    lo = x.quantile(low)
    hi = x.quantile(high)
    return x.clip(lower=lo, upper=hi)

# ----------------------------- loading -----------------------------

def load_inputs(norm_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teu_m = _read_tsv(os.path.join(norm_dir, 'teu_port_month.tsv'))
    teu_q = _read_tsv(os.path.join(norm_dir, 'teu_port_quarter.tsv'))
    tons  = _read_tsv(os.path.join(norm_dir, 'tons_port_month.tsv'))
    lpr   = _read_tsv(os.path.join(norm_dir, 'l_proxy.tsv'))

    # enforce dtypes and basic sanity
    for df, nm in [(teu_m,'teu_port_month'), (tons,'tons_port_month'), (lpr,'l_proxy')]:
        df['year'] = _to_int64(df['year'])
        df['month'] = _to_int64(df['month'])
        if 'month_index' in df.columns:
            df['month_index'] = _to_int64(df['month_index'])
        if not df['month'].between(1,12).all():
            bad = df.loc[~df['month'].between(1,12), ['port','year','month']].head(5)
            raise SystemExit(f"[FATAL] {nm} has month outside 1..12. Examples: {bad.to_dict(orient='records')}")
        if 'month_index' in df.columns:
            comp = (df['year']*12 + df['month']).astype('Int64')
            if not comp.equals(df['month_index'].astype('Int64')):
                raise SystemExit(f"[FATAL] {nm} month_index mismatch with year*12+month")

    if not teu_q.empty:
        if set(teu_q['quarter'].dropna().unique()) - {"Q1","Q2","Q3","Q4"}:
            raise SystemExit("[FATAL] teu_port_quarter has invalid quarter labels")

    # key uniqueness
    if not teu_m.empty:
        _assert_unique(teu_m, ['port','year','month'], 'teu_port_month')
    if not teu_q.empty:
        _assert_unique(teu_q, ['port','year','quarter'], 'teu_port_quarter')
    _assert_unique(tons,  ['port','year','month'], 'tons_port_month')
    _assert_unique(lpr,   ['port','terminal','year','month'], 'l_proxy')

    return teu_m, teu_q, tons, lpr

# ----------------------------- w builder -----------------------------

def build_w(teu_m: pd.DataFrame, teu_q: pd.DataFrame, tons: pd.DataFrame,
            winsor_low: float, winsor_high: float) -> pd.DataFrame:
    """Return month-level table with w (winsorized & rebased), r (tons/teu), and w_source."""
    # month universe from tons
    month_univ = tons[['port','year','month','month_index']].drop_duplicates().copy()

    # monthly ratio where monthly TEU exist
    r_m = month_univ.merge(teu_m[['port','year','month','teu_p_m']], on=['port','year','month'], how='left') \
                    .merge(tons[['port','year','month','tons_p_m','tons_source']], on=['port','year','month'], how='left')
    r_m['r_monthly'] = np.where((r_m['teu_p_m']>0) & (r_m['tons_p_m'].notna()), r_m['tons_p_m']/r_m['teu_p_m'], np.nan)

    # quarterly fallback: r_{p,q} = sum(tons_p_m)/teu_p_q then broadcast to months
    if teu_q.empty:
        r_m['r_quarterly'] = np.nan
    else:
        t_q = tons.copy()
        t_q['quarter'] = t_q['month'].apply(_quarter_from_month)
        t_sum = t_q.groupby(['port','year','quarter'], as_index=False)['tons_p_m'].sum(min_count=1)
        rq = t_sum.merge(teu_q, on=['port','year','quarter'], how='left')
        rq['r_quarterly_val'] = np.where((rq['teu_p_q']>0) & (rq['tons_p_m'].notna()), rq['tons_p_m']/rq['teu_p_q'], np.nan)
        rq_expanded = month_univ.copy()
        rq_expanded['quarter'] = rq_expanded['month'].apply(_quarter_from_month)
        rq_expanded = rq_expanded.merge(rq[['port','year','quarter','r_quarterly_val']], on=['port','year','quarter'], how='left')
        r_m = r_m.merge(rq_expanded[['port','year','month','r_quarterly_val']], on=['port','year','month'], how='left')
        r_m.rename(columns={'r_quarterly_val':'r_quarterly'}, inplace=True)

    # choose r: prefer monthly else quarterly
    r_m['r'] = r_m['r_monthly']
    r_m.loc[r_m['r'].isna(), 'r'] = r_m.loc[r_m['r'].isna(), 'r_quarterly']
    r_m['w_source'] = np.where(r_m['r_monthly'].notna(), 'monthly', np.where(r_m['r_quarterly'].notna(), 'quarterly', 'na'))

    # winsorize + rebase by (port,year) using transform (no FutureWarning)
    r_m['r_winsor'] = r_m.groupby(['port','year'])['r'].transform(lambda s: _winsorize_series(s, winsor_low, winsor_high))
    mu = r_m.groupby(['port','year'])['r_winsor'].transform('mean')
    r_m['w'] = np.where((mu.isna()) | (mu==0), 1.0, r_m['r_winsor']/mu)

    # keep useful columns
    out = r_m[['port','year','month','month_index','r','r_winsor','w','w_source','teu_p_m','tons_p_m','tons_source']].copy()
    return out

# ----------------------------- Pi builder -----------------------------

def build_pi_mixbase(lpr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Pi_port_month_mixbase, terminal_shares_port_quarter)."""
    df = lpr.copy()
    df['teu_i_m_pos'] = np.where(df['teu_i_m']>0, df['teu_i_m'], 0.0)
    if 'quarter' not in df.columns:
        df['quarter'] = df['month'].apply(_quarter_from_month)
    g = df.groupby(['port','terminal','year','quarter'], as_index=False)['teu_i_m_pos'].sum(min_count=1)
    tot = g.groupby(['port','year','quarter'], as_index=False)['teu_i_m_pos'].sum(min_count=1).rename(columns={'teu_i_m_pos':'teu_sum_pq'})
    shares = g.merge(tot, on=['port','year','quarter'], how='left')
    shares['share_i_pq'] = np.where((shares['teu_sum_pq']>0) & (shares['teu_i_m_pos'].notna()), shares['teu_i_m_pos']/shares['teu_sum_pq'], 0.0)

    pi_i_y = (lpr[['port','terminal','year','pi_teu_per_hour_i_y']].drop_duplicates())
    months = lpr[['port','year','month','month_index']].drop_duplicates().copy()
    months['quarter'] = months['month'].apply(_quarter_from_month)

    sh = shares.merge(pi_i_y, on=['port','terminal','year'], how='left')
    sh['prod'] = sh['share_i_pq'] * sh['pi_teu_per_hour_i_y']
    pi_q = sh.groupby(['port','year','quarter'], as_index=False)['prod'].sum(min_count=1).rename(columns={'prod':'Pi_mixbase_pq'})

    pi_m = months.merge(pi_q, on=['port','year','quarter'], how='left')
    pi_m.rename(columns={'Pi_mixbase_pq':'Pi_mixbase_p_m'}, inplace=True)
    return pi_m, shares[['port','terminal','year','quarter','share_i_pq']]

# ----------------------------- LP assemblers -----------------------------

def build_port_tables(wtab: pd.DataFrame, pi_m: pd.DataFrame, lpr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    L_port = lpr.groupby(['port','year','month','month_index'], as_index=False)['l_hours_i_m'].sum(min_count=1).rename(columns={'l_hours_i_m':'L_port_m'})

    port = (wtab.merge(pi_m[['port','year','month','month_index','Pi_mixbase_p_m']],
                       on=['port','year','month','month_index'], how='left')
                .merge(L_port, on=['port','year','month','month_index'], how='left'))
    port['LP_mix'] = port['w'] * port['Pi_mixbase_p_m']
    port['LP_id']  = np.where((port.get('teu_p_m',np.nan)>0) & (port['L_port_m']>0), port['teu_p_m']/port['L_port_m'], np.nan)

    port_id = port[['port','year','month','month_index','teu_p_m','L_port_m','LP_id']].copy().sort_values(['port','year','month']).reset_index(drop=True)

    port_mix = port[['port','year','month','month_index','r','r_winsor','w','w_source','teu_p_m','tons_p_m','tons_source','Pi_mixbase_p_m','LP_mix']].copy()
    port_mix = port_mix.sort_values(['port','year','month']).reset_index(drop=True)
    return port_mix, port_id


def build_terminal_tables(wtab: pd.DataFrame, lpr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # bring w and r_winsor, w_source to terminal months
    tt = lpr.merge(wtab[['port','year','month','w','r_winsor','w_source']], on=['port','year','month'], how='left')
    tt.rename(columns={'pi_teu_per_hour_i_y':'Pi_i_y'}, inplace=True)
    mask_valid = (tt['teu_i_m']>0) & (tt['l_hours_i_m']>0)
    tt['LP_mix'] = np.where(mask_valid, tt['w'] * tt['Pi_i_y'], np.nan)

    t_month = tt[['port','terminal','year','month','month_index','quarter','operating','Pi_i_y','w','r_winsor','w_source','teu_i_m','l_hours_i_m','LP_mix']].copy()
    t_month = t_month.sort_values(['port','terminal','year','month']).reset_index(drop=True)

    # quarterly aggregate (mean over months in quarter)
    t_q = t_month.copy()
    t_q['quarter'] = t_q['month'].apply(_quarter_from_month)
    t_quarter = (t_q.groupby(['port','terminal','year','quarter'], as_index=False)
                   .agg(Pi_i_y=('Pi_i_y','first'), w=('w','mean'), r_winsor=('r_winsor','mean'),
                        teu_i_m=('teu_i_m','sum'), l_hours_i_m=('l_hours_i_m','sum'), LP_mix=('LP_mix','mean')))
    t_quarter = t_quarter.sort_values(['port','terminal','year','quarter']).reset_index(drop=True)
    return t_month, t_quarter

# ----------------------------- QA -----------------------------

def qa_bundle(port_mix: pd.DataFrame, port_id: pd.DataFrame, t_month: pd.DataFrame, wtab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(name, ok, note):
        rows.append({'check': name, 'ok': bool(ok), 'note': note})

    add('unique_port_month', not port_mix.duplicated(['port','year','month']).any(), 'port mix unique by (p,y,m)')
    add('unique_terminal_month', not t_month.duplicated(['port','terminal','year','month']).any(), 'terminal mix unique by (p,i,y,m)')

    agg = port_mix.groupby(['port','year'], as_index=False).agg(mu_LP=('LP_mix','mean'), mu_Pi=('Pi_mixbase_p_m','mean'))
    agg['delta'] = agg['mu_LP'] - agg['mu_Pi']
    for _, r in agg.iterrows():
        rows.append({'check':'annual_preservation','ok': True, 'note': f"{r['port']} {int(r['year'])}: meanLP={r['mu_LP']:.4f}, meanPi={r['mu_Pi']:.4f}, delta={r['delta']:.4f}"})

    if {'w_source'}.issubset(wtab.columns):
        dist = wtab.groupby(['port','year','w_source']).size().reset_index(name='n')
        for _, r in dist.iterrows():
            rows.append({'check':'w_source_dist','ok': True, 'note': f"{r['port']} {int(r['year'])} {r['w_source']}: n={int(r['n'])}"})

    rows.append({'check':'w_na_count','ok': True, 'note': f"w NA count: {int(wtab['w'].isna().sum())}"})
    rows.append({'check':'Pi_na_count','ok': True, 'note': f"Pi NA count (port months): {int(port_mix['Pi_mixbase_p_m'].isna().sum())}"})

    return pd.DataFrame(rows)

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--norm_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--winsor_low', type=float, default=0.01)
    ap.add_argument('--winsor_high', type=float, default=0.99)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    teu_m, teu_q, tons, lpr = load_inputs(args.norm_dir)

    wtab = build_w(teu_m, teu_q, tons, args.winsor_low, args.winsor_high)
    pi_m, shares = build_pi_mixbase(lpr)

    port_mix, port_id = build_port_tables(wtab, pi_m, lpr)
    t_month, t_quarter = build_terminal_tables(wtab, lpr)

    # unified panel (mixedfreq)
    port_panel = port_mix.copy()
    port_panel['level'] = 'port'
    port_panel['terminal'] = pd.NA
    port_panel = port_panel.merge(port_id[['port','year','month','LP_id']], on=['port','year','month'], how='left')
    port_panel.rename(columns={'teu_p_m':'TEU','tons_p_m':'tons','r':'tons_per_teu','Pi_mixbase_p_m':'Pi'}, inplace=True)
    port_panel['L_hours'] = pd.NA
    port_panel['freq'] = 'M'

    term_panel = t_month.copy()
    term_panel['level'] = 'terminal'
    term_panel.rename(columns={'teu_i_m':'TEU','Pi_i_y':'Pi','LP_mix':'LP_mix','l_hours_i_m':'L_hours'}, inplace=True)
    term_panel['tons'] = pd.NA
    term_panel['tons_per_teu'] = pd.NA
    term_panel['LP_id'] = pd.NA
    term_panel['freq'] = 'M'

    common_cols = ['level','port','terminal','year','month','month_index']
    measure_cols = ['TEU','tons','tons_per_teu','w','w_source','r_winsor','Pi','L_hours','LP_mix','LP_id']

    # align columns via reindex (prevents KeyError even if a column is missing in one branch)
    final_cols = common_cols + measure_cols
    port_panel = port_panel.reindex(columns=final_cols)
    term_panel = term_panel.reindex(columns=final_cols)

    lp_panel = pd.concat([port_panel, term_panel], ignore_index=True)

    qa = qa_bundle(port_mix, port_id, t_month, wtab)

    _write_tsv(port_mix, os.path.join(args.out_dir, 'LP_port_month_mixadjusted.tsv'))
    _write_tsv(port_id,  os.path.join(args.out_dir, 'LP_port_month_identity.tsv'))
    _write_tsv(t_month,  os.path.join(args.out_dir, 'LP_terminal_month_mixadjusted.tsv'))
    _write_tsv(t_quarter,os.path.join(args.out_dir, 'LP_terminal_quarter_mixadjusted.tsv'))
    _write_tsv(lp_panel, os.path.join(args.out_dir, 'LP_panel_mixedfreq.tsv'))
    _write_tsv(qa,       os.path.join(args.out_dir, 'qa_lp_report.tsv'))

    meta = {
        'winsor_low': args.winsor_low,
        'winsor_high': args.winsor_high,
        'rows': {
            'port_mix': len(port_mix),
            'port_id': len(port_id),
            'terminal_month': len(t_month),
            'terminal_quarter': len(t_quarter),
            'lp_panel': len(lp_panel)
        }
    }
    with open(os.path.join(args.out_dir, '_meta_lp_build.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('[lp] Wrote LP panel artifacts to', args.out_dir)

if __name__ == '__main__':
    main()

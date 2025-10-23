#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build_LP_Panel_S5_Stack.py
---------------------------------
Purpose:
  Stack the six S4 LP series into a single long panel file: LP_Panel.tsv

Inputs (defaults):
  --haifa_m     Data/LP/LP_Haifa_port_month.tsv
  --ashdod_m    Data/LP/LP_Ashdod_port_month.tsv
  --haifa_legacy_q   Data/LP/LP_Haifa_Legacy_quarter.tsv
  --haifa_sipg_q     Data/LP/LP_Haifa_SIPG_quarter.tsv
  --ashdod_legacy_q  Data/LP/LP_Ashdod_Legacy_quarter.tsv
  --ashdod_hct_q     Data/LP/LP_Ashdod_HCT_quarter.tsv

Outputs:
  - LP_Panel.tsv   (unified schema for all six series)
  - S5_qa.tsv      (stack-level QA: uniqueness, spans, w means, NA rates)
  - _meta_s5.json  (counts, spans, parameters)

Usage (from project root):
  python "Data/LP/Build_LP_Panel_S5_Stack.py"     --out_dir "Data/LP"
"""
import argparse, os, json, re
import numpy as np
import pandas as pd

TAB = '\t'

def _read(path):
    return pd.read_csv(path, sep=TAB)

def _write(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=TAB, index=False)

def _to_Int64(series):
    return pd.to_numeric(series, errors='coerce').astype('Int64')

def _qcode(qstr):
    m = re.match(r'^\s*Q([1-4])\s*$', str(qstr))
    return int(m.group(1)) if m else np.nan

def _monthly_transform(df, series_id, port_name):
    # required cols (as produced by S4)
    need = {'port','year','month','month_index','TEU_port_m','tons_port_m','w','Pi_port_q','LP_mix'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[{series_id}] Missing monthly columns: {miss}")

    out = pd.DataFrame({
        'series_id': series_id,
        'level': 'port',
        'freq': 'M',
        'port': df['port'],
        'terminal': pd.NA,
        'year': _to_Int64(df['year']),
        'month': _to_Int64(df['month']),
        'quarter': pd.NA,
        'month_index': _to_Int64(df['month_index']),
        'quarter_index': pd.Series([pd.NA]*len(df), dtype='Int64'),
        'TEU': df['TEU_port_m'],
        'tons': df['tons_port_m'],
        'L_hours': df['L_hours_port_m'] if 'L_hours_port_m' in df.columns else pd.Series([pd.NA]*len(df)),
        'w': df['w'],
        'Pi': df['Pi_port_q'],
        'LP': df['LP_mix'],
        'LP_id': df['LP_id'] if 'LP_id' in df.columns else pd.Series([pd.NA]*len(df)),
        'tons_source': df['tons_source'] if 'tons_source' in df.columns else pd.Series([pd.NA]*len(df))
    })
    # enforce port name if needed
    if port_name is not None:
        out['port'] = port_name
    return out

def _quarterly_transform(df, series_id):
    # required cols (as produced by S4)
    need = {'port','terminal','year','quarter','TEU_i_q','w','Pi_teu_per_hour_i_y','LP_mix'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[{series_id}] Missing quarterly columns: {miss}")

    qidx = _to_Int64(df['year'])*4 + pd.to_numeric(df['quarter'].map(_qcode), errors='coerce').astype('Int64')

    out = pd.DataFrame({
        'series_id': series_id,
        'level': 'terminal',
        'freq': 'Q',
        'port': df['port'],
        'terminal': df['terminal'],
        'year': _to_Int64(df['year']),
        'month': pd.Series([pd.NA]*len(df), dtype='Int64'),
        'quarter': df['quarter'],
        'month_index': pd.Series([pd.NA]*len(df), dtype='Int64'),
        'quarter_index': qidx,
        'TEU': df['TEU_i_q'],
        'tons': pd.Series([pd.NA]*len(df)),
        'L_hours': pd.Series([pd.NA]*len(df)),
        'w': df['w'],
        'Pi': df['Pi_teu_per_hour_i_y'],
        'LP': df['LP_mix'],
        'LP_id': pd.Series([pd.NA]*len(df)),
        'tons_source': pd.Series([pd.NA]*len(df))
    })
    return out

def _span_monthly(df):
    ym = df['year'].astype('Int64')*100 + df['month'].astype('Int64')
    return int(ym.min()), int(ym.max())

def _span_quarterly(df):
    q = pd.to_numeric(df['quarter'].map(_qcode), errors='coerce').astype('Int64')
    yq = df['year'].astype('Int64')*10 + q
    return int(yq.min()), int(yq.max())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--haifa_m', default='Data/LP/LP_Haifa_port_month.tsv')
    ap.add_argument('--ashdod_m', default='Data/LP/LP_Ashdod_port_month.tsv')
    ap.add_argument('--haifa_legacy_q', default='Data/LP/LP_Haifa_Legacy_quarter.tsv')
    ap.add_argument('--haifa_sipg_q', default='Data/LP/LP_Haifa_SIPG_quarter.tsv')
    ap.add_argument('--ashdod_legacy_q', default='Data/LP/LP_Ashdod_Legacy_quarter.tsv')
    ap.add_argument('--ashdod_hct_q', default='Data/LP/LP_Ashdod_HCT_quarter.tsv')
    ap.add_argument('--out_dir', default='Data/LP')
    args = ap.parse_args()

    # Read inputs
    df_hm = _read(args.haifa_m)
    df_am = _read(args.ashdod_m)
    df_hlq = _read(args.haifa_legacy_q)
    df_hsq = _read(args.haifa_sipg_q)
    df_alq = _read(args.ashdod_legacy_q)
    df_ahq = _read(args.ashdod_hct_q)

    # Transform
    hm = _monthly_transform(df_hm, 'Haifa_port_M', 'Haifa')
    am = _monthly_transform(df_am, 'Ashdod_port_M', 'Ashdod')
    hlq = _quarterly_transform(df_hlq, 'Haifa_Legacy_Q')
    hsq = _quarterly_transform(df_hsq, 'Haifa_SIPG_Q')
    alq = _quarterly_transform(df_alq, 'Ashdod_Legacy_Q')
    ahq = _quarterly_transform(df_ahq, 'Ashdod_HCT_Q')

    # Concat
    cols = ['series_id','level','freq','port','terminal',
            'year','month','quarter','month_index','quarter_index',
            'TEU','tons','L_hours','w','Pi','LP','LP_id','tons_source']
    panel = pd.concat([hm, am, hlq, hsq, alq, ahq], ignore_index=True)[cols]

    # QA
    qa_rows = []
    def add(check, ok, note): qa_rows.append({'check':check, 'ok':bool(ok), 'note':note})

    # Counts per series
    counts = panel.groupby('series_id').size().reset_index(name='n')
    for _, r in counts.iterrows():
        add(f"rows_{r['series_id']}", True, f"n={int(r['n'])}")

    # Uniqueness
    m = panel[panel['freq']=='M']
    if not m.empty:
        add('unique_monthly_series', not m.duplicated(['series_id','year','month']).any(), f"n={len(m)}")
    q = panel[panel['freq']=='Q']
    if not q.empty:
        add('unique_quarterly_series', not q.duplicated(['series_id','year','quarter']).any(), f"n={len(q)}")

    # Spans (informational)
    spans = []
    if not hm.empty: spans.append(('Haifa_port_M',) + _span_monthly(hm))
    if not am.empty: spans.append(('Ashdod_port_M',) + _span_monthly(am))
    if not hlq.empty: spans.append(('Haifa_Legacy_Q',) + _span_quarterly(hlq))
    if not hsq.empty: spans.append(('Haifa_SIPG_Q',) + _span_quarterly(hsq))
    if not alq.empty: spans.append(('Ashdod_Legacy_Q',) + _span_quarterly(alq))
    if not ahq.empty: spans.append(('Ashdod_HCT_Q',) + _span_quarterly(ahq))

    spans_df = pd.DataFrame(spans, columns=['series_id','span_min','span_max']) if spans else pd.DataFrame(columns=['series_id','span_min','span_max'])

    # w means per (port,year) — informational
    port_year_w = panel.groupby(['freq','port','year'])['w'].mean().reset_index()
    max_dev = float((port_year_w['w'] - 1.0).abs().max()) if len(port_year_w) else np.nan
    add('mean_w_port_year≈1', True, f"max |mean(w)-1|={max_dev:.3g}")

    # Forbidden ports (none expected)
    bad = panel[panel['port'].isin(['Eilat'])]
    add('no_forbidden_ports', bad.empty, f"Eilat_rows={len(bad)}")

    # NA rates by series (TEU/Pi/LP)
    for sid, grp in panel.groupby('series_id'):
        msg = f"NA TEU={int(grp['TEU'].isna().sum())}, NA Pi={int(grp['Pi'].isna().sum())}, NA LP={int(grp['LP'].isna().sum())}"
        add(f'na_rates_{sid}', True, msg)

    qa = pd.DataFrame(qa_rows)

    # Write outputs
    out_panel = os.path.join(args.out_dir, 'LP_Panel.tsv')
    _write(panel, out_panel)

    out_qa = os.path.join(args.out_dir, 'S5_qa.tsv')
    _write(qa, out_qa)

    out_spans = os.path.join(args.out_dir, 'S5_spans.tsv')
    _write(spans_df, out_spans)

    meta = {
        'inputs': {
            'haifa_m': args.haifa_m, 'ashdod_m': args.ashdod_m,
            'haifa_legacy_q': args.haifa_legacy_q, 'haifa_sipg_q': args.haifa_sipg_q,
            'ashdod_legacy_q': args.ashdod_legacy_q, 'ashdod_hct_q': args.ashdod_hct_q
        },
        'outputs': {'LP_Panel': out_panel, 'S5_qa': out_qa, 'S5_spans': out_spans},
        'schema': ['series_id','level','freq','port','terminal','year','month','quarter','month_index','quarter_index',
                   'TEU','tons','L_hours','w','Pi','LP','LP_id','tons_source']
    }
    with open(os.path.join(args.out_dir, '_meta_s5.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("[S5] Wrote LP_Panel.tsv, S5_qa.tsv, S5_spans.tsv and _meta_s5.json to", args.out_dir)

if __name__ == '__main__':
    main()

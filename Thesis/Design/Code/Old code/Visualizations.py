
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizations — Econ H191 (Ports) : L, LP, TEU, Tons
----------------------------------------------------
Reads copies of input TSVs from `Design/Input Data` and saves figures into
`Design/Output Data/visuals` (created if missing).

Files (expected names):
  - LP_panel copy.tsv
  - L_Proxy copy.tsv
  - teu_monthly_plus_quarterly_by_port copy.tsv
  - monthly_output_by_1000_tons_ports_and_terminals copy.tsv

Usage (from repo root or from this script's folder):
  $ python Visualizations.py --run-all
  # or run a subset:
  $ python Visualizations.py --make A1 A2 B3 B4

Notes:
- Pure matplotlib (no seaborn). 
- Default colors/styles only.
- Robust to minor column naming variations described in the project reports.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- Paths & Config --------------------------

SCRIPT_PATH = Path(__file__).resolve()
DESIGN_DIR  = SCRIPT_PATH.parents[1]        # .../Design
IN_DIR      = DESIGN_DIR / "Input Data"
OUT_DIR     = DESIGN_DIR / "Output Data" / "visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LP_PANEL_F  = IN_DIR / "LP_panel copy.tsv"
L_PROXY_F   = IN_DIR / "L_Proxy copy.tsv"
TEU_PORT_F  = IN_DIR / "teu_monthly_plus_quarterly_by_port copy.tsv"
TONS_F      = IN_DIR / "monthly_output_by_1000_tons_ports_and_terminals copy.tsv"


# Known events (approximate commissioning / privatization anchors)
EVENTS = {
    "Haifa": {
        "bayport_commission": "2021-09-01",   # SIPG Bayport
        "privatization":      "2023-01-01",   # Haifa Legacy privatization close
    },
    "Ashdod": {
        "hct_commission":     "2022-11-01",   # Ashdod Southport (HCT / TIL) effective
    },
}
ENTRANTS = {"Haifa": "Bayport", "Ashdod": "HCT"}
LEGACY_TAG = "Legacy"


# -------------------------- Utilities --------------------------

def _parse_month(col: pd.Series) -> pd.Series:
    """Coerce month column to pandas Period[M] (accepts YYYY-MM, date, or int month with year elsewhere)."""
    if np.issubdtype(col.dtype, np.datetime64):
        return col.dt.to_period("M")
    try:
        # try timestamps / strings
        s = pd.to_datetime(col, errors="coerce")
        if s.notna().any():
            return s.dt.to_period("M")
    except Exception:
        pass
    # if it's likely integer month, return as-is (caller will combine with year)
    return col

def _ensure_date(df: pd.DataFrame, year_col="year", month_col="month", out_name="date") -> pd.DataFrame:
    """Ensure a monthly timestamp column named `out_name` exists."""
    df = df.copy()
    if out_name in df.columns:
        return df
    y_ok = year_col in df.columns
    m_ok = month_col in df.columns
    if y_ok and m_ok:
        y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
        m = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
        df[out_name] = pd.to_datetime(
            y.astype(str) + "-" + m.astype(str) + "-01", errors="coerce"
        )
    elif month_col in df.columns:
        m = _parse_month(df[month_col])
        if isinstance(m.dtype, pd.PeriodDtype):
            df[out_name] = m.dt.to_timestamp()
        else:
            df[out_name] = pd.to_datetime(m, errors="coerce")
    else:
        # last resort: search for a date-like column
        for c in df.columns:
            if "date" in c.lower() or "month" in c.lower():
                s = pd.to_datetime(df[c], errors="coerce")
                if s.notna().any():
                    df[out_name] = s
                    break
    if out_name not in df.columns:
        raise ValueError("Could not infer a monthly date column.")
    return df

def _savefig(name: str):
    path = OUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path.relative_to(DESIGN_DIR)}")

def _annotate_events(ax: plt.Axes, port: str):
    if port not in EVENTS: 
        return
    for label, d in EVENTS[port].items():
        try:
            x = pd.to_datetime(d)
            ax.axvline(x, linestyle="--", linewidth=1)
            ax.text(x, ax.get_ylim()[1], f" {label}", rotation=90, va="top", ha="left", fontsize=8)
        except Exception:
            pass

def _index_series(s: pd.Series, base_date: pd.Timestamp) -> pd.Series:
    """Index to 100 at base_date (first non-null from <= base if missing)."""
    s = s.sort_index()
    if base_date not in s.index or pd.isna(s.loc[base_date]):
        # fallback to last non-na prior to base
        prior = s.loc[:base_date].dropna()
        if prior.empty:
            return s* np.nan
        base_val = prior.iloc[-1]
    else:
        base_val = s.loc[base_date]
    return (s / base_val) * 100.0

def _rolling_corr(x: pd.Series, y: pd.Series, win: int = 12) -> pd.Series:
    """Rolling correlation with window win (months)."""
    # align
    df = pd.concat([x, y], axis=1).dropna()
    if df.empty:
        return df.iloc[:,0] * np.nan
    # compute rolling correlation
    return df[x.name].rolling(win).corr(df[y.name])

def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd

# -------------------------- Loaders --------------------------

@dataclass
class DataBundle:
    lp: pd.DataFrame
    lproxy: pd.DataFrame
    teu_port: Optional[pd.DataFrame]
    tons: Optional[pd.DataFrame]

def load_all() -> DataBundle:
    # LP panel
    lp = pd.read_csv(LP_PANEL_F, sep="\t")
    lp = _ensure_date(lp, year_col="year", month_col="month", out_name="date")
    # normalize column names
    rename = {}
    for c in lp.columns:
        cl = c.lower()
        if cl == "lp_mix": rename[c] = "LP_mix"
        if cl == "lp_id":  rename[c] = "LP_id"
        if cl == "l_hours": rename[c] = "L_hours"
    if rename: lp = lp.rename(columns=rename)
    # L Proxy
    lproxy = pd.read_csv(L_PROXY_F, sep="\t")
    lproxy = _ensure_date(lproxy, year_col="year", month_col="month", out_name="date")
    # TEU/tons raw (optional; LP_panel already contains most of what we need)
    teu_df = None
    if TEU_PORT_F.exists():
        teu_df = pd.read_csv(TEU_PORT_F, sep="\t")
    tons_df = None
    if TONS_F.exists():
        tons_df = pd.read_csv(TONS_F, sep="\t")
    return DataBundle(lp=lp, lproxy=lproxy, teu_port=teu_df, tons=tons_df)


# -------------------------- A) Sanity & Coverage --------------------------

def viz_A1_missingness(bundle: DataBundle):
    """Missingness barlines by series, by level, and by month."""
    lp = bundle.lp.copy()
    lp["metric_LP_mix_na"] = lp["LP_mix"].isna()
    lp["metric_LP_id_na"]  = lp.get("LP_id", pd.Series(False, index=lp.index)).isna()
    lp["metric_TEU_na"]    = lp["TEU"].isna() if "TEU" in lp.columns else False
    lp["metric_tons_na"]   = lp["tons"].isna() if "tons" in lp.columns else False
    lp["metric_w_na"]      = lp["w"].isna() if "w" in lp.columns else False

    # by level (port vs terminal)
    grp = lp.groupby("level")[["metric_LP_mix_na","metric_TEU_na","metric_tons_na","metric_w_na"]].mean().fillna(0.0)
    ax = grp.mul(100).plot(kind="bar")
    ax.set_ylabel("% missing")
    ax.set_title("A1: Missingness by level (LP_panel)")
    _savefig("A1_missingness_by_level")

    # by port/terminal entity
    if "port" in lp.columns:
        grp2 = lp.groupby(["level","port"])[["metric_LP_mix_na","metric_TEU_na","metric_tons_na","metric_w_na"]].mean().mul(100).unstack(0)
        grp2.plot(kind="bar", figsize=(10,4))
        plt.ylabel("% missing")
        plt.title("A1: Missingness by port (LP_panel)")
        _savefig("A1_missingness_by_port")

    # by month (share of NAs)
    lp["ym"] = lp["date"].dt.to_period("M").dt.to_timestamp()
    bym = lp.groupby("ym")[["metric_LP_mix_na","metric_TEU_na","metric_tons_na","metric_w_na"]].mean().mul(100)
    ax = bym.plot()
    ax.set_ylabel("% missing")
    ax.set_title("A1: Missingness share over time")
    _annotate_events(ax, "Haifa")
    _annotate_events(ax, "Ashdod")
    _savefig("A1_missingness_over_time")


def viz_A2_calendar_heatmaps(bundle: DataBundle):
    """Calendar heatmaps for LP_mix, L_hours, TEU, tons, w at port level."""
    lp_port = bundle.lp[bundle.lp["level"].str.lower().eq("port")].copy()
    if lp_port.empty:
        print("[warn] No port-level rows in LP_panel for A2.")
        return
    lp_port["Year"] = lp_port["date"].dt.year
    lp_port["Month"] = lp_port["date"].dt.month

    metrics = [("LP_mix","LP_mix"), ("TEU","TEU"), ("tons","tons"), ("w","w")]
    for port in sorted(lp_port["port"].unique()):
        dfp = lp_port[lp_port["port"] == port]
        for col, label in metrics:
            if col not in dfp.columns:
                continue
            piv = dfp.pivot_table(index="Year", columns="Month", values=col, aggfunc="mean")
            fig, ax = plt.subplots(figsize=(8,4))
            im = ax.imshow(piv.values, aspect="auto", origin="lower")
            ax.set_xticks(np.arange(0,12)); ax.set_xticklabels(range(1,13))
            ax.set_yticks(np.arange(0,len(piv.index))); ax.set_yticklabels(piv.index)
            ax.set_title(f"A2: Calendar heatmap — {label} — {port}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _savefig(f"A2_calendar_{label}_{port}")


# -------------------------- B) Levels & Trends --------------------------

def _plot_series(ax, df, col, title, port=None):
    ax.plot(df["date"], df[col])
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel(col)
    if port: _annotate_events(ax, port)

def viz_B3_small_multiples_port(bundle: DataBundle):
    """Four time series per port: TEU, tons, L_hours (sum terminals), LP_mix."""
    lp = bundle.lp.copy()
    lp_port = lp[lp["level"].str.lower()=="port"]
    lp_term = lp[lp["level"].str.lower()=="terminal"]
    # L_hours at port from terminals
    L_port = (lp_term.groupby(["port","date"])["L_hours"].sum().reset_index()
                    .rename(columns={"L_hours":"L_port_hours"}))

    merged = lp_port.merge(L_port, on=["port","date"], how="left")
    for port in sorted(merged["port"].unique()):
        dfp = merged[merged["port"]==port].sort_values("date")
        # TEU
        fig, ax = plt.subplots()
        _plot_series(ax, dfp, "TEU", f"B3: TEU — {port}", port)
        _savefig(f"B3_port_TEU_{port}")
        # tons
        if "tons" in dfp.columns:
            fig, ax = plt.subplots()
            _plot_series(ax, dfp, "tons", f"B3: tons — {port}", port)
            _savefig(f"B3_port_tons_{port}")
        # L_hours (summed terminals)
        fig, ax = plt.subplots()
        _plot_series(ax, dfp, "L_port_hours", f"B3: L_hours (sum terminals) — {port}", port)
        _savefig(f"B3_port_Lhours_{port}")
        # LP_mix
        fig, ax = plt.subplots()
        _plot_series(ax, dfp, "LP_mix", f"B3: LP_mix — {port}", port)
        _savefig(f"B3_port_LPmix_{port}")


def viz_B4_small_multiples_terminal(bundle: DataBundle):
    """Time series per terminal: TEU_i_m, L_hours_i_m, LP_mix."""
    lp_term = bundle.lp[bundle.lp["level"].str.lower()=="terminal"].copy()
    for (port, term), g in lp_term.groupby(["port","terminal"]):
        g = g.sort_values("date")
        # TEU
        if "TEU" in g.columns:
            fig, ax = plt.subplots()
            ax.plot(g["date"], g["TEU"])
            ax.set_title(f"B4: TEU — {port} / {term}")
            ax.set_xlabel("Date"); ax.set_ylabel("TEU")
            _annotate_events(ax, port)
            _savefig(f"B4_terminal_TEU_{port}_{term}")
        # L_hours
        fig, ax = plt.subplots()
        ax.plot(g["date"], g["L_hours"])
        ax.set_title(f"B4: L_hours — {port} / {term}")
        ax.set_xlabel("Date"); ax.set_ylabel("hours")
        _annotate_events(ax, port)
        _savefig(f"B4_terminal_Lhours_{port}_{term}")
        # LP_mix
        fig, ax = plt.subplots()
        ax.plot(g["date"], g["LP_mix"])
        ax.set_title(f"B4: LP_mix — {port} / {term}")
        ax.set_xlabel("Date"); ax.set_ylabel("LP_mix")
        _annotate_events(ax, port)
        _savefig(f"B4_terminal_LPmix_{port}_{term}")


def viz_B5_indexed_levels(bundle: DataBundle, base="2021-06-01"):
    """Index TEU/tons/L_hours/LP_mix to 100 at a base month (per port)."""
    base_ts = pd.to_datetime(base)
    lp = bundle.lp.copy()
    lp_port = lp[lp["level"].str.lower()=="port"].copy()
    lp_term = lp[lp["level"].str.lower()=="terminal"].copy()

    # port-level series
    for port, g in lp_port.groupby("port"):
        g = g.sort_values("date").set_index("date")
        metrics = [c for c in ["TEU","tons","LP_mix"] if c in g.columns]
        # L_hours from terminals
        Lg = lp_term[lp_term["port"]==port].groupby("date")["L_hours"].sum().rename("L_hours")
        G = g.join(Lg, how="left")
        for col in metrics + ["L_hours"]:
            if col not in G.columns: 
                continue
            fig, ax = plt.subplots()
            ser = _index_series(G[col], base_ts)
            ax.plot(ser.index, ser.values)
            ax.set_title(f"B5: Indexed {col}=100 @ {base[:7]} — {port}")
            ax.set_ylabel("Index (100=base)"); ax.set_xlabel("Date")
            _annotate_events(ax, port)
            _savefig(f"B5_index_{col}_{port}")

def viz_B6_smoothing_growth(bundle: DataBundle):
    """12m trailing average and MoM/YoY growth for TEU, tons, LP_mix (port level)."""
    lp_port = bundle.lp[bundle.lp["level"].str.lower()=="port"].copy().sort_values("date")
    metrics = [c for c in ["TEU","tons","LP_mix"] if c in lp_port.columns]
    for port, g in lp_port.groupby("port"):
        g = g.set_index("date")
        for col in metrics:
            s = g[col].astype(float)
            # trailing avg
            fig, ax = plt.subplots()
            ax.plot(s.index, s.rolling(12, min_periods=3).mean())
            ax.set_title(f"B6: 12m trailing average {col} — {port}")
            _annotate_events(ax, port)
            _savefig(f"B6_trailing12_{col}_{port}")
            # MoM
            fig, ax = plt.subplots()
            ax.plot(s.index, s.pct_change()*100)
            ax.set_title(f"B6: MoM % change {col} — {port}")
            ax.set_ylabel("%"); _annotate_events(ax, port)
            _savefig(f"B6_mom_{col}_{port}")
            # YoY
            fig, ax = plt.subplots()
            ax.plot(s.index, s.pct_change(12)*100)
            ax.set_title(f"B6: YoY % change {col} — {port}")
            ax.set_ylabel("%"); _annotate_events(ax, port)
            _savefig(f"B6_yoy_{col}_{port}")


# -------------------------- C) Composition & Shares --------------------------

def viz_C7_stacked_TEU_shares(bundle: DataBundle):
    """Stacked area: terminal shares of port TEU over time."""
    lp_term = bundle.lp[bundle.lp["level"].str.lower()=="terminal"].copy()
    for port, g in lp_term.groupby("port"):
        piv = g.pivot_table(index="date", columns="terminal", values="TEU", aggfunc="sum").sort_index()
        if piv.empty: 
            continue
        totals = piv.sum(axis=1).replace({0: np.nan})
        shares = piv.div(totals, axis=0)
        fig, ax = plt.subplots(figsize=(10,4))
        x = shares.index
        ax.stackplot(x, shares.T.values, labels=shares.columns)
        ax.set_title(f"C7: Terminal shares of port TEU — {port}")
        ax.set_ylim(0,1); ax.legend(loc="upper left", fontsize=8)
        _annotate_events(ax, port)
        _savefig(f"C7_stacked_TEU_shares_{port}")


def viz_C8_stacked_L_shares(bundle: DataBundle):
    """Stacked area: terminal shares of port L_hours."""
    lp_term = bundle.lp[bundle.lp["level"].str.lower()=="terminal"].copy()
    for port, g in lp_term.groupby("port"):
        piv = g.pivot_table(index="date", columns="terminal", values="L_hours", aggfunc="sum").sort_index()
        if piv.empty: 
            continue
        totals = piv.sum(axis=1).replace({0: np.nan})
        shares = piv.div(totals, axis=0)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.stackplot(shares.index, shares.T.values, labels=shares.columns)
        ax.set_title(f"C8: Terminal shares of port L_hours — {port}")
        ax.set_ylim(0,1); ax.legend(loc="upper left", fontsize=8)
        _annotate_events(ax, port)
        _savefig(f"C8_stacked_L_shares_{port}")


# -------------------------- D) Mix & Productivity --------------------------

def viz_D9_w_over_time(bundle: DataBundle):
    """w over time (port-level)."""
    lp_port = bundle.lp[bundle.lp["level"].str.lower()=="port"].copy()
    for port, g in lp_port.groupby("port"):
        if "w" not in g.columns:
            continue
        fig, ax = plt.subplots()
        ax.plot(g["date"], g["w"])
        ax.set_title(f"D9: w over time — {port}")
        _annotate_events(ax, port)
        _savefig(f"D9_w_time_{port}")


def viz_D10_scatter_LP_vs_w(bundle: DataBundle, level="port"):
    """Scatter LP_mix vs w (port or terminal)."""
    df = bundle.lp[bundle.lp["level"].str.lower()==level.lower()].copy()
    if "w" not in df.columns:
        print("[warn] w not found for D10.")
        return
    for key, g in df.groupby("port" if level=="terminal" else "port"):
        fig, ax = plt.subplots()
        ax.scatter(g["w"], g["LP_mix"], s=10, alpha=0.6)
        ax.set_xlabel("w (tons_per_teu rebased)")
        ax.set_ylabel("LP_mix")
        ax.set_title(f"D10: LP_mix vs w — {key} ({level})")
        _savefig(f"D10_scatter_LP_vs_w_{level}_{key}")


def viz_D11_identity_vs_mix(bundle: DataBundle, level="port"):
    """Lines: LP_id vs LP_mix to show divergence."""
    df = bundle.lp[bundle.lp["level"].str.lower()==level.lower()].copy()
    if "LP_id" not in df.columns:
        print("[warn] LP_id not found for D11; skipping.")
        return
    for key, g in df.groupby("port" if level=="terminal" else "port"):
        g = g.sort_values("date")
        fig, ax = plt.subplots()
        ax.plot(g["date"], g["LP_id"], label="LP_id (TEU/L)")
        ax.plot(g["date"], g["LP_mix"], label="LP_mix (mix-adjusted)")
        ax.set_title(f"D11: Identity vs Mix — {key} ({level})")
        ax.legend()
        _savefig(f"D11_id_vs_mix_{level}_{key}")


# -------------------------- E) Event-window raw views --------------------------

def _relative_time(df: pd.DataFrame, event_date: pd.Timestamp) -> pd.Series:
    k = ((df["date"] - event_date) / np.timedelta64(1, "M")).round().astype("Int64")
    return k

def viz_E12_relative_time_means(bundle: DataBundle, pre=12, post=12):
    """Mean LP_mix by relative month k around events (per port)."""
    df = bundle.lp.copy()
    for port, events in EVENTS.items():
        if port not in df["port"].unique():
            continue
        for elabel, d in events.items():
            ed = pd.to_datetime(d)
            tmp = df[df["port"]==port].copy()
            tmp["k"] = _relative_time(tmp, ed)
            tmp = tmp[(tmp["k"]>=-pre)&(tmp["k"]<=post)]
            g = tmp.groupby("k")["LP_mix"].agg(["mean","count","std"]).reset_index()
            g["se"] = g["std"]/np.sqrt(g["count"].replace(0,np.nan))
            fig, ax = plt.subplots()
            ax.plot(g["k"], g["mean"])
            ax.fill_between(g["k"], g["mean"]-1.96*g["se"], g["mean"]+1.96*g["se"], alpha=0.2)
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.set_title(f"E12: LP_mix mean around {elabel} — {port}")
            ax.set_xlabel("k (months relative to event)")
            _savefig(f"E12_relative_time_{port}_{elabel}")


def viz_E13_legacy_vs_entrant_diff(bundle: DataBundle, pre=12, post=12):
    """Difference-of-means LP_mix: entrants vs legacy in pre/post windows (per port)."""
    df = bundle.lp[bundle.lp["level"].str.lower()=="terminal"].copy()
    if df.empty:
        print("[warn] No terminal rows for E13.")
        return
    for port, events in EVENTS.items():
        elabel, d = list(events.items())[0]
        ed = pd.to_datetime(d)
        tmp = df[df["port"]==port].copy()
        tmp["k"] = _relative_time(tmp, ed)
        pre_win = tmp[(tmp["k"]>=-pre)&(tmp["k"]<=-1)]
        post_win= tmp[(tmp["k"]>=1)&(tmp["k"]<=post)]
        # entrant and legacy sets
        entrant_pat = ENTRANTS.get(port, "")
        is_entrant = tmp["terminal"].str.contains(entrant_pat, case=False, na=False)
        is_legacy  = tmp["terminal"].str.contains(LEGACY_TAG, case=False, na=False)
        pre_diff  = post_diff = np.nan
        if not pre_win[is_entrant].empty and not pre_win[is_legacy].empty:
            pre_diff = pre_win[is_entrant]["LP_mix"].mean() - pre_win[is_legacy]["LP_mix"].mean()
        if not post_win[is_entrant].empty and not post_win[is_legacy].empty:
            post_diff = post_win[is_entrant]["LP_mix"].mean() - post_win[is_legacy]["LP_mix"].mean()
        fig, ax = plt.subplots()
        ax.bar(["Pre(-12,-1)","Post(+1,+12)"], [pre_diff, post_diff])
        ax.set_title(f"E13: Entrant−Legacy mean LP_mix — {port}")
        ax.set_ylabel("Δ mean LP_mix")
        _savefig(f"E13_diff_means_{port}")


# -------------------------- F) Heterogeneity & Distribution --------------------------

def viz_F14_yearly_boxplots(bundle: DataBundle):
    """Yearly boxplots of LP_mix by terminal."""
    df = bundle.lp[bundle.lp["level"].str.lower()=="terminal"].copy()
    if df.empty: 
        print("[warn] No terminal rows for F14.")
        return
    df["Year"] = df["date"].dt.year
    for term, g in df.groupby("terminal"):
        piv = [g[g["Year"]==y]["LP_mix"].dropna().values for y in sorted(g["Year"].unique())]
        labels = [str(y) for y in sorted(g["Year"].unique())]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.boxplot(piv, labels=labels, showfliers=False)
        ax.set_title(f"F14: LP_mix yearly boxplots — {term}")
        ax.set_xlabel("Year"); ax.set_ylabel("LP_mix")
        _savefig(f"F14_box_year_{term}")


def viz_F15_stacked_density(bundle: DataBundle, bins=40):
    """Simple 'ridgeline-style' via stacked density lines (no external libs)."""
    df = bundle.lp[bundle.lp["level"].str.lower()=="terminal"].copy()
    if df.empty: 
        print("[warn] No terminal rows for F15.")
        return
    terms = list(sorted(df["terminal"].unique()))
    # common x grid from pooled data
    pooled = df["LP_mix"].dropna().values
    if pooled.size == 0: 
        return
    xgrid = np.linspace(np.nanmin(pooled), np.nanmax(pooled), bins)
    fig, ax = plt.subplots(figsize=(8, 0.5*len(terms)+2))
    offset = 0.0
    for term in terms:
        vals = df[df["terminal"]==term]["LP_mix"].dropna().values
        if vals.size < 5:
            continue
        hist, edges = np.histogram(vals, bins=bins, range=(xgrid.min(), xgrid.max()), density=True)
        centers = 0.5*(edges[:-1]+edges[1:])
        ax.plot(centers, hist + offset, linewidth=1.0)
        ax.text(centers[0], offset, f" {term}", va="bottom", fontsize=8)
        offset += hist.max()*0.8 + 0.05
    ax.set_title("F15: Stacked density (LP_mix) by terminal")
    ax.set_yticks([]); ax.set_xlabel("LP_mix")
    _savefig("F15_stacked_density_terminals")


# -------------------------- G) Relationships & Frontiers --------------------------

def viz_G16_scatter_logTEU_vs_LP(bundle: DataBundle):
    """Scatter log(TEU) vs LP_mix by terminal, with pre/post markers."""
    df = bundle.lp[bundle.lp["level"].str.lower()=="terminal"].copy()
    if df.empty or "TEU" not in df.columns: 
        print("[warn] Missing TEU for G16.")
        return
    df = df.copy()
    df["logTEU"] = np.log(df["TEU"].replace({0: np.nan}))
    for (port, term), g in df.groupby(["port","terminal"]):
        fig, ax = plt.subplots()
        ax.scatter(g["logTEU"], g["LP_mix"], s=10, alpha=0.6)
        ax.set_title(f"G16: log(TEU) vs LP_mix — {port}/{term}")
        ax.set_xlabel("log(TEU)"); ax.set_ylabel("LP_mix")
        _savefig(f"G16_logTEU_vs_LP_{port}_{term}")


def viz_G17_rolling_corr(bundle: DataBundle, win=12, level="terminal"):
    """Rolling 12m correlation between TEU and LP_mix (per entity)."""
    df = bundle.lp[bundle.lp["level"].str.lower()==level.lower()].copy()
    if df.empty or "TEU" not in df.columns:
        print("[warn] Missing TEU for G17.")
        return
    for key, g in df.groupby("terminal" if level=="terminal" else "port"):
        g = g.sort_values("date").set_index("date")
        rc = _rolling_corr(g["TEU"].astype(float), g["LP_mix"].astype(float), win=win)
        fig, ax = plt.subplots()
        ax.plot(rc.index, rc.values)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"G17: Rolling {win}m corr(TEU, LP_mix) — {key}")
        _savefig(f"G17_rolling_corr_{level}_{key}")


# -------------------------- H) Outliers & Integrity --------------------------

def viz_H18_zscore_spikes(bundle: DataBundle, level="terminal"):
    """Z-score spike flags (|z|>3) per metric, per entity."""
    df = bundle.lp[bundle.lp["level"].str.lower()==level.lower()].copy()
    metrics = [c for c in ["TEU","tons","L_hours","LP_mix"] if c in df.columns]
    for key, g in df.groupby("terminal" if level=="terminal" else "port"):
        g = g.sort_values("date")
        for col in metrics:
            z = _zscore(g[col].astype(float))
            fig, ax = plt.subplots()
            ax.plot(g["date"], g[col], linewidth=1.0)
            mask = z.abs() > 3
            ax.scatter(g.loc[mask, "date"], g.loc[mask, col], s=25, marker="x")
            ax.set_title(f"H18: {col} with |z|>3 spikes — {key}")
            _savefig(f"H18_spikes_{col}_{level}_{key}")


def viz_H19_break_check(bundle: DataBundle, level="terminal"):
    """Break-check: first-difference around known discontinuities (visual)."""
    df = bundle.lp[bundle.lp["level"].str.lower()==level.lower()].copy()
    for key, g in df.groupby("terminal" if level=="terminal" else "port"):
        g = g.sort_values("date").set_index("date")
        for col in [c for c in ["LP_mix","L_hours"] if c in g.columns]:
            d = g[col].diff()
            fig, ax = plt.subplots()
            ax.plot(d.index, d.values)
            ax.set_title(f"H19: Δ {col} — {key}")
            _savefig(f"H19_breakcheck_d_{col}_{level}_{key}")


# -------------------------- Orchestrator --------------------------

ALL_TASKS = {
    "A1": viz_A1_missingness,
    "A2": viz_A2_calendar_heatmaps,
    "B3": viz_B3_small_multiples_port,
    "B4": viz_B4_small_multiples_terminal,
    "B5": viz_B5_indexed_levels,
    "B6": viz_B6_smoothing_growth,
    "C7": viz_C7_stacked_TEU_shares,
    "C8": viz_C8_stacked_L_shares,
    "D9": viz_D9_w_over_time,
    "D10": lambda b: (viz_D10_scatter_LP_vs_w(b, "port"), viz_D10_scatter_LP_vs_w(b, "terminal")),
    "D11": lambda b: (viz_D11_identity_vs_mix(b, "port"), viz_D11_identity_vs_mix(b, "terminal")),
    "E12": viz_E12_relative_time_means,
    "E13": viz_E13_legacy_vs_entrant_diff,
    "F14": viz_F14_yearly_boxplots,
    "F15": viz_F15_stacked_density,
    "G16": viz_G16_scatter_logTEU_vs_LP,
    "G17": lambda b: (viz_G17_rolling_corr(b, level="terminal"), viz_G17_rolling_corr(b, level="port")),
    "H18": viz_H18_zscore_spikes,
    "H19": viz_H19_break_check,
}

def run(tasks: Iterable[str] = ("A1","A2","B3","B4","B5","B6","C7","C8","D9","D10","D11","E12","E13","F14","F15","G16","G17","H18","H19")):
    bundle = load_all()
    for t in tasks:
        fn = ALL_TASKS.get(t)
        if fn is None:
            print(f"[skip] Unknown task {t}")
            continue
        print(f"[run] {t} ...")
        fn(bundle)

def main():
    p = argparse.ArgumentParser(description="Generate visualizations for Econ H191 Ports panel.")
    p.add_argument("--run-all", action="store_true", help="Run all visualizations.")
    p.add_argument("--make", nargs="+", help="Specify a subset of tasks to make (e.g., A1 B3 C7).")
    args = p.parse_args()
    if args.run_all:
        run()
    elif args.make:
        run(args.make)
    else:
        print("Usage examples:\n  python Visualizations.py --run-all\n  python Visualizations.py --make A1 B3 C7")

if __name__ == "__main__":
    main()

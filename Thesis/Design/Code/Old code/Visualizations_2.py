#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizations_2 — Two overview line charts (Haifa & Ashdod) + LP-only overlays
--------------------------------------------------------------------------------
Fixes:
- Robustly coalesces duplicate columns created by merges (e.g., TEU_teuopt,
  TEU_teuopt_x, ...), avoiding Series/DataFrame dtype errors.
- Extends plotting index through the end of 2024 if data exists.
- Keeps terminal LP (Legacy/Bayport/HCT) separate where available.

Inputs (expected in Design/Input Data/):
  - LP_panel copy.tsv
  - L_Proxy copy.tsv
  - teu_monthly_plus_quarterly_by_port copy.tsv (optional, fills TEU)
  - monthly_output_by_1000_tons_ports_and_terminals copy.tsv (optional, fills tons)

Outputs:
  - Design/Output Data/visuals_2/Overview_{Port}.png
  - Design/Output Data/visuals_2/LP_only_{Port}.png
  - Design/Output Data/visuals_2/port_month_panel.tsv

Pure matplotlib (no seaborn). One figure per chart.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paths ----------
SCRIPT_PATH = Path(__file__).resolve()
DESIGN_DIR  = SCRIPT_PATH.parents[1]
IN_DIR      = DESIGN_DIR / "Input Data"
OUT_DIR     = DESIGN_DIR / "Output Data" / "visuals_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LP_PANEL_F  = IN_DIR / "LP_panel copy.tsv" # LP_panel_mixedfreq copy
L_PROXY_F   = IN_DIR / "L_Proxy copy.tsv"
TEU_PORT_F  = IN_DIR / "teu_monthly_plus_quarterly_by_port copy.tsv"   # optional
TONS_F      = IN_DIR / "monthly_output_by_1000_tons_ports_and_terminals copy.tsv"  # optional

PORTS = ["Haifa", "Ashdod"]

# ---------- Helpers ----------

def to_month_start(s):
    """Parse various date formats to month start timestamps (YYYY-MM-01)."""
    s = pd.to_datetime(s, errors="coerce", utc=False)
    # Convert to monthly period then back to timestamp ensures month start
    return pd.to_datetime(s.dt.to_period("M").astype(str), errors="coerce")

def coalesce_first(df: pd.DataFrame, patterns: list[str], numeric=True) -> pd.Series:
    """
    Return the first non-null across columns whose names match any of `patterns`.
    - Any pattern ending with '*' is treated as a prefix (e.g., 'TEU_teuopt*').
    - Order in `patterns` matters.
    """
    cols: list[str] = []
    for pat in patterns:
        if pat.endswith("*"):
            pref = pat[:-1]
            cols.extend([c for c in df.columns if c.startswith(pref)])
        else:
            if pat in df.columns:
                cols.append(pat)
    # dedupe preserve order
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    if not ordered:
        return pd.Series(np.nan, index=df.index)
    vals = df[ordered].copy()
    if numeric:
        for c in ordered:
            vals[c] = pd.to_numeric(vals[c], errors="coerce")
    return vals.bfill(axis=1).iloc[:, 0]

def load_lp_panel():
    df = pd.read_csv(LP_PANEL_F, sep="\t")
    # Harmonize columns
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "lp_mix": rename[c] = "LP_mix"
        if cl == "lp_id": rename[c] = "LP_id"
        if cl == "l_hours": rename[c] = "L_hours"
    if rename:
        df = df.rename(columns=rename)

    # Create date if needed
    if "date" not in df.columns:
        if {"year", "month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01", errors="coerce")
        else:
            for c in df.columns:
                if "date" in c.lower() or "month" in c.lower():
                    df["date"] = pd.to_datetime(df[c], errors="coerce")
                    break
    df["date"] = to_month_start(df["date"])
    return df

def load_lproxy():
    df = pd.read_csv(L_PROXY_F, sep="\t")
    # Find hours column
    hour_col = None
    for c in df.columns:
        if "hour" in c.lower():
            hour_col = c; break
    if hour_col is None:
        raise ValueError("Could not find an hours column in L_Proxy copy.tsv")
    # Build date
    if "date" not in df.columns:
        if {"year","month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01", errors="coerce")
        else:
            for c in df.columns:
                if "date" in c.lower() or "month" in c.lower():
                    df["date"] = pd.to_datetime(df[c], errors="coerce")
                    break
    df["date"] = to_month_start(df["date"])
    df = df.rename(columns={hour_col: "L_hours"})
    return df[["port","terminal","date","L_hours"]]

def load_tons_optional():
    if not TONS_F.exists():
        return None
    t = pd.read_csv(TONS_F, sep="\t")
    # Expect columns like: PortOrTerminal, Month-Year, tons_k
    pcol = None; dcol = None; vcol = None
    for c in t.columns:
        lc = c.lower()
        if lc in {"portorterminal","port"}:
            pcol = c
        if "month" in lc or "date" in lc:
            dcol = c
        if "tons" in lc:
            vcol = c
    if pcol is None or dcol is None or vcol is None:
        return None
    t = t.rename(columns={pcol:"port", dcol:"date", vcol:"tons"})
    t["date"] = to_month_start(t["date"])
    t = t[t["port"].isin(PORTS)]
    # keep only port-level rows (drop terminals if present by pattern)
    return t[["port","date","tons"]]

def load_teu_optional():
    if not TEU_PORT_F.exists():
        return None
    df = pd.read_csv(TEU_PORT_F, sep="\t")
    # Heuristic schema handling
    cols = {c.lower(): c for c in df.columns}
    pcol = cols.get("port") or cols.get("portname") or list(df.columns)[0]
    # Period OR separate Year/Month
    if "period" in cols:
        df["date"] = to_month_start(df[cols["period"]])
    elif {"year","monthindex"}.issubset(cols.keys()):
        df["date"] = pd.to_datetime(df[cols["year"]].astype(int).astype(str) + "-" + df[cols["monthindex"]].astype(int).astype(str) + "-01", errors="coerce")
    elif {"year","month"}.issubset(cols.keys()):
        df["date"] = pd.to_datetime(df[cols["year"]].astype(int).astype(str) + "-" + df[cols["month"]].astype(int).astype(str) + "-01", errors="coerce")
    else:
        # try best-effort scan
        for c in df.columns:
            if "date" in c.lower() or "month" in c.lower():
                df["date"] = pd.to_datetime(df[c], errors="coerce")
                break
    df["date"] = to_month_start(df["date"])

    # Value column
    vcol = None
    for c in df.columns:
        if c.lower() == "teu":
            vcol = c; break
    if vcol is None:
        for c in df.columns:
            if "teu" in c.lower():
                vcol = c; break
    if vcol is None:
        return None

    out = df.rename(columns={pcol:"port", vcol:"TEU"})[["port","date","TEU"]]
    out = out[out["port"].isin(PORTS)]
    return out

def index_to_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s * np.nan
    base = s.dropna().iloc[0]
    return (s / base) * 100.0

# ---------- Build panel ----------

def build_port_month_panel():
    lp = load_lp_panel()
    lproxy = load_lproxy()
    tons_opt = load_tons_optional()
    teu_opt  = load_teu_optional()

    # Split levels
    has_level = "level" in lp.columns
    port_df = lp[lp["level"].str.lower()=="port"].copy() if has_level else lp.iloc[0:0].copy()
    term_df = lp[lp["level"].str.lower()=="terminal"].copy() if has_level else lp.iloc[0:0].copy()

    # Terminal aggregates by port-date
    term_agg = pd.DataFrame(columns=["port","date","TEU_term","LP_term_avg","LP_term_wavg"])
    if not term_df.empty:
        term_agg = term_df.groupby(["port","date"]).agg(
            TEU_term=("TEU","sum"),
            LP_term_avg=("LP_mix","mean"),
        ).reset_index()
        if "TEU" in term_df.columns:
            sums = term_df.groupby(["port","date"])["TEU"].sum().rename("sum_TEU")
            tmp = term_df[["port","date","LP_mix","TEU"]].merge(sums, on=["port","date"], how="left")
            tmp["w"] = tmp["TEU"] / tmp["sum_TEU"]
            tmp["contrib"] = tmp["w"] * tmp["LP_mix"]
            wavg = tmp.groupby(["port","date"])["contrib"].sum().rename("LP_term_wavg").reset_index()
            term_agg = term_agg.merge(wavg, on=["port","date"], how="left")
        else:
            term_agg["LP_term_wavg"] = np.nan

    # Port-level subset (keep only relevant)
    cols_keep = [c for c in ["port","date","TEU","tons","LP_mix"] if c in port_df.columns]
    port_sel = port_df[cols_keep].copy() if not port_df.empty else pd.DataFrame(columns=["port","date"])

    # L (hours) at port level
    L_port = lproxy.groupby(["port","date"])["L_hours"].sum().reset_index()

    # Merge everything (outer) on port/date
    panel = port_sel.merge(term_agg, on=["port","date"], how="outer")
    panel = panel.merge(L_port, on=["port","date"], how="outer")

    # Optional: tons and TEU from external tables
    if tons_opt is not None and not tons_opt.empty:
        panel = panel.merge(tons_opt, on=["port","date"], how="left", suffixes=("","_tonsopt"))
    if teu_opt is not None and not teu_opt.empty:
        panel = panel.merge(teu_opt, on=["port","date"], how="left", suffixes=("","_teuopt"))

    # Coalesce duplicates safely
    panel["TEU"]  = coalesce_first(panel, ["TEU", "TEU_term", "TEU_teuopt*"])
    panel["tons"] = coalesce_first(panel, ["tons", "tons_tonsopt*"])

    # LP: prefer port-level LP_mix; else terminal weighted average, else terminal mean
    panel["LP_port"] = coalesce_first(panel, ["LP_mix", "LP_term_wavg", "LP_term_avg"], numeric=True)

    # Keep only requested ports
    panel = panel[panel["port"].isin(PORTS)].copy()
    panel = panel.sort_values(["port","date"])

    # Minimal visible columns
    visible = ["port","date","TEU","tons","L_hours","LP_port"]
    for c in visible:
        if c not in panel.columns:
            panel[c] = np.nan
    panel = panel[visible]

    # Prepare terminal LP slice for plotting (Legacy vs new terminals)
    term_lp = pd.DataFrame(columns=["port","terminal","date","LP_mix"])
    if not term_df.empty:
        term_lp = term_df[["port","terminal","date","LP_mix"]].copy()

    # Determine date span for plotting (extend to end of 2024 if earlier)
    max_dates = [panel["date"].max()]
    for tbl in [term_lp, tons_opt, teu_opt, lproxy]:
        if tbl is not None and not getattr(tbl, "empty", False):
            max_dates.append(tbl["date"].max())
    overall_max = pd.to_datetime(max_dates).max()
    end_2024 = pd.Timestamp("2024-12-01")
    plot_end = max(end_2024, overall_max) if pd.notnull(overall_max) else end_2024

    min_dates = [panel["date"].min()]
    for tbl in [term_lp, tons_opt, teu_opt, lproxy]:
        if tbl is not None and not getattr(tbl, "empty", False):
            min_dates.append(tbl["date"].min())
    plot_start = pd.to_datetime(min_dates).min()

    return panel, term_lp, plot_start, plot_end

# ---------- Plotting ----------

def plot_overview(panel: pd.DataFrame, term_lp: pd.DataFrame, plot_start, plot_end, out_dir: Path = OUT_DIR):
    # Generate a continuous monthly index for each port to guarantee axis coverage
    monthly_index = pd.date_range(plot_start, plot_end, freq="MS")

    for port, g in panel.groupby("port"):
        g = g.sort_values("date").set_index("date").reindex(monthly_index)
        # Build indices
        series = {
            "TEU (index=100)": index_to_100(g["TEU"]),
            "tons (index=100)": index_to_100(g["tons"]),
            "L_hours (index=100)": index_to_100(g["L_hours"]),
            "LP_port (index=100)": index_to_100(g["LP_port"]),
        }
        fig, ax = plt.subplots(figsize=(10,4))
        for label, s in series.items():
            ax.plot(s.index, s.values, label=label, linewidth=1.2)
        ax.set_title(f"{port} — TEU, tons, L, LP (indexed to 100 at first available)")
        ax.set_xlabel("Date"); ax.set_ylabel("Index (100=first observed)")
        ax.legend(loc="lower left", ncol=2, fontsize=9)
        fig.savefig(out_dir / f"Overview_{port}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_dir.name}/Overview_{port}.png")

def plot_lp_only(term_lp: pd.DataFrame, panel: pd.DataFrame, plot_start, plot_end, out_dir: Path = OUT_DIR):
    """LP-only plot: LP_port, plus Legacy and new terminal LP series if present."""
    monthly_index = pd.date_range(plot_start, plot_end, freq="MS")

    for port in PORTS:
        # Port-level LP
        g = panel[panel["port"]==port].sort_values("date").set_index("date").reindex(monthly_index)
        fig, ax = plt.subplots(figsize=(10,4))
        # Port LP
        ax.plot(index_to_100(g["LP_port"]).index, index_to_100(g["LP_port"]).values,
                label="LP_port (index=100)", linewidth=1.5)

        # Terminal LP (Legacy vs entrant names heuristic)
        tlp = term_lp[term_lp["port"]==port].copy()
        if not tlp.empty:
            # Map terminal names to simplified labels
            lab_map = {}
            for t in sorted(tlp["terminal"].dropna().unique()):
                low = t.lower()
                if "legacy" in low or "old" in low:
                    lab_map[t] = "LP_Legacy (index=100)"
                elif "bay" in low:        # Haifa Bayport
                    lab_map[t] = "LP_Bayport (index=100)"
                elif "hct" in low or "h.c.t" in low:  # Ashdod HCT
                    lab_map[t] = "LP_HCT (index=100)"
                else:
                    lab_map[t] = f"LP_{t} (index=100)"
            for t, lab in lab_map.items():
                s = tlp[tlp["terminal"]==t].set_index("date")["LP_mix"].reindex(monthly_index)
                s = index_to_100(s)
                ax.plot(s.index, s.values, label=lab, linewidth=1.2)

        ax.set_title(f"{port} — LP series only (indexed to 100)")
        ax.set_xlabel("Date"); ax.set_ylabel("Index (100=first observed)")
        ax.legend(loc="upper left", ncol=3, fontsize=9)
        fig.savefig(out_dir / f"LP_only_{port}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_dir.name}/LP_only_{port}.png")

# ---------- Main ----------

def main():
    panel, term_lp, plot_start, plot_end = build_port_month_panel()
    # Save the panel we constructed for reproducibility
    panel_out = OUT_DIR / "port_month_panel.tsv"
    panel.to_csv(panel_out, sep="\t", index=False)
    print(f"[wrote] {panel_out.relative_to(DESIGN_DIR)}")
    # Plots
    plot_overview(panel, term_lp, plot_start, plot_end, OUT_DIR)
    plot_lp_only(term_lp, panel, plot_start, plot_end, OUT_DIR)

if __name__ == "__main__":
    main()

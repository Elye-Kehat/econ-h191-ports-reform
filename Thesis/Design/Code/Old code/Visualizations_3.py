#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizations_3 — Overview & LP-only plots using LP_panel_mixedfreq
--------------------------------------------------------------------
Drop-in update of Visualizations_2.py to read the new mixed-frequency
panel (monthly ports + monthly/quarterly terminals). Robust to the
"copy.tsv" filenames in Design/Input Data/. Pure matplotlib.

Inputs (looked for in this order under Design/Input Data/):
  - LP_panel_mixedfreq copy.tsv  (preferred)
  - LP_panel_mixedfreq.tsv
  - LP_panel copy.tsv
  - LP_panel.tsv
  - L_Proxy copy.tsv
  - teu_monthly_plus_quarterly_by_port copy.tsv (optional, for TEU fill)
  - monthly_output_by_1000_tons_ports_and_terminals copy.tsv (optional, tons fill)

Outputs:
  - Design/Output Data/visuals_3/Overview_{Port}.png
  - Design/Output Data/visuals_3/LP_only_{Port}.png
  - Design/Output Data/visuals_3/port_month_panel.tsv
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
OUT_DIR     = DESIGN_DIR / "Output Data" / "visuals_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate filenames (first one that exists wins)
LP_PANEL_CANDIDATES = [
    "LP_panel_mixedfreq copy.tsv",
    "LP_panel_mixedfreq.tsv",
    "LP_panel copy.tsv",
    "LP_panel.tsv",
]
L_PROXY_CANDIDATES = ["L_Proxy copy.tsv", "L_Proxy.tsv"]
TEU_PORT_CANDIDATES = ["teu_monthly_plus_quarterly_by_port copy.tsv", "teu_monthly_plus_quarterly_by_port.tsv"]
TONS_CANDIDATES = ["monthly_output_by_1000_tons_ports_and_terminals copy.tsv",
                   "monthly_output_by_1000_tons_ports_and_terminals.tsv"]

def first_existing(basenames):
    for nm in basenames:
        p = IN_DIR / nm
        if p.exists():
            return p
    return None

LP_PANEL_F = first_existing(LP_PANEL_CANDIDATES)
L_PROXY_F  = first_existing(L_PROXY_CANDIDATES)
TEU_PORT_F = first_existing(TEU_PORT_CANDIDATES)   # optional
TONS_F     = first_existing(TONS_CANDIDATES)       # optional

if LP_PANEL_F is None:
    raise FileNotFoundError("Could not find LP panel file in Design/Input Data/. "
                            f"Tried: {LP_PANEL_CANDIDATES}")

if L_PROXY_F is None:
    raise FileNotFoundError("Could not find L_Proxy file in Design/Input Data/. "
                            f"Tried: {L_PROXY_CANDIDATES}")

# ---------- Helpers ----------

def to_month_start(s):
    """Parse to month starts (YYYY-MM-01)."""
    s = pd.to_datetime(s, errors="coerce", utc=False)
    return pd.to_datetime(s.dt.to_period("M").astype(str), errors="coerce")

def index_to_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s * np.nan
    base = s.dropna().iloc[0]
    return (s / base) * 100.0

def coalesce_first(df: pd.DataFrame, patterns: list[str], numeric=True) -> pd.Series:
    """
    Return the first non-null across columns whose names match any of `patterns`.
    Pattern ending with '*' is treated as a prefix (e.g., 'TEU_teuopt*').
    """
    cols: list[str] = []
    for pat in patterns:
        if pat.endswith("*"):
            pref = pat[:-1]
            cols.extend([c for c in df.columns if c.startswith(pref)])
        else:
            if pat in df.columns:
                cols.append(pat)
    # dedupe while preserving order
    seen = set(); ordered = []
    for c in cols:
        if c not in seen:
            ordered.append(c); seen.add(c)
    if not ordered:
        return pd.Series(np.nan, index=df.index)
    vals = df[ordered].copy()
    if numeric:
        for c in ordered:
            vals[c] = pd.to_numeric(vals[c], errors="coerce")
    return vals.bfill(axis=1).iloc[:, 0]

# ---------- Loaders ----------

def load_lp_panel():
    df = pd.read_csv(LP_PANEL_F, sep="\t")
    # Normalize column names we use
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "lp_mix": ren[c] = "LP_mix"
        if cl == "lp_id": ren[c] = "LP_id"
        if cl == "l_hours": ren[c] = "L_hours"
    if ren:
        df = df.rename(columns=ren)

    # Build a date column
    if "date" not in df.columns:
        if {"year", "month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" +
                                        df["month"].astype(int).astype(str) + "-01",
                                        errors="coerce")
        else:
            for c in df.columns:
                if "date" in c.lower() or "month" in c.lower():
                    df["date"] = pd.to_datetime(df[c], errors="coerce")
                    break
    df["date"] = to_month_start(df["date"])

    # If ports list not provided, derive from data
    ports = sorted(df["port"].dropna().unique().tolist()) if "port" in df.columns else []
    return df, ports

def load_lproxy():
    df = pd.read_csv(L_PROXY_F, sep="\t")
    hour_col = None
    for c in df.columns:
        if "hour" in c.lower():
            hour_col = c; break
    if hour_col is None:
        raise ValueError("Could not find an hours column in L_Proxy file")
    if "date" not in df.columns:
        if {"year","month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" +
                                        df["month"].astype(int).astype(str) + "-01", errors="coerce")
        else:
            for c in df.columns:
                if "date" in c.lower() or "month" in c.lower():
                    df["date"] = pd.to_datetime(df[c], errors="coerce")
                    break
    df["date"] = to_month_start(df["date"])
    df = df.rename(columns={hour_col: "L_hours"})
    return df[["port","terminal","date","L_hours"]]

def load_tons_optional():
    if TONS_F is None:
        return None
    t = pd.read_csv(TONS_F, sep="\t")
    pcol = None; dcol = None; vcol = None
    for c in t.columns:
        lc = c.lower()
        if lc in {"portorterminal","port","location"}: pcol = c
        if "month" in lc or "date" in lc: dcol = c
        if "tons" in lc: vcol = c
    if pcol is None or dcol is None or vcol is None:
        return None
    t = t.rename(columns={pcol:"port", dcol:"date", vcol:"tons"})
    t["date"] = to_month_start(t["date"])
    # Keep only recognizable ports
    return t[["port","date","tons"]]

def load_teu_optional():
    if TEU_PORT_F is None:
        return None
    df = pd.read_csv(TEU_PORT_F, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    pcol = cols.get("port") or cols.get("portname") or list(df.columns)[0]
    if "period" in cols:
        df["date"] = to_month_start(df[cols["period"]])
    elif {"year","month"}.issubset(cols.keys()):
        df["date"] = pd.to_datetime(df[cols["year"]].astype(int).astype(str) + "-" +
                                    df[cols["month"]].astype(int).astype(str) + "-01", errors="coerce")
    else:
        for c in df.columns:
            if "date" in c.lower() or "month" in c.lower():
                df["date"] = pd.to_datetime(df[c], errors="coerce"); break
        df["date"] = to_month_start(df["date"])
    vcol = None
    for c in df.columns:
        if c.lower() == "teu": vcol = c; break
    if vcol is None:
        for c in df.columns:
            if "teu" in c.lower(): vcol = c; break
    if vcol is None:
        return None
    out = df.rename(columns={pcol:"port", vcol:"TEU"})[["port","date","TEU"]]
    return out

# ---------- Build panel for plotting ----------

def build_port_month_panel():
    lp, ports_from_data = load_lp_panel()
    lproxy = load_lproxy()
    tons_opt = load_tons_optional()
    teu_opt  = load_teu_optional()

    # Derive ports list:
    if ports_from_data:
        PORTS = ports_from_data
    else:
        PORTS = ["Haifa","Ashdod","Eilat"]

    # Partition by level if present
    if "level" in lp.columns:
        port_df = lp[lp["level"].str.lower()=="port"].copy()
        term_df = lp[lp["level"].str.lower()=="terminal"].copy()
    else:
        # Fallback: guess by presence of terminal
        term_mask = lp.get("terminal", pd.Series(index=lp.index, dtype=object)).notna()
        term_df = lp[term_mask].copy()
        port_df = lp[~term_mask].copy()

    # Terminal aggregates by port-date
    term_agg = pd.DataFrame(columns=["port","date","TEU_term","LP_term_avg","LP_term_wavg"])
    if not term_df.empty:
        term_agg = term_df.groupby(["port","date"]).agg(
            TEU_term=("TEU","sum") if "TEU" in term_df.columns else ("port","size"),
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

    # Port subset
    cols_keep = [c for c in ["port","date","TEU","tons","LP_mix"] if c in port_df.columns]
    port_sel = port_df[cols_keep].copy() if not port_df.empty else pd.DataFrame(columns=["port","date"])

    # L (hours) at port level
    L_port = lproxy.groupby(["port","date"])["L_hours"].sum().reset_index()

    # Merge (outer) on port/date
    panel = port_sel.merge(term_agg, on=["port","date"], how="outer")
    panel = panel.merge(L_port, on=["port","date"], how="outer")

    # Optional fills
    if tons_opt is not None and not tons_opt.empty:
        panel = panel.merge(tons_opt, on=["port","date"], how="left", suffixes=("","_tonsopt"))
    if teu_opt is not None and not teu_opt.empty:
        panel = panel.merge(teu_opt, on=["port","date"], how="left", suffixes=("","_teuopt"))

    # Coalesce duplicates safely
    panel["TEU"]  = coalesce_first(panel, ["TEU", "TEU_term", "TEU_teuopt*"])
    panel["tons"] = coalesce_first(panel, ["tons", "tons_tonsopt*"])

    # LP: prefer port-level LP_mix; else terminal weighted average, else terminal mean
    panel["LP_port"] = coalesce_first(panel, ["LP_mix", "LP_term_wavg", "LP_term_avg"], numeric=True)

    # Keep only ports present in the LP panel to avoid empty plots
    panel = panel[panel["port"].isin(PORTS)].copy()
    panel = panel.sort_values(["port","date"])

    # Terminal LP slice for plotting detail
    term_lp = pd.DataFrame(columns=["port","terminal","date","LP_mix"])
    if not term_df.empty:
        # ensure 'terminal' exists
        if "terminal" in term_df.columns:
            term_lp = term_df[["port","terminal","date","LP_mix"]].copy()

    # Determine date span for plotting
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

    return panel, term_lp, plot_start, plot_end, PORTS

# ---------- Plotting ----------

def plot_overview(panel: pd.DataFrame, term_lp: pd.DataFrame, plot_start, plot_end, ports, out_dir: Path = OUT_DIR):
    monthly_index = pd.date_range(plot_start, plot_end, freq="MS")
    for port, g in panel.groupby("port"):
        g = g.sort_values("date").set_index("date").reindex(monthly_index)
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

def plot_lp_only(term_lp: pd.DataFrame, panel: pd.DataFrame, plot_start, plot_end, ports, out_dir: Path = OUT_DIR):
    monthly_index = pd.date_range(plot_start, plot_end, freq="MS")
    for port in ports:
        g = panel[panel["port"]==port].sort_values("date").set_index("date").reindex(monthly_index)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(index_to_100(g["LP_port"]).index, index_to_100(g["LP_port"]).values,
                label="LP_port (index=100)", linewidth=1.5)

        tlp = term_lp[term_lp["port"]==port].copy()
        if not tlp.empty and "terminal" in tlp.columns:
            lab_map = {}
            for t in sorted(tlp["terminal"].dropna().unique()):
                low = str(t).lower()
                if "legacy" in low or "old" in low:
                    lab_map[t] = "LP_Legacy (index=100)"
                elif "bay" in low:
                    lab_map[t] = "LP_Bayport (index=100)"
                elif "hct" in low or "h.c.t" in low:
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
    panel, term_lp, plot_start, plot_end, ports = build_port_month_panel()
    # Save panel used for plotting
    panel_out = OUT_DIR / "port_month_panel.tsv"
    panel.to_csv(panel_out, sep="\t", index=False)
    print(f"[wrote] {panel_out.relative_to(DESIGN_DIR)}")
    # Plots
    plot_overview(panel, term_lp, plot_start, plot_end, ports, OUT_DIR)
    plot_lp_only(term_lp, panel, plot_start, plot_end, ports, OUT_DIR)

if __name__ == "__main__":
    main()

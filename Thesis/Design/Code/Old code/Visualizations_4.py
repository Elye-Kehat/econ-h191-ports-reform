#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Paths ----------------
SCRIPT_PATH = Path(__file__).resolve()
DESIGN_DIR  = SCRIPT_PATH.parents[1]
IN_DIR      = DESIGN_DIR / "Input Data"
OUT_DIR     = DESIGN_DIR / "Output Data" / "visuals_4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LP_PANEL_FILES = [
    "LP_panel_mixedfreq copy.tsv",
    "LP_panel_mixedfreq.tsv",
    "LP_panel copy.tsv",
    "LP_panel.tsv",
]
L_PROXY_FILES = ["L_Proxy copy.tsv", "L_Proxy.tsv"]

def first_existing(basenames):
    for nm in basenames:
        p = IN_DIR / nm
        if p.exists():
            return p
    return None

LP_PANEL_F = first_existing(LP_PANEL_FILES)
L_PROXY_F  = first_existing(L_PROXY_FILES)

if LP_PANEL_F is None:
    raise FileNotFoundError("LP panel not found in Design/Input Data/. Tried: " + ", ".join(LP_PANEL_FILES))
if L_PROXY_F is None:
    raise FileNotFoundError("L_Proxy not found in Design/Input Data/. Tried: " + ", ".join(L_PROXY_FILES))

# ---------------- Helpers ----------------

CUTOVER = {"Haifa": "2021-09", "Ashdod": "2022-07"}  # YYYY-MM
YLIM = (0, 300)

def to_month_start(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=False)
    return pd.to_datetime(s.dt.to_period("M").astype(str), errors="coerce")

def index_to_100(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.dropna().empty:
        return vals * np.nan
    base = vals.dropna().iloc[0]
    if pd.isna(base) or base == 0:
        return vals * np.nan
    return (vals / base) * 100.0

def first_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan

def dedupe(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty: return df
    cols = [c for c in df.columns if c not in keys]
    agg = {c: first_nonnull for c in cols}
    return (df.groupby(keys, dropna=False, as_index=False).agg(agg))[keys + cols]

def classify_terminal(term: str) -> str:
    t = str(term or "").lower()
    if "legacy" in t or "old" in t:
        return "Legacy"
    if "bay" in t:
        return "Bayport"
    if "hct" in t or "southport" in t or "til" in t:
        return "HCT"
    return "Other"

def parse_cutover_date(port: str) -> pd.Timestamp:
    ym = CUTOVER.get(port, None)
    if ym is None:
        return pd.Timestamp("2100-01-01")
    return pd.to_datetime(ym + "-01")

# ---------------- Load ----------------

def load_lp_panel() -> pd.DataFrame:
    df = pd.read_csv(LP_PANEL_F, sep="\t")
    # normalize names
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "lp_mix": ren[c] = "LP_mix"
        if cl == "lp_id": ren[c] = "LP_id"
        if cl == "l_hours": ren[c] = "L_hours"
    if ren: df = df.rename(columns=ren)

    if "date" not in df.columns:
        if {"year","month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" +
                                        df["month"].astype(int).astype(str) + "-01",
                                        errors="coerce")
        else:
            for c in df.columns:
                if "date" in c.lower() or "month" in c.lower():
                    df["date"] = pd.to_datetime(df[c], errors="coerce"); break
    df["date"] = to_month_start(df["date"])

    # Eilat exclusion
    df = df[~df["port"].astype(str).str.contains("Eilat", case=False, na=False)].copy()
    return df

def load_lproxy() -> pd.DataFrame:
    df = pd.read_csv(L_PROXY_F, sep="\t")
    # find hours column
    hour_col = None
    for c in df.columns:
        if "hour" in c.lower():
            hour_col = c; break
    if hour_col is None:
        raise ValueError("L_Proxy.tsv: could not find an 'hours' column.")

    if "date" not in df.columns:
        if {"year","month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-" +
                                        df["month"].astype(int).astype(str) + "-01", errors="coerce")
        else:
            for c in df.columns:
                if "date" in c.lower() or "month" in c.lower():
                    df["date"] = pd.to_datetime(df[c], errors="coerce"); break
    df["date"] = to_month_start(df["date"])
    df = df.rename(columns={hour_col: "L_hours"})
    out = df[["port","terminal","date","L_hours"]].copy()
    out = out[~out["port"].astype(str).str.contains("Eilat", case=False, na=False)]
    return out.groupby(["port","date"], dropna=False, as_index=False)["L_hours"].sum()

# ---------------- Build tidy panel ----------------

def build_port_level_panel(lp_all: pd.DataFrame, lproxy_sum: pd.DataFrame) -> pd.DataFrame:
    is_port = lp_all.get("level", "").astype(str).str.lower().eq("port")
    port_df = lp_all.loc[is_port, ["port","date","TEU","tons","LP_mix"]].copy()
    port_df = port_df.rename(columns={"LP_mix": "LP_port"})
    port_df = dedupe(port_df, ["port","date"])
    panel = port_df.merge(lproxy_sum, on=["port","date"], how="left")
    return panel

def build_terminal_lp(lp_all: pd.DataFrame) -> pd.DataFrame:
    is_term = lp_all.get("level", "").astype(str).str.lower().eq("terminal")
    term_df = lp_all.loc[is_term, ["port","terminal","date","LP_mix"]].copy()
    term_df["terminal_class"] = term_df["terminal"].apply(classify_terminal)
    term_df = dedupe(term_df, ["port","terminal","date"])
    return term_df

# ---------------- Plotting ----------------

def plot_overview(panel: pd.DataFrame, ports: list[str], out_dir: Path = OUT_DIR):
    for port in ports:
        g = panel[panel["port"]==port].copy()
        if g.empty: 
            print(f"[skip] {port}: no port-level panel rows.")
            continue
        g = dedupe(g, ["port","date"]).sort_values("date").set_index("date")
        g = g[~g.index.duplicated(keep="first")]

        series = {
            "TEU (index=100)": index_to_100(g["TEU"]),
            "tons (index=100)": index_to_100(g["tons"]),
            "L_hours (index=100)": index_to_100(g["L_hours"]),
            "LP_port (index=100)": index_to_100(g["LP_port"]),
        }

        fig, ax = plt.subplots(figsize=(11,4))
        for label, s in series.items():
            ax.plot(s.index, s.values, label=label, linewidth=1.2)
        ax.set_title(f"{port} — TEU, tons, L, LP (indexed to 100 at first available)")
        ax.set_xlabel("Date"); ax.set_ylabel("Index (100=first observed)")
        ax.set_ylim(0, 300)
        ax.legend(loc="lower left", ncol=2, fontsize=9, framealpha=0.9)
        fig.savefig(out_dir / f"Overview_{port}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_dir.name}/Overview_{port}.png")

def plot_lp_split(lp_port_panel: pd.DataFrame, term_lp: pd.DataFrame, ports: list[str], out_dir: Path = OUT_DIR):
    for port in ports:
        g = lp_port_panel[lp_port_panel["port"]==port].copy()
        if g.empty:
            print(f"[skip] {port}: no LP_port data."); 
            continue
        g = dedupe(g, ["port","date"]).sort_values("date").set_index("date")
        g = g[~g.index.duplicated(keep="first")]

        cut_date = parse_cutover_date(port)
        pre = g[g.index < cut_date]["LP_port"]

        tt = term_lp[term_lp["port"]==port].copy()
        tt["terminal_class"] = tt["terminal"].apply(classify_terminal)

        lines = {}
        post = tt[tt["date"] >= cut_date].copy()
        if not post.empty:
            for cls in ["Legacy", "Bayport", "HCT"]:
                sub = post[post["terminal_class"]==cls]
                if sub.empty: 
                    continue
                s = (sub.groupby("date", dropna=False)["LP_mix"].mean().sort_index())
                lines[f"LP_{cls} (index=100)"] = index_to_100(s)

        fig, ax = plt.subplots(figsize=(11,4))
        if not pre.dropna().empty:
            ax.plot(index_to_100(pre).index, index_to_100(pre).values, label="LP_port (index=100)", linewidth=1.8)

        for lab, s in lines.items():
            ax.plot(s.index, s.values, label=lab, linewidth=1.4)

        ax.set_title(f"{port} — LP series only (indexed to 100)")
        ax.set_xlabel("Date"); ax.set_ylabel("Index (100=first observed)")
        ax.set_ylim(0, 300)
        ax.legend(loc="upper left", ncol=3, fontsize=9, framealpha=0.9)
        fig.savefig(out_dir / f"LP_only_split_{port}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_dir.name}/LP_only_split_{port}.png")

# ---------------- Main ----------------

def main():
    lp_all = load_lp_panel()
    lproxy_sum = load_lproxy()
    port_panel = build_port_level_panel(lp_all, lproxy_sum)
    term_lp    = build_terminal_lp(lp_all)

    ports = sorted(port_panel["port"].dropna().unique().tolist())

    port_panel_out = OUT_DIR / "port_level_panel.tsv"
    port_panel.to_csv(port_panel_out, sep="\t", index=False)
    print(f"[wrote] {port_panel_out.relative_to(DESIGN_DIR)}")

    plot_overview(port_panel, ports, OUT_DIR)
    plot_lp_split(port_panel, term_lp, ports, OUT_DIR)

if __name__ == "__main__":
    main()

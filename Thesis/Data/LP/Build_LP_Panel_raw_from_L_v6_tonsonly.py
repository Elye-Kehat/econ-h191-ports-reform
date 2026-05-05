#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build_LP_Panel_raw_from_L_v6_tonsonly.py
=======================================

Purpose
-------
Extend the v5 raw-from-L LP builder so the mixed-frequency LP panel also contains
quarterly aggregate-port LP rows for the post-terminal-data era.

Why v6 exists
-------------
v5 fixed the pre-reform monthly port branch and aligned the port-specific monthly
cutoffs with the competition design. But it still stacked only:

    - monthly port rows
    - quarterly terminal rows

That meant the LP panel had no direct quarterly aggregate-port objects. For the
Model 1A privatization pivot, that is the missing ingredient.

Core v6 change
--------------
1) Keep the v5 monthly port branch unchanged.
2) Keep the quarterly terminal branch unchanged.
3) Add a new quarterly aggregate-port branch built by aggregating quarterly
   terminal numerators and denominators first:

       TEU_port_q     = sum_i TEU_i_q
       L_hours_port_q = sum_i L_hours_i_q
       tons_port_q    = sum_i tons_i_q
       LP_port_q      = TEU_port_q / L_hours_port_q

   Important: do NOT average terminal LP values.
4) Stack the new quarterly port rows into LP_Panel.tsv as:

       Haifa_port_Q
       Ashdod_port_Q

This keeps the active LP definition as raw throughput over labor while providing the
continuous aggregate-port quarterly objects required downstream by Model 1A v8.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TAB = "\t"


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=TAB, engine="python")


def _write(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=TAB, index=False)


def _to_int64(x):
    return pd.to_numeric(x, errors="coerce").astype("Int64")


def _q_from_m(m: int) -> str:
    return f"Q{(int(m) - 1) // 3 + 1}"


def _qcode(qstr: str) -> int:
    m = re.match(r"^\s*Q([1-4])\s*$", str(qstr))
    return int(m.group(1)) if m else np.nan


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return np.where((n > 0) & (d > 0), n / d, np.nan)


def _require_columns(df: pd.DataFrame, cols: List[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _encode_yq(year: pd.Series, quarter: pd.Series) -> pd.Series:
    return _to_int64(year) * 10 + pd.to_numeric(quarter.map(_qcode), errors="coerce").astype("Int64")


def find_thesis_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur] + list(cur.parents):
        if (candidate / "Data").exists() and (candidate / "Design").exists():
            return candidate
    raise FileNotFoundError("Could not find thesis root containing both Data/ and Design/.")


def _parse_yyyymm(value: int | str) -> int:
    s = str(value).strip()
    if not re.fullmatch(r"\d{6}", s):
        raise ValueError(f"Expected YYYYMM, got {value!r}")
    return int(s)


def _apply_port_specific_monthly_cutoffs(teu_m: pd.DataFrame, cutoffs: Dict[str, int], monthly_start: int) -> pd.DataFrame:
    out = teu_m.copy()
    out["ym"] = _to_int64(out["year"]) * 100 + _to_int64(out["month"])
    out = out[out["ym"] >= monthly_start].copy()

    parts = []
    for port, end_ym in cutoffs.items():
        sub = out[(out["port"] == port) & (out["ym"] <= end_ym)].copy()
        parts.append(sub)

    if not parts:
        return out.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True).sort_values(["port", "year", "month"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Monthly port LP: pre-reform port-level object
# -----------------------------------------------------------------------------

def build_monthly_port_raw(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tons_m = _read(args.s1_port_month_tons)
    teu_m = _read(args.s2_port_month_teu)
    lproxy = _read(args.lproxy)

    _require_columns(tons_m, ["port", "year", "month", "month_index", "tons_port_m"], "S1_port_month_tons")
    _require_columns(teu_m, ["port", "year", "month", "month_index", "TEU_port_m"], "S2_port_month_teu")
    _require_columns(lproxy, ["port", "year", "month", "L_hours_i_m"], "L_Proxy_v6")

    tons_m = tons_m.copy()
    teu_m = teu_m.copy()
    lproxy = lproxy.copy()

    for df in (tons_m, teu_m, lproxy):
        df["year"] = _to_int64(df["year"])
        df["month"] = _to_int64(df["month"])

    labor_p_m = (
        lproxy.groupby(["port", "year", "month"], as_index=False)["L_hours_i_m"]
        .sum(min_count=1)
        .rename(columns={"L_hours_i_m": "L_hours_port_m"})
    )

    cutoffs = {
        "Haifa": _parse_yyyymm(args.haifa_monthly_end),
        "Ashdod": _parse_yyyymm(args.ashdod_monthly_end),
    }
    teu_m = _apply_port_specific_monthly_cutoffs(teu_m, cutoffs, _parse_yyyymm(args.monthly_start))

    out = (
        teu_m.merge(
            labor_p_m[["port", "year", "month", "L_hours_port_m"]],
            on=["port", "year", "month"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            tons_m[["port", "year", "month", "month_index", "quarter", "tons_port_m", "tons_source"]],
            on=["port", "year", "month"],
            how="left",
            suffixes=("", "_tons"),
            validate="many_to_one",
        )
    )

    out["year"] = _to_int64(out["year"])
    out["month"] = _to_int64(out["month"])
    out["month_index"] = _to_int64(out["month_index"])
    if "quarter" not in out.columns or out["quarter"].isna().all():
        out["quarter"] = out["month"].apply(_q_from_m)

    out["LP_raw"] = _safe_ratio(out["TEU_port_m"], out["L_hours_port_m"])
    keep = [
        "port", "year", "month", "month_index", "quarter",
        "TEU_port_m", "tons_port_m", "tons_source",
        "L_hours_port_m", "LP_raw",
    ]
    out = out[keep].sort_values(["port", "year", "month"]).reset_index(drop=True)

    qa_rows = []
    for p in sorted(out["port"].dropna().unique().tolist()):
        sub = out[out["port"] == p].copy()
        n_rows = len(sub)
        n_labor_na = int(sub["L_hours_port_m"].isna().sum())
        n_lp_na = int(sub["LP_raw"].isna().sum())
        max_ym = int((sub["year"] * 100 + sub["month"]).max()) if n_rows else np.nan
        cutoff = cutoffs[p]

        qa_rows.append({
            "check": f"unique_{p}_port_month",
            "ok": bool(not sub.duplicated(["port", "year", "month"]).any()),
            "note": f"n={n_rows}",
        })
        qa_rows.append({
            "check": f"na_monthly_labor_{p}",
            "ok": bool(n_labor_na < n_rows),
            "note": f"{n_labor_na} NA labor rows of {n_rows}",
        })
        qa_rows.append({
            "check": f"na_lp_months_{p}",
            "ok": bool(n_lp_na < n_rows),
            "note": f"{n_lp_na} NA LP rows of {n_rows}",
        })
        qa_rows.append({
            "check": f"cutoff_{p}_monthly_branch",
            "ok": bool(pd.isna(max_ym) or max_ym <= cutoff),
            "note": f"max ym={max_ym}, cutoff={cutoff}",
        })

        if n_rows == 0:
            raise ValueError(
                f"Monthly port LP has zero rows for {p}. Check the port-specific cutoff and upstream S2 monthly TEU coverage."
            )
        if n_lp_na == n_rows:
            raise ValueError(
                f"Monthly port LP is entirely missing for {p}. Check year/month alignment across S2_port_month_teu and the L_Proxy."
            )
        if max_ym > cutoff:
            raise ValueError(
                f"Monthly port LP for {p} extends beyond its cutoff. max ym={max_ym}, cutoff={cutoff}."
            )

    qa = pd.DataFrame(qa_rows)
    return out, qa


# -----------------------------------------------------------------------------
# Quarterly terminal LP: post-reform terminal-level object
# -----------------------------------------------------------------------------

def build_quarterly_terminal_raw(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    teu_tq = _read(args.s2_term_quarter_teu)
    lproxy = _read(args.lproxy)

    _require_columns(teu_tq, ["port", "terminal", "year", "quarter", "TEU_i_q"], "S2_terminal_quarter_teu")
    _require_columns(lproxy, ["port", "terminal", "year", "month", "L_hours_i_m"], "L_Proxy_v6")

    lproxy = lproxy.copy()
    lproxy["year"] = _to_int64(lproxy["year"])
    lproxy["month"] = _to_int64(lproxy["month"])
    if "quarter" not in lproxy.columns or lproxy["quarter"].isna().all():
        lproxy["quarter"] = lproxy["month"].apply(_q_from_m)

    labor_i_q = (
        lproxy.groupby(["port", "terminal", "year", "quarter"], as_index=False)["L_hours_i_m"]
        .sum(min_count=1)
        .rename(columns={"L_hours_i_m": "L_hours_i_q"})
    )

    out = teu_tq.merge(labor_i_q, on=["port", "terminal", "year", "quarter"], how="left")

    if args.s1_terminal_quarter_tons and Path(args.s1_terminal_quarter_tons).exists():
        t_tq = _read(args.s1_terminal_quarter_tons)
        if {"port", "terminal", "year", "quarter", "tons_i_q"}.issubset(set(t_tq.columns)):
            out = out.merge(
                t_tq[["port", "terminal", "year", "quarter", "tons_i_q"]],
                on=["port", "terminal", "year", "quarter"],
                how="left",
            )
        else:
            out["tons_i_q"] = np.nan
    else:
        out["tons_i_q"] = np.nan

    out["LP_raw"] = _safe_ratio(out["TEU_i_q"], out["L_hours_i_q"])

    out["yq"] = _encode_yq(out["year"], out["quarter"])
    qs = int(str(args.quarterly_start)[:4]) * 10 + int(str(args.quarterly_start)[-1])
    qe = int(str(args.quarterly_end)[:4]) * 10 + int(str(args.quarterly_end)[-1])
    out = out[(out["yq"] >= qs) & (out["yq"] <= qe)].copy()

    keep = ["port", "terminal", "year", "quarter", "TEU_i_q", "tons_i_q", "L_hours_i_q", "LP_raw"]
    out = out[keep].sort_values(["port", "terminal", "year", "quarter"]).reset_index(drop=True)

    qa_rows = [{
        "check": "unique_terminal_quarter",
        "ok": bool(not out.duplicated(["port", "terminal", "year", "quarter"]).any()),
        "note": f"n={len(out)}",
    }]

    for term in ["Haifa-Legacy", "Haifa-Bayport", "Ashdod-Legacy", "Ashdod-HCT"]:
        sub = out[out["terminal"] == term].copy()
        qa_rows.append({
            "check": f"na_lp_quarters_{term}",
            "ok": True,
            "note": f"{int(sub['LP_raw'].isna().sum())} NA of {len(sub)}",
        })

    qa = pd.DataFrame(qa_rows)
    return out, qa


# -----------------------------------------------------------------------------
# Quarterly aggregate-port LP built from quarterly terminal numerators/denominators
# -----------------------------------------------------------------------------

def build_quarterly_port_raw(quarterly_term: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q = quarterly_term.copy()
    _require_columns(q, ["port", "year", "quarter", "TEU_i_q", "tons_i_q", "L_hours_i_q"], "quarterly_term_for_port_agg")

    out = (
        q.groupby(["port", "year", "quarter"], as_index=False)
        .agg(
            TEU_port_q=("TEU_i_q", "sum"),
            tons_port_q=("tons_i_q", "sum"),
            L_hours_port_q=("L_hours_i_q", "sum"),
        )
    )
    out["LP_raw"] = _safe_ratio(out["TEU_port_q"], out["L_hours_port_q"])
    keep = ["port", "year", "quarter", "TEU_port_q", "tons_port_q", "L_hours_port_q", "LP_raw"]
    out = out[keep].sort_values(["port", "year", "quarter"]).reset_index(drop=True)

    qa_rows = [
        {
            "check": "unique_port_quarter",
            "ok": bool(not out.duplicated(["port", "year", "quarter"]).any()),
            "note": f"n={len(out)}",
        },
        {
            "check": "na_lp_port_quarter",
            "ok": bool(out["LP_raw"].notna().any()),
            "note": f"{int(out['LP_raw'].isna().sum())} NA of {len(out)}",
        },
    ]

    for port in ["Haifa", "Ashdod"]:
        sub = out[out["port"] == port].copy()
        qa_rows.append({
            "check": f"na_quarterly_port_lp_{port}",
            "ok": bool(len(sub) > 0 and sub["LP_raw"].notna().any()),
            "note": f"{int(sub['LP_raw'].isna().sum())} NA of {len(sub)}",
        })

    return out, pd.DataFrame(qa_rows)


# -----------------------------------------------------------------------------
# Stack panel in the same mixed-frequency schema used by Model 1A
# -----------------------------------------------------------------------------

def write_component_files(monthly_port: pd.DataFrame, quarterly_term: pd.DataFrame, quarterly_port: pd.DataFrame, out_dir: str):
    _write(monthly_port[monthly_port["port"] == "Haifa"].copy(), os.path.join(out_dir, "LP_Haifa_port_month.tsv"))
    _write(monthly_port[monthly_port["port"] == "Ashdod"].copy(), os.path.join(out_dir, "LP_Ashdod_port_month.tsv"))
    _write(quarterly_port[quarterly_port["port"] == "Haifa"].copy(), os.path.join(out_dir, "LP_Haifa_port_quarter.tsv"))
    _write(quarterly_port[quarterly_port["port"] == "Ashdod"].copy(), os.path.join(out_dir, "LP_Ashdod_port_quarter.tsv"))

    mapping = {
        "Haifa-Legacy": "LP_Haifa_Legacy_quarter.tsv",
        "Haifa-Bayport": "LP_Haifa_SIPG_quarter.tsv",
        "Ashdod-Legacy": "LP_Ashdod_Legacy_quarter.tsv",
        "Ashdod-HCT": "LP_Ashdod_HCT_quarter.tsv",
    }
    for term, fname in mapping.items():
        _write(quarterly_term[quarterly_term["terminal"] == term].copy(), os.path.join(out_dir, fname))


def stack_panel(monthly_port: pd.DataFrame, quarterly_term: pd.DataFrame, quarterly_port: pd.DataFrame) -> pd.DataFrame:
    hm = monthly_port[monthly_port["port"] == "Haifa"].copy().reset_index(drop=True)
    am = monthly_port[monthly_port["port"] == "Ashdod"].copy().reset_index(drop=True)

    def _monthly_transform(df: pd.DataFrame, series_id: str, port_name: str) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        idx = df.index
        out = pd.DataFrame({
            "series_id": pd.Series([series_id] * len(df), index=idx),
            "level": pd.Series(["port"] * len(df), index=idx),
            "freq": pd.Series(["M"] * len(df), index=idx),
            "port": pd.Series([port_name] * len(df), index=idx),
            "terminal": pd.Series([pd.NA] * len(df), index=idx),
            "year": _to_int64(df["year"]),
            "month": _to_int64(df["month"]),
            "quarter": df["quarter"],
            "month_index": _to_int64(df["month_index"]),
            "quarter_index": pd.Series([pd.NA] * len(df), dtype="Int64", index=idx),
            "TEU": df["TEU_port_m"],
            "tons": df["tons_port_m"],
            "L_hours": df["L_hours_port_m"],
            "w": pd.Series([1.0] * len(df), index=idx),
            "Pi": pd.Series([pd.NA] * len(df), index=idx),
            "LP": df["LP_raw"],
            "LP_id": df["LP_raw"],
            "tons_source": df["tons_source"] if "tons_source" in df.columns else pd.Series([pd.NA] * len(df), index=idx),
            "lp_definition": pd.Series(["raw_teu_over_labor"] * len(df), index=idx),
        })
        return out.reset_index(drop=True)

    def _quarterly_transform(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        idx = df.index
        qidx = _to_int64(df["year"]) * 4 + pd.to_numeric(df["quarter"].map(_qcode), errors="coerce").astype("Int64")
        out = pd.DataFrame({
            "series_id": pd.Series([series_id] * len(df), index=idx),
            "level": pd.Series(["terminal"] * len(df), index=idx),
            "freq": pd.Series(["Q"] * len(df), index=idx),
            "port": df["port"],
            "terminal": df["terminal"],
            "year": _to_int64(df["year"]),
            "month": pd.Series([pd.NA] * len(df), dtype="Int64", index=idx),
            "quarter": df["quarter"],
            "month_index": pd.Series([pd.NA] * len(df), dtype="Int64", index=idx),
            "quarter_index": qidx,
            "TEU": df["TEU_i_q"],
            "tons": df["tons_i_q"],
            "L_hours": df["L_hours_i_q"],
            "w": pd.Series([1.0] * len(df), index=idx),
            "Pi": pd.Series([pd.NA] * len(df), index=idx),
            "LP": df["LP_raw"],
            "LP_id": df["LP_raw"],
            "tons_source": pd.Series([pd.NA] * len(df), index=idx),
            "lp_definition": pd.Series(["raw_teu_over_labor"] * len(df), index=idx),
        })
        return out.reset_index(drop=True)

    def _quarterly_port_transform(df: pd.DataFrame, series_id: str, port_name: str) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        idx = df.index
        qidx = _to_int64(df["year"]) * 4 + pd.to_numeric(df["quarter"].map(_qcode), errors="coerce").astype("Int64")
        out = pd.DataFrame({
            "series_id": pd.Series([series_id] * len(df), index=idx),
            "level": pd.Series(["port"] * len(df), index=idx),
            "freq": pd.Series(["Q"] * len(df), index=idx),
            "port": pd.Series([port_name] * len(df), index=idx),
            "terminal": pd.Series([pd.NA] * len(df), index=idx),
            "year": _to_int64(df["year"]),
            "month": pd.Series([pd.NA] * len(df), dtype="Int64", index=idx),
            "quarter": df["quarter"],
            "month_index": pd.Series([pd.NA] * len(df), dtype="Int64", index=idx),
            "quarter_index": qidx,
            "TEU": df["TEU_port_q"],
            "tons": df["tons_port_q"],
            "L_hours": df["L_hours_port_q"],
            "w": pd.Series([1.0] * len(df), index=idx),
            "Pi": pd.Series([pd.NA] * len(df), index=idx),
            "LP": df["LP_raw"],
            "LP_id": df["LP_raw"],
            "tons_source": pd.Series([pd.NA] * len(df), index=idx),
            "lp_definition": pd.Series(["raw_teu_over_labor"] * len(df), index=idx),
        })
        return out.reset_index(drop=True)

    pieces = [
        _monthly_transform(hm, "Haifa_port_M", "Haifa"),
        _monthly_transform(am, "Ashdod_port_M", "Ashdod"),
        _quarterly_port_transform(quarterly_port[quarterly_port["port"] == "Haifa"].copy(), "Haifa_port_Q", "Haifa"),
        _quarterly_port_transform(quarterly_port[quarterly_port["port"] == "Ashdod"].copy(), "Ashdod_port_Q", "Ashdod"),
        _quarterly_transform(quarterly_term[quarterly_term["terminal"] == "Haifa-Legacy"].copy(), "Haifa_Legacy_Q"),
        _quarterly_transform(quarterly_term[quarterly_term["terminal"] == "Haifa-Bayport"].copy(), "Haifa_SIPG_Q"),
        _quarterly_transform(quarterly_term[quarterly_term["terminal"] == "Ashdod-Legacy"].copy(), "Ashdod_Legacy_Q"),
        _quarterly_transform(quarterly_term[quarterly_term["terminal"] == "Ashdod-HCT"].copy(), "Ashdod_HCT_Q"),
    ]

    cols = [
        "series_id", "level", "freq", "port", "terminal",
        "year", "month", "quarter", "month_index", "quarter_index",
        "TEU", "tons", "L_hours", "w", "Pi", "LP", "LP_id", "tons_source", "lp_definition",
    ]
    panel = pd.concat(pieces, ignore_index=True)[cols]
    return panel.sort_values(["series_id", "year", "month", "quarter"]).reset_index(drop=True)


def build_qa(panel: pd.DataFrame, qa_month: pd.DataFrame, qa_quarter: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, ok: bool, note: str):
        rows.append({"check": check, "ok": bool(ok), "note": note})

    rows.extend(qa_month.to_dict(orient="records"))
    rows.extend(qa_quarter.to_dict(orient="records"))

    m = panel[panel["freq"] == "M"].copy()
    q = panel[panel["freq"] == "Q"].copy()
    q_port = panel[(panel["freq"] == "Q") & (panel["level"] == "port")].copy()

    add("unique_monthly_series", not m.duplicated(["series_id", "year", "month"]).any(), f"n={len(m)}")
    add("unique_quarterly_series", not q.duplicated(["series_id", "year", "quarter"]).any(), f"n={len(q)}")
    add(
        "unique_port_quarterly_series",
        not q_port.duplicated(["series_id", "year", "quarter"]).any(),
        f"n={len(q_port)}",
    )

    for sid, grp in panel.groupby("series_id"):
        add(
            f"na_rates_{sid}",
            True,
            f"NA TEU={int(grp['TEU'].isna().sum())}, NA L_hours={int(grp['L_hours'].isna().sum())}, NA LP={int(grp['LP'].isna().sum())}"
        )

    for sid in ["Haifa_port_Q", "Ashdod_port_Q"]:
        grp = panel[panel["series_id"] == sid].copy()
        add(
            f"nonmissing_{sid}",
            bool(len(grp) > 0 and grp["LP"].notna().any()),
            f"{int(grp['LP'].isna().sum())} NA LP of {len(grp)} rows",
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    thesis_root = find_thesis_root(script_dir)

    parser.add_argument("--s1_port_month_tons", default=str(thesis_root / "Data" / "LP" / "S1_port_month_tons.tsv"))
    parser.add_argument("--s1_terminal_quarter_tons", default=str(thesis_root / "Data" / "LP" / "S1_terminal_quarter_tons.tsv"))
    parser.add_argument("--s2_port_month_teu", default=str(thesis_root / "Data" / "LP" / "S2_port_month_teu.tsv"))
    parser.add_argument("--s2_term_quarter_teu", default=str(thesis_root / "Data" / "LP" / "S2_terminal_quarter_teu.tsv"))
    parser.add_argument("--lproxy", default=str(thesis_root / "Data" / "L_proxy" / "common_rule_v6_tonsonly" / "L_Proxy_commonrule_v6_tonsonly.tsv"))
    parser.add_argument("--out", default=str(thesis_root / "Data" / "LP" / "raw_from_l_v6_tonsonly"))

    parser.add_argument("--monthly_start", type=int, default=201801)
    parser.add_argument("--haifa_monthly_end", type=int, default=202108)
    parser.add_argument("--ashdod_monthly_end", type=int, default=202209)

    parser.add_argument("--quarterly_start", type=str, default="2021Q3")
    parser.add_argument("--quarterly_end", type=str, default="2024Q4")
    parser.add_argument("--also-write-canonical", action="store_true", help="Also overwrite Data/LP/LP_Panel.tsv with the new build")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    monthly_port, qa_m = build_monthly_port_raw(args)
    quarterly_term, qa_q = build_quarterly_terminal_raw(args)
    quarterly_port, qa_q_port = build_quarterly_port_raw(quarterly_term)

    write_component_files(monthly_port, quarterly_term, quarterly_port, args.out)

    panel = stack_panel(monthly_port, quarterly_term, quarterly_port)
    qa = build_qa(panel, qa_m, pd.concat([qa_q, qa_q_port], ignore_index=True))

    panel_path = os.path.join(args.out, "LP_Panel.tsv")
    qa_path = os.path.join(args.out, "LP_raw_qa.tsv")
    _write(panel, panel_path)
    _write(qa, qa_path)

    meta = {
        "builder": "Build_LP_Panel_raw_from_L_v6_tonsonly.py",
        "definition": {
            "monthly_port": "LP_{p,m} = TEU_{p,m} / L_{p,m}",
            "quarterly_terminal": "LP_{i,q} = TEU_{i,q} / L_{i,q}",
            "quarterly_port": "LP_{p,q} = sum_i TEU_{i,q} / sum_i L_{i,q}",
            "quarterly_labor": "L_{i,q} = sum_{m in q} L_{i,m}",
        },
        "why_v6": [
            "keeps the v5 monthly port labor merge fix and port-specific monthly cutoffs",
            "keeps the quarterly terminal LP branch unchanged",
            "adds a quarterly aggregate-port branch built from aggregated terminal numerators and denominators",
            "stacks Haifa_port_Q and Ashdod_port_Q into LP_Panel.tsv for downstream aggregate-port analysis",
        ],
        "inputs": {
            "s1_port_month_tons": args.s1_port_month_tons,
            "s1_terminal_quarter_tons": args.s1_terminal_quarter_tons,
            "s2_port_month_teu": args.s2_port_month_teu,
            "s2_term_quarter_teu": args.s2_term_quarter_teu,
            "lproxy": args.lproxy,
        },
        "monthly_cutoffs": {
            "monthly_start": args.monthly_start,
            "haifa_monthly_end": args.haifa_monthly_end,
            "ashdod_monthly_end": args.ashdod_monthly_end,
        },
        "outputs": {
            "LP_Panel": panel_path,
            "LP_raw_qa": qa_path,
        },
    }

    meta_path = os.path.join(args.out, "_meta_lp_raw_v6_tonsonly.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.also_write_canonical:
        canonical = thesis_root / "Data" / "LP" / "LP_Panel.tsv"
        _write(panel, str(canonical))
        meta["outputs"]["canonical_LP_Panel"] = str(canonical)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("[LP_raw_v6_tonsonly] Wrote raw throughput-over-labor LP artifacts to", args.out)


if __name__ == "__main__":
    main()

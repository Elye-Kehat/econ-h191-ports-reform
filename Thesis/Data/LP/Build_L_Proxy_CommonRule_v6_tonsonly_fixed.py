#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build_L_Proxy_CommonRule_v6_tonsonly.py
======================================

Purpose
-------
Build a new labor proxy that preserves the existing annual KPI anchor but removes
TEU from within-year labor movement.

This file is the direct implementation of the revised design discussed in the
thesis workflow:

    1) Keep the annual labor anchor:
           H_{i,y} = TEU_{i,y} / Pi^{work}_{i,y}

    2) Remove TEU from within-year labor movement.
       Quarterly labor is allocated using terminal-quarter tons only:
           L_{i,q} = H_{i,y} * tons_{i,q} / sum_{r in y} tons_{i,r}

    3) Monthly labor is then obtained using monthly tons shares within each
       quarter, rather than quarter-flat monthlyization:
           L_{i,m} = L_{i,q} * tons_{i,m} / tons_{i,q}
       with a simple equal-split fallback only when a positive-labor quarter has
       zero / missing monthly tons.

Why this version exists
-----------------------
The v5 common-rule labor proxy still used a TEU-linked within-year activity rule:

    activity_{i,q} = share_{i,p,q} * tons_{p,q}

where the terminal share itself came from terminal-quarter TEU. That means TEU was
entering both the labor denominator and the LP numerator. The goal of v6 is to keep
TEU only at the annual anchor and to let tons alone determine within-year labor
movement.

Key design choices
------------------
1) Annual anchor is unchanged
   We keep the KPI-based annual anchor because it is still the cleanest way to get
   annual implied labor hours from the available data.

2) Quarterly allocation is tons-only
   This is the main conceptual rewrite. Once annual hours are fixed, quarter weights
   come only from terminal-quarter tons shares within the terminal-year.

3) Monthlyization also uses tons-only
   Even though Model 1A v7 is quarterly only, we still write a monthly labor proxy for
   pipeline compatibility. Unlike v5, we do not spread quarter labor flatly across the
   three months unless monthly tons information is unavailable inside a positive-labor
   quarter.

4) Entrant KPI backfill remains deterministic
   The user explicitly did not want arbitrary startup hyperparameters. Therefore this
   version keeps the existing deterministic KPI backfill logic:
       entrant KPI in 2021 / 2022 = alpha_2023 * legacy KPI same year
   when the entrant KPI is missing and both entrant-2023 and legacy-2023 are observed.

5) No silent redefinition of output concepts
   This file is only about labor. LP will still be built downstream as raw TEU over
   labor.

Primary outputs
---------------
By default, outputs are written to:
    Data/L_proxy/common_rule_v6_tonsonly/

and include:
    labor_hours_monthly_terminal_commonrule_v6_tonsonly.tsv
    labor_hours_monthly_port_commonrule_v6_tonsonly.tsv
    labor_anchor_terminal_year_commonrule_v6_tonsonly.tsv
    L_Proxy_commonrule_v6_tonsonly.tsv
    lproxy_commonrule_v6_tonsonly_qa.tsv
    lproxy_commonrule_v6_tonsonly_old_new_anchor_compare.tsv
    _meta_l_proxy_commonrule_v6_tonsonly.json

Optional convenience
--------------------
If --also-write-canonical is passed, the script also writes:
    Data/L_proxy/L_Proxy.tsv

so downstream code that expects the canonical labor proxy path can use the new build
without manual copying.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"
PORTS_KEEP = {"Haifa", "Ashdod"}
ENTRANT_BY_PORT = {"Haifa": "Haifa-Bayport", "Ashdod": "Ashdod-HCT"}
LEGACY_BY_PORT = {"Haifa": "Haifa-Legacy", "Ashdod": "Ashdod-Legacy"}

TONS_NAME_MAP = {
    "Haifa": ("Haifa", None),
    "Ashdod": ("Ashdod", None),
    "Haifa SIPG": ("Haifa", "Haifa-Bayport"),
    "Ashdod HCT": ("Ashdod", "Ashdod-HCT"),
}

TEU_NAME_MAP = {
    "Haifa": ("Haifa", None),
    "Ashdod": ("Ashdod", None),
    "Haifa SIPG": ("Haifa", "Haifa-Bayport"),
    "Ashdod HCT": ("Ashdod", "Ashdod-HCT"),
}

KPI_TERMINAL_ALIASES = {
    "Haifa": "Haifa-Legacy",
    "Ashdod": "Ashdod-Legacy",
    "Haifa-Legacy": "Haifa-Legacy",
    "Ashdod-Legacy": "Ashdod-Legacy",
    "Haifa SIPG": "Haifa-Bayport",
    "Haifa-Bayport": "Haifa-Bayport",
    "Bayport": "Haifa-Bayport",
    "Ashdod HCT": "Ashdod-HCT",
    "Southport": "Ashdod-HCT",
    "HCT": "Ashdod-HCT",
    "Ashdod-HCT": "Ashdod-HCT",
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    for d in DASHES:
        s = s.replace(d, "-")
    return s


def find_thesis_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur] + list(cur.parents):
        if (candidate / "Data").exists() and (candidate / "Design").exists():
            return candidate
    raise FileNotFoundError("Could not find thesis root containing both Data/ and Design/.")


def parse_yyyymm(value: str) -> Tuple[int, int]:
    s = str(value).strip()
    if re.fullmatch(r"\d{6}", s):
        return int(s[:4]), int(s[4:6])
    raise ValueError(f"Expected YYYYMM, got {value!r}")


def parse_mm_yyyy(series: pd.Series) -> pd.DataFrame:
    s = series.astype(str).str.strip()
    out = s.str.extract(r"^(?P<month>\d{2})-(?P<year>\d{4})$")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def quarter_from_month(m: int) -> str:
    if pd.isna(m):
        return np.nan
    return f"Q{((int(m) - 1) // 3) + 1}"


def quarter_end_month(q: str) -> int:
    q = normalize_text(q).upper()
    mapping = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12, "1": 3, "2": 6, "3": 9, "4": 12}
    if q not in mapping:
        raise ValueError(f"Could not parse quarter value: {q!r}")
    return mapping[q]


def quarter_index(year: int, quarter: str) -> int:
    qnum = int(normalize_text(quarter).upper().replace("Q", ""))
    return int(year) * 10 + qnum


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def coalesce_teu(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if "TEU" in df.columns:
        out = pd.to_numeric(df["TEU"], errors="coerce")
    if "TEU_thousands" in df.columns:
        out = out.fillna(pd.to_numeric(df["TEU_thousands"], errors="coerce") * 1000.0)
    return out.astype(float)


def choose_first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exist:\n" + "\n".join(str(p) for p in paths))


# -----------------------------------------------------------------------------
# Stage A: monthly tons with full terminal coverage
# -----------------------------------------------------------------------------

def load_monthly_tons(tons_path: Path, sample_start: int, sample_end: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build port-month tons and all-terminal month tons directly from the raw tons
    file.

    Important logic:
    - Canonical port totals come from raw port rows when available.
    - Entrant terminal tons come from observed terminal rows.
    - Legacy terminal tons are the residual:
          tons_legacy = tons_port - tons_entrant
    - Before entrant activity begins, legacy tons equal the full port total.

    This mirrors the robust logic already adopted in Stage 1 v2 of the LP pipeline.
    """
    raw = read_tsv(tons_path).copy()
    need = {"PortOrTerminal", "Month-Year", "tons_k"}
    miss = need - set(raw.columns)
    if miss:
        raise ValueError(f"Tons file missing columns: {sorted(miss)}")

    parsed = parse_mm_yyyy(raw["Month-Year"])
    raw["year"] = parsed["year"]
    raw["month"] = parsed["month"]
    raw["month_index"] = raw["year"] * 100 + raw["month"]
    raw["quarter"] = raw["month"].map(quarter_from_month)
    raw["name"] = raw["PortOrTerminal"].map(normalize_text)
    raw["tons"] = pd.to_numeric(raw["tons_k"], errors="coerce") * 1000.0

    raw = raw.loc[raw["name"].isin(TONS_NAME_MAP)].copy()
    raw["port"] = raw["name"].map(lambda x: TONS_NAME_MAP[x][0])
    raw["terminal"] = raw["name"].map(lambda x: TONS_NAME_MAP[x][1])
    raw = raw.loc[(raw["month_index"] >= sample_start) & (raw["month_index"] <= sample_end)].copy()

    port_rows = (
        raw.loc[raw["terminal"].isna(), ["port", "year", "month", "month_index", "quarter", "tons"]]
        .groupby(["port", "year", "month", "month_index", "quarter"], as_index=False)["tons"]
        .sum(min_count=1)
        .rename(columns={"tons": "tons_port_m"})
    )

    entrant_rows = (
        raw.loc[raw["terminal"].notna(), ["port", "terminal", "year", "month", "month_index", "quarter", "tons"]]
        .groupby(["port", "terminal", "year", "month", "month_index", "quarter"], as_index=False)["tons"]
        .sum(min_count=1)
        .rename(columns={"tons": "tons_i_m"})
    )

    frames = []
    qa_rows = []

    for port in sorted(PORTS_KEEP):
        entrant = ENTRANT_BY_PORT[port]
        legacy = LEGACY_BY_PORT[port]

        base = port_rows.loc[port_rows["port"].eq(port)].copy()
        ent = entrant_rows.loc[(entrant_rows["port"].eq(port)) & (entrant_rows["terminal"].eq(entrant))].copy()
        ent = ent.rename(columns={"tons_i_m": "tons_entrant_m"})

        merged = base.merge(
            ent[["port", "year", "month", "month_index", "quarter", "tons_entrant_m"]],
            on=["port", "year", "month", "month_index", "quarter"],
            how="left",
        )
        merged["tons_entrant_m"] = merged["tons_entrant_m"].fillna(0.0)
        merged["tons_legacy_raw_m"] = merged["tons_port_m"] - merged["tons_entrant_m"]
        merged["legacy_negative_flag"] = merged["tons_legacy_raw_m"] < -1e-6
        merged["tons_legacy_m"] = merged["tons_legacy_raw_m"].clip(lower=0.0)

        leg = merged[["port", "year", "month", "month_index", "quarter", "tons_port_m", "tons_legacy_m"]].copy()
        leg["terminal"] = legacy
        leg = leg.rename(columns={"tons_legacy_m": "tons_i_m"})
        leg["tons_source"] = np.where(merged["tons_entrant_m"] > 0, "residual_port_minus_entrant", "full_port_pre_entry")

        ent_out = merged[["port", "year", "month", "month_index", "quarter", "tons_port_m", "tons_entrant_m"]].copy()
        ent_out["terminal"] = entrant
        ent_out = ent_out.rename(columns={"tons_entrant_m": "tons_i_m"})
        ent_out = ent_out[ent_out["tons_i_m"] > 0].copy()
        ent_out["tons_source"] = "observed_entrant_terminal"

        frames.append(leg)
        frames.append(ent_out)
        qa_rows.append(
            merged[["port", "year", "month", "month_index", "tons_port_m", "tons_entrant_m", "tons_legacy_raw_m", "legacy_negative_flag"]]
        )

    terminal_tons = pd.concat(frames, ignore_index=True)
    terminal_tons = terminal_tons.sort_values(["port", "terminal", "year", "month"]).reset_index(drop=True)
    qa_tons = pd.concat(qa_rows, ignore_index=True).sort_values(["port", "year", "month"]).reset_index(drop=True)

    return port_rows.sort_values(["port", "year", "month"]).reset_index(drop=True), terminal_tons, qa_tons


# -----------------------------------------------------------------------------
# Stage B: TEU tables and terminal-month TEU monthlyization
# -----------------------------------------------------------------------------

def load_teu_panel(teu_path: Path, sample_start: int, sample_end: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_tsv(teu_path).copy()
    need = {"Port", "Period", "Freq", "Year"}
    miss = need - set(raw.columns)
    if miss:
        raise ValueError(f"TEU panel missing columns: {sorted(miss)}")

    raw["name"] = raw["Port"].map(normalize_text)
    raw = raw.loc[raw["name"].isin(TEU_NAME_MAP)].copy()
    raw["port"] = raw["name"].map(lambda x: TEU_NAME_MAP[x][0])
    raw["terminal"] = raw["name"].map(lambda x: TEU_NAME_MAP[x][1])
    raw["TEU_val"] = coalesce_teu(raw)

    # Monthly port TEU
    m = raw.loc[raw["Freq"].astype(str).str.lower().eq("monthly") & raw["terminal"].isna()].copy()
    if "MonthIndex" in m.columns:
        m["month_index"] = pd.to_numeric(m["MonthIndex"], errors="coerce").astype("Int64")
        m["month"] = (m["month_index"] % 100).astype("Int64")
    else:
        parsed = parse_mm_yyyy(m["Period"])
        m["month"] = parsed["month"]
        m["month_index"] = m["Year"].astype("Int64") * 100 + m["month"]
    m["year"] = pd.to_numeric(m["Year"], errors="coerce").astype("Int64")
    m["quarter"] = m["month"].map(quarter_from_month)
    m = m.loc[(m["month_index"] >= sample_start) & (m["month_index"] <= sample_end)].copy()
    m_port = (
        m[["port", "year", "month", "month_index", "quarter", "TEU_val"]]
        .rename(columns={"TEU_val": "TEU_port_m_obs"})
        .sort_values(["port", "year", "month"])
    )

    # Quarterly terminal TEU
    q = raw.loc[raw["Freq"].astype(str).str.lower().eq("quarterly")].copy()
    q["year"] = pd.to_numeric(q["Year"], errors="coerce").astype("Int64")
    q["quarter"] = q["Period"].astype(str).str.extract(r"(Q\d)", expand=False).map(normalize_text)
    q = q.loc[q["quarter"].notna()].copy()
    q["month_index"] = q["year"] * 100 + q["quarter"].map(quarter_end_month)
    q = q.loc[(q["month_index"] >= sample_start) & (q["month_index"] <= sample_end)].copy()

    q_legacy = (
        q.loc[q["terminal"].isna(), ["port", "year", "quarter", "TEU_val"]]
        .groupby(["port", "year", "quarter"], as_index=False)["TEU_val"]
        .sum(min_count=1)
        .rename(columns={"TEU_val": "TEU_i_q"})
    )

    # Pre-entry legacy quarterly TEU from monthly port TEU for 2018-2019.
    pre_q = (
        m_port.loc[m_port["year"].isin([2018, 2019])]
        .groupby(["port", "year", "quarter"], as_index=False)["TEU_port_m_obs"]
        .sum(min_count=1)
        .rename(columns={"TEU_port_m_obs": "TEU_i_q"})
    )

    q_legacy_full = pd.concat([q_legacy, pre_q], ignore_index=True)
    q_legacy_full = q_legacy_full.sort_values(["port", "year", "quarter"]).drop_duplicates(["port", "year", "quarter"], keep="last")

    legacy_frames = []
    for port in sorted(PORTS_KEEP):
        d = q_legacy_full.loc[q_legacy_full["port"].eq(port)].copy()
        d["terminal"] = LEGACY_BY_PORT[port]
        legacy_frames.append(d)
    q_legacy_term = pd.concat(legacy_frames, ignore_index=True)

    q_ent = (
        q.loc[q["terminal"].notna(), ["port", "terminal", "year", "quarter", "TEU_val"]]
        .groupby(["port", "terminal", "year", "quarter"], as_index=False)["TEU_val"]
        .sum(min_count=1)
        .rename(columns={"TEU_val": "TEU_i_q"})
    )

    q_term = pd.concat([q_legacy_term, q_ent], ignore_index=True)
    q_term = q_term.sort_values(["port", "terminal", "year", "quarter"]).reset_index(drop=True)
    return m_port, q_term


def monthlyize_terminal_teu(terminal_tons: pd.DataFrame, q_term: pd.DataFrame, m_port: pd.DataFrame) -> pd.DataFrame:
    """
    Monthlyize terminal-quarter TEU to terminal-month TEU using monthly tons shares
    within quarter.

    This monthlyization is used only to compute annual TEU totals for the labor
    anchor and to carry a schema-compatible monthly TEU field through L_Proxy.
    The headline LP object remains quarterly after reform.
    """
    t = terminal_tons.copy()
    q = q_term.copy()
    q["TEU_i_q"] = pd.to_numeric(q["TEU_i_q"], errors="coerce").fillna(0.0)

    t = t.merge(q, on=["port", "terminal", "year", "quarter"], how="left")
    t["TEU_i_q"] = t["TEU_i_q"].fillna(0.0)

    sum_q = (
        t.groupby(["port", "terminal", "year", "quarter"], as_index=False)["tons_i_m"]
        .sum(min_count=1)
        .rename(columns={"tons_i_m": "tons_i_q"})
    )
    t = t.merge(sum_q, on=["port", "terminal", "year", "quarter"], how="left")

    t["tons_weight_m_in_q"] = np.where(t["tons_i_q"] > 0, t["tons_i_m"] / t["tons_i_q"], np.nan)
    zero_q_mask = t["tons_i_q"].fillna(0.0).eq(0.0) & t["TEU_i_q"].fillna(0.0).gt(0.0)
    fallback_equal_q = pd.Series(np.where(zero_q_mask, 1.0 / 3.0, np.nan), index=t.index, dtype="float64")
    t["tons_weight_m_in_q"] = t["tons_weight_m_in_q"].fillna(fallback_equal_q)
    t["tons_weight_m_in_q"] = t["tons_weight_m_in_q"].fillna(0.0)
    t["TEU_i_m_model"] = t["TEU_i_q"] * t["tons_weight_m_in_q"]

    out = []
    for port in sorted(PORTS_KEEP):
        legacy = LEGACY_BY_PORT[port]
        entrant = ENTRANT_BY_PORT[port]

        base = t.loc[(t["port"].eq(port)) & (t["terminal"].eq(legacy))].copy()
        ent_month = (
            terminal_tons.loc[(terminal_tons["port"].eq(port)) & (terminal_tons["terminal"].eq(entrant)), ["year", "month", "tons_i_m"]]
            .rename(columns={"tons_i_m": "tons_entrant_m"})
        )
        base = base.merge(ent_month, on=["year", "month"], how="left")
        base["tons_entrant_m"] = base["tons_entrant_m"].fillna(0.0)

        m_obs = m_port.loc[m_port["port"].eq(port), ["year", "month", "TEU_port_m_obs"]].copy()
        base = base.merge(m_obs, on=["year", "month"], how="left")
        use_obs = base["TEU_port_m_obs"].notna() & base["tons_entrant_m"].eq(0.0)
        base["TEU_i_m"] = np.where(use_obs, base["TEU_port_m_obs"], base["TEU_i_m_model"])
        base["teu_month_source"] = np.where(use_obs, "observed_port_month_pre_entry", "quarter_benchmarked_by_tons")
        out.append(base)

        e = t.loc[(t["port"].eq(port)) & (t["terminal"].eq(entrant))].copy()
        e["TEU_i_m"] = e["TEU_i_m_model"]
        e["teu_month_source"] = "quarter_benchmarked_by_tons"
        out.append(e)

    out = pd.concat(out, ignore_index=True).sort_values(["port", "terminal", "year", "month"]).reset_index(drop=True)

    port_month_teu = (
        out.groupby(["port", "year", "month"], as_index=False)["TEU_i_m"]
        .sum(min_count=1)
        .rename(columns={"TEU_i_m": "TEU_port_m"})
    )
    out = out.merge(port_month_teu, on=["port", "year", "month"], how="left")

    qsum = (
        out.groupby(["port", "year", "quarter"], as_index=False)["TEU_i_q"]
        .sum(min_count=1)
        .rename(columns={"TEU_i_q": "TEU_port_q"})
    )
    qterm = out[["port", "terminal", "year", "quarter", "TEU_i_q"]].drop_duplicates().merge(qsum, on=["port", "year", "quarter"], how="left")
    qterm["share_i_p_q"] = np.where(qterm["TEU_port_q"] > 0, qterm["TEU_i_q"] / qterm["TEU_port_q"], 0.0)
    out = out.merge(qterm[["port", "terminal", "year", "quarter", "share_i_p_q"]], on=["port", "terminal", "year", "quarter"], how="left")
    out["share_i_p_q"] = out["share_i_p_q"].fillna(0.0)

    return out[[
        "port", "terminal", "year", "month", "month_index", "quarter",
        "tons_port_m", "tons_i_m", "tons_i_q", "tons_weight_m_in_q",
        "TEU_i_q", "TEU_i_m", "TEU_port_m", "share_i_p_q", "teu_month_source"
    ]]


# -----------------------------------------------------------------------------
# Stage C: KPI handling and annual anchor
# -----------------------------------------------------------------------------

def standardize_kpi_columns(kpi: pd.DataFrame) -> pd.DataFrame:
    kpi = kpi.copy()
    original = {c: normalize_text(c).lower() for c in kpi.columns}
    kpi = kpi.rename(columns=original)

    rename_map = {}
    for c in ["port", "harbor", "port_name"]:
        if c in kpi.columns:
            rename_map[c] = "port"
            break
    for c in ["terminal", "entity", "unit", "series", "operator"]:
        if c in kpi.columns:
            rename_map[c] = "terminal"
            break
    for c in ["year", "y"]:
        if c in kpi.columns:
            rename_map[c] = "year"
            break
    for c in [
        "teu_per_work_hour", "teu_per_hour", "pi_teu_per_hour", "kpi_teu_per_hour",
        "teu/work_hour", "teu_per_workhr"
    ]:
        if c in kpi.columns:
            rename_map[c] = "teu_per_work_hour"
            break

    kpi = kpi.rename(columns=rename_map)
    required = {"port", "terminal", "year", "teu_per_work_hour"}
    missing = required - set(kpi.columns)
    if missing:
        raise ValueError(
            "KPI table must contain at least port, terminal, year, teu_per_work_hour "
            f"after normalization. Missing: {sorted(missing)}"
        )

    kpi = kpi[["port", "terminal", "year", "teu_per_work_hour"]].copy()
    kpi["port"] = kpi["port"].map(normalize_text)
    kpi["terminal"] = kpi["terminal"].map(lambda x: KPI_TERMINAL_ALIASES.get(normalize_text(x), normalize_text(x)))
    mask = kpi["terminal"].isin(["Haifa", "Ashdod"])
    kpi.loc[mask, "terminal"] = kpi.loc[mask, "terminal"].map(LEGACY_BY_PORT)
    kpi["year"] = pd.to_numeric(kpi["year"], errors="coerce").astype("Int64")
    kpi["teu_per_work_hour"] = pd.to_numeric(kpi["teu_per_work_hour"], errors="coerce")
    kpi = kpi.dropna(subset=["year"]).copy()
    kpi["year"] = kpi["year"].astype(int)
    return kpi.sort_values(["port", "terminal", "year"]).reset_index(drop=True)


def backfill_entrant_kpis_deterministic(kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic entrant KPI backfill retained from v5.

    For each port, if the entrant has a 2023 work-hour KPI and the legacy terminal
    has a 2023 work-hour KPI, define:
        alpha = entrant_2023 / legacy_2023

    Then if entrant KPI is missing in 2021 or 2022 but legacy KPI exists in that
    same year, backfill with:
        entrant_y = alpha * legacy_y

    This is parameter-free and preserves the user's preference against arbitrary
    startup hyperparameters.
    """
    k = kpis.copy()
    rows = []

    for port in sorted(PORTS_KEEP):
        entrant = ENTRANT_BY_PORT[port]
        legacy = LEGACY_BY_PORT[port]

        entrant_2023 = k.loc[(k["port"].eq(port)) & (k["terminal"].eq(entrant)) & (k["year"].eq(2023)), "teu_per_work_hour"].dropna()
        legacy_2023 = k.loc[(k["port"].eq(port)) & (k["terminal"].eq(legacy)) & (k["year"].eq(2023)), "teu_per_work_hour"].dropna()
        if entrant_2023.empty or legacy_2023.empty or float(legacy_2023.iloc[0]) <= 0:
            continue

        alpha = float(entrant_2023.iloc[0]) / float(legacy_2023.iloc[0])

        for year in [2021, 2022]:
            mask_have = (k["port"].eq(port)) & (k["terminal"].eq(entrant)) & (k["year"].eq(year))
            have = k.loc[mask_have, "teu_per_work_hour"]
            if not have.empty and have.notna().any():
                continue

            legacy_same_year = k.loc[(k["port"].eq(port)) & (k["terminal"].eq(legacy)) & (k["year"].eq(year)), "teu_per_work_hour"].dropna()
            if legacy_same_year.empty or float(legacy_same_year.iloc[0]) <= 0:
                continue

            rows.append({
                "port": port,
                "terminal": entrant,
                "year": int(year),
                "teu_per_work_hour": alpha * float(legacy_same_year.iloc[0]),
                "kpi_backfill_source": "alpha2023_times_legacy_sameyear",
            })

    if rows:
        add = pd.DataFrame(rows)
        if "kpi_backfill_source" not in k.columns:
            k["kpi_backfill_source"] = np.nan
        k = pd.concat([k, add], ignore_index=True, sort=False)
        k["has_work"] = k["teu_per_work_hour"].notna().astype(int)
        k = k.sort_values(["port", "terminal", "year", "has_work"]).drop_duplicates(["port", "terminal", "year"], keep="last")
        k = k.drop(columns=["has_work"])
    else:
        if "kpi_backfill_source" not in k.columns:
            k["kpi_backfill_source"] = np.nan

    return k.sort_values(["port", "terminal", "year"]).reset_index(drop=True)


def load_kpis(kpi_path: Path) -> pd.DataFrame:
    kpi = read_tsv(kpi_path)
    kpi = standardize_kpi_columns(kpi)
    kpi = backfill_entrant_kpis_deterministic(kpi)
    return kpi


def compute_annual_anchor(teu_monthly: pd.DataFrame, kpis: pd.DataFrame) -> pd.DataFrame:
    annual_teu = (
        teu_monthly.groupby(["port", "terminal", "year"], as_index=False)["TEU_i_m"]
        .sum(min_count=1)
        .rename(columns={"TEU_i_m": "TEU_i_y"})
    )
    ann = annual_teu.merge(kpis, on=["port", "terminal", "year"], how="left")
    ann["anchor_mode"] = "work_only"
    ann["Pi_teu_per_hour_i_y"] = ann["teu_per_work_hour"]
    ann["H_annual_i_y"] = ann["TEU_i_y"] / ann["Pi_teu_per_hour_i_y"]
    return ann.sort_values(["port", "terminal", "year"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Stage D: tons-only within-year labor allocation
# -----------------------------------------------------------------------------

def build_quarter_grid_from_tons(terminal_tons: pd.DataFrame, annual_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Build the quarter grid used for labor allocation.

    This is the crucial v6 rewrite: quarter weights are based only on terminal-quarter
    tons within the terminal-year. TEU shares do not enter here.
    """
    q = (
        terminal_tons.groupby(["port", "terminal", "year", "quarter"], as_index=False)
        .agg(
            tons_i_q=("tons_i_m", lambda s: s.sum(min_count=1)),
            months_in_q=("month", "nunique"),
        )
        .sort_values(["port", "terminal", "year", "quarter"])
        .reset_index(drop=True)
    )
    q["quarter_index"] = q.apply(lambda r: quarter_index(int(r["year"]), str(r["quarter"])), axis=1)

    annual_tons = (
        q.groupby(["port", "terminal", "year"], as_index=False)["tons_i_q"]
        .sum(min_count=1)
        .rename(columns={"tons_i_q": "tons_i_y"})
    )
    q = q.merge(annual_tons, on=["port", "terminal", "year"], how="left")
    q = q.merge(annual_anchor[[c for c in ["port", "terminal", "year", "TEU_i_y", "H_annual_i_y"] if c in annual_anchor.columns]], on=["port", "terminal", "year"], how="left")

    q["tons_weight_q_in_y"] = np.where(q["tons_i_y"] > 0, q["tons_i_q"] / q["tons_i_y"], np.nan)

    # Deterministic fallback for positive-TEU years with missing / zero tons everywhere.
    qcount = (
        q.groupby(["port", "terminal", "year"], as_index=False)["quarter"]
        .count()
        .rename(columns={"quarter": "quarters_in_y"})
    )
    q = q.merge(qcount, on=["port", "terminal", "year"], how="left")
    fallback_mask = q["tons_weight_q_in_y"].isna() & q["TEU_i_y"].fillna(0.0).gt(0.0) & q["quarters_in_y"].gt(0)
    q["tons_weight_q_fallback_equal"] = np.where(fallback_mask, 1.0 / q["quarters_in_y"], np.nan)
    q["tons_weight_q_in_y"] = q["tons_weight_q_in_y"].fillna(q["tons_weight_q_fallback_equal"]).fillna(0.0)

    q["L_hours_i_q"] = q["H_annual_i_y"] * q["tons_weight_q_in_y"]
    q["labor_weight_source_q"] = np.where(
        q["tons_i_y"].fillna(0.0) > 0,
        "tons_only_terminal_quarter_share_in_year",
        np.where(fallback_mask, "equal_quarter_fallback_positive_teu_year", "zero_or_missing_activity"),
    )
    return q


def build_monthly_labor_from_tons(terminal_tons: pd.DataFrame, teu_monthly: pd.DataFrame, annual_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly terminal labor.

    Quarterly labor comes from annual anchor * tons-only quarter share.
    Monthly labor then uses monthly tons shares inside each quarter.
    """
    quarter_grid = build_quarter_grid_from_tons(terminal_tons, annual_anchor)

    base = terminal_tons.merge(
        teu_monthly[[c for c in [
            "port", "terminal", "year", "month", "month_index", "quarter",
            "TEU_i_m", "TEU_i_q", "TEU_port_m", "share_i_p_q", "teu_month_source"
        ] if c in teu_monthly.columns]],
        on=["port", "terminal", "year", "month", "month_index", "quarter"],
        how="left",
    )
    base = base.merge(
        annual_anchor[[c for c in ["port", "terminal", "year", "TEU_i_y", "Pi_teu_per_hour_i_y", "H_annual_i_y", "anchor_mode", "kpi_backfill_source"] if c in annual_anchor.columns]],
        on=["port", "terminal", "year"],
        how="left",
    )
    base = base.merge(
        quarter_grid[[
            "port", "terminal", "year", "quarter", "quarter_index", "tons_i_q", "tons_i_y",
            "tons_weight_q_in_y", "L_hours_i_q", "labor_weight_source_q"
        ]],
        on=["port", "terminal", "year", "quarter"],
        how="left",
    )

    # Monthly tons shares within quarter. This is still tons-only.
    base["tons_weight_m_in_q"] = np.where(base["tons_i_q"] > 0, base["tons_i_m"] / base["tons_i_q"], np.nan)

    # Equal split only when quarter labor is positive but monthly tons information is absent.
    month_count = (
        base.groupby(["port", "terminal", "year", "quarter"], as_index=False)["month"]
        .count()
        .rename(columns={"month": "months_in_q"})
    )
    base = base.merge(month_count, on=["port", "terminal", "year", "quarter"], how="left")
    fallback_mask = base["tons_weight_m_in_q"].isna() & base["L_hours_i_q"].fillna(0.0).gt(0.0) & base["months_in_q"].gt(0)
    base["tons_weight_m_fallback_equal"] = np.where(fallback_mask, 1.0 / base["months_in_q"], np.nan)
    base["tons_weight_m_in_q"] = base["tons_weight_m_in_q"].fillna(base["tons_weight_m_fallback_equal"]).fillna(0.0)

    base["L_hours_i_m"] = base["L_hours_i_q"] * base["tons_weight_m_in_q"]
    base["w_i_m_in_year"] = np.where(base["H_annual_i_y"].fillna(0.0) > 0, base["L_hours_i_m"] / base["H_annual_i_y"], 0.0)
    base["labor_weight_source"] = np.where(
        base["tons_i_q"].fillna(0.0) > 0,
        "tons_only_month_share_in_quarter",
        np.where(fallback_mask, "equal_month_fallback_positive_labor_quarter", "zero_or_missing_activity"),
    )

    keep = [
        "port", "terminal", "year", "month", "month_index", "quarter",
        "TEU_port_m", "share_i_p_q", "TEU_i_m",
        "tons_port_m", "tons_i_m", "tons_i_q", "tons_i_y",
        "TEU_i_q", "TEU_i_y",
        "Pi_teu_per_hour_i_y", "H_annual_i_y",
        "quarter_index", "tons_weight_q_in_y", "L_hours_i_q",
        "tons_weight_m_in_q", "L_hours_i_m", "w_i_m_in_year",
        "anchor_mode", "labor_weight_source_q", "labor_weight_source",
        "teu_month_source", "kpi_backfill_source",
    ]
    out = base[[c for c in keep if c in base.columns]].copy()
    return out.sort_values(["port", "terminal", "year", "month"]).reset_index(drop=True)


def build_port_month_labor(monthly_labor: pd.DataFrame) -> pd.DataFrame:
    port_month = (
        monthly_labor.groupby(["port", "year", "month", "month_index", "quarter"], as_index=False)
        .agg(
            tons_port_m=("tons_port_m", "first"),
            TEU_port_m=("TEU_port_m", "first"),
            L_hours_port_m=("L_hours_i_m", "sum"),
        )
        .sort_values(["port", "year", "month"])
        .reset_index(drop=True)
    )
    return port_month


def build_old_new_anchor_compare(annual_anchor: pd.DataFrame, base_lproxy_path: Optional[Path]) -> pd.DataFrame:
    if base_lproxy_path is None or (not base_lproxy_path.exists()):
        return pd.DataFrame()

    try:
        old = read_tsv(base_lproxy_path).copy()
    except Exception:
        return pd.DataFrame()

    required = {"port", "terminal", "year", "H_annual_i_y", "Pi_teu_per_hour_i_y"}
    if not required.issubset(set(old.columns)):
        return pd.DataFrame()

    old_ann = (
        old.groupby(["port", "terminal", "year"], as_index=False)
        .agg(H_old=("H_annual_i_y", "first"), Pi_old=("Pi_teu_per_hour_i_y", "first"))
    )
    cmp = annual_anchor.merge(old_ann, on=["port", "terminal", "year"], how="left")
    cmp["H_ratio_new_old"] = cmp["H_annual_i_y"] / cmp["H_old"]
    cmp["Pi_ratio_new_old"] = cmp["Pi_teu_per_hour_i_y"] / cmp["Pi_old"]
    return cmp.sort_values(["port", "terminal", "year"]).reset_index(drop=True)


def build_qa(
    port_month_tons: pd.DataFrame,
    terminal_tons: pd.DataFrame,
    teu_monthly: pd.DataFrame,
    annual_anchor: pd.DataFrame,
    monthly_labor: pd.DataFrame,
    tons_residual_qa: pd.DataFrame,
    anchor_compare: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, detail: str):
        rows.append({"check": check, "passed": bool(passed), "detail": str(detail)})

    add("tons_port_month_unique", not port_month_tons.duplicated(["port", "year", "month"]).any(), "port-month tons keys unique")
    add("tons_terminal_month_unique", not terminal_tons.duplicated(["port", "terminal", "year", "month"]).any(), "terminal-month tons keys unique")
    add("teu_terminal_month_unique", not teu_monthly.duplicated(["port", "terminal", "year", "month"]).any(), "terminal-month TEU keys unique")
    add("anchor_terminal_year_unique", not annual_anchor.duplicated(["port", "terminal", "year"]).any(), "terminal-year anchor keys unique")
    add("lproxy_terminal_month_unique", not monthly_labor.duplicated(["port", "terminal", "year", "month"]).any(), "terminal-month labor keys unique")

    neg_count = int(tons_residual_qa["legacy_negative_flag"].sum()) if "legacy_negative_flag" in tons_residual_qa.columns else 0
    add("legacy_tons_negative_before_clip", neg_count == 0, f"{neg_count} port-months had entrant tons greater than port tons before clipping")

    anchor_eval = annual_anchor.copy()
    pos_teu = anchor_eval["TEU_i_y"].fillna(0.0) > 0.0
    missing_pi = int((pos_teu & anchor_eval["Pi_teu_per_hour_i_y"].isna()).sum())
    add("missing_kpi_anchor_positive_teu_years", missing_pi == 0, f"{missing_pi} positive-TEU terminal-years missing Pi_teu_per_hour_i_y")

    missing_h = int((pos_teu & anchor_eval["H_annual_i_y"].isna()).sum())
    add("missing_annual_hours_anchor_positive_teu_years", missing_h == 0, f"{missing_h} positive-TEU terminal-years missing H_annual_i_y")

    annual_l = (
        monthly_labor.groupby(["port", "terminal", "year"], as_index=False)["L_hours_i_m"]
        .sum(min_count=1)
        .rename(columns={"L_hours_i_m": "L_hours_sum_y"})
    )
    annual_chk = annual_l.merge(annual_anchor[["port", "terminal", "year", "H_annual_i_y"]], on=["port", "terminal", "year"], how="left")
    annual_chk["abs_gap_hours"] = (annual_chk["L_hours_sum_y"] - annual_chk["H_annual_i_y"]).abs()
    max_gap_hours = float(annual_chk["abs_gap_hours"].max()) if not annual_chk.empty else 0.0
    add("annual_hours_addup", max_gap_hours <= 1e-8, f"max |sum_m L_hours_i_m - H_annual_i_y| = {max_gap_hours}")

    annual_w = (
        monthly_labor.groupby(["port", "terminal", "year"], as_index=False)["w_i_m_in_year"]
        .sum()
        .rename(columns={"w_i_m_in_year": "w_sum_y"})
    )
    annual_w = annual_w.merge(annual_anchor[["port", "terminal", "year", "TEU_i_y"]], on=["port", "terminal", "year"], how="left")
    annual_w["relevant"] = annual_w["TEU_i_y"].fillna(0.0) > 0.0
    max_gap_w = float((annual_w.loc[annual_w["relevant"], "w_sum_y"] - 1.0).abs().max()) if annual_w["relevant"].any() else 0.0
    add("annual_weight_addup_positive_teu_years", max_gap_w <= 1e-8, f"max |sum_m w_i_m_in_year - 1| over positive-TEU years = {max_gap_w}")

    if not anchor_compare.empty:
        continuing = anchor_compare["H_old"].notna() & anchor_compare["H_annual_i_y"].notna() & (anchor_compare["H_old"] > 0)
        small_ratios = anchor_compare.loc[continuing & (anchor_compare["H_ratio_new_old"] < 0.5)]
        add("old_new_anchor_sanity_ratio", small_ratios.empty, f"{len(small_ratios)} terminal-years have H_new / H_old < 0.5")

    return pd.DataFrame(rows)


def build_meta(
    tons_path: Path,
    teu_path: Path,
    kpi_path: Path,
    base_lproxy_path: Optional[Path],
    args: argparse.Namespace,
    outputs: Dict[str, Path],
    dataframes: Dict[str, pd.DataFrame],
) -> dict:
    inputs = {
        "tons": {"path": str(tons_path), "sha256": sha256sum(tons_path)},
        "teu": {"path": str(teu_path), "sha256": sha256sum(teu_path)},
        "kpis": {"path": str(kpi_path), "sha256": sha256sum(kpi_path)},
    }
    if base_lproxy_path is not None and base_lproxy_path.exists():
        inputs["base_lproxy"] = {"path": str(base_lproxy_path), "sha256": sha256sum(base_lproxy_path)}

    return {
        "implementation": "common_rule_v6_tonsonly",
        "formula_main": {
            "annual_anchor": "H_{i,y} = TEU_{i,y} / Pi^{work}_{i,y}",
            "quarter_weight": "tons_weight_q_in_y = tons_{i,q} / sum_{r in y} tons_{i,r}",
            "quarter_labor": "L_{i,q} = H_{i,y} * tons_weight_q_in_y",
            "month_weight": "tons_weight_m_in_q = tons_{i,m} / tons_{i,q}",
            "month_labor": "L_{i,m} = L_{i,q} * tons_weight_m_in_q",
        },
        "why_v6": [
            "keeps TEU only at the annual anchor",
            "removes TEU from within-year labor movement",
            "allocates both quarter and month labor using tons-only shares",
            "keeps entrant KPI backfill deterministic rather than introducing startup hyperparameters",
        ],
        "options": {
            "sample_start": args.sample_start,
            "sample_end": args.sample_end,
            "also_write_canonical": bool(args.also_write_canonical),
        },
        "inputs": inputs,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "row_counts": {k: int(len(v)) for k, v in dataframes.items()},
    }


def main():
    ap = argparse.ArgumentParser(description="Build common-rule monthly labor proxy v6 with tons-only within-year allocation")
    ap.add_argument("--tons", type=str, default=None, help="Path to monthly_output_by_1000_tons_ports_and_terminals.tsv")
    ap.add_argument("--teu", type=str, default=None, help="Path to teu_monthly_plus_quarterly_by_port.tsv")
    ap.add_argument("--kpis", type=str, default=None, help="Path to containers_kpis_annual_wide_filled.tsv")
    ap.add_argument("--base-lproxy", type=str, default=None, help="Optional existing Data/L_proxy/L_Proxy.tsv for comparison")
    ap.add_argument("--out", type=str, default=None, help="Output directory; defaults to Data/L_proxy/common_rule_v6_tonsonly")
    ap.add_argument("--sample-start", type=str, default="201801")
    ap.add_argument("--sample-end", type=str, default="202412")
    ap.add_argument("--also-write-canonical", action="store_true", help="Also overwrite Data/L_proxy/L_Proxy.tsv with the new build")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    thesis_root = find_thesis_root(script_dir)

    default_tons = thesis_root / "Data" / "Output" / "monthly_output_by_1000_tons_ports_and_terminals.tsv"
    default_teu = thesis_root / "Data" / "Output" / "teu_monthly_plus_quarterly_by_port.tsv"
    default_kpis = thesis_root / "Data" / "L_proxy" / "containers_kpis_annual_wide_filled.tsv"
    default_base_lproxy = thesis_root / "Data" / "L_proxy" / "L_Proxy.tsv"
    default_out = thesis_root / "Data" / "L_proxy" / "common_rule_v6_tonsonly"

    tons_path = Path(args.tons) if args.tons else default_tons
    teu_path = Path(args.teu) if args.teu else default_teu
    kpi_path = Path(args.kpis) if args.kpis else default_kpis
    base_lproxy_path = Path(args.base_lproxy) if args.base_lproxy else default_base_lproxy
    out_dir = Path(args.out) if args.out else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    sy, sm = parse_yyyymm(args.sample_start)
    ey, em = parse_yyyymm(args.sample_end)
    sample_start = sy * 100 + sm
    sample_end = ey * 100 + em

    tons_path = choose_first_existing([tons_path])
    teu_path = choose_first_existing([teu_path])
    kpi_path = choose_first_existing([kpi_path])

    port_month_tons, terminal_tons, tons_residual_qa = load_monthly_tons(tons_path, sample_start, sample_end)
    m_port, q_term = load_teu_panel(teu_path, sample_start, sample_end)
    teu_monthly = monthlyize_terminal_teu(terminal_tons, q_term, m_port)
    kpis = load_kpis(kpi_path)
    annual_anchor = compute_annual_anchor(teu_monthly, kpis)
    monthly_labor = build_monthly_labor_from_tons(terminal_tons, teu_monthly, annual_anchor)
    port_month_labor = build_port_month_labor(monthly_labor)
    anchor_compare = build_old_new_anchor_compare(annual_anchor, base_lproxy_path)

    qa = build_qa(
        port_month_tons=port_month_tons,
        terminal_tons=terminal_tons,
        teu_monthly=teu_monthly,
        annual_anchor=annual_anchor,
        monthly_labor=monthly_labor,
        tons_residual_qa=tons_residual_qa,
        anchor_compare=anchor_compare,
    )

    outputs = {
        "terminal_month_labor": out_dir / "labor_hours_monthly_terminal_commonrule_v6_tonsonly.tsv",
        "port_month_labor": out_dir / "labor_hours_monthly_port_commonrule_v6_tonsonly.tsv",
        "annual_anchor": out_dir / "labor_anchor_terminal_year_commonrule_v6_tonsonly.tsv",
        "lproxy": out_dir / "L_Proxy_commonrule_v6_tonsonly.tsv",
        "qa": out_dir / "lproxy_commonrule_v6_tonsonly_qa.tsv",
        "anchor_compare": out_dir / "lproxy_commonrule_v6_tonsonly_old_new_anchor_compare.tsv",
        "meta": out_dir / "_meta_l_proxy_commonrule_v6_tonsonly.json",
    }

    write_tsv(monthly_labor, outputs["terminal_month_labor"])
    write_tsv(port_month_labor, outputs["port_month_labor"])
    write_tsv(annual_anchor, outputs["annual_anchor"])
    write_tsv(monthly_labor, outputs["lproxy"])
    write_tsv(qa, outputs["qa"])
    write_tsv(anchor_compare, outputs["anchor_compare"])

    if args.also_write_canonical:
        canonical = thesis_root / "Data" / "L_proxy" / "L_Proxy.tsv"
        write_tsv(monthly_labor, canonical)
        outputs["canonical_lproxy"] = canonical

    meta = build_meta(
        tons_path=tons_path,
        teu_path=teu_path,
        kpi_path=kpi_path,
        base_lproxy_path=base_lproxy_path,
        args=args,
        outputs=outputs,
        dataframes={
            "port_month_tons": port_month_tons,
            "terminal_tons": terminal_tons,
            "monthly_terminal_teu": teu_monthly,
            "annual_anchor": annual_anchor,
            "monthly_labor": monthly_labor,
            "port_month_labor": port_month_labor,
            "qa": qa,
            "anchor_compare": anchor_compare,
        },
    )
    outputs["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[L_proxy_v6_tonsonly] Thesis root: {thesis_root}")
    for key, path in outputs.items():
        print(f"[L_proxy_v6_tonsonly] Wrote ({key}): {path}")


if __name__ == "__main__":
    main()

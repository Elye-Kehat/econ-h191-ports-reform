#!/usr/bin/env python3
"""
Build_KL_Panel.py

Build a clean monthly K/L panel from the new interpolation_02 capital outputs
and the current labor proxy.

Main design choice
------------------
This version intentionally builds a *minimal* analysis panel with one unique
Haifa port-level series, to avoid the ambiguity that broke Model_1B when the
older KL panel contained many Haifa port-level variants (HPC / IPC / SIPG /
cluster × scenario).

Inputs
------
1. Data/L_proxy/L_Proxy.tsv
   Terminal × month labor proxy with at least:
     port, terminal, year, month, L_hours_i_m

2. Data/K/Interpolation Output/interpolation_02_monthly_entity_series_long.tsv
   Final productive-capital entity × month outputs from interpolation_02 with:
     entity, month, K_productive_kNIS
   Expected entities:
     HPC, IPC, SIPG, HAIFA_TOTAL

Outputs
-------
Written to Data/KL/:
  - KL_Panel_monthly.tsv
  - KL_monthly_series_long.tsv
  - KL_monthly_series_wide.tsv
  - kl_build_manifest.json

Series constructed
------------------
1. Haifa_Legacy_KL   (terminal level)
   K = HPC productive capital
   L = Haifa-Legacy labor hours

2. Haifa_Bayport_KL  (terminal level)
   K = SIPG productive capital
   L = Haifa-Bayport labor hours

3. Haifa_port_KL     (port level)
   K = HAIFA_TOTAL productive capital
   L = total Haifa port labor hours = Haifa-Legacy + Haifa-Bayport

Why this is the preferred panel
-------------------------------
- It uses the *new* productive-capital series directly from interpolation_02.
- It aligns terminal capital with terminal labor where that is conceptually
  meaningful:
      HPC  <-> Haifa-Legacy
      SIPG <-> Haifa-Bayport
- It builds exactly one Haifa port-level series, so downstream econometrics do
  not face the old many-series ambiguity for the label "Haifa port".
- It produces only the core K/L objects needed for the Haifa-side econometrics.

Important limitation
--------------------
This panel remains Haifa-only because the uploaded interpolation_02 capital
outputs are Haifa-only. Ashdod K/L competition designs cannot be estimated
until Ashdod capital series exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_thesis_root(start: Optional[Path] = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for p in [here] + list(here.parents):
        if (p / "Data").exists():
            return p
    raise RuntimeError("Could not locate thesis root (expected a parent containing Data/).")


def month_to_year_month(s: str) -> tuple[int, int]:
    p = pd.Period(str(s), freq="M")
    return int(p.year), int(p.month)


def safe_log_ratio(k: pd.Series, l: pd.Series) -> tuple[pd.Series, pd.Series]:
    kl = pd.Series(np.nan, index=k.index, dtype="float64")
    log_kl = pd.Series(np.nan, index=k.index, dtype="float64")
    good = (pd.to_numeric(k, errors="coerce") > 0) & (pd.to_numeric(l, errors="coerce") > 0)
    kl.loc[good] = pd.to_numeric(k.loc[good], errors="coerce") / pd.to_numeric(l.loc[good], errors="coerce")
    log_kl.loc[good] = np.log(kl.loc[good])
    return kl, log_kl


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_l_proxy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"L_Proxy file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    require_columns(df, ["port", "terminal", "year", "month", "L_hours_i_m"], "L_Proxy.tsv")
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="raise").astype(int)
    df["L_hours_i_m"] = pd.to_numeric(df["L_hours_i_m"], errors="coerce")
    return df


def load_k_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"interpolation_02 long K file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    require_columns(df, ["entity", "month", "K_productive_kNIS"], "interpolation_02_monthly_entity_series_long.tsv")
    yy_mm = df["month"].map(month_to_year_month)
    df["year"] = [y for y, _ in yy_mm]
    df["month_num"] = [m for _, m in yy_mm]
    df["K_productive_kNIS"] = pd.to_numeric(df["K_productive_kNIS"], errors="coerce")
    return df


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def build_terminal_l(df_l: pd.DataFrame, port: str, terminal: str) -> pd.DataFrame:
    out = df_l.loc[(df_l["port"] == port) & (df_l["terminal"] == terminal), ["port", "terminal", "year", "month", "L_hours_i_m"]].copy()
    if out.empty:
        raise ValueError(f"No labor-proxy rows found for port={port!r}, terminal={terminal!r}.")
    out = out.rename(columns={"L_hours_i_m": "L"})
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def build_port_l(df_l: pd.DataFrame, port: str) -> pd.DataFrame:
    out = (
        df_l.loc[df_l["port"] == port, ["port", "year", "month", "L_hours_i_m"]]
        .groupby(["port", "year", "month"], as_index=False)["L_hours_i_m"]
        .sum()
        .rename(columns={"L_hours_i_m": "L"})
    )
    if out.empty:
        raise ValueError(f"No labor-proxy rows found for port={port!r}.")
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def extract_entity_k(df_k: pd.DataFrame, entity: str) -> pd.DataFrame:
    out = df_k.loc[df_k["entity"] == entity, ["year", "month_num", "K_productive_kNIS"]].copy()
    if out.empty:
        raise ValueError(f"No capital rows found for entity={entity!r} in interpolation_02 long output.")
    out = out.rename(columns={"month_num": "month", "K_productive_kNIS": "K"})
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def make_series(
    *,
    series_id: str,
    level: str,
    port: str,
    terminal: str | None,
    l_df: pd.DataFrame,
    k_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(
        l_df,
        k_df,
        on=["year", "month"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"Merged K/L series is empty for {series_id}. Check month overlap between K and L inputs.")

    merged["KL"], merged["log_KL"] = safe_log_ratio(merged["K"], merged["L"])
    merged["series_id"] = series_id
    merged["level"] = level
    merged["freq"] = "M"
    merged["port"] = port
    merged["terminal"] = terminal if terminal is not None else np.nan
    merged["month_index"] = merged["year"] * 12 + merged["month"]

    out_cols = [
        "series_id",
        "level",
        "freq",
        "port",
        "terminal",
        "year",
        "month",
        "month_index",
        "K",
        "L",
        "KL",
        "log_KL",
    ]
    return merged[out_cols].sort_values(["year", "month"]).reset_index(drop=True)


def build_long_panel(df_l: pd.DataFrame, df_k: pd.DataFrame) -> pd.DataFrame:
    # Labor objects
    l_legacy = build_terminal_l(df_l, port="Haifa", terminal="Haifa-Legacy")
    l_bayport = build_terminal_l(df_l, port="Haifa", terminal="Haifa-Bayport")
    l_haifa = build_port_l(df_l, port="Haifa")

    # Capital objects
    k_hpc = extract_entity_k(df_k, "HPC")
    k_sipg = extract_entity_k(df_k, "SIPG")
    k_total = extract_entity_k(df_k, "HAIFA_TOTAL")

    # Main analysis series
    s_legacy = make_series(
        series_id="Haifa_Legacy_KL",
        level="terminal",
        port="Haifa",
        terminal="Haifa-Legacy",
        l_df=l_legacy,
        k_df=k_hpc,
    )

    s_bayport = make_series(
        series_id="Haifa_Bayport_KL",
        level="terminal",
        port="Haifa",
        terminal="Haifa-Bayport",
        l_df=l_bayport,
        k_df=k_sipg,
    )

    s_port = make_series(
        series_id="Haifa_port_KL",
        level="port",
        port="Haifa",
        terminal=None,
        l_df=l_haifa,
        k_df=k_total,
    )

    panel = pd.concat([s_legacy, s_bayport, s_port], ignore_index=True)
    panel = panel.sort_values(["series_id", "year", "month"]).reset_index(drop=True)
    return panel


def build_wide_kl(panel: pd.DataFrame) -> pd.DataFrame:
    wide = panel.pivot_table(index=["year", "month"], columns="series_id", values="KL", aggfunc="first").reset_index()
    wide["month_str"] = [f"{int(y):04d}-{int(m):02d}" for y, m in zip(wide["year"], wide["month"])]
    # Move month_str forward for convenience
    cols = ["month_str", "year", "month"] + [c for c in wide.columns if c not in {"month_str", "year", "month"}]
    return wide[cols].sort_values(["year", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    thesis_root = find_thesis_root(Path(__file__))
    l_proxy_path = thesis_root / "Data" / "L_proxy" / "L_Proxy.tsv"
    k_long_path = thesis_root / "Data" / "K" / "Interpolation Output" / "interpolation_02_monthly_entity_series_long.tsv"
    out_dir = thesis_root / "Data" / "KL"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = out_dir / "KL_Panel_monthly.tsv"
    long_path = out_dir / "KL_monthly_series_long.tsv"
    wide_path = out_dir / "KL_monthly_series_wide.tsv"
    manifest_path = out_dir / "kl_build_manifest.json"

    print("=== Build_KL_Panel from interpolation_02 ===")
    print(f"THESIS_ROOT : {thesis_root}")
    print(f"L input     : {l_proxy_path}")
    print(f"K input     : {k_long_path}")

    df_l = load_l_proxy(l_proxy_path)
    df_k = load_k_long(k_long_path)

    panel = build_long_panel(df_l, df_k)
    wide = build_wide_kl(panel)

    panel.to_csv(panel_path, sep="\t", index=False)
    panel.to_csv(long_path, sep="\t", index=False)
    wide.to_csv(wide_path, sep="\t", index=False)

    manifest = {
        "builder": "Build_KL_Panel.py",
        "design_note": (
            "Uses interpolation_02 productive capital outputs directly and builds only the three core "
            "Haifa analysis series to avoid the old many-series ambiguity in the K/L panel."
        ),
        "inputs": {
            "L_proxy": str(l_proxy_path),
            "K_interpolation_long": str(k_long_path),
        },
        "outputs": {
            "KL_panel_monthly": str(panel_path),
            "KL_monthly_series_long": str(long_path),
            "KL_monthly_series_wide": str(wide_path),
        },
        "series_ids": sorted(panel["series_id"].unique().tolist()),
        "rows": {
            "panel_rows": int(len(panel)),
            "wide_rows": int(len(wide)),
        },
        "coverage": {
            sid: {
                "start": f"{int(g.iloc[0]['year']):04d}-{int(g.iloc[0]['month']):02d}",
                "end": f"{int(g.iloc[-1]['year']):04d}-{int(g.iloc[-1]['month']):02d}",
                "n_rows": int(len(g)),
                "n_valid_KL": int(g["KL"].notna().sum()),
            }
            for sid, g in panel.groupby("series_id", sort=True)
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote panel : {panel_path}")
    print(f"Wrote long  : {long_path}")
    print(f"Wrote wide  : {wide_path}")
    print(f"Wrote manifest: {manifest_path}")
    print("Series built:", ", ".join(sorted(panel["series_id"].unique().tolist())))


if __name__ == "__main__":
    main()

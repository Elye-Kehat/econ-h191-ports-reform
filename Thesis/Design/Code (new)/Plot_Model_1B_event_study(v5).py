from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Plot_Model_1B_event_study(v5).py
#
# Purpose:
#   Make clearer Model 1B K/L figures and helper tables for the corrected
#   Haifa-only K/L scope.
#
# Main change relative to v4:
#   v4 plotted Haifa-Legacy and Haifa-Bayport placebo coefficients together.
#   In the Haifa-only two-terminal design, those two coefficient paths are
#   largely the same contrast with the sign reversed. v5 therefore emphasizes
#   the single interpretable contrast:
#
#       Legacy minus Bayport, normalized to event month m = -1.
#
# Outputs:
#   Figures in:
#       Design/Output (new)/Model_1B/Figures/v5
#
#   Helper tables in:
#       Design/Output (new)/Model_1B/Tables/v5
#
# Figures:
#   1. raw log(K/L) paths for Haifa-Legacy, Haifa-Bayport, and Haifa port
#   2. normalized raw Legacy-minus-Bayport gap around privatization
#   3. event-study contrast plots for baseline and relaxed NYT/TWFE
#   4. optional diagnostic mirror plot for the placebo symmetry
#
# Usage:
#   python Plot_Model_1B_event_study(v5).py
#   python Plot_Model_1B_event_study(v5).py --spec all
#   python Plot_Model_1B_event_study(v5).py --jmin -12 --jmax 24
# =============================================================================


PRIV_EVENT_YEAR = 2023
PRIV_EVENT_MONTH = 1
BAYPORT_OPEN_YEAR = 2021
BAYPORT_OPEN_MONTH = 9

SERIES_LABELS = {
    "Haifa_Legacy_KL": "Haifa-Legacy",
    "Haifa_Bayport_KL": "Haifa-Bayport",
    "Haifa_port_KL": "Haifa port",
}

SPEC_DISPLAY = {
    "baseline": "Baseline",
    "relaxed_tr": "Relaxed+Tr",
}

DESIGN_DISPLAY = {
    "NYT": "Not-yet-treated (NYT)",
    "TWFE": "Conventional event study (TWFE)",
}


# -----------------------------------------------------------------------------
# Path and file helpers
# -----------------------------------------------------------------------------


def find_thesis_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise RuntimeError("Could not locate Thesis root. Expected directories: Data/ and Design/.")


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s


def read_tsv_if_exists(path: Path, required: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        print(f"[SKIP] Missing file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, sep = "\t")
    if required is not None:
        missing = set(required).difference(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    return df


def add_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors = "coerce")
    out["month"] = pd.to_numeric(out["month"], errors = "coerce")
    out = out.dropna(subset = ["year", "month"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["date"] = pd.to_datetime(dict(year = out["year"], month = out["month"], day = 1))
    out["month_serial"] = out["year"] * 12 + out["month"]
    return out


def event_serial(year: int, month: int) -> int:
    return int(year) * 12 + int(month)


def add_event_time(df: pd.DataFrame, year: int = PRIV_EVENT_YEAR, month: int = PRIV_EVENT_MONTH) -> pd.DataFrame:
    out = add_date_cols(df)
    out["event_time"] = out["month_serial"] - event_serial(year, month)
    return out


def choose_tick_step(mmin: int, mmax: int) -> int:
    span = mmax - mmin
    if span <= 18:
        return 1
    if span <= 30:
        return 2
    if span <= 48:
        return 3
    return 4


def finite_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors = "coerce").replace([np.inf, -np.inf], np.nan)


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------


def load_kl_panel(path: Path) -> pd.DataFrame:
    required = {"series_id", "year", "month", "log_KL", "KL", "K", "L"}
    df = read_tsv_if_exists(path, required = required)
    if df.empty:
        raise FileNotFoundError(f"No K/L panel found at {path}")
    df = add_event_time(df)
    for c in ["K", "L", "KL", "log_KL"]:
        df[c] = finite_series(df[c])
    return df


def load_dynamic(path: Path, design_label: str) -> pd.DataFrame:
    required = {
        "table_group",
        "reform",
        "target",
        "target_key",
        "spec_name",
        "event_time",
        "beta",
        "se",
    }
    df = read_tsv_if_exists(path, required = required)
    if df.empty:
        return df
    out = df.copy()
    out["design"] = design_label
    out["m"] = pd.to_numeric(out["event_time"], errors = "coerce")
    out["beta"] = finite_series(out["beta"])
    out["se"] = finite_series(out["se"])
    out = out[np.isfinite(out["m"])].copy()
    out["m"] = out["m"].astype(int)
    return out


def dynamic_paths(base_dir: Path, relaxed_dir: Path) -> dict[tuple[str, str], Path]:
    return {
        ("baseline", "NYT"): base_dir / "model1b_kl_dynamic_betas_all.tsv",
        ("baseline", "TWFE"): base_dir / "model1b_kl_dynamic_betas_all_twfe.tsv",
        ("relaxed_tr", "NYT"): relaxed_dir / "model1b_kl_dynamic_betas_all_relaxed.tsv",
        ("relaxed_tr", "TWFE"): relaxed_dir / "model1b_kl_dynamic_betas_all_relaxed_twfe.tsv",
    }


# -----------------------------------------------------------------------------
# Raw data plots
# -----------------------------------------------------------------------------


def plot_raw_log_kl(kl: pd.DataFrame, fig_dir: Path, tables_dir: Path) -> None:
    d = kl[kl["series_id"].isin(SERIES_LABELS)].copy()
    d = d.sort_values(["series_id", "date"])

    fig, ax = plt.subplots(figsize = (11, 6))

    for sid, lab in SERIES_LABELS.items():
        g = d[d["series_id"] == sid].copy()
        if g.empty:
            continue
        ax.plot(g["date"], g["log_KL"], marker = "o", linewidth = 1, markersize = 3, label = lab)

    ax.axvline(pd.Timestamp(BAYPORT_OPEN_YEAR, BAYPORT_OPEN_MONTH, 1), linewidth = 1, linestyle = "--")
    ax.axvline(pd.Timestamp(PRIV_EVENT_YEAR, PRIV_EVENT_MONTH, 1), linewidth = 1, linestyle = "-")
    ax.axhline(0.0, linewidth = 0.8)

    ax.set_title("Model 1B input: raw monthly log(K/L) paths")
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("log(K/L)")
    ax.legend(loc = "best", fontsize = 9)
    fig.autofmt_xdate()
    fig.tight_layout()

    out = fig_dir / "model1b_raw_log_kl_paths_v5.png"
    fig.savefig(out, dpi = 220)
    plt.close(fig)
    print(f"[OK] Saved: {out}")

    d.to_csv(tables_dir / "model1b_raw_log_kl_plot_data_v5.tsv", sep = "\t", index = False)


def make_gap_data(kl: pd.DataFrame) -> pd.DataFrame:
    d = kl[kl["series_id"].isin(["Haifa_Legacy_KL", "Haifa_Bayport_KL"])].copy()
    wide = (
        d.pivot_table(
            index = ["year", "month", "date", "event_time"],
            columns = "series_id",
            values = "log_KL",
            aggfunc = "first",
        )
        .reset_index()
    )

    needed = ["Haifa_Legacy_KL", "Haifa_Bayport_KL"]
    for c in needed:
        if c not in wide.columns:
            wide[c] = np.nan

    wide["legacy_minus_bayport"] = wide["Haifa_Legacy_KL"] - wide["Haifa_Bayport_KL"]
    ref = wide.loc[wide["event_time"] == -1, "legacy_minus_bayport"].dropna()
    if ref.empty:
        wide["legacy_minus_bayport_norm_m_minus_1"] = np.nan
    else:
        wide["legacy_minus_bayport_norm_m_minus_1"] = wide["legacy_minus_bayport"] - float(ref.iloc[0])
    return wide


def plot_normalized_gap(kl: pd.DataFrame, fig_dir: Path, tables_dir: Path, jmin: int, jmax: int | None) -> None:
    gap = make_gap_data(kl)
    if gap.empty:
        print("[SKIP] Could not build Legacy-minus-Bayport gap data.")
        return

    mmax = int(gap["event_time"].max()) if jmax is None else int(jmax)
    d = gap[(gap["event_time"] >= jmin) & (gap["event_time"] <= mmax)].copy()
    d = d.dropna(subset = ["legacy_minus_bayport_norm_m_minus_1"])
    if d.empty:
        print("[SKIP] No valid normalized gap observations in requested event-time range.")
        return

    fig, ax = plt.subplots(figsize = (10, 5.5))
    ax.plot(
        d["event_time"],
        d["legacy_minus_bayport_norm_m_minus_1"],
        marker = "o",
        linewidth = 1,
        markersize = 4,
        label = "Legacy minus Bayport, normalized to m = -1",
    )
    ax.axhline(0.0, linewidth = 1)
    ax.axvline(0.0, linewidth = 1)

    step = choose_tick_step(int(d["event_time"].min()), int(d["event_time"].max()))
    ax.set_xticks(list(range(int(d["event_time"].min()), int(d["event_time"].max()) + 1, step)))
    ax.set_title("Model 1B: raw normalized K/L gap around Haifa privatization")
    ax.set_xlabel("Event time m, months relative to January 2023")
    ax.set_ylabel("log(K/L) gap relative to m = -1")
    ax.legend(loc = "best", fontsize = 9)
    fig.tight_layout()

    out = fig_dir / "model1b_normalized_legacy_minus_bayport_gap_v5.png"
    fig.savefig(out, dpi = 220)
    plt.close(fig)
    print(f"[OK] Saved: {out}")

    gap.to_csv(tables_dir / "model1b_normalized_gap_plot_data_v5.tsv", sep = "\t", index = False)


# -----------------------------------------------------------------------------
# Event-study plots
# -----------------------------------------------------------------------------


def select_legacy_contrast(df: pd.DataFrame, spec_name: str, design: str) -> pd.DataFrame:
    d = df[
        (df["table_group"] == "privatization") &
        (df["reform"] == "haifa_priv") &
        (df["target"] == "Haifa-Legacy") &
        (df["spec_name"] == spec_name) &
        (df["design"] == design)
    ].copy()
    return d.sort_values("m")


def plot_event_study_contrast(
    df: pd.DataFrame,
    spec_name: str,
    design: str,
    fig_dir: Path,
    tables_dir: Path,
    jmin: int,
    jmax: int | None,
) -> None:
    d = select_legacy_contrast(df, spec_name = spec_name, design = design)
    if d.empty:
        print(f"[SKIP] No Legacy event-study rows for spec={spec_name}, design={design}")
        return

    mmax = int(d["m"].max()) if jmax is None else int(jmax)
    d = d[(d["m"] >= jmin) & (d["m"] <= mmax)].copy()
    d = d.dropna(subset = ["beta"])
    if d.empty:
        print(f"[SKIP] No valid beta rows after clipping for spec={spec_name}, design={design}")
        return

    valid_se = d["se"].notna() & np.isfinite(d["se"]) & (d["se"] > 0)
    all_valid_se = bool(valid_se.all())

    fig, ax = plt.subplots(figsize = (10.5, 5.8))
    ax.axhline(0.0, linewidth = 1)
    ax.axvline(0.0, linewidth = 1)

    if all_valid_se:
        ax.errorbar(
            d["m"],
            d["beta"],
            yerr = 1.96 * d["se"],
            fmt = "o-",
            linewidth = 1,
            markersize = 4,
            capsize = 3,
            label = "Legacy minus Bayport contrast",
        )
    else:
        ax.plot(
            d["m"],
            d["beta"],
            marker = "o",
            linewidth = 1,
            markersize = 4,
            label = "Legacy minus Bayport contrast",
        )
        ax.text(
            0.02,
            0.96,
            "SE unavailable or invalid. Baseline dynamic model is saturated.",
            transform = ax.transAxes,
            ha = "left",
            va = "top",
            fontsize = 9,
            bbox = {"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    step = choose_tick_step(int(d["m"].min()), int(d["m"].max()))
    ax.set_xticks(list(range(int(d["m"].min()), int(d["m"].max()) + 1, step)))

    ax.set_title(
        "Model 1B: Haifa privatization K/L contrast, "
        f"{DESIGN_DISPLAY.get(design, design)}, {SPEC_DISPLAY.get(spec_name, spec_name)}"
    )
    ax.set_xlabel("Event time m, months relative to January 2023. m = -1 omitted")
    ax.set_ylabel("Coefficient on log(K/L)")
    ax.legend(loc = "best", fontsize = 9)
    fig.tight_layout()

    fname = sanitize_filename(f"model1b_eventstudy_legacy_contrast_{design.lower()}_{spec_name}_v5.png")
    out = fig_dir / fname
    fig.savefig(out, dpi = 220)
    plt.close(fig)
    print(f"[OK] Saved: {out}")

    data_out = tables_dir / sanitize_filename(f"model1b_eventstudy_legacy_contrast_{design.lower()}_{spec_name}_plot_data_v5.tsv")
    d.to_csv(data_out, sep = "\t", index = False)


def plot_mirror_diagnostic(
    df: pd.DataFrame,
    spec_name: str,
    design: str,
    fig_dir: Path,
    tables_dir: Path,
    jmin: int,
    jmax: int | None,
) -> None:
    d = df[
        (df["table_group"] == "privatization") &
        (df["reform"] == "haifa_priv") &
        (df["spec_name"] == spec_name) &
        (df["design"] == design) &
        (df["target"].isin(["Haifa-Legacy", "Haifa-Bayport"]))
    ].copy()
    if d.empty:
        return
    mmax = int(d["m"].max()) if jmax is None else int(jmax)
    d = d[(d["m"] >= jmin) & (d["m"] <= mmax)].copy()
    if d.empty:
        return

    wide = d.pivot_table(index = "m", columns = "target", values = "beta", aggfunc = "first").reset_index()
    for c in ["Haifa-Legacy", "Haifa-Bayport"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide["legacy_plus_bayport"] = wide["Haifa-Legacy"] + wide["Haifa-Bayport"]
    wide["negative_bayport"] = -wide["Haifa-Bayport"]

    fig, ax = plt.subplots(figsize = (10, 5.2))
    ax.plot(wide["m"], wide["Haifa-Legacy"], marker = "o", linewidth = 1, markersize = 4, label = "Legacy coefficient")
    ax.plot(wide["m"], wide["negative_bayport"], marker = "x", linewidth = 1, markersize = 4, label = "Negative Bayport placebo coefficient")
    ax.axhline(0.0, linewidth = 1)
    ax.axvline(0.0, linewidth = 1)
    ax.set_title(
        "Model 1B mirror diagnostic: Legacy coefficient vs negative Bayport coefficient, "
        f"{DESIGN_DISPLAY.get(design, design)}, {SPEC_DISPLAY.get(spec_name, spec_name)}"
    )
    ax.set_xlabel("Event time m")
    ax.set_ylabel("Coefficient on log(K/L)")
    ax.legend(loc = "best", fontsize = 9)
    fig.tight_layout()

    fname = sanitize_filename(f"model1b_mirror_diagnostic_{design.lower()}_{spec_name}_v5.png")
    out = fig_dir / fname
    fig.savefig(out, dpi = 220)
    plt.close(fig)
    print(f"[OK] Saved: {out}")

    data_out = tables_dir / sanitize_filename(f"model1b_mirror_diagnostic_{design.lower()}_{spec_name}_v5.tsv")
    wide.to_csv(data_out, sep = "\t", index = False)


# -----------------------------------------------------------------------------
# Helper tables
# -----------------------------------------------------------------------------


def combine_files(paths: list[Path], source_names: list[str]) -> pd.DataFrame:
    frames = []
    for path, source in zip(paths, source_names):
        if not path.exists():
            continue
        df = pd.read_csv(path, sep = "\t")
        df["source_file"] = path.name
        df["source_spec_family"] = source
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index = True)


def write_summary_tables(base_dir: Path, relaxed_dir: Path, tables_dir: Path) -> None:
    static_paths = [
        base_dir / "model1b_kl_static_betas_all_twfe.tsv",
        relaxed_dir / "model1b_kl_static_betas_all_relaxed_twfe.tsv",
        relaxed_dir / "model1b_kl_static_betas_relaxed_tr_relaxed_twfe.tsv",
    ]
    static_sources = ["baseline", "relaxed_tr", "relaxed_tr"]
    static = combine_files(static_paths, static_sources)
    if not static.empty:
        static = static.drop_duplicates()
        for c in ["beta", "se", "pvalue", "r2"]:
            if c in static.columns:
                static[c] = finite_series(static[c])
        static.to_csv(tables_dir / "model1b_static_did_combined_v5.tsv", sep = "\t", index = False)
        print(f"[OK] Saved: {tables_dir / 'model1b_static_did_combined_v5.tsv'}")

    window_paths = [
        base_dir / "model1b_kl_window_betas_all.tsv",
        base_dir / "model1b_kl_window_betas_all_twfe.tsv",
        relaxed_dir / "model1b_kl_window_betas_all_relaxed.tsv",
        relaxed_dir / "model1b_kl_window_betas_all_relaxed_twfe.tsv",
        relaxed_dir / "model1b_kl_window_betas_relaxed_tr_relaxed.tsv",
        relaxed_dir / "model1b_kl_window_betas_relaxed_tr_relaxed_twfe.tsv",
    ]
    window_sources = ["baseline", "baseline", "relaxed_tr", "relaxed_tr", "relaxed_tr", "relaxed_tr"]
    windows = combine_files(window_paths, window_sources)
    if not windows.empty:
        windows = windows.drop_duplicates()
        for c in ["beta", "se", "pvalue", "r2"]:
            if c in windows.columns:
                windows[c] = finite_series(windows[c])
        windows.to_csv(tables_dir / "model1b_window_estimates_combined_v5.tsv", sep = "\t", index = False)
        print(f"[OK] Saved: {tables_dir / 'model1b_window_estimates_combined_v5.tsv'}")

    pretrend_paths = [
        base_dir / "model1b_kl_pretrend_tests_all.tsv",
        base_dir / "model1b_kl_pretrend_tests_all_twfe.tsv",
        relaxed_dir / "model1b_kl_pretrend_tests_all_relaxed.tsv",
        relaxed_dir / "model1b_kl_pretrend_tests_all_relaxed_twfe.tsv",
        relaxed_dir / "model1b_kl_pretrend_tests_relaxed_tr_relaxed.tsv",
        relaxed_dir / "model1b_kl_pretrend_tests_relaxed_tr_relaxed_twfe.tsv",
    ]
    pretrend_sources = ["baseline", "baseline", "relaxed_tr", "relaxed_tr", "relaxed_tr", "relaxed_tr"]
    pre = combine_files(pretrend_paths, pretrend_sources)
    if not pre.empty:
        pre = pre.drop_duplicates()
        for c in ["f_stat", "pvalue", "r2", "df_num", "df_denom"]:
            if c in pre.columns:
                pre[c] = finite_series(pre[c])
        pre.to_csv(tables_dir / "model1b_pretrend_tests_combined_v5.tsv", sep = "\t", index = False)
        print(f"[OK] Saved: {tables_dir / 'model1b_pretrend_tests_combined_v5.tsv'}")

    # A compact diagnostics table for quick inspection.
    rows = []
    if not windows.empty:
        for _, r in windows.iterrows():
            rows.append({
                "kind": "window",
                "design": r.get("design", ""),
                "target": r.get("target", ""),
                "spec_name": r.get("spec_name", ""),
                "window": r.get("window", ""),
                "beta": r.get("beta", np.nan),
                "se": r.get("se", np.nan),
                "pvalue": r.get("pvalue", np.nan),
                "n_obs": r.get("n_obs", np.nan),
                "r2": r.get("r2", np.nan),
                "source_spec_family": r.get("source_spec_family", ""),
            })
    if rows:
        diag = pd.DataFrame(rows)
        diag.to_csv(tables_dir / "model1b_compact_window_diagnostics_v5.tsv", sep = "\t", index = False)
        print(f"[OK] Saved: {tables_dir / 'model1b_compact_window_diagnostics_v5.tsv'}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec",
        default = "all",
        choices = ["baseline", "relaxed_tr", "all"],
        help = "Which spec to plot. Default is all.",
    )
    ap.add_argument(
        "--design",
        default = "all",
        choices = ["NYT", "TWFE", "all"],
        help = "Which design to plot. Default is all.",
    )
    ap.add_argument("--jmin", type = int, default = -12, help = "Minimum event time m to show.")
    ap.add_argument("--jmax", type = int, default = None, help = "Maximum event time m to show.")
    ap.add_argument(
        "--no-mirror-diagnostic",
        action = "store_true",
        help = "Skip the mirror diagnostic plots.",
    )
    args = ap.parse_args()

    thesis_root = find_thesis_root(Path(__file__).resolve())
    kl_path = thesis_root / "Data" / "KL" / "KL_Panel_monthly.tsv"
    base_dir = thesis_root / "Design" / "Output (new)" / "Model_1B"
    relaxed_dir = thesis_root / "Design" / "Output (new)" / "Model_1B_relaxed"
    fig_dir = base_dir / "Figures" / "v5"
    tables_dir = base_dir / "Tables" / "v5"
    fig_dir.mkdir(parents = True, exist_ok = True)
    tables_dir.mkdir(parents = True, exist_ok = True)

    print("=== Plot_Model_1B_event_study(v5) ===")
    print(f"THESIS_ROOT: {thesis_root}")
    print(f"K/L panel  : {kl_path}")
    print(f"Figures    : {fig_dir}")
    print(f"Tables     : {tables_dir}")

    kl = load_kl_panel(kl_path)
    plot_raw_log_kl(kl, fig_dir = fig_dir, tables_dir = tables_dir)
    plot_normalized_gap(kl, fig_dir = fig_dir, tables_dir = tables_dir, jmin = args.jmin, jmax = args.jmax)

    specs = ["baseline", "relaxed_tr"] if args.spec == "all" else [args.spec]
    designs = ["NYT", "TWFE"] if args.design == "all" else [args.design]

    paths = dynamic_paths(base_dir = base_dir, relaxed_dir = relaxed_dir)
    dyn_frames = []
    for spec in specs:
        for design in designs:
            path = paths[(spec, design)]
            df = load_dynamic(path, design_label = design)
            if df.empty:
                continue
            dyn_frames.append(df)

    if dyn_frames:
        dyn = pd.concat(dyn_frames, ignore_index = True)
        dyn.to_csv(tables_dir / "model1b_dynamic_inputs_used_v5.tsv", sep = "\t", index = False)

        for spec in specs:
            for design in designs:
                plot_event_study_contrast(
                    dyn,
                    spec_name = spec,
                    design = design,
                    fig_dir = fig_dir,
                    tables_dir = tables_dir,
                    jmin = args.jmin,
                    jmax = args.jmax,
                )
                if not args.no_mirror_diagnostic:
                    plot_mirror_diagnostic(
                        dyn,
                        spec_name = spec,
                        design = design,
                        fig_dir = fig_dir,
                        tables_dir = tables_dir,
                        jmin = args.jmin,
                        jmax = args.jmax,
                    )
    else:
        print("[WARN] No dynamic event-study files found. Raw K/L plots and summary tables only.")

    write_summary_tables(base_dir = base_dir, relaxed_dir = relaxed_dir, tables_dir = tables_dir)

    print("\nDone.")
    print(f"Figures written to: {fig_dir}")
    print(f"Tables written to : {tables_dir}")


if __name__ == "__main__":
    main()

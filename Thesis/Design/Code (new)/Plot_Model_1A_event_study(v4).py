#!/usr/bin/env python3
"""
Plot_Model_1A_event_study.py  (v4)

Goal:
    Produce clean Model 1A event-study plots consistent with the v4 thesis
    architecture and Emi's feedback on dynamic presentation.

What this script now does:
    - uses event time m on the x-axis (applied notation)
    - plots dynamic coefficients against m
    - uses the v4 target universe
    - respects the revised design split:
        * competition: NYT and TWFE
        * privatization: NYT and TWFE
    - defaults to the preferred spec ("porttr")
    - skips aggregate targets automatically if they are absent upstream

Default outputs:
    Exactly 4 figures total (for one chosen spec):
      1. competition   — NYT
      2. competition   — TWFE
      3. privatization — NYT
      4. privatization — TWFE

Input files:
    - Design/Output (new)/Model_1A/model1a_lp_dynamic_betas_all.tsv
    - Design/Output (new)/Model_1A/model1a_lp_dynamic_betas_all_twfe.tsv

Usage:
    python Plot_Model_1A_event_study.py
    python Plot_Model_1A_event_study.py --spec baseline
    python Plot_Model_1A_event_study.py --spec porttr
    python Plot_Model_1A_event_study.py --spec all
    python Plot_Model_1A_event_study.py --jmin -24 --rotate 45
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def find_thesis_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "Data").exists() and (p / "Design").exists():
            return p
    raise RuntimeError("Could not locate Thesis root (expected directories: Data/ and Design/).")


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    return s


def choose_tick_step(mmin: int, mmax: int) -> int:
    span = mmax - mmin
    if span <= 18:
        return 1
    if span <= 30:
        return 2
    if span <= 48:
        return 3
    return 4


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_dynamic(path: Path, design_label: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep = "\t")

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
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["design"] = design_label
    df["m"] = pd.to_numeric(df["event_time"], errors = "coerce")
    df["beta"] = pd.to_numeric(df["beta"], errors = "coerce")
    df["se"] = pd.to_numeric(df["se"], errors = "coerce")

    df = df[np.isfinite(df["m"])].copy()
    df["m"] = df["m"].astype(int)

    return df


# ---------------------------------------------------------------------
# Display metadata
# ---------------------------------------------------------------------

SPEC_DISPLAY = {
    "baseline": "Baseline",
    "porttr": "Preferred (+Tr)",
}

DESIGN_DISPLAY = {
    "NYT": "Not-yet-treated (NYT)",
    "TWFE": "Conventional event-study (TWFE)",
}

TABLE_GROUP_DISPLAY = {
    "competition": "Competition entry",
    "privatization": "Haifa privatization",
}

TARGET_ORDER = {
    ("competition", "NYT"): [
        "Haifa-Legacy",
        "Haifa aggregate",
    ],
    ("competition", "TWFE"): [
        "Haifa-Legacy",
        "Haifa aggregate",
        "Ashdod-Legacy",
        "Ashdod aggregate",
    ],
    ("privatization", "NYT"): [
        "Haifa-Legacy",
        "Haifa-Bayport",
        "Haifa aggregate",
    ],
    ("privatization", "TWFE"): [
        "Haifa-Legacy",
        "Haifa-Bayport",
        "Haifa aggregate",
    ],
}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def get_targets_for_plot(df: pd.DataFrame, table_group: str, design: str) -> list[str]:
    preferred = TARGET_ORDER[(table_group, design)]
    present = set(df["target"].dropna().unique())
    return [t for t in preferred if t in present]


def make_plot(
    df: pd.DataFrame,
    table_group: str,
    design: str,
    spec_name: str,
    out_path: Path,
    mmin: int,
    mmax_override: int | None,
    rotate: int,
) -> None:
    d = df[
        (df["table_group"] == table_group) &
        (df["design"] == design) &
        (df["spec_name"] == spec_name)
    ].copy()

    if d.empty:
        print(f"[SKIP] No rows for table_group={table_group}, design={design}, spec={spec_name}")
        return

    targets = get_targets_for_plot(d, table_group = table_group, design = design)
    if not targets:
        print(f"[SKIP] No target rows available for table_group={table_group}, design={design}, spec={spec_name}")
        return

    d = d[d["target"].isin(targets)].copy()

    mmax_data = int(d["m"].max())
    mmax = int(mmax_override) if mmax_override is not None else mmax_data

    d = d[(d["m"] >= mmin) & (d["m"] <= mmax)].copy()
    if d.empty:
        print(f"[SKIP] All rows clipped out for table_group={table_group}, design={design}, spec={spec_name}")
        return

    plt.figure(figsize = (11, 6))
    ax = plt.gca()

    ax.axhline(0.0, linewidth = 1)
    ax.axvline(0.0, linewidth = 1)

    for target in targets:
        g = d[d["target"] == target].sort_values("m").copy()
        if g.empty:
            continue

        m = g["m"].to_numpy()
        beta = g["beta"].to_numpy()
        se = g["se"].to_numpy()
        yerr = 1.96 * se

        ax.errorbar(
            m,
            beta,
            yerr = yerr,
            fmt = "o-",
            linewidth = 1,
            markersize = 4,
            capsize = 2,
            label = target,
        )

    step = choose_tick_step(mmin, mmax)
    ticks = list(range(mmin, mmax + 1, step))
    ax.set_xticks(ticks)

    if rotate != 0:
        for lab in ax.get_xticklabels():
            lab.set_rotation(rotate)
            lab.set_ha("right")

    ax.set_xlabel("Event time m (months relative to reform; m = -1 omitted)")
    ax.set_ylabel("Coefficient on ln(LP)")

    title_left = TABLE_GROUP_DISPLAY[table_group]
    title_mid = DESIGN_DISPLAY[design]
    title_right = SPEC_DISPLAY.get(spec_name, spec_name)
    ax.set_title(f"Model 1A: {title_left} — {title_mid} — {title_right}")

    ax.legend(loc = "best", fontsize = 9)
    plt.tight_layout()

    out_path.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(out_path, dpi = 200)
    plt.close()
    print(f"[OK] Saved: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec",
        default = "porttr",
        choices = ["baseline", "porttr", "all"],
        help = "Which spec to plot. Default is 'porttr' (preferred).",
    )
    ap.add_argument(
        "--jmin",
        type = int,
        default = -12,
        help = "Minimum event time m to show on x-axis (default -12).",
    )
    ap.add_argument(
        "--jmax",
        type = int,
        default = None,
        help = "Maximum event time m to show on x-axis. Default None = use max available.",
    )
    ap.add_argument(
        "--rotate",
        type = int,
        default = 0,
        help = "Rotate x tick labels (degrees).",
    )
    args = ap.parse_args()

    thesis_root = find_thesis_root(Path(__file__).resolve())
    model_dir = thesis_root / "Design" / "Output (new)" / "Model_1A"
    fig_dir = model_dir / "Figures" / "v4"
    fig_dir.mkdir(parents = True, exist_ok = True)

    nyt_path = model_dir / "model1a_lp_dynamic_betas_all.tsv"
    twfe_path = model_dir / "model1a_lp_dynamic_betas_all_twfe.tsv"

    if not nyt_path.exists():
        raise FileNotFoundError(f"Missing NYT dynamic betas: {nyt_path}")
    if not twfe_path.exists():
        raise FileNotFoundError(f"Missing TWFE dynamic betas: {twfe_path}")

    df_nyt = load_dynamic(nyt_path, design_label = "NYT")
    df_twfe = load_dynamic(twfe_path, design_label = "TWFE")
    df = pd.concat([df_nyt, df_twfe], ignore_index = True)

    if args.spec == "all":
        specs_to_plot = ["baseline", "porttr"]
    else:
        specs_to_plot = [args.spec]

    combos = [
        ("competition", "NYT"),
        ("competition", "TWFE"),
        ("privatization", "NYT"),
        ("privatization", "TWFE"),
    ]

    for spec_name in specs_to_plot:
        for table_group, design in combos:
            fname = sanitize_filename(
                f"model1a_lp_eventstudy_{table_group}_{design.lower()}_{spec_name}_v4.png"
            )
            out_path = fig_dir / fname

            make_plot(
                df = df,
                table_group = table_group,
                design = design,
                spec_name = spec_name,
                out_path = out_path,
                mmin = args.jmin,
                mmax_override = args.jmax,
                rotate = args.rotate,
            )

    print(f"\nDone. Figures written to: {fig_dir}")


if __name__ == "__main__":
    main()



# =============================================================================
# EVALUATION NOTE AFTER FIRST RUN OF Plot_Model_1A_event_study(v4)
#
# Summary:
# The v4 plotting script is functioning as intended and is aligned with the
# revised Model 1A architecture and Emi's feedback on dynamic presentation.
# No clear code bug is evident from the first run. The remaining odd-looking
# features of the figures are consistent with the current interim LP data and
# known saturation / collinearity issues upstream.
#
# What worked:
# 1. The script ran cleanly and saved the intended four default figures:
#    - competition NYT
#    - competition TWFE
#    - privatization NYT
#    - privatization TWFE
#
# 2. The plotted target universe matches the current v4 logic:
#    - Competition NYT: Haifa only
#    - Competition TWFE: Haifa + Ashdod
#    - Privatization NYT: Haifa-Legacy + Haifa-Bayport
#    - Privatization TWFE: Haifa-Legacy + Haifa-Bayport
#
# 3. The applied notation is improved relative to the older plotting script:
#    event time is shown as m on the x-axis, and dynamic coefficients are
#    graphed directly against event time, which is closer to the applied
#    literature style requested by Emi.
#
# 4. Titles, legends, and output filenames appear to be mapped correctly to
#    the design (NYT vs TWFE), reform group, and specification.
#
# Non-bug features that reflect current interim LP data:
# 5. The competition NYT figure still shows a sharp abnormal movement around
#    the omitted-period boundary and very limited visible uncertainty bars.
#    This is consistent with the already-known saturated / collinear interim
#    LP setup and does not by itself indicate a plotting bug.
#
# 6. In the privatization NYT figure, Haifa-Legacy and Haifa-Bayport largely
#    overlap visually. This is expected under the current interim LP data
#    construction and is consistent with prior debugging of Model_1A(v4).
#
# 7. Aggregate lines are absent from the figures because aggregate specs remain
#    disabled upstream in Model_1A(v4). This is expected and not a bug here.
#
# Minor presentation note:
# 8. Because m = -1 is omitted, the connected line visually jumps directly from
#    m = -2 to m = 0. This is not a code error, but it can make the treatment
#    break appear sharper than it is. A future cosmetic refinement could break
#    the line at the omitted bin rather than connect across it.
#
# Bottom line:
# Plot_Model_1A_event_study(v4) is working correctly for the current pipeline.
# No further code change is necessary before moving on, unless a later cosmetic
# refinement is desired for presentation quality.
# =============================================================================
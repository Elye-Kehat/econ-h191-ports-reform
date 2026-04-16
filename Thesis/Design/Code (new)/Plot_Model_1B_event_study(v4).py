from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Plot_Model_1B_event_study(v4).py
#
# Goal:
#   Produce clean Model 1B event-study plots consistent with the v4 thesis
#   architecture.
#
# What this script does:
#   - uses event time m on the x-axis
#   - plots dynamic coefficients against m
#   - uses the revised target universe
#   - reads both baseline and relaxed dynamic outputs
#   - defaults to plotting one spec at a time
#
# Default outputs:
#   For one chosen spec, exactly 4 figures:
#     1. competition   — NYT
#     2. competition   — TWFE
#     3. privatization — NYT
#     4. privatization — TWFE
#
# Usage:
#   python Plot_Model_1B_event_study(v4).py
#   python Plot_Model_1B_event_study(v4).py --spec baseline
#   python Plot_Model_1B_event_study(v4).py --spec relaxed_tr
#   python Plot_Model_1B_event_study(v4).py --spec all
# =============================================================================


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


SPEC_DISPLAY = {
    "baseline": "Baseline",
    "relaxed_tr": "Relaxed+Tr",
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
    ax.set_ylabel("Coefficient on ln(K/L)")

    title_left = TABLE_GROUP_DISPLAY[table_group]
    title_mid = DESIGN_DISPLAY[design]
    title_right = SPEC_DISPLAY.get(spec_name, spec_name)
    ax.set_title(f"Model 1B: {title_left} — {title_mid} — {title_right}")

    ax.legend(loc = "best", fontsize = 9)
    plt.tight_layout()

    out_path.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(out_path, dpi = 200)
    plt.close()
    print(f"[OK] Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec",
        default = "baseline",
        choices = ["baseline", "relaxed_tr", "all"],
        help = "Which spec to plot. Default is 'baseline'.",
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
    base_dir = thesis_root / "Design" / "Output (new)" / "Model_1B"
    relaxed_dir = thesis_root / "Design" / "Output (new)" / "Model_1B_relaxed"
    fig_dir = base_dir / "Figures" / "v4"
    fig_dir.mkdir(parents = True, exist_ok = True)

    path_map = {
        "baseline": {
            "NYT": base_dir / "model1b_kl_dynamic_betas_all.tsv",
            "TWFE": base_dir / "model1b_kl_dynamic_betas_all_twfe.tsv",
        },
        "relaxed_tr": {
            "NYT": relaxed_dir / "model1b_kl_dynamic_betas_all_relaxed.tsv",
            "TWFE": relaxed_dir / "model1b_kl_dynamic_betas_all_relaxed_twfe.tsv",
        },
    }

    specs_to_plot = ["baseline", "relaxed_tr"] if args.spec == "all" else [args.spec]

    combos = [
        ("competition", "NYT"),
        ("competition", "TWFE"),
        ("privatization", "NYT"),
        ("privatization", "TWFE"),
    ]

    for spec_name in specs_to_plot:
        frames = []
        for design in ["NYT", "TWFE"]:
            path = path_map[spec_name][design]
            if not path.exists():
                print(f"[SKIP] Missing dynamic file for spec={spec_name}, design={design}: {path}")
                continue
            frames.append(load_dynamic(path, design_label = design))

        if not frames:
            print(f"[SKIP] No dynamic inputs found for spec={spec_name}")
            continue

        df = pd.concat(frames, ignore_index = True)

        for table_group, design in combos:
            fname = sanitize_filename(
                f"model1b_kl_eventstudy_{table_group}_{design.lower()}_{spec_name}_v4.png"
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

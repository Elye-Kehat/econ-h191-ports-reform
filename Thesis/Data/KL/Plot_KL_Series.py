from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLE_START = "2018-01"
SAMPLE_END = "2024-12"


def read_tsv(path):
    return pd.read_csv(path, sep = "\t")


def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any of these files:\n" + "\n".join(str(p) for p in paths)
    )


def make_month_string(year_col, month_col):
    year_col = pd.to_numeric(year_col, errors = "coerce").astype("Int64")
    month_col = pd.to_numeric(month_col, errors = "coerce").astype("Int64")
    return year_col.astype(str) + "-" + month_col.astype(str).str.zfill(2)


def prep_plot(df):
    df = df.copy().sort_values("month").reset_index(drop = True)
    df["date"] = pd.PeriodIndex(df["month"], freq = "M").to_timestamp(how = "end")
    return df


def plot_single(df, title, outpath):
    df = prep_plot(df)

    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(df["date"], df["KL_kNIS_per_hour"], linewidth = 2)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("K/L (thousands of NIS per labor-hour)")
    ax.grid(True, alpha = 0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi = 200, bbox_inches = "tight")
    plt.close(fig)


def plot_all(df, outpath):
    df = df.copy()

    label_map = {
        "LEGACY": "Legacy",
        "ENTRANT": "Entrant",
        "HAIFA_TOTAL": "Haifa total",
    }

    fig, ax = plt.subplots(figsize = (12, 6))

    for series in ["LEGACY", "ENTRANT", "HAIFA_TOTAL"]:
        d = prep_plot(df.loc[df["series"] == series].copy())
        ax.plot(d["date"], d["KL_kNIS_per_hour"], linewidth = 2, label = label_map[series])

    ax.set_title("Monthly K/L: All Main Haifa Series")
    ax.set_xlabel("Month")
    ax.set_ylabel("K/L (thousands of NIS per labor-hour)")
    ax.grid(True, alpha = 0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi = 200, bbox_inches = "tight")
    plt.close(fig)


def main():
    kl_dir = Path(__file__).resolve().parent
    data_dir = kl_dir.parent
    vis_dir = kl_dir / "Visualization"
    vis_dir.mkdir(parents = True, exist_ok = True)

    k_candidates = [
        data_dir / "K" / "Interpolation Output" / "interpolation_02_monthly_entity_series_wide.tsv",
        data_dir / "K" / "interpolation_02_monthly_entity_series_wide.tsv",
        data_dir / "K" / "Output" / "interpolation_02_monthly_entity_series_wide.tsv",
    ]
    l_candidates = [
        data_dir / "L_proxy" / "L_Proxy.tsv",
        data_dir / "L_Proxy" / "L_Proxy.tsv",
    ]

    k_path = find_first_existing(k_candidates)
    l_path = find_first_existing(l_candidates)

    k = read_tsv(k_path)
    l = read_tsv(l_path)

    k = k[["month", "K_HPC_kNIS", "K_SIPG_kNIS", "K_HAIFA_TOTAL_kNIS"]].copy()
    for col in ["K_HPC_kNIS", "K_SIPG_kNIS", "K_HAIFA_TOTAL_kNIS"]:
        k[col] = pd.to_numeric(k[col], errors = "coerce")

    l = l.copy()
    l["L_hours_i_m"] = pd.to_numeric(l["L_hours_i_m"], errors = "coerce")
    l["month_str"] = make_month_string(l["year"], l["month"])

    haifa_l = l.loc[l["port"] == "Haifa"].copy()

    terminal_month = (
        haifa_l.groupby(["terminal", "month_str"], as_index = False)["L_hours_i_m"]
        .sum()
        .rename(columns = {"month_str": "month", "L_hours_i_m": "L_hours"})
    )

    l_legacy = (
        terminal_month.loc[terminal_month["terminal"] == "Haifa-Legacy", ["month", "L_hours"]]
        .assign(series = "LEGACY")
    )

    l_entrant = (
        terminal_month.loc[terminal_month["terminal"] == "Haifa-Bayport", ["month", "L_hours"]]
        .assign(series = "ENTRANT")
    )

    l_haifa_total = (
        terminal_month.groupby("month", as_index = False)["L_hours"]
        .sum()
        .assign(series = "HAIFA_TOTAL")
    )

    l_final = pd.concat([l_legacy, l_entrant, l_haifa_total], ignore_index = True)

    k_long = pd.concat(
        [
            k[["month", "K_HPC_kNIS"]].rename(columns = {"K_HPC_kNIS": "K_kNIS"}).assign(series = "LEGACY"),
            k[["month", "K_SIPG_kNIS"]].rename(columns = {"K_SIPG_kNIS": "K_kNIS"}).assign(series = "ENTRANT"),
            k[["month", "K_HAIFA_TOTAL_kNIS"]].rename(columns = {"K_HAIFA_TOTAL_kNIS": "K_kNIS"}).assign(series = "HAIFA_TOTAL"),
        ],
        ignore_index = True
    )

    kl = k_long.merge(l_final, on = ["series", "month"], how = "left")
    kl["KL_kNIS_per_hour"] = np.where(kl["L_hours"] > 0, kl["K_kNIS"] / kl["L_hours"], np.nan)
    kl = kl.sort_values(["series", "month"]).reset_index(drop = True)

    kl_wide = kl.pivot(index = "month", columns = "series", values = "KL_kNIS_per_hour").reset_index()
    kl_wide = kl_wide.rename(
        columns = {
            "LEGACY": "KL_LEGACY_kNIS_per_hour",
            "ENTRANT": "KL_ENTRANT_kNIS_per_hour",
            "HAIFA_TOTAL": "KL_HAIFA_TOTAL_kNIS_per_hour",
        }
    )

    kl.to_csv(kl_dir / "KL_monthly_series_long.tsv", sep = "\t", index = False)
    kl_wide.to_csv(kl_dir / "KL_monthly_series_wide.tsv", sep = "\t", index = False)

    plot_single(
        kl.loc[kl["series"] == "LEGACY"].copy(),
        "Monthly K/L: Legacy (Haifa-Legacy)",
        vis_dir / "plot_kl_legacy.png"
    )

    plot_single(
        kl.loc[kl["series"] == "ENTRANT"].copy(),
        "Monthly K/L: Entrant (Haifa-Bayport)",
        vis_dir / "plot_kl_entrant.png"
    )

    plot_single(
        kl.loc[kl["series"] == "HAIFA_TOTAL"].copy(),
        "Monthly K/L: Haifa as a Whole",
        vis_dir / "plot_kl_haifa_total.png"
    )

    plot_all(kl, vis_dir / "plot_kl_all_series.png")

    manifest = {
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "k_source": str(k_path),
        "l_source": str(l_path),
        "series_mapping": {
            "LEGACY": {
                "K": "K_HPC_kNIS",
                "L": "Haifa-Legacy",
            },
            "ENTRANT": {
                "K": "K_SIPG_kNIS",
                "L": "Haifa-Bayport",
            },
            "HAIFA_TOTAL": {
                "K": "K_HAIFA_TOTAL_kNIS",
                "L": "Haifa-Legacy + Haifa-Bayport",
            },
        },
        "outputs": [
            "KL_monthly_series_long.tsv",
            "KL_monthly_series_wide.tsv",
            "Visualization/plot_kl_legacy.png",
            "Visualization/plot_kl_entrant.png",
            "Visualization/plot_kl_haifa_total.png",
            "Visualization/plot_kl_all_series.png",
        ],
    }

    with open(kl_dir / "kl_build_manifest.json", "w", encoding = "utf-8") as f:
        json.dump(manifest, f, indent = 2)

    print("Done.")
    print(f"Wrote: {kl_dir / 'KL_monthly_series_long.tsv'}")
    print(f"Wrote: {kl_dir / 'KL_monthly_series_wide.tsv'}")
    print(f"Wrote: {vis_dir / 'plot_kl_legacy.png'}")
    print(f"Wrote: {vis_dir / 'plot_kl_entrant.png'}")
    print(f"Wrote: {vis_dir / 'plot_kl_haifa_total.png'}")
    print(f"Wrote: {vis_dir / 'plot_kl_all_series.png'}")
    print(f"Wrote: {kl_dir / 'kl_build_manifest.json'}")


if __name__ == "__main__":
    main()
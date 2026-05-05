import pandas as pd
import numpy as np

old = pd.read_csv("Data/LP/LP_Panel.tsv", sep="\t")
new = pd.read_csv("Data/LP/common_rule_v5/LP_Panel.tsv", sep="\t")

keys = ["series_id", "year", "month", "quarter"]
cmp = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)

print(cmp["_merge"].value_counts(dropna=False))

for col in ["LP", "w", "Pi", "TEU", "tons"]:
    co = f"{col}_old"
    cn = f"{col}_new"
    if co in cmp.columns and cn in cmp.columns:
        d = (cmp[cn] - cmp[co]).abs()
        print(col, {
            "n_changed": int((d.fillna(0) > 1e-12).sum()),
            "max_abs_diff": float(d.max(skipna=True)) if d.notna().any() else 0.0
        })


import pandas as pd

df = pd.read_csv("Data/LP/common_rule_v5/LP_Panel.tsv", sep="\t")

q = df[df["freq"] == "Q"].copy()
q = q[q["terminal"].notna()].copy()

chk = (
    q.groupby(["port", "year", "quarter"])["w"]
      .nunique(dropna=True)
      .reset_index(name="n_unique_w")
)

print(chk["n_unique_w"].value_counts(dropna=False).sort_index())
print(chk[chk["n_unique_w"] <= 1].head(20))



import pandas as pd
import numpy as np

df = pd.read_csv("Data/LP/common_rule_v5/LP_Panel.tsv", sep="\t")
q = df[df["freq"] == "Q"].copy()

haifa = q[q["port"] == "Haifa"].copy()
wide = haifa.pivot_table(index=["year","quarter"], columns="terminal", values="LP")

wide["log_gap"] = np.log(wide["Haifa-Legacy"]) - np.log(wide["Haifa-Bayport"])
print(wide["log_gap"].describe())
print(wide["log_gap"].diff().describe())
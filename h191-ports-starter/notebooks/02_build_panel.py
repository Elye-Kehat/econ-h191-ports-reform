# %% [markdown]
# # 02 — Build Canonical Port-Month Panel
# Once raw/external datasets exist, this builds `data/processed/ports_panel_month.parquet`.

# %%
from h191.etl import build_ports_panel_month
p = build_ports_panel_month()
p

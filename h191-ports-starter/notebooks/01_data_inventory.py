# %% [markdown]
# # 01 — Data Inventory & Source-to-Field Map
# This notebook lists datasets, adds them to `data_catalog.yaml`, and previews raw files.

# %%
import pandas as pd
from pathlib import Path
from h191.paths import RAW_DIR, EXTERNAL_DIR, REGISTRY

print("RAW:", RAW_DIR)
print("EXTERNAL:", EXTERNAL_DIR)
pd.read_csv(REGISTRY).tail()

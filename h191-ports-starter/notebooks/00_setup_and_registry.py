# %% [markdown]
# # 00 — Project Setup & Data Registry
# 
# This notebook validates the environment and shows how to record a download
# into the registry. Use VS Code's Jupyter to run cells.

# %%
import sys, platform
import pandas as pd
from pathlib import Path

print(platform.python_version(), sys.executable)

# %% [markdown]
# ## Inspect data registry

# %%
from h191.paths import REGISTRY
import pandas as pd

df = pd.read_csv(REGISTRY)
df.head()

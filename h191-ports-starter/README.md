# H191 Ports — VS Code Starter

This repository is a **research-grade, reproducible** starter focused on **downloading data and storing it cleanly** for your Econ H191 thesis (Ports Reform → K/L → LP).

## Quick start
1. Install **VS Code** with the **Python** and **Jupyter** extensions.
2. Install **Mambaforge** (or Miniconda).
3. Create the environment:
   ```bash
   conda env create -f env.yml
   conda activate h191
   pre-commit install
   ```
4. Open this folder in VS Code. Select the `h191` interpreter.
5. Test the downloader (example):
   ```bash
   python scripts/download.py      --name cbs_example      --url https://example.com/data.csv      --dest-subdir raw      --filename data.csv
   ```

## Data layout
- `data/raw/` — exact copies of downloads, one subfolder per dataset (`data/raw/<name>/...`).
- `data/external/` — data you export from portals manually.
- `data/interim/` — temporary working files (never commit large files unless using DVC/LFS).
- `data/processed/` — cleaned/analysis-ready parquet/csv.
- `data/_registry.csv` — **append-only ledger** of every download (source, hash, size, timestamp).
- `secrets/` — place credentials here (e.g., `.env`); **never commit**.

## Notebooks
Use Jupyter inside VS Code. Keep heavy logic inside `src/h191/` and call it from notebooks.

## Reproducibility
- `env.yml` pins the Python toolchain.
- `pre-commit` ensures formatting/linting.
- Optional: initialize **DVC** (`dvc init`) and add a remote (`dvc remote add -d ...`) if you want cloud storage for large data.


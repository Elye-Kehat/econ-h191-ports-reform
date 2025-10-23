# Data Directory

- `raw/` — pristine downloads by dataset name (one folder per dataset).
- `external/` — manually exported files from portals (document the steps).
- `interim/` — working files during cleaning/joins.
- `processed/` — final, analysis-ready outputs (parquet/csv).

**Registry:** Every automated download appends one row to `_registry.csv` with SHA-256, size, content-type, and timestamp.

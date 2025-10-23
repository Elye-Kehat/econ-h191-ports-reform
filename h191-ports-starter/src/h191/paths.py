from pathlib import Path

# Project root = parent of this file's parent (src/h191/ -> src -> project root)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REGISTRY = DATA_DIR / "_registry.csv"
SECRETS_DIR = ROOT / "secrets"

def ensure_dirs() -> None:
    for p in [DATA_DIR, RAW_DIR, EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, SECRETS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

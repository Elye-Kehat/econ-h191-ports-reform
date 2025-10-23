import yaml
from pathlib import Path
from typing import Dict, Any

from .paths import ROOT

CATALOG_PATH = ROOT / "data_catalog.yaml"

def load_catalog() -> Dict[str, Any]:
    if not CATALOG_PATH.exists():
        return {}
    return yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}

from __future__ import annotations

from pathlib import Path
import pandas as pd
from ..paths import RAW_DIR, EXTERNAL_DIR, PROCESSED_DIR
from ..datasets.schema import DTYPES

def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                pass
    return df

def build_ports_panel_month(output: Path | None = None) -> Path:
    """Stub builder: creates an empty canonical table if real inputs not yet present.
    Replace with joins/cleaning once the first raw datasets are downloaded.
    """
    cols = list(DTYPES.keys())
    df = pd.DataFrame(columns=cols)
    df['port_id'] = pd.Categorical([])
    df['date'] = pd.to_datetime(pd.Series([], dtype='datetime64[ns]'))
    df['year'] = pd.Series([], dtype='int32')
    df['month'] = pd.Series([], dtype='int8')
    df = coerce_dtypes(df)

    out = output or (PROCESSED_DIR / "ports_panel_month.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out

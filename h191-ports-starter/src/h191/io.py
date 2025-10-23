from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from .paths import RAW_DIR, REGISTRY, ensure_dirs


@dataclass
class DownloadResult:
    path: Path
    bytes: int
    sha256: str
    content_type: str
    url: str


def _sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def download(url: str, dest_dir: Path, filename: Optional[str] = None) -> DownloadResult:
    ensure_dirs()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"

    out_path = dest_dir / filename

    with _session().get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        ctype = r.headers.get("content-type", mimetypes.guess_type(filename)[0] or "application/octet-stream")
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {filename}") as pbar:
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    sha = _sha256_file(out_path)
    size = out_path.stat().st_size

    return DownloadResult(path=out_path, bytes=size, sha256=sha, content_type=ctype, url=url)


def append_registry(dataset_id: str, name: str, url: str, dest_subdir: str, filename: str, result: DownloadResult, notes: str = "") -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY.exists():
        REGISTRY.write_text("dataset_id,name,url,dest_subdir,filename,bytes,sha256,content_type,fetched_at_iso,notes\n", encoding="utf-8")
    line = f"{dataset_id},{name},{url},{dest_subdir},{filename},{result.bytes},{result.sha256},{result.content_type},{datetime.now(timezone.utc).isoformat()},{notes}\n"
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(line)

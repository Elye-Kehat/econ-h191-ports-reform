#!/usr/bin/env python
"""Register an existing file (e.g., manual export) into data/_registry.csv."""
from __future__ import annotations

import argparse, hashlib, mimetypes
from pathlib import Path
from datetime import datetime, timezone
from h191.io import append_registry, DownloadResult

def sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--file", required=True, help="path to the local file to register")
    ap.add_argument("--url", default="manual://local")
    ap.add_argument("--dest-subdir", choices=["raw","external","interim","processed"], default="external")
    ap.add_argument("--notes", default="manual import")
    args = ap.parse_args()

    p = Path(args.file).resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    sha = sha256_file(p)
    res = DownloadResult(path=p, bytes=p.stat().st_size, sha256=sha, content_type=ctype, url=args.url)
    append_registry(args.dataset_id, args.name, args.url, args.dest_subdir, p.name, res, notes=args.notes)
    print(f"Registered {p}")

if __name__ == "__main__":
    main()

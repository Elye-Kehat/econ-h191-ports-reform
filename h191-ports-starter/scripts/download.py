#!/usr/bin/env python
"""
CLI: Download a dataset to data/<dest_subdir>/<name>/filename and log to data/_registry.csv

Usage:
  python scripts/download.py --name cbs_va --url https://example.com/va.csv --dest-subdir raw --filename va.csv --dataset-id CBS_VA_2025 --notes "CBS VA table"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from h191.paths import RAW_DIR, DATA_DIR
from h191.io import download, append_registry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="dataset short name, used as subfolder")
    ap.add_argument("--url", required=True, help="download URL")
    ap.add_argument("--dest-subdir", choices=["raw", "external"], default="raw")
    ap.add_argument("--filename", default=None, help="override output filename")
    ap.add_argument("--dataset-id", default=None, help="stable ID for the dataset registry")
    ap.add_argument("--notes", default="", help="freeform notes for the registry")
    args = ap.parse_args()

    base = DATA_DIR / args.dest_subdir / args.name
    base.mkdir(parents=True, exist_ok=True)
    result = download(args.url, base, filename=args.filename)

    dataset_id = args.dataset_id if args.dataset_id else args.name
    append_registry(
        dataset_id=dataset_id,
        name=args.name,
        url=args.url,
        dest_subdir=args.dest_subdir,
        filename=result.path.name,
        result=result,
        notes=args.notes,
    )
    print(f"Saved: {result.path} ({result.bytes} bytes)")

if __name__ == "__main__":
    main()

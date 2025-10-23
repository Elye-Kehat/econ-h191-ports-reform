
#!/usr/bin/env python3
"""
build_lp_v7_runner.py
A tiny wrapper that fixes common relative-path mistakes and then executes
Data/LP/build_lp_mixadjusted_qflex_v7.py with the corrected arguments.

Usage (from your project root):
  python Data/LP/build_lp_v7_runner.py --teu Output/teu_monthly_plus_quarterly_by_port.tsv \
      --tons Output/monthly_output_by_1000_tons_ports_and_terminals.tsv \
      --l-proxy L_proxy/L_Proxy.tsv --granularity auto --force
"""
import os, sys, runpy
from pathlib import Path

def resolve(p: str) -> str:
    """If p doesn't exist, try common 'Data/...' prefixed locations."""
    if not p:
        return p
    P = Path(p)
    if P.exists():
        return str(P)
    # Try relative to CWD
    cand = Path.cwd() / P
    if cand.exists():
        return str(cand)
    # Try under Data/
    data = Path.cwd() / "Data" / P
    if data.exists():
        return str(data)
    # If starts with Output/, L_proxy/, LP/ -> prefix Data/
    first = P.parts[0] if P.parts else ""
    if first in {"Output", "L_proxy", "LP"}:
        data2 = Path.cwd() / "Data" / P
        if data2.exists():
            return str(data2)
    # Try just filename inside Data/Output or Data/L_proxy when a directory hint is missing
    if not P.parent or str(P.parent) == ".":
        for sub in ("Output", "L_proxy", "LP"):
            cand2 = Path.cwd() / "Data" / sub / P.name
            if cand2.exists():
                return str(cand2)
    return str(P)  # give back original, builder will error clearly if still wrong

def main():
    # Collect args as-is, then patch paths for the 3 file flags if present
    argv = sys.argv[1:]
    # Map of flags to the indices after the flag (value positions)
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag in {"--teu", "--tons", "--l-proxy"} and i+1 < len(argv):
            argv[i+1] = resolve(argv[i+1])
            i += 2
        else:
            i += 1

    # Compose sys.argv to execute the real builder as __main__
    target = Path("Data/LP/build_lp_mixadjusted_qflex_v7.py")
    if not target.exists():
        # also support running the wrapper from inside Data/LP/
        alt = Path.cwd() / "build_lp_mixadjusted_qflex_v7.py"
        if alt.exists():
            target = alt
    sys.argv = [str(target.name)] + argv
    runpy.run_path(str(target), run_name="__main__")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Regenerate every table and figure of the paper from the stored results under results/ (no retrieval, no GPU, no Java).

  python paper/make_all.py            # regenerate into paper/output/, then compare byte for byte with paper/expected/
  python paper/make_all.py --no-check # regenerate only

Each step runs in its own Python process, in this order (later steps read the outputs of earlier ones):
  s51.py                    Tables 3 and A3 (T4), F1_data, T3b, T3 (Gate 1 channels, Section 5.3); checks the Figure 1 caption
  f1_split.py               Figures 2 and 3 (F1_under1M, F1_over1M) from F1_data.csv
  s52_53.py                 Tables 4, 6, A4, A5 (T5, T7, T8, T6), T5b, F2_data, F3_data, pool_and_weighting.md, checks_s52_53.csv
  f_sens.py                 Figure 4 (F_sens) from F2_data.csv, F3_data.csv, T8.csv
  appendix.py               Tables A1, A2, A6-A11 (TA1, TA2, TB1-TB4, TC1, TC2), FB1_data
  gate2_twochannel.py       Table 5 (T6b)
Every script cross-checks its numbers and stops with a non-zero exit status on a mismatch.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "output")
EXPECTED = os.path.join(HERE, "expected")

STEPS = [["s51.py"], ["f1_split.py"], ["s52_53.py"], ["f_sens.py"], ["appendix.py"],
         ["gate2_twochannel.py"]]


def files_under(d):
    return sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs)


def compare():
    exp, got = files_under(EXPECTED), files_under(OUT)
    missing = [f for f in exp if f not in got]
    extra = [f for f in got if f not in exp]
    differ = [f for f in exp if f in got and not filecmp.cmp(os.path.join(EXPECTED, f), os.path.join(OUT, f), shallow=False)]
    same = len(exp) - len(missing) - len(differ)
    print(f"\ncompare paper/output with paper/expected: {same} of {len(exp)} files byte-identical", flush=True)
    for label, lst in (("missing", missing), ("not expected", extra), ("different", differ)):
        for f in lst:
            print(f"  {label}: {f}", flush=True)
    return not (missing or extra or differ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true", help="do not compare the outputs with paper/expected/")
    a = ap.parse_args()
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    env = dict(os.environ, MPLBACKEND="Agg")
    for step in STEPS:
        t0 = time.time()
        print(f"\n=== {' '.join(step)}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, step[0])] + step[1:], cwd=ROOT, env=env)
        if r.returncode != 0:
            raise SystemExit(f"{' '.join(step)} failed (exit status {r.returncode})")
        print(f"=== {' '.join(step)}: done in {time.time() - t0:.0f} s", flush=True)
    if not a.no_check and not compare():
        raise SystemExit("outputs differ from paper/expected/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download the 13 BEIR and 12 BRIGHT collections of the paper into data/raw/ and verify them against the paper's files.

  python scripts/download_data.py                      # everything (about 17 GB for BEIR, 0.5 GB for BRIGHT)
  python scripts/download_data.py --beir scifact nfcorpus
  python scripts/download_data.py --bright             # BRIGHT only (--bright pony biology: some domains)
  python scripts/download_data.py --verify-only        # hash the files already in data/raw

BEIR: the official zips (https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip), extracted to
data/raw/beir/<name>/ (corpus.jsonl, queries.jsonl, qrels/test.tsv).
BRIGHT: the Hugging Face dataset xlangai/BRIGHT at revision 3066d29c9651a576c8aba4832d249807b181ecae, folders documents/
and examples/ only, into data/raw/bright/.
After the download, the SHA-256 of every file the pipeline reads is compared with configs/data_checksums.json (the files
used for the paper); a mismatch is reported and the exit status is 1. The Gate 1 sidecars of the paper record the
corpus hash, so they only load against identical corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

RAW = "data/raw"
BEIR = ["arguana", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa", "nfcorpus", "nq", "quora", "scidocs", "scifact",
        "trec-covid", "webis-touche2020"]
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
BRIGHT_REPO = "xlangai/BRIGHT"
BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
CHECKSUMS = "configs/data_checksums.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download_beir(name):
    dst = os.path.join(RAW, "beir", name)
    if all(os.path.exists(os.path.join(dst, f)) for f in ("corpus.jsonl", "queries.jsonl", os.path.join("qrels", "test.tsv"))):
        print(f"[beir] {name}: present, skipped", flush=True)
        return
    os.makedirs(os.path.join(RAW, "beir"), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.join(RAW, "beir")) as tmp:
        zpath = os.path.join(tmp, f"{name}.zip")
        url = BEIR_URL.format(name=name)
        print(f"[beir] {name}: downloading {url}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():  # refuse absolute paths and parent references
                if m.startswith("/") or ".." in m.split("/"):
                    raise SystemExit(f"unexpected path in {zpath}: {m}")
            z.extractall(tmp)
        src = os.path.join(tmp, name)
        if not os.path.isdir(src):
            raise SystemExit(f"{zpath} has no folder {name}/")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
    print(f"[beir] {name}: extracted to {dst}", flush=True)


def download_bright(domains=None):
    from huggingface_hub import snapshot_download
    dst = os.path.join(RAW, "bright")
    print(f"[bright] {BRIGHT_REPO}@{BRIGHT_REVISION}: documents/ and examples/", flush=True)
    snapshot_download(repo_id=BRIGHT_REPO, repo_type="dataset", revision=BRIGHT_REVISION, local_dir=dst,
                      allow_patterns=[f"{f}/{d}-*" for d in domains for f in ("documents", "examples")] if domains else ["documents/*", "examples/*"])


def verify(prefixes=None):
    want = json.load(open(CHECKSUMS, encoding="utf-8"))["files"]
    bad, missing, ok = [], [], 0
    for rel, info in sorted(want.items()):
        if prefixes and not any(rel.startswith(p) for p in prefixes):
            continue
        path = os.path.join(RAW, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        got = sha256(path)
        if got != info["sha256"]:
            bad.append(rel)
            print(f"[verify] MISMATCH {rel}: {got} (expected {info['sha256']})", flush=True)
        else:
            ok += 1
    print(json.dumps({"verified": ok, "mismatched": bad, "missing": missing}), flush=True)
    return not bad and not missing


def main():
    global RAW
    ap = argparse.ArgumentParser()
    ap.add_argument("--beir", nargs="*", default=None, help="BEIR collections (folder names as on the UKP server); none given: all 13")
    ap.add_argument("--bright", nargs="*", default=None, help="BRIGHT domains (e.g. pony biology); none given: all 12")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--raw-dir", default="data/raw", help="target folder (default data/raw, where the pipeline reads)")
    a = ap.parse_args()
    RAW = a.raw_dir
    everything = a.beir is None and a.bright is None
    beir = BEIR if (everything or a.beir == []) else (a.beir or [])
    prefixes = None
    if not everything:
        prefixes = [f"beir/{n}/" for n in beir]
        if a.bright == []:
            prefixes.append("bright/")
        elif a.bright:
            prefixes += [f"bright/{f}/{d}-" for d in a.bright for f in ("documents", "examples")]
    if not a.verify_only:
        for n in beir:
            if n not in BEIR:
                raise SystemExit(f"unknown BEIR collection {n} (one of {BEIR})")
            download_beir(n)
        if everything or a.bright is not None:
            download_bright(a.bright or None)
    if not verify(prefixes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

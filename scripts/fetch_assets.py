#!/usr/bin/env python3
"""Download the prebuilt assets of the paper from Google Drive, verify their SHA-256 and unpack them into data/cache/.

  python scripts/fetch_assets.py                    # all five archives (3.8 GB download, about 4.3 GB unpacked)
  python scripts/fetch_assets.py gate1 sidecars     # only some (keys below)
  python scripts/fetch_assets.py --from-dir ~/Downloads   # use archives downloaded by hand (same file names), read in place
  python scripts/fetch_assets.py --verify-only      # check the archives already in --archive-dir, unpack nothing

Keys (configs/assets.json holds the Google Drive file ids, sizes and SHA-256):
  gate1     crve_gate1_assets.tar.gz   pools, sparse lexical profiles and their Terrier indexes   data/cache/gate1/
  sidecars  crve_s2_sidecars.tar.gz    S2 context sidecars                                          data/cache/sidecars/S2/
  ppmi      crve_ppmi_over1m.tar.gz    PPMI lists of the five collections over 1M documents         data/cache/gate1/
  llm       crve_qe_llm_cache.tar.gz   LLM-Q2E keyword cache                                         data/cache/qe_llm/raw/
  vocab     crve_qe_vocab.tar.gz       BGE-QE vocabulary sidecars                                    data/cache/qe_vocab/

Archives are downloaded with gdown (requirements.txt) into --archive-dir (default data/archives/, ignored by git) and kept
there unless --delete-archives is given; an archive already present with the right SHA-256 is not downloaded again. With
--from-dir the archives are checked and unpacked where they are (nothing is copied or deleted). Every archive holds only
paths under data/cache/ and is unpacked into the repository root. Exit status 1 if a checksum differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CONFIG = "configs/assets.json"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(file_id: str, out: str) -> None:
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is not installed (pip install -r requirements.txt), or download the archives by hand and use --from-dir")
    tmp = out + ".part"
    try:
        got = gdown.download(id=file_id, output=tmp, quiet=False)
    except Exception as e:  # gdown raises on missing permission or quota; the file can still be fetched in a browser
        got = None
        print(f"gdown: {type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}", flush=True)
    if not got or not os.path.exists(tmp):
        raise SystemExit(f"download failed; get https://drive.google.com/uc?id={file_id} in a browser and rerun with --from-dir")
    os.replace(tmp, out)


def extract(path: str, dest: str = ".") -> int:
    """Unpacks a release archive; every member must be a regular file or directory under data/cache/."""
    n = 0
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            name = os.path.normpath(m.name)
            if not (m.isfile() or m.isdir()) or os.path.isabs(m.name) or not (name == "data" or name == os.path.join("data", "cache")
                                                                               or name.startswith(os.path.join("data", "cache") + os.sep)):
                raise SystemExit(f"{path}: unexpected member {m.name!r}")
        tar.extractall(dest, members=members, filter="data")
        n = sum(m.isfile() for m in members)
    return n


def main():
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    by_key = {a["key"]: a for a in cfg["archives"]}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help=f"archives to fetch: {', '.join(by_key)} (default: all)")
    ap.add_argument("--archive-dir", default="data/archives")
    ap.add_argument("--from-dir", default=None, help="read the archives in place from this directory instead of downloading them")
    ap.add_argument("--dest", default=".", help="directory to unpack into (default: the repository root)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--delete-archives", action="store_true", help="remove each archive from --archive-dir after unpacking it")
    a = ap.parse_args()
    keys = a.keys or list(by_key)
    unknown = [k for k in keys if k not in by_key]
    if unknown:
        ap.error(f"unknown archive keys {unknown}; choose from {list(by_key)}")
    if not a.from_dir:
        os.makedirs(a.archive_dir, exist_ok=True)
    bad = []
    for k in keys:
        arc = by_key[k]
        path = os.path.join(a.archive_dir, arc["name"])
        if a.from_dir:
            path = os.path.join(os.path.expanduser(a.from_dir), arc["name"])
            if not os.path.exists(path):
                raise SystemExit(f"{path} not found")
        elif not (os.path.exists(path) and os.path.getsize(path) == arc["bytes"] and sha256(path) == arc["sha256"]):
            if a.verify_only:
                bad.append(arc["name"]); print(f"[{k}] {path} missing or incomplete", flush=True); continue
            print(f"[{k}] downloading {arc['name']} ({arc['bytes'] / 1e9:.2f} GB) from Google Drive", flush=True)
            download(arc["google_drive_id"], path)
        size, digest = os.path.getsize(path), sha256(path)
        if size != arc["bytes"] or digest != arc["sha256"]:
            bad.append(arc["name"])
            print(f"[{k}] CHECKSUM MISMATCH {path}: {size} bytes, sha256 {digest} (expected {arc['bytes']}, {arc['sha256']})", flush=True)
            continue
        print(f"[{k}] {arc['name']}: sha256 ok", flush=True)
        if a.verify_only:
            continue
        n = extract(path, a.dest)
        print(f"[{k}] unpacked {n} files into {os.path.join(a.dest, 'data/cache')}", flush=True)
        if a.delete_archives and not a.from_dir:
            os.remove(path)
    if bad:
        print("failed: " + ", ".join(bad), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

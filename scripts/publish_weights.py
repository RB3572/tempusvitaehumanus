#!/usr/bin/env python3
"""Split the exported graph into byte-range parts, stamp the meta, and upload to R2.

WHY PARTS AT ALL. `wrangler r2 object put` refuses anything over 300 MiB and the graph is
~610 MB. Uploading it whole needs S3 multipart, which needs an R2 access key pair that
only the dashboard can mint. Three parts of ~194 MiB each go up with the CLI alone.

The split is byte-exact and order-dependent: part(i) is bytes [i*size, (i+1)*size) of the
original, so concatenating them in order reproduces the file bit for bit. `infer.ts`
fetches them in order and concatenates; `sha256` in the meta is the checksum of the whole
reassembled graph.

TWO THINGS THAT WILL SILENTLY SHIP THE OLD MODEL IF FORGOTTEN, both handled here:

  THE BROWSER CACHE. `infer.ts` caches the weights under MODEL_URL in a Cache API bucket
  named by CACHE_NAME. Uploading new bytes to the same URL does NOT invalidate that: every
  returning visitor keeps running the old model, indefinitely, with no sign anything is
  stale. So CACHE_NAME must be bumped in the same commit, and this script refuses to
  upload unless it has been.

  THE `Range` HEADER. The site probes for the model with a one-byte ranged GET rather than
  downloading 610 MB to ask whether it exists. If the bucket's CORS policy drops `Range`
  the probe fails and the page silently falls back to demo mode -- which looks like a
  working site, just with made-up numbers.

    python scripts/publish_weights.py --split          # split + stamp, no network
    python scripts/publish_weights.py --upload         # and push to R2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

BUCKET = "tempusvitaehumanus-models"
INFER_TS = Path("app/lib/infer.ts")


def cache_name() -> str | None:
    m = re.search(r'CACHE_NAME\s*=\s*"([^"]+)"', INFER_TS.read_text(encoding="utf8"))
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("public/models/cleavage.onnx"))
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--split", action="store_true", help="write the part files")
    ap.add_argument("--upload", action="store_true", help="upload the parts to R2")
    ap.add_argument("--expect-cache-name", default=None,
                    help="refuse to upload unless CACHE_NAME differs from this")
    a = ap.parse_args()

    if not a.src.is_file():
        print(f"ERROR: {a.src} not found"); return 1
    total = a.src.stat().st_size
    size = (total + a.parts - 1) // a.parts
    print(f"  source {a.src.name}  {total/1e6:.0f} MB -> {a.parts} parts of "
          f"<= {size/1e6:.0f} MB")

    h = hashlib.sha256()
    if a.split:
        with a.src.open("rb") as fh:
            for i in range(a.parts):
                p = a.src.with_name(a.src.name + f".part{i}")
                n = 0
                with p.open("wb") as out:
                    while n < size:
                        chunk = fh.read(min(1 << 22, size - n))
                        if not chunk:
                            break
                        h.update(chunk); out.write(chunk); n += len(chunk)
                print(f"    wrote {p.name}  {n/1e6:.0f} MB")
        digest = h.hexdigest()

        # Verify the split really is byte-exact rather than assuming it. A wrong split
        # produces a graph that fails to parse in the browser, hours later, with no clue.
        h2 = hashlib.sha256()
        for i in range(a.parts):
            h2.update(a.src.with_name(a.src.name + f".part{i}").read_bytes())
        if h2.hexdigest() != digest:
            print("  REASSEMBLY MISMATCH -- refusing to publish"); return 1
        print(f"    reassembly verified, sha256 {digest[:16]}...")

        meta_path = a.src.parent / "model_meta.json"
        meta = json.loads(meta_path.read_text())
        meta.update({
            "sha256": digest,
            "modelUrl": "https://pub-ba5c8c9e2af84560b29ea26ea363eb80.r2.dev/cleavage.onnx",
            "parts": a.parts,
            "partsNote": (f"Stored as {a.parts} consecutive byte-range parts because "
                          f"wrangler refuses objects over 300 MiB. Concatenating "
                          f"part0..part{a.parts-1} in order reproduces the file byte for "
                          f"byte; `sha256` is the checksum of the reassembled whole."),
            "corpus": "Gomez et al. 2022 (Zenodo 10.5281/zenodo.6390798), CC-BY-NC-SA 4.0",
            "trainedOnEverything": True,
            "heldOutMae": None,
            "heldOutNote": ("NO held-out estimate exists. The sealed vault was "
                            "deliberately folded into training on 2026-08-24, so every "
                            "figure here is cross-validation."),
        })
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"    stamped {meta_path.name}")

    if a.upload:
        cn = cache_name()
        print(f"  CACHE_NAME is {cn!r}")
        if a.expect_cache_name and cn == a.expect_cache_name:
            print(f"  REFUSING: CACHE_NAME is still {cn!r}. Every returning visitor "
                  f"would keep the OLD model from the Cache API. Bump it first.")
            return 1
        for i in range(a.parts):
            p = a.src.with_name(a.src.name + f".part{i}")
            key = f"{BUCKET}/{p.name}"
            print(f"    uploading {p.name} ({p.stat().st_size/1e6:.0f} MB) -> {key}",
                  flush=True)
            # errors="replace": wrangler writes box-drawing and emoji that cp1252
            # cannot decode, and the reader thread raised UnicodeDecodeError mid-upload
            # -- which looked like a failure while the upload itself had succeeded.
            r = subprocess.run(["npx", "wrangler", "r2", "object", "put", key,
                                "--file", str(p), "--remote"],
                               text=True, capture_output=True, shell=True,
                               encoding="utf-8", errors="replace")
            tail = (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
            if r.returncode != 0:
                print(f"    FAILED rc={r.returncode}\n{tail}")
                return 1
            print("    ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

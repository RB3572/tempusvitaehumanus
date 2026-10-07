#!/usr/bin/env python3
"""Fail if a number the site asserts disagrees with the measurement it came from.

    python scripts/check_claims.py            # from TempusVitaeHumanus/

WHY THIS EXISTS
The gallery claimed the register-token switch "dropped that ratio to 1.4x" while the
trunk actually staged for deployment measured **8.2x**. Nothing was wrong with the
measurement and nothing was wrong with the deploy pipeline -- the figure was simply typed
into prose once, against a trunk that was later replaced, and prose does not get
recomputed. The same shape of error has already happened twice on this project: an "100%
interval" label that survived a mass recalibration, and a meta that shipped with no
accuracy figure because the score reader was still looking for the mouse project's
filenames.

So the numbers now live in named constants with the measurement file beside them, and this
script compares the two. It is cheap enough to run before every publish and it refuses
rather than warns, because a warning in a long deploy log is a number nobody checks.

WHAT IT DOES NOT DO
It cannot verify prose. "Four of the five outlier tokens are gone" is a sentence a human
has to keep true. It checks the figures that have a file to be checked against:

  * the artifact ratios in ExplanationGallery.tsx, against analysis/artifact_ratio.json
  * valMae in public/models/model_meta.json, against the analysis champion record

A claim whose source file is missing is reported as UNCHECKED, not as passing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
ROOT = SITE.parent
TOL = 0.051          # ratios are quoted to 1 dp, so allow half a display unit


def num(src: str, key: str):
    m = re.search(rf"\b{key}\s*:\s*([0-9]+(?:\.[0-9]+)?)", src)
    return float(m.group(1)) if m else None


def txt(src: str, key: str):
    m = re.search(rf"\b{key}\s*:\s*\"([^\"]*)\"", src)
    return m.group(1) if m else None


def main() -> int:
    bad, unchecked = [], []

    gal = SITE / "app" / "components" / "ExplanationGallery.tsx"
    src = gal.read_text(encoding="utf8")
    ar_path = ROOT / "analysis" / "artifact_ratio.json"
    if not ar_path.exists():
        unchecked.append(f"{ar_path.name} missing -- artifact ratios UNCHECKED")
    elif "ARTIFACT" not in src:
        unchecked.append("ExplanationGallery.tsx has no ARTIFACT constant -- the "
                         "artifact figures are in prose only and cannot be checked")
    else:
        ar = json.loads(ar_path.read_text())
        trunks = ar["trunks"]
        want_key = txt(src, "measuredTrunk")
        claims = {"deployedRatio": (want_key, "ratio_mean"),
                  "deployedTokens": (want_key, "mean_tokens_above_2x"),
                  "stockRatio": ("stock", "ratio_mean"),
                  "stockTokens": ("stock", "mean_tokens_above_2x"),
                  "fixedRatio": ("fixed", "ratio_mean"),
                  "phase2Ratio": ("phase2", "ratio_mean")}
        for cname, (tkey, field) in claims.items():
            claimed = num(src, cname)
            if claimed is None:
                continue
            if tkey is None or tkey not in trunks:
                unchecked.append(f"{cname}: no row '{tkey}' in {ar_path.name} "
                                 f"(rows: {sorted(trunks)})")
                continue
            actual = float(trunks[tkey][field])
            if abs(claimed - actual) > TOL:
                bad.append(f"{cname}: site says {claimed}, {ar_path.name} row "
                           f"'{tkey}'.{field} is {actual:.3f}")
            else:
                print(f"  ok  {cname} = {claimed}  (measured {actual:.3f}, "
                      f"row '{tkey}')")
        n = ar.get("n_frames")
        if n is not None and n < 24:
            bad.append(f"artifact_ratio.json was computed on only {n} frames; the "
                       f"published figure should come from a larger sample")

    meta_p = SITE / "public" / "models" / "model_meta.json"
    champ = None
    for cand in ("champion_96k.json", "champion_96k_s025.json"):
        if (ROOT / "analysis" / cand).exists():
            champ = ROOT / "analysis" / cand
            break
    if not meta_p.exists():
        unchecked.append("model_meta.json missing -- valMae UNCHECKED")
    elif champ is None:
        unchecked.append("no analysis/champion_*.json -- valMae UNCHECKED")
    else:
        meta = json.loads(meta_p.read_text())
        rec = json.loads(champ.read_text())
        mv, rv = meta.get("valMae"), rec.get("mae_embryo")
        if mv is None:
            bad.append("model_meta.json has no valMae -- the site would show no "
                       "accuracy figure at all")
        elif rv is None:
            unchecked.append(f"{champ.name} has no mae_embryo")
        elif abs(float(mv) - float(rv)) > 1e-4:
            bad.append(f"valMae: meta says {float(mv):.4f} h, {champ.name} says "
                       f"{float(rv):.4f} h")
        else:
            print(f"  ok  valMae = {float(mv):.4f} h  (matches {champ.name})")

    for u in unchecked:
        print(f"  UNCHECKED  {u}")
    if bad:
        print("\n  CLAIMS DISAGREE WITH MEASUREMENTS:")
        for b in bad:
            print(f"    - {b}")
        print("\n  Fix the site copy or re-measure. Do not publish.")
        return 1
    print("\n  every checkable claim matches its measurement."
          + ("  (some UNCHECKED above)" if unchecked else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

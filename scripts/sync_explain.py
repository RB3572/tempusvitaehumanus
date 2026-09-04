#!/usr/bin/env python3
"""Copy freshly rendered saliency triptychs into the site and rewrite the gallery list.

WHY THIS IS A SCRIPT AND NOT A HAND-EDIT. `ExplanationGallery.tsx` carries a hardcoded
array of ten entries, each with the embryo id, its true hours and the model's PREDICTED
hours. Those predictions come from whichever trunk rendered the images. Ship a new trunk
and forget this file and the page shows a caption -- "predicted 11.72 h" -- that the
deployed model does not produce, on an image the deployed model did not make. Nothing
errors; the page just quietly lies about its own output.

So the entries are generated from `index.json`, which `attention_map.py` writes beside the
PNGs, and the trunk tag is written into the file as a comment so the next person can see
at a glance which model these came from.

    python3 -m training.attention_map --tag ssl_vitl_96k \
        --backbone vit_large_patch14_reg4_dinov2.lvd142m \
        --weights cache/ssl_vitl_96k/backbone.pt --n 10 --out analysis/attention_96k
    python scripts/sync_explain.py --src ../analysis/attention_96k --tag ssl_vitl_96k
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

GALLERY = Path("app/components/ExplanationGallery.tsx")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="attention_map.py --out directory (must hold index.json)")
    ap.add_argument("--tag", required=True, help="trunk tag, recorded in the file")
    ap.add_argument("--dest", type=Path, default=Path("public/explain"))
    a = ap.parse_args()

    idx = a.src / "index.json"
    if not idx.is_file():
        print(f"ERROR: no {idx}"); return 1
    rows = json.loads(idx.read_text())

    a.dest.mkdir(parents=True, exist_ok=True)
    for old in a.dest.glob("*_gradient.png"):
        old.unlink()
    items = []
    for j, r in enumerate(rows):
        name = f"{j:02d}_{r['embryo']}_gradient.png"
        src = a.src / name
        if not src.is_file():
            # attention_map names the file with the row index; fall back to a glob so a
            # naming change here fails loudly instead of silently shipping nine images.
            cand = sorted(a.src.glob(f"{j:02d}_*_gradient.png"))
            if not cand:
                print(f"ERROR: no rendered PNG for row {j} ({r['embryo']})"); return 1
            src = cand[0]
            name = src.name
        shutil.copy2(src, a.dest / name)
        items.append(f'  {{ file: "{name}", embryo: "{r["embryo"]}", '
                     f'trueH: {r["true_h"]:.1f}, predH: {r["pred_h"]:.2f} }},')

    text = GALLERY.read_text(encoding="utf8")
    block = ("const ITEMS: Item[] = [\n" + "\n".join(items) + "\n];")
    new, n = re.subn(r"const ITEMS: Item\[\] = \[.*?\];", block, text, flags=re.S)
    if n != 1:
        print(f"ERROR: matched the ITEMS array {n} times, expected 1"); return 1
    stamp = f"// Rendered from trunk `{a.tag}`. Regenerate with scripts/sync_explain.py.\n"
    new = re.sub(r"^// Rendered from trunk .*\n", "", new, flags=re.M)
    new = new.replace("type Item = ", stamp + "type Item = ")
    GALLERY.write_text(new, encoding="utf8")
    print(f"  copied {len(items)} triptychs -> {a.dest}")
    print(f"  rewrote {GALLERY} from {a.tag}")
    for it in items:
        print("   ", it.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Model hosting — RESOLVED

**Status: done. The site is live at <https://tempusvitaehumanus.rishib.com/> with real
weights.** This file was a request for help; it is kept as a record of what the answer
turned out to be, because the failed options are worth not retrying.

**This file was inherited from the sibling MOUSE project and described that site**, down
to its hostname, its 48-bin output and its scores. Corrected 2026-09-03; if a number here
looks like it belongs to another corpus, it probably did.

The weights are served from Cloudflare, with Vercel building the site from GitHub `main`
and `NEXT_PUBLIC_MODEL_URL` pointing at the object URL.

---

## The artifact, as shipped

| | |
|---|---|
| File | `public/models/cleavage.onnx` (git-ignored, never committed) |
| Size | **~609 MB** — fp16 weights, fp32 activations, halved from 1217 MB |
| Input / output | `image [1,1,224,224]` → `log_posterior [1,64]` — **64 bins over 0–42 h**, not the mouse project's 48 over 0–18 |
| Recipe | frozen DINOv2 ViT-L/14 **with 4 register tokens**, temporal SSL on the pre-cleavage window; 3-seed 256×2 head; **posterior-mean readout, no fitted quantile, no TTA** |
| Why no TTA | measured on this trunk at **−0.001 h**. The mouse champion used TTA-8; carrying it over would cost eight trunk passes per prediction in a browser for nothing. |

`model_meta.json` is small and **is** committed; it carries the readout parameters and
the published scores, and the page reads them from it.

---

## What was ruled out — do not retry these

Each was tested, not assumed. Every one looks like it works until a browser tries it.

| Option | Why not |
|---|---|
| Commit to the repo | GitHub rejects blobs over 100 MB. |
| Git LFS (free tier) | 1 GB storage, 1 GB/month bandwidth. One visitor exhausts it. |
| **GitHub Release asset** | **CORS.** The URL 302s from `github.com` to `release-assets.githubusercontent.com` and **neither hop sends `Access-Control-Allow-Origin`**. `curl` fetches it happily; a browser `fetch()` is blocked. Verified with an `Origin` header on both hops. |
| `raw.githubusercontent.com` | Does send `ACAO: *`, but caps files at 100 MB. |
| GitHub Pages | 100 MB per file, 1 GB per site. |
| Full fp16 conversion | `onnxconverter_common` ran 3.6 h and 2.4 h and never finished on an 11,142-node graph. The initializers-only route below is what worked. |
| int8 dynamic quantisation | Finished in 0.4 min, moved the decoded prediction 0.35 h — 35× the parity bar — **and produced a larger file**, because of the dequantisation nodes it inserts. |

---

## Two things that will break it again

**`AllowedHeaders` must include `Range`.** The site probes for the model with a one-byte
ranged GET rather than downloading 610 MB to ask whether it exists. Without `Range` in
the CORS policy that probe fails and the page silently drops to demo mode.

**`NEXT_PUBLIC_MODEL_URL` needs a redeploy, not just a save.** Next.js inlines
`NEXT_PUBLIC_*` at build time. Changing the variable in the Vercel dashboard without
rebuilding changes nothing on the deployed site.

---

## How to tell it is working

Open the site. **The absence of the amber "Demo output" badge means the model was
found.** The badge instead shows the execution provider and a timing — `webgpu` when the
browser supports it, `wasm` otherwise (wasm is slow but correct for a ViT-L —
and this site runs ONE view, not eight, so it is markedly faster than the mouse site).
The headline should read a plausible number of hours captioned "posterior mean".

First load fetches 610 MB with a progress readout, then caches it in the browser's Cache
API, so later visits are instant.

**Do not swap in a smaller model to make the first load nicer.** The graph is the exact
recipe every published number describes. Substituting a smaller trunk would put a model
on the page that the reported score does not evaluate.

The score for what is shipped: **1.290 h per-embryo MAE** [1.212, 1.382], 6-fold grouped
CV over 536 patient↔slide components, 3 seeds, against a 6.81 h guess-the-median
baseline. There is **no held-out number** — the sealed vault was deliberately folded into
training on 2026-08-24, so everything is cross-validation and the meta says so.

**Bump `CACHE_NAME` in `app/lib/infer.ts` whenever the weights change.** The browser
caches the graph under `MODEL_URL`; new bytes at the same URL do not invalidate it, so
every returning visitor keeps the old model with no sign anything is stale.
`scripts/publish_weights.py --upload` refuses to run if the name has not moved.

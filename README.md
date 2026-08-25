# Tempus Vitae Humanus

The public site for the **human zygote cleavage-time model** — predicts how many hours
remain until first cleavage from a single still of a one-cell human embryo, and shows
which region of the image the prediction depended on.

Live at **<https://tempusvitaehumanus.rishib.com>** (Vercel builds from `main`, Cloudflare
serves the weights).

The model, the training code and the measured results live in the parent research repo
(`VisionModelHuman/`). This repository is only the website.

---

## What the page does

Everything runs **in the visitor's browser**. Images are never uploaded.

1. **Drop a frame** of a one-cell human embryo (JPEG/PNG/TIFF).
2. `app/lib/preprocess.ts` reproduces the training-time preprocessing exactly — middle
   slice, 0.5/99.5 percentile stretch, bilinear resize to 224×224. If this drifts from
   `training/build_image_cache.py`, every prediction is quietly wrong, so keep them in step.
3. `onnxruntime-web` runs the exported graph and returns a **distribution over 64 ordered
   time bins**, not a single number.
4. `app/lib/decode.ts` turns that into a point estimate, an interval, and the CDF the
   charts draw.

## Where the model looks

`app/components/ExplanationGallery.tsx` shows, for ten real corpus frames spanning 0.1 h
to 41.5 h before division: the input, a saliency map, and the frame masked to the top
quarter of that map.

**These are precomputed, and the reason is not laziness.** The map is a *gradient*
saliency map — the derivative of the predicted hours with respect to each image patch,
which answers "which regions, if changed, would move this prediction". That needs a
backward pass, and `onnxruntime-web` has no autograd, so it cannot be computed in the
browser for an uploaded image. They are rendered offline by
`training/attention_map.py`.

**The backbone's own attention is not shown, deliberately.** It is free to compute, it
would work live in the browser, and it makes a good-looking picture — and on this model it
explains nothing. Scored against occlusion (mask a region, measure how far the prediction
actually moves, no assumptions at all), raw CLS-to-patch attention reached a rank
correlation of **0.00**. The gradient map reached **0.38**. Showing the free one and
calling it an explanation would have been the easy and wrong choice.

## Running locally

```bash
npm install
npm run dev
```

The page needs a model to do anything useful. Point it at one with
`NEXT_PUBLIC_MODEL_URL`, or drop a graph at `public/models/cleavage.onnx` (git-ignored —
never commit weights, GitHub rejects blobs over 100 MB).

```bash
# from the research repo, once a champion is chosen
python scripts/export_onnx.py --ckpt <checkpoint> --code ../training --out public/models
```

`public/models/model_meta.json` is small, **is** committed, and carries the bin edges,
readout parameters and published scores that the page displays. Update it whenever the
model changes or the page will report stale numbers next to fresh predictions.

## Hosting

Weights are too large for GitHub and are served from a CORS-enabled object store.
`HOSTING-HANDOFF.md` records every option that was tried and rejected — GitHub Releases
fail on CORS, raw.githubusercontent caps at 100 MB, Git LFS's free tier is exhausted by a
single visitor. Read it before trying any of them again.

## Honesty rules this site follows

Carried over from the research project, where each was learned the hard way:

- **Every score is quoted against the baseline** — the error of always guessing the
  training median, which is **6.83 h** on this corpus. A model that does not beat that has
  learned nothing, and a number without it means nothing.
- **Per embryo, never per frame.** 66,573 frames but only 687 embryos; frame-level
  intervals are about an order of magnitude too tight.
- **The distribution is shown, not just its mean.** Uncertainty here is genuinely
  heteroscedastic — minutes before division the answer is nearly certain, a day out it is
  vague — and a single number hides that.
- **This is a research tool, not a clinical or diagnostic instrument**, and not a basis for
  any decision about an embryo.

## Data

Trained on the open corpus of Gomez et al. (2022), *A time-lapse embryo dataset for
morphokinetic parameter prediction*, Data in Brief 42:108258 — 704 EmbryoScope recordings
from CHU Nantes, Zenodo `10.5281/zenodo.6390798`, **CC-BY-NC-SA 4.0**. Non-commercial and
share-alike; that licence constrains what may be done with anything derived from it.

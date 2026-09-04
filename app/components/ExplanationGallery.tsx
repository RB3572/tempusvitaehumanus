"use client";

import { useState } from "react";

/**
 * Where the model looks when it judges the time.
 *
 * WHY THESE ARE PRECOMPUTED AND NOT LIVE
 * The map shown here is a gradient saliency map: the derivative of the predicted hours
 * with respect to each image patch, which answers "which regions, if changed, would move
 * THIS prediction". Computing it needs a backward pass, and onnxruntime-web has no
 * autograd — so it cannot be produced in the browser for an uploaded image. These are
 * rendered offline by `training/attention_map.py` on real corpus frames.
 *
 * WHY NOT SHOW THE BACKBONE'S ATTENTION INSTEAD, WHICH IS FREE
 * Raw CLS-to-patch attention falls out of the forward pass at no cost and makes a pretty
 * picture, but measured against occlusion — masking a region and watching how far the
 * prediction actually moves — it reaches only 0.19 on the deployed trunk, against 0.46
 * for the gradient map. It largely shows what DINOv2 attends to, not what our prediction
 * depends on. Only the gradient map is shown, and both numbers are quoted rather than
 * hidden.
 *
 * These figures are rendered with `--bundle`, i.e. with the DEPLOYED 3-seed ensemble, not
 * with a head refit for the occasion. Without that flag attention_map.py fits its own
 * single head and the captions differ from the shipped model by up to 0.36 h — a caption
 * reading "predicted 11.35 h" for a model that does not produce it.
 */

// Rendered from trunk `ssl_vitl_96k (deployed bundle, sigma 0.25)`. Regenerate with scripts/sync_explain.py.
type Item = { file: string; embryo: string; trueH: number; predH: number };

const ITEMS: Item[] = [
  { file: "00_LK523-2_f66_gradient.png", embryo: "LK523-2", trueH: 0.1, predH: 0.69 },
  { file: "01_GSS052-6_f84_gradient.png", embryo: "GSS052-6", trueH: 3.0, predH: 3.06 },
  { file: "02_HE444-4_f67_gradient.png", embryo: "HE444-4", trueH: 5.8, predH: 6.02 },
  { file: "03_LK584-2_f96_gradient.png", embryo: "LK584-2", trueH: 8.8, predH: 8.82 },
  { file: "04_LL854-1_f53_gradient.png", embryo: "LL854-1", trueH: 11.7, predH: 11.54 },
  { file: "05_LC161-2-5_f44_gradient.png", embryo: "LC161-2-5", trueH: 14.6, predH: 14.54 },
  { file: "06_VF269-7_f34_gradient.png", embryo: "VF269-7", trueH: 17.6, predH: 17.48 },
  { file: "07_RI273-6_f6_gradient.png", embryo: "RI273-6", trueH: 20.6, predH: 20.96 },
  { file: "08_GSS052-2_f5_gradient.png", embryo: "GSS052-2", trueH: 23.9, predH: 23.85 },
  { file: "09_RM855-3_f1_gradient.png", embryo: "RM855-3", trueH: 41.5, predH: 38.45 },
];

export default function ExplanationGallery() {
  const [i, setI] = useState(4);
  const it = ITEMS[i];
  const err = it.predH - it.trueH;

  return (
    <section style={{ marginTop: 34 }}>
      <h2
        style={{
          fontSize: 15,
          fontWeight: 800,
          letterSpacing: "-0.01em",
          margin: "0 0 6px",
        }}
      >
        Where the model looks
      </h2>
      <p
        style={{
          margin: "0 0 16px",
          fontSize: 13,
          fontWeight: 600,
          color: "#747474",
          maxWidth: "78ch",
          lineHeight: 1.65,
        }}
      >
        For each frame: the input, a saliency map of how strongly each region moves the
        predicted time, and the same frame with everything outside the top quarter of that
        map removed. On most frames the hot region sits on the zygote, usually on the
        pronuclei, and not on the dish or the well wall. The 41.5 h frame is the honest
        exception — there an air bubble outside the embryo also lights up: not the map&rsquo;s
        hottest region, which stays on the zygote, but bright enough to survive into the
        top quarter. It is the furthest-from-division example in the entire corpus, the
        regime where the zygote itself carries least information, and it is shown rather
        than quietly dropped: a gallery of only the clean cases would be evidence of
        nothing.
      </p>

      <div
        style={{
          display: "flex",
          gap: 6,
          flexWrap: "wrap",
          marginBottom: 14,
        }}
      >
        {ITEMS.map((i2, k) => (
          <button
            key={i2.file}
            onClick={() => setI(k)}
            style={{
              padding: "5px 10px",
              fontSize: 11.5,
              fontWeight: 700,
              borderRadius: 7,
              cursor: "pointer",
              border: "1px solid var(--border)",
              background: k === i ? "#1b1b1b" : "transparent",
              color: k === i ? "#fff" : "#666",
            }}
            title={`${i2.embryo} — true ${i2.trueH} h`}
          >
            {i2.trueH.toFixed(1)} h
          </button>
        ))}
      </div>

      <figure style={{ margin: 0 }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`/explain/${it.file}`}
          alt={`Saliency map for embryo ${it.embryo}: input, saliency, and explanation mask`}
          style={{
            width: "100%",
            height: "auto",
            borderRadius: 10,
            border: "1px solid var(--border)",
            background: "#fff",
          }}
        />
        <figcaption
          style={{
            marginTop: 9,
            fontSize: 11.5,
            fontWeight: 600,
            color: "#8a8a85",
            lineHeight: 1.6,
          }}
        >
          Embryo <strong>{it.embryo}</strong> · true {it.trueH.toFixed(2)} h · predicted{" "}
          {it.predH.toFixed(2)} h ({err >= 0 ? "+" : ""}
          {err.toFixed(2)} h). Saliency is the gradient of the predicted time with respect
          to each image patch.
        </figcaption>
      </figure>

      <div
        style={{
          marginTop: 16,
          padding: "12px 14px",
          borderRadius: 9,
          background: "#faf9f6",
          border: "1px solid var(--border)",
          fontSize: 11.5,
          fontWeight: 600,
          color: "#6b6b66",
          lineHeight: 1.7,
          maxWidth: "84ch",
        }}
      >
        <strong style={{ color: "#3a3a36" }}>How we know this map is real.</strong> Any
        saliency method can produce a confident, meaningless picture, so all three were
        scored against occlusion — masking a region and measuring how far the prediction
        actually moves, which makes no assumptions at all. On the trunk deployed here, the
        gradient map above tracks occlusion at a rank correlation of{" "}
        <strong>0.46</strong> (mean over 8 frames). The backbone&rsquo;s own attention —
        free to compute, and what most published figures show — reaches{" "}
        <strong>0.19</strong> against the same yardstick: not nothing, but under half the
        gradient map&rsquo;s agreement, which is why it is not the map shown here.
        An earlier version of this model scored 0.38 and 0.00 on the same two measures;
        these are the numbers for the model you are actually running.
      </div>
      <div
        style={{
          marginTop: 10,
          padding: "12px 14px",
          borderRadius: 9,
          background: "#faf9f6",
          border: "1px solid var(--border)",
          fontSize: 11.5,
          fontWeight: 600,
          color: "#6b6b66",
          lineHeight: 1.7,
          maxWidth: "84ch",
        }}
      >
        <strong style={{ color: "#3a3a36" }}>Why the backbone has register tokens.</strong>{" "}
        An earlier version of these figures showed a bright blob on empty dish below the
        embryo. It was not the model reading a hidden timestamp — nothing in that region
        correlates with elapsed time (max |r| = 0.09 over 3,000 frames). It was a single
        high-norm <em>artifact token</em>: a plain DINOv2 ViT hijacks one low-information
        patch as global scratch space, giving it a norm <strong>11.8×</strong> the median.
        Switching to the four-register variant, which exists precisely for this, dropped
        that ratio to <strong>1.4×</strong> with no token above threshold — and cost no
        accuracy.
      </div>
    </section>
  );
}

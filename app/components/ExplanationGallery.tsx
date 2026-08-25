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
 * Because it would be a lie. Raw CLS-to-patch attention falls out of the forward pass at
 * no cost and makes a pretty picture, but measured against occlusion — masking a region
 * and watching how far the prediction actually moves — it scored a rank correlation of
 * 0.00 on this model. It shows what DINOv2 attends to, not what our prediction depends
 * on. The gradient map scored 0.38 against the same yardstick. Only the second one is
 * shown, and the number is quoted rather than hidden.
 */

type Item = { file: string; embryo: string; trueH: number; predH: number };

const ITEMS: Item[] = [
  { file: "00_LK523-2_f66_gradient.png", embryo: "LK523-2", trueH: 0.1, predH: 2.81 },
  { file: "01_PC758-2_f77_gradient.png", embryo: "PC758-2", trueH: 3.0, predH: 3.54 },
  { file: "02_GE294-4_f62_gradient.png", embryo: "GE294-4", trueH: 5.8, predH: 6.13 },
  { file: "03_PN636-1-6_f3_gradient.png", embryo: "PN636-1-6", trueH: 8.8, predH: 8.74 },
  { file: "04_DE604-3_f56_gradient.png", embryo: "DE604-3", trueH: 11.7, predH: 11.72 },
  { file: "05_DM1046-12_f27_gradient.png", embryo: "DM1046-12", trueH: 14.6, predH: 14.4 },
  { file: "06_OJ319-6_f29_gradient.png", embryo: "OJ319-6", trueH: 17.6, predH: 17.77 },
  { file: "07_RM126-6_f14_gradient.png", embryo: "RM126-6", trueH: 20.6, predH: 20.3 },
  { file: "08_MA488-3_f12_gradient.png", embryo: "MA488-3", trueH: 23.9, predH: 24.31 },
  { file: "09_RM855-3_f1_gradient.png", embryo: "RM855-3", trueH: 41.5, predH: 40.02 },
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
        map removed. The hot region sits on the zygote — usually on the pronuclei — and not
        on the dish, the well wall, or on debris such as the air bubble in the 41.5 h
        frame.
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
        actually moves, which makes no assumptions at all. The gradient map above tracks
        occlusion at a rank correlation of <strong>0.38</strong>. The backbone&rsquo;s own
        attention, which is free to compute and is what most published figures show,
        scored <strong>0.00</strong> against the same yardstick — so it is not shown here,
        because on this model it explains nothing.
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

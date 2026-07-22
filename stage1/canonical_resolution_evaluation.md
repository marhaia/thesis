# Canonical Analysis Resolution — Candidate Evaluation

This note records the evidence behind the canonical analysis resolution used by
the Stage-1 visual-feature extractor (`CANONICAL_LONG_SIDE` in
`stage1/visual_complexity.py`). It exists so the choice is reproducible and
auditable; it is not runtime code.

## Why a canonical resolution exists

Several of the eight visual features use fixed pixel-scale operators (Canny at a
fixed sigma, fixed-scale Gaussian / steerable pyramids, area-normalised element
counting). On the raw screenshot these operators make the feature values depend
on the native pixel resolution: the **same layout** rendered at 1x / 2x / 3x
produced materially different feature vectors, which made the downstream
headline score resolution-dependent (deterministic, not flaky). The diagnosis
attributed essentially the entire headline gap to `feature_congestion`
(amplified by the `content_presence` max gate), with `edge_density` and
`interactive_element_density` also inherently pixel-scale dependent.

The fix computes all eight features on a fixed, aspect-ratio-preserving
canonical resolution (resize so the long side equals `CANONICAL_LONG_SIDE`),
applied once before feature extraction. The original image and its coordinate
system are untouched for element detection, bounding boxes, overlays, target
selection and the Jokinen search model.

## Native resolution of the reference corpus (UEyes, 1,485 GUI images)

Long-side percentiles (px):

| set | min | p25 | median | p75 | max |
|-----|-----|-----|--------|-----|-----|
| all (n=1485) | 460 | 1024 | 1188 | 1920 | 3950 |
| desktop (495) | 460 | 1024 | 1185 | 1187 | 3950 |
| mobile (495) | 960 | 1920 | 1920 | 1920 | 2340 |
| web (495) | 720 | 1024 | 1200 | 1280 | 1896 |

Fraction of the corpus with long side ≤ candidate (i.e. would be **upscaled**,
which is information-preserving; the remainder is **downscaled**, which loses
detail):

| candidate | ≤ candidate (upscaled) | > candidate (downscaled) |
|-----------|------------------------|--------------------------|
| 1024 | 29.8 % | 70.2 % |
| 1280 | 69.7 % | 30.3 % |
| 1440 | 70.9 % | 29.1 % |

## Candidate comparison

Evaluated on three deterministic synthetic layouts (hard-edged UI, text/line
UI, gradient/low-detail UI) rendered at 1x/2x/3x, plus a deterministic UEyes
sample across all three categories.

**Primary objective — scale gap of the score-driving features**
(`feature_congestion`, `edge_density`, `interactive_element_density`), worst-case
relative 1x/2x/3x gap after canonicalisation:

| candidate | mean | worst |
|-----------|------|-------|
| 1024 | 0.72 % | 4.43 % |
| 1280 | 0.73 % | 4.38 % |
| 1440 | 1.32 % | 6.66 % |

1024 and 1280 are tied (< 5 %); 1440 is clearly worse because the larger upscale
of small inputs re-introduces interpolation artefacts.

**Tie-breaker — information loss and cost** (native→canonical feature drift and
per-image wall-clock on the UEyes sample):

- 1024 downscales ~70 % of the corpus and drifts up to ~44 % on
  high-resolution mobile screenshots; cost ≈ 6.5 s/image.
- 1280 downscales only ~30 % of the corpus and drifts far less
  (e.g. 0 % for the many native-1280 web images); cost ≈ 10 s/image.
- 1440 adds ~25 % cost over 1280 for only a marginal preservation gain and
  loses on the primary scale-gap objective.

## Decision rule and selection

Select the smallest candidate that (a) minimises the score-driving-feature scale
gap **and** (b) does not force the majority of the corpus to be downscaled.

- 1024 fails (b): it downscales 70 % of the corpus, with up to 44 % drift.
- 1440 fails (a): worse scale gap, plus 25 % higher cost.
- **1280 satisfies both** — tied for the best scale gap, downscales only 30 %,
  and sits just above the corpus median long side (1188 px). **Selected.**

This is an evidence-based selection, not a default: 1024 tied 1280 on the
primary objective and lost only on information preservation.

## Scope / limitations

- Applies to general/source-domain GUI analysis only. Automotive transfer and
  UIED/Rico remain out of scope; Driver Glance stays disabled / Future Work.
- The current `feature_norms.json` was built at native resolution. It is **not**
  modified by this change. Because canonicalisation makes the 1x/2x/3x raw
  vectors materially equal, the headline scale-invariance test passes under the
  existing norms; a full norm rebuild at the canonical resolution is a separate,
  later step and is intentionally **not** performed here.

# ViCoP: Study-isolated UEyes reference (candidate v1)

## Scope and motivation

The September 30, 2026 final stimulus pool contains 75 distinct UEyes GUI
screenshots (72 main-study and 3 practice). The archived production reference
used all 1,404 GUI images from the official UEyes `Train` partition, while
excluding all 81 official `Test` images. Cross-checking each of the 75
packaged screenshots against the **original per-image SHA-256s** and the
canonical `image_types.csv` identified exactly **24 Train images** (22 main,
2 practice) and **51 Test images**.

A reference based on all 1,404 Train images would not be disjoint from the final
study pool. Therefore the study-isolated candidate uses **official UEyes Train
minus those 24 final study images**, for **1,380 reference images**: exactly
**460 desktop + 460 mobile + 460 web**. The complete official 81-image Test
partition remains excluded. No study screenshot is present in the new reference.

**Caveat:** This only removes overlap with *empirical feature-reference
estimation*. It does **not** show whether the pretrained UMSI++ checkpoint saw
Train examples during original saliency-model training. Do not claim complete
model-training independence or external validation from this filtering alone.
The reference values are corpus-relative descriptors, **not a validated
human cognitive-load measure**.

## Protected technical semantics

- The raw `v8` and `s5` extraction code and their source rows are unchanged.
- Visual features use the original canonical 1,280-pixel long-side policy.
- Saliency features use native input (no new image resize or UMSI run).
- UMSI++ weights, checkpoint hash, HCEye coefficients, and Stage-2 context
  contract remain unchanged.
- Statistics use sample standard deviation (`ddof=1`) and linear p5/p25/p50/
  p75/p95 quantiles. The frozen reference-build runtime declared NumPy 1.26.4.
- The original reference and its September 1, 2026 audit documents are
  preserved as historical evidence; the production reference pack must point
  to this successor only after required release gates.

## Inputs and audit

`stage1/data/results/study_stimulus_split_75.csv` records the full 75-stimulus
resolution (study ID, category, role, original filename/split/block, hash and
packaged path); `stage1/data/results/study_stimulus_train_exclusions.json`
records all 24 excluded Train images. The CSV is protected from accidental
re-selection by originating from the already exported final pool manifest.

Inputs and checksum identity:

| Input | SHA-256 |
|---|---|
| UEyes `image_types.csv` | `fc22d679849862c0f0476b1d9c270a50383c62304c7635b87cb18a2559a25c68` |
| Visual feature rows (1,485) | `b9769935527a58cf7c9e158a6d3ed0bcd33edaa37aa3dcc40b3ed273c453804a` |
| Saliency feature rows (1,485) | `1f576987fa080bbc7c34062ce1dde86c41c16eaa1fd17be9e00adac4e6a01638` |
| Final stimulus ZIP (75 images) | `6bbf0fcfdb32e00181a8a7c05a3c84a7fb95e018a1344e2e27ff6d48b8e6c154` |
| Final study pool manifest | `e77faface786f339d28bf9683cf4b6b0981ce4abe3c7787f0bf48d40c21628e8` |

The reaggregation first reproduced the complete previous 1,404-row reference
across all 13 features and 10 statistics per feature (maximum absolute
floating-point error at most 1e-12), then excluded the 24 locked stimulus
original identities, then rebuilt the 1,380-row reference. The new reference
population is cryptographically bound to a sorted `filename|category|SHA256`
manifest; every output is hashed in `study_isolated_reference_manifest.json`.

## Drift and interpretation

Relative changes in feature-reference means range from −0.496% to +0.433%.
For the 75 stimulus images, the largest absolute change in the 13
**piecewise-interpolated reference percentiles** was 0.5483 percentile points
(975 feature comparisons). This is **not** a regression of the final x19,
HCEye h6 or experimental Layout Complexity Index: that score-bearing replay
must run with the real model stack in the frozen study/release environment.

## Release requirements

Candidate generation and local statistical checks **do not constitute a new
Stage-1 technical freeze**. Before promoting:

1. Re-run the full `python -m pytest -q` suite on the exact candidate tree.
2. Re-aggregate in the frozen Python/NumPy environment (NumPy 1.26.4), compare
   every output/statistic, and regenerate candidate hashes if bytes differ.
3. Run full UMSI++/EasyOCR Stage-1 x19/h6/layout index regression against
   predeclared representative screens; raw visual/saliency feature extraction
   must remain unchanged. Investigate any score changes attributable to new
   reference percentiles; do not require the final score to be byte-identical.
4. Verify the modified reference pack plus detached SHA-256, source/environment
   identities, complete artifact list and the active audit/release gates.
5. Review scientific wording, including the original UMSI++ model-training
   dependence, and agree the updated freeze identity before study analysis.

The production `main` branch and original release are **not** changed by
building or inspecting this candidate bundle.

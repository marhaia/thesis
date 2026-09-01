# UEyes Development Reference and Held-Out Stimulus Candidates

## Purpose

The production reference distributions and the planned study stimuli must not
use the same UEyes images. This repository therefore applies the official
UEyes `Train/Test` field as a strict development/evaluation boundary:

| Role | Official partition | Desktop | Mobile | Web | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Development reference | `Train` | 468 | 468 | 468 | 1,404 |
| Held-out evaluation pool | `Test` | 27 | 27 | 27 | 81 |

Poster images are outside the GUI reference scope. The 1,404 Train images are
used only to estimate corpus-relative feature distributions. No Test image
contributes to a reference mean, standard deviation, quantile, percentile, or
score anchor.

This is not a universal GUI norm and it is not a human cognitive-load norm.
The Test images are held-out images from UEyes, not an external validation
dataset.

## What changed technically

The UMSI++ checkpoint, model architecture, saliency postprocessing, canonical
visual-feature extraction, and raw `v8`/`s5` feature definitions are unchanged.
Only the corpus used to estimate the production reference distributions was
changed from all 1,485 authorized GUI images to the 1,404 official Train GUI
images. Consequently, percentile-normalized inputs to the project-specific
`h6` block and the experimental layout-complexity index can change slightly.
The change is therefore score-bearing and is protected by new identity hashes,
drift evidence, and regression tests.

The exact old-versus-new norm comparison is stored in
`stage1/data/results/development_reference_norm_drift.json`.

## Candidate stimulus selection

The complete official Test pool remains held out from reference estimation.
Within each of the three GUI categories, the repository ranks Test images by a
seeded SHA-256 key inside each official Test block. A predeclared rotating quota
selects 20 candidates per category and retains seven ordered reserves:

| Category | Block 53 | Block 54 | Block 55 | Selected | Reserves |
| --- | ---: | ---: | ---: | ---: | ---: |
| Desktop | 6 | 7 | 7 | 20 | 7 |
| Mobile | 7 | 6 | 7 | 20 | 7 |
| Web | 7 | 7 | 6 | 20 | 7 |

Across categories, each Test block contributes exactly 20 selected candidates.
This policy prevents score-based or appearance-based cherry-picking while
preserving category and block coverage.

The tracked selection file
`stage1/data/results/ueyes_stimulus_selection_v1.csv` records candidates and
reserves. Its current status is **candidate selection; manual eligibility
review pending**. It must not be described as the final study set yet.

## Manual eligibility review

Review must be completed without viewing pipeline scores or participant
outcomes. A candidate may be excluded only for a predeclared operational
reason, such as a corrupt/unreadable file, an incorrect GUI category, duplicate
content, personal or sensitive information, or incompatibility with the
predeclared study task. Every exclusion must record the filename and reason.
The next ranked reserve from the same category and official Test block replaces
an excluded candidate. The selection must not be optimized for desired scores.

## Reproduction

The builder verifies the official split, all 1,485 authorized GUI filenames,
categories, source-row identities, and on-disk image SHA-256 values before it
writes any output. Statistics use NumPy 1.26.4, sample standard deviation
(`ddof=1`), and linear percentiles.

For the optional drift comparison, recover the previous all-GUI reference from
the frozen Stage-1 tag:

```bash
git show stage1-technical-freeze-v1:stage1/data/results/feature_norms.json \
  > /private/tmp/feature_norms_all_gui_v1.json
```

Then run:

```bash
python3 stage1/tools/build_ueyes_development_reference.py \
  --types-csv <UEYES_ROOT>/image_types.csv \
  --images-dir <UEYES_ROOT>/images \
  --visual-rows <VISUAL_RUN>/canonical_visual_feature_rows.csv \
  --saliency-rows <SALIENCY_RUN>/saliency_feature_rows.csv \
  --current-feature-norms /private/tmp/feature_norms_all_gui_v1.json \
  --output-dir <NEW_OUTPUT_DIRECTORY>
```

The required input and output hashes are recorded in
`stage1/data/results/development_reference_manifest.json`; the active production
artifacts are additionally bound by `stage1/reference_pack_manifest.json` and
its detached SHA-256 file.

## Claim boundary

Appropriate wording:

> Corpus-relative development references were estimated exclusively from the
> official UEyes Train partition (1,404 GUI images). The complete official
> UEyes Test partition (81 GUI images) was excluded from reference estimation
> and used only as the held-out source pool for study-stimulus candidates.

Do not call these values universal norms, population norms, validated
cognitive-load norms, or independent external validation.

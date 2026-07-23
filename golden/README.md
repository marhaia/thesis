# UMSI++ TF1/Keras2 ↔ TF2/Keras3 golden numerical comparison

Isolated audit built from accepted commit `6a17288`. **It changes no production
code, no norms, and no protected branch.** It only adds these files and a
push-triggered workflow that runs entirely in x86_64 Linux CI containers (the
local Apple-Silicon host cannot run TensorFlow 1.14).

## What it does
1. **Phase 0 — freeze protocol** (`manifest.py`): records audit branch/commit,
   digest-pinned container images, upstream commit, checkpoint + fixture SHAs,
   comparison stages, metrics and the **pre-registered thresholds**
   (`common.THRESHOLDS`) *before* any prediction runs.
2. **Phase 1 — run both models** on the same 5 pinned fixtures and the same
   authoritative checkpoint:
   - `legacy_runner.py` executes the **unmodified upstream**
     `singleduration_models.UMSI` + `sal_imp_utilities` preprocessing under
     Python 3.7.9 / TF 1.14.0 / Keras 2.3.1;
   - `modern_runner.py` executes the repo's own `saliency.umsi_model` under
     Python 3.9.6 / TF 2.16.2 / Keras 3.10.0.
   Each records the preprocessing tensor, the raw 512×512 heatmap, the aux
   `out_classif` (numerical only), its genuine end-to-end map, and 3
   same-process + 3 fresh-process repeats (full hashes + max diffs).
3. **Phase 2 — compare** (`compare.py`): applies ONE shared postprocess
   (`shared_postprocess.py`: squeeze → identical resize → identical min-max) to
   BOTH raw tensors, computes every metric, checks each pre-registered
   threshold, and emits the verdict. Saliency features and the 6 HCEye outputs
   are computed with the repo's own code and are labelled **PROVISIONAL**.

## Deliverables (uploaded as real GitHub Actions artifacts)
- `umsi_tf1_tf2_golden_6a17288.json` — strict JSON, full provenance + verdict
- `umsi_tf1_tf2_golden_6a17288.log` — complete concatenated stdout/stderr
- `umsi_tf1_tf2_golden_arrays_6a17288.npz` — raw + shared maps, both envs
- `SHA256SUMS` — SHA-256 + byte size of all deliverables

## Scope
An x86_64 CPU comparison is a **source-matched legacy numerical golden run, not
a claim of bitwise CUDA 9 reproduction.** All `cognitive_load_index` values are
provisional (the saliency norms are stale); they measure downstream numerical
impact only.

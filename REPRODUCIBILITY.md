# Reproducibility

This document defines how to reproduce the technical pipeline without
expanding its scientific claims. Reproduction establishes source, environment,
artifact, and output identity; it does not establish construct validity or
turn the exploratory layout-complexity index into a cognitive-load measure.

## Authoritative identity

A released version is identified by all of the following:

- an annotated Git tag;
- the full commit SHA and Git tree SHA bound by that tag;
- a clean source tree;
- the SHA-256 manifest distributed with the release archive; and
- the pinned runtime, UMSI++, EasyOCR, reference-pack, and normalization
  identities exposed by the application.

An untagged or dirty checkout is development state, even when its tests pass.

## Frozen runtime

The recorded production runtime is macOS 26.3 on arm64 with Python 3.9.6 and
the 97 distributions pinned in
`stage1/environment/requirements-macos-arm64-python3.9.lock.txt`. Verify it in
strict mode before collecting release evidence:

```bash
STAGE1_RUNTIME_VERIFICATION=strict python -c \
  "from stage1.reproducibility import verify_runtime_environment; print(verify_runtime_environment())"
```

GitHub CI uses the declared metadata-only mode because the hosted lightweight
job does not contain the full model stack. It is a regression gate, not a
substitute for strict release replay.

## External model artifacts

The UMSI++ and EasyOCR model files are not stored in Git. Their required file
names, byte sizes, and SHA-256 identities are pinned in the repository and are
verified before score-bearing model construction. Install them with:

```bash
python scripts/download_weights.py
python scripts/download_easyocr_models.py
```

The production route fails closed when a required exact artifact, runtime
identity, saliency result, OCR result, x19 value, or layout-score input is
unavailable or invalid.

## Test levels

1. Run the authoritative regression collection:

   ```bash
   python -m pytest -q
   ```

2. Run heavy and evidence gates only through their documented isolated
   commands. In particular, the Step-2b/P9 evaluator must start in a fresh
   process before TensorFlow/Keras has been imported.

3. Resolve the conditional UEyes-corpus test against the authorized external
   corpus during release verification. A local skip is not a failure, but the
   final release record must state whether the external corpus was verified.

4. Run the public score-bearing route with the verified UMSI++ and EasyOCR
   artifacts. Across all 15 Stage-2 combinations, the packed float32 x19 bytes
   and packed float64 layout-score bytes must be identical for the same image.

5. Repeat representative white, gray, simple-interface, complex-interface,
   malformed-input, missing-artifact, and optional-diagnostic fault probes.

## Candidate-bound successor evidence

Any active evidence contract with lifecycle `CURRENT_SUCCESSOR` and a declared
candidate-replay requirement is release-blocking until that replay has been run
against the exact final candidate commit. The out-of-tree replay report must:

- record the exact final candidate commit;
- finish with the contract-required PASS verdict;
- identify every external input by filename, size, and cryptographic hash; and
- be listed by filename, size, and SHA-256 in both the handoff manifest and
  package checksum file.

An immutable historical report remains valid evidence for its own historical
contract, but it cannot substitute for a fresh replay explicitly required by
the active successor contract. The replay must occur only after the final
candidate commit exists so the report can bind that identity without creating
a self-referential Git commit.

## Documentation-only changes

A documentation cleanup is accepted only when the active score-bearing source
files remain byte-identical or when any non-computational metadata edit is
explicitly listed. The complete tests and real-model x19/layout parity replay
must then pass. Any x19 or layout byte change rejects the cleanup.

## Source archives

A Git archive has no `.git` directory. Bind it to the audited commit and mark
the tree as exported before executing the same test collection:

```bash
export STAGE1_SOURCE_COMMIT=<full-40-character-commit-sha>
export STAGE1_SOURCE_TREE_STATE=exported
STAGE1_RUNTIME_VERIFICATION=metadata_only python -m pytest -q
```

The final release archive must be reconstructed independently and its Git tree
must equal the tagged tree before a clean-room verdict is accepted.

When a curated release tree is derived from a separate development repository,
the handoff must also make the declared relationship independently verifiable.
It must include either the required development Git objects or an exact source
archive plus an allowlisted source-to-release blob manifest. The manifest must
record the development commit/tree, every mapped path, both blob identities,
and whether the bytes are equal; the source artifact and mapping manifest must
themselves be hash-bound by the handoff. Commit/tree strings without resolvable
objects or a reconstructable source artifact are descriptive metadata only and
must be labelled `UNVERIFIED`, not established provenance.

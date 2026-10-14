# ViCoP study-reference and score validation — 2026-10-14

Overall status: **PASS WITH WARNINGS**

This record covers the study-isolated 1,380-image UEyes reference, the real
Stage-1 API execution for all 75 final study stimuli, and a filesystem-isolated
clean-room replay. During validation, no commit, merge, tag, or push was
performed; any later branch freeze is identified by its Git commit.

## Scope and identity

- Working tree: `/Users/Q682780/Thesis_G_study_reference_1380`
- Branch: `fix/study-reference-isolation-1380`
- Base commit: `9220c45597f58075164839b0458966d9b554d204`
- Source state recorded by the export: `dirty`
- Runtime: Python 3.9.6 on macOS 26.7.1 arm64, strict verification of 97
  locked distributions
- Stimulus ZIP SHA-256:
  `6bbf0fcfdb32e00181a8a7c05a3c84a7fb95e018a1344e2e27ff6d48b8e6c154`
- Final-pool manifest SHA-256:
  `e77faface786f339d28bf9683cf4b6b0981ce4abe3c7787f0bf48d40c21628e8`
- Reference-source rows ZIP SHA-256:
  `3d6f3c9312b0d20737251ab156183eaaaf2139f375fdff7542409790f6cac328`
- Official UEyes split CSV SHA-256:
  `fc22d679849862c0f0476b1d9c270a50383c62304c7635b87cb18a2559a25c68`
- Reference population SHA-256:
  `9ef889847f63be8fa2a41653e3a06d4aa992384594d814c57ca1a8f82000b923`
- Feature norms SHA-256:
  `2e9861f04ea80ad03ac14d0a8f5c61522f8150b814dcb31de54a36042f340e3d`
- Reference-pack manifest SHA-256:
  `00537e81365bde48e830e111de74bab16e6c54fccbfaf5dc7ebcfb6641f40581`
- Runtime manifest SHA-256:
  `846e84a3cd35dedd054e05df49fba1081114ff04a66f328a9b3bfb744f5abe02`

## Gate results

| Gate | Status | Evidence |
| --- | --- | --- |
| Frozen environment discovery | PASS | Existing production environment recovered; Python 3.9.6 and locked package versions used from an isolated clone. |
| Strict runtime identity | PASS | macOS 26.7.1 arm64, Python 3.9.6, 97 locked distributions verified. |
| `pip check` | WARN | `grpcio 1.80.0` reports an internal x86_64 wheel metadata tag. Its extension is a universal x86_64/arm64 Mach-O binary, imports as 1.80.0, and real inference passed. |
| Model identity | PASS | UMSI++ `f4290c3f…e9fe`; EasyOCR detector `4a5efbfb…db17`; recognizer `e2272681…29e8`. |
| Reference rebuild | PASS | Original 1,404-row statistics reproduced at maximum absolute error 0; 75 stimulus bytes resolved; 24 Train overlaps excluded; 1,380 rows remain, 460 per category, zero study overlap. |
| Reference artifact identity | PASS | Seven clean-room outputs are byte-identical to the active candidate. |
| Targeted reference/reproducibility tests | PASS | 147 passed, 14 warnings. |
| Complete regression suite | PASS | 641 passed, 1 conditionally skipped, 23 warnings in 333.70 s. The skipped full-corpus deterministic-ID test was then run with the local UEyes corpus: 1 passed. Effective coverage: 642/642. |
| Real API sample E2E | PASS | M01, W17, and D22 returned HTTP 200 with real UMSI++ and EasyOCR, exact v8+s5+h6=x19, 1,380-reference identity, finite values, and exact agreement with the final export. |
| Invalid-input fail-closed check | PASS | A different ZIP exited 1 on SHA mismatch and created no output directory. |
| Full study score export | PASS | 75/75 unique images: 72 main, 3 practice, and 25 per platform. Every response had 19 finite float32 values and a consistent experimental layout index. |
| Export checksum verification | PASS | CSV, JSON, and export manifest all match `SHA256SUMS`. |
| Git isolation | PASS | No commit, merge, tag, or push was performed during validation. No write command was issued against `/Users/Q682780/Thesis_G`; its pre-existing dirty state was not altered. |

## Final score artifacts

Output directory:
`stage1/data/results/study_scores_75_2026-10-14/`

| File | SHA-256 |
| --- | --- |
| `vicop_study_scores_75.csv` | `2ae770692c02007b5308d46eeb2e4e803b4d11712caaba4d44ee60ab236adcba` |
| `vicop_study_scores_75.json` | `25f728a674d4e1e83396ad753f5946750b39ee7ff90af210e58f7ab6e6bd8955` |
| `vicop_study_scores_75_manifest.json` | `faecc6da7f133147b13141e515e9a534ec7f3e37de061d24600a56950521e370` |
| `SHA256SUMS` | `19e59db2da592558b64dea0cd2059efc13914e075b7b9b112afdf0a2deca9ea2` |

The canonical score-row digest is
`f70cd6164d2e53c556d5862a43716f4ba9544ba94d8521f0f28f9ff7f80c8ae3`.
The CSV contains 75 data rows and 83 columns. The experimental layout index
ranges from 10.686704516410828 to 65.03403782844543, with mean
39.827088594436646. These are project-specific descriptors and an
experimental index, not a validated human cognitive-load measure.

## Clean-room reproduction and independence level

The clean room was a fresh local Git checkout at base commit `9220c455`, to
which the exact candidate patch and candidate-only artifacts were copied. The
tracked patch digest was identical in both trees:
`c225ffbca8dfa45f8c670e427deea551d62fa48525f910a4c30b7cba0d088198`.
All 18 untracked source/reference artifacts were byte-identical; their
canonical hash-list digest was
`55ad7f13308c9d953b5c51b0a778bebff29202232d836aa420b272a6b7c7cdb8`.

Independence classification: **PASS WITH LIMITATIONS — filesystem-isolated,
deterministic replay**.

What was independent:

- fresh checkout and fresh process;
- fresh reference aggregation from the source-row ZIP, official split CSV,
  exclusion manifest, and final stimulus ZIP;
- byte comparison of all seven regenerated reference artifacts;
- full strict regression run in the clean checkout;
- fresh real-API inference for one mobile, one web, and one desktop screen,
  compared exactly with the 75-screen export.

What was deliberately shared:

- the same Mac, OS, cloned frozen Python environment, model bytes, input ZIPs,
  and candidate source changes;
- the reference rebuild re-aggregated already extracted v8/s5 source rows; it
  did not independently re-extract all 1,485 source images;
- it did not retrain UMSI++ or recreate EasyOCR models;
- the complete 75-screen inference was executed once in the isolated working
  project; the clean room independently replayed three cross-platform screens.

This is strong evidence for deterministic integration and reference isolation,
but not an independent machine/build/model-training replication.

## Resolved failures and operator events

No unresolved FAIL or BLOCKED gate remains. The following failures were
fail-closed and resolved before the final artifacts were accepted:

1. The first score-export attempt stopped after 42 screens because its audit
   check compared float32 API values with pre-cast float64 values. No output was
   written. The check was corrected to exact IEEE-754 float32 casting; the
   second run passed 75/75.
2. The first strict clean-room regression run was 640 passed, 1 skipped, 1
   failed. The sole failure was a test fixture still simulating macOS 26.3
   against the updated 26.7.1 runtime manifest. The fixture was corrected, its
   focused test passed, and the complete suite then passed.
3. The first clean-room rebuild invocation supplied the study split where the
   tool requires the official UEyes split CSV. It failed on the expected SHA
   guard before producing artifacts. The corrected invocation passed.
4. An independent post-export validator initially used a non-existent HCEye
   field name and raised `KeyError`. The corrected validator passed; the export
   files were not changed.

## Release state

The data are ready for local statistical analysis with the identities above.
At validation time the candidate was uncommitted. A subsequent commit or
branch push may freeze this exact candidate, but promotion to `main`, tagging,
or replacement of historical freezes requires an explicit separate decision.

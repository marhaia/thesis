# ARCHIVED / NON-RUNNABLE — upstream HCEye saliency programs

`test.py` and `dynamic_test.py` are retained only as provenance from the
upstream research code. They are not Stage-1 runtime code, automated tests,
reproducibility evidence, or acceptance authority.

They intentionally are not runnable from the frozen repository environment:
the original programs require unshipped checkpoints and datasets, the optional
`pytorch_ssim` dependency, and developer-machine paths. The authoritative
Stage-1 HCEye work is limited to the separately tested coefficient provenance
and project-specific screenshot proxies in `hceye/hceye_features.py`.

Do not invoke these two archived files as repository tests.

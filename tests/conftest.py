"""Pytest configuration for the test suite.

`test_pipeline.py` is a retained HISTORICAL / INCOMPLETE demo, not an automated
test or current pipeline evidence. It does not compute UMSI++ saliency features:
five zero placeholders make its combined values unsuitable for identification
as the audited Stage-1 x19. Exclude it from collection so the real
smoke/regression suite in `test_smoke.py` runs cleanly. The historical artifact
can still be inspected directly with `python tests/test_pipeline.py`.
"""

collect_ignore = ["test_pipeline.py"]

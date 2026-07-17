"""Pytest configuration for the test suite.

`test_pipeline.py` is an interactive end-to-end DEMO script (it takes an
`image_path` argument and mainly prints values), not an automated test. Exclude
it from collection so the real smoke/regression suite in `test_smoke.py` runs
cleanly. The demo can still be run directly: `python tests/test_pipeline.py`.
"""

collect_ignore = ["test_pipeline.py"]

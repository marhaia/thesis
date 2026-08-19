"""ARCHIVED LEGACY TOOL — excluded from the active Stage-1/Stage-2-v1 contract.

The former endpoint-scale generator depended on retired fail-open saliency,
neutral OCR fallbacks, task/profile modifiers, and response fields that no
longer exist. It must not be executed or cited as current pipeline evidence.

Current scale and endpoint evidence is provided by the maintained canonical
scale regression tests and the real-weight acceptance replay. Reintroducing a
generated endpoint matrix would require a separately reviewed implementation
that preserves mandatory Stage-1 inputs and the qualitative Stage-2-v1
boundary.
"""

import sys


ARCHIVE_MESSAGE = (
    "endpoint_scale_matrix.py is archived and cannot generate current "
    "pipeline evidence. Use the maintained canonical-scale and acceptance "
    "tests instead."
)


def main() -> int:
    print(ARCHIVE_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

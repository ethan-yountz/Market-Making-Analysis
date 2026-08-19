"""Compatibility entry point for the persistent L2 recorder.

The canonical command is ``python scripts/record_lob.py``.  This numbered
entry point remains so the recorder still fits the 00-05 research workflow;
both commands accept the same arguments and use the same implementation.
"""

from record_lob import main


if __name__ == "__main__":
    main()

"""
Layer 3 import bootstrap.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAYER3_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = LAYER3_ROOT / "pipeline"
VALIDATION_DIR = LAYER3_ROOT / "validation"
TOOLS_DIR = LAYER3_ROOT / "tools"


def setup_layer3_paths(*, include_validation: bool = True, include_tools: bool = False) -> None:
    dirs = [PIPELINE_DIR]
    if include_validation:
        dirs.append(VALIDATION_DIR)
    if include_tools:
        dirs.append(TOOLS_DIR)
    dirs.append(LAYER3_ROOT)
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)

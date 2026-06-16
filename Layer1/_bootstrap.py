"""
Layer 1 import bootstrap.

Adds pipeline / archive / tools subfolders to sys.path for Layer 1 imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAYER1_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = LAYER1_ROOT / "pipeline"
ARCHIVE_DIR = LAYER1_ROOT / "archive"
TOOLS_DIR = LAYER1_ROOT / "tools"


def setup_layer1_paths(
    *,
    include_archive: bool = True,
    include_tools: bool = False,
) -> None:
    """Prepend Layer 1 directories to sys.path ."""
    dirs = [PIPELINE_DIR]
    if include_archive:
        dirs.append(ARCHIVE_DIR)
    if include_tools:
        dirs.append(TOOLS_DIR)
    dirs.append(LAYER1_ROOT)
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)

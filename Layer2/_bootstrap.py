"""
Layer 2 import bootstrap.

Adds pipeline / validation / viz folders to sys.path so scripts can
be run directly after the Layer2 folder cleanup.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAYER2_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = LAYER2_ROOT / "pipeline"
VALIDATION_DIR = LAYER2_ROOT / "validation"
VIZ_DIR = LAYER2_ROOT / "viz"
ARCHIVE_DIR = LAYER2_ROOT / "archive"


def setup_layer2_paths(
    *,
    include_validation: bool = True,
    include_viz: bool = False,
    include_archive: bool = False,
) -> None:
    """Prepend Layer 2 directories to sys.path (idempotent)."""
    dirs = [PIPELINE_DIR]
    if include_validation:
        dirs.append(VALIDATION_DIR)
    if include_viz:
        dirs.append(VIZ_DIR)
    if include_archive:
        dirs.append(ARCHIVE_DIR)
    dirs.append(LAYER2_ROOT)

    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)

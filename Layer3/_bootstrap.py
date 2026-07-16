"""
Layer 3 import bootstrap.

The Layer 3 source is split into pipeline, validation, and tools folders, but
the scripts are usually run directly from the repository root. Direct script
execution does not make sibling folders importable, so entry points can call
these helpers before importing Layer 1/2/3 modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAYER3_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LAYER3_ROOT.parent
LAYER1_ROOT = REPO_ROOT / "Layer1"
LAYER2_ROOT = REPO_ROOT / "Layer2"
PIPELINE_DIR = LAYER3_ROOT / "pipeline"
VALIDATION_DIR = LAYER3_ROOT / "validation"
TOOLS_DIR = LAYER3_ROOT / "tools"
LAYER1_PIPELINE_DIR = LAYER1_ROOT / "pipeline"
LAYER1_ARCHIVE_DIR = LAYER1_ROOT / "archive"
LAYER1_TOOLS_DIR = LAYER1_ROOT / "tools"
LAYER2_PIPELINE_DIR = LAYER2_ROOT / "pipeline"
LAYER2_VALIDATION_DIR = LAYER2_ROOT / "validation"
LAYER2_ARCHIVE_DIR = LAYER2_ROOT / "archive"


def setup_layer3_paths(*, include_validation: bool = True, include_tools: bool = False) -> None:
    """Prepend Layer 3 source folders needed by direct script execution."""
    dirs = [PIPELINE_DIR]
    if include_validation:
        dirs.append(VALIDATION_DIR)
    if include_tools:
        dirs.append(TOOLS_DIR)
    dirs.append(LAYER3_ROOT)
    _prepend_existing(dirs)


def setup_layer1_paths(*, include_archive: bool = True, include_tools: bool = False) -> None:
    """Prepend Layer 1 folders used by Layer 3 beat-sync validation."""
    dirs = [LAYER1_PIPELINE_DIR]
    if include_archive:
        dirs.append(LAYER1_ARCHIVE_DIR)
    if include_tools:
        dirs.append(LAYER1_TOOLS_DIR)
    dirs.append(LAYER1_ROOT)
    _prepend_existing(dirs)


def setup_layer2_paths(*, include_validation: bool = True, include_archive: bool = False) -> None:
    """Prepend Layer 2 folders used by the handcrafted A0 baseline arm."""
    dirs = [LAYER2_PIPELINE_DIR]
    if include_validation:
        dirs.append(LAYER2_VALIDATION_DIR)
    if include_archive:
        dirs.append(LAYER2_ARCHIVE_DIR)
    dirs.append(LAYER2_ROOT)
    _prepend_existing(dirs)


def _prepend_existing(dirs: list[Path]) -> None:
    # Insert in reverse so the caller's first directory keeps highest priority.
    for d in reversed(dirs):
        if not d.exists():
            continue
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)

"""
BayesOpt import bootstrap.

Adds the BayesOpt folder and its sub-directories to sys.path so that
scripts can be run directly from any working directory.

Usage (at the top of any script in this layer):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    # or simply:
    import _bootstrap; _bootstrap.setup_bayesopt_paths()
"""
from __future__ import annotations

import sys
from pathlib import Path

BAYESOPT_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = BAYESOPT_ROOT / "tools"


def setup_bayesopt_paths(*, include_tools: bool = True) -> None:
    """Prepend BayesOpt directories to sys.path (idempotent)."""
    dirs = [BAYESOPT_ROOT]
    if include_tools:
        dirs.append(TOOLS_DIR)

    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)

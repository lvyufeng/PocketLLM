"""Make the package importable from a bare source checkout, subprocesses included.

The Python package lives under ``python/`` rather than at the repository root.
pytest's ``pythonpath`` setting in ``pyproject.toml`` fixes that for *this*
process, but several tests here run their probe in a fresh interpreter --
``tests/abi/test_no_runtime_import.py`` and the boundary tests both do, because
``sys.modules`` is global to a session and a leaked runtime has to be caught in
an interpreter that imported nothing else.

A child process inherits the environment, not ``sys.path``, so the path is
exported once here instead of being threaded through every ``subprocess.run``
call site.
"""

from __future__ import annotations

import os
import pathlib

PYTHON = pathlib.Path(__file__).resolve().parent.parent / "python"

_existing = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(PYTHON) if not _existing else f"{PYTHON}{os.pathsep}{_existing}"
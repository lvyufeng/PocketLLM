"""Fail the documentation build when ``docs/llms.txt`` no longer matches the nav.

Registered under ``hooks:`` in ``mkdocs.yml``. ``mkdocs build --strict`` -- which is what
``.github/workflows/pages.yml`` runs -- promotes this warning to an error, so a nav edit that does
not regenerate the index fails the build rather than publishing a stale one.

That guard has to live in the build rather than in the test suite, because **CI runs no pytest**:
the pages workflow builds the docs and that is all. The build is already this repository's link
checker (``validation.links.not_found``), so an index check belongs in the same place.

The generator is loaded from ``scripts/gen_llms_txt.py`` rather than reimplemented, so there is
exactly one definition of what the file should contain.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.llms_txt_staleness")


def _generator(config):
    """Import ``scripts/gen_llms_txt.py`` from the repository the config lives in."""
    script = Path(config["config_file_path"]).resolve().parent / "scripts" / "gen_llms_txt.py"
    spec = importlib.util.spec_from_file_location("_pocketllm_gen_llms_txt", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def on_config(config):
    """Warn when the checked-in index differs from the one the nav generates."""
    index = Path(config["docs_dir"]) / "llms.txt"
    if not index.exists():
        log.warning(
            "docs/llms.txt is missing; generate it with `python scripts/gen_llms_txt.py`"
        )
        return config

    try:
        expected = _generator(config).generate(Path(config["config_file_path"]))
    except Exception as exc:  # noqa: BLE001 -- a hook must not break the build on its own error
        log.warning("could not check docs/llms.txt against the nav: %s", exc)
        return config

    if index.read_text(encoding="utf-8") != expected:
        log.warning(
            "docs/llms.txt is stale: the nav no longer matches it. "
            "Run `python scripts/gen_llms_txt.py` and commit the result."
        )
    return config

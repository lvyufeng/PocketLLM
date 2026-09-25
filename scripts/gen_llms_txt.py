#!/usr/bin/env python3
"""Generate ``docs/llms.txt`` from the documentation site's navigation.

``llms.txt`` is the machine-readable index of the published site: one line per page, with the
absolute URL an agent or a script can fetch without guessing a path. It is generated rather than
hand-maintained because the site publishes 90-odd pages and a hand-written list drifts the moment a
page is added.

``mkdocs.yml``'s ``nav`` is the source because ``mkdocs build --strict`` already makes it a
complete one: ``validation.nav.omitted_files`` is a warning there, and ``--strict`` promotes
warnings to errors, so a document missing from the nav fails the build rather than shipping
unreachable. That is what makes the nav safe to derive an index from.

Usage::

    python scripts/gen_llms_txt.py            # write docs/llms.txt
    python scripts/gen_llms_txt.py --check    # exit 1 if the checked-in file is stale

``hooks/llms_txt_staleness.py`` calls :func:`generate` on every docs build, so the check runs in CI
as well — see the note in ``CLAUDE.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "mkdocs.yml"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "llms.txt"

_HEADER = """\
# PocketLLM

> {description}

This is the index of every page the documentation site publishes. It is generated from `mkdocs.yml`
by `scripts/gen_llms_txt.py`; edit the nav, not this file, and re-run the script.
"""

# ``mkdocs.yml`` carries Python object tags for its Markdown extensions (the slugify function and
# the emoji index). They are irrelevant here and a plain SafeLoader refuses them, so the loader
# maps them to None rather than importing the objects they name -- which would make this script
# depend on mkdocs-material and on a working plugin import for a file that only needs the nav.
_IGNORED_TAGS = (
    "tag:yaml.org,2002:python/object/apply:",
    "tag:yaml.org,2002:python/name:",
    "tag:yaml.org,2002:python/object:",
)


def _loader() -> type:
    class _Loader(yaml.SafeLoader):
        pass

    for prefix in _IGNORED_TAGS:
        _Loader.add_multi_constructor(prefix, lambda loader, suffix, node: None)
    return _Loader


def page_url(site_url: str, target: str) -> str:
    """The absolute URL ``target`` is published at.

    Mirrors ``use_directory_urls: true``: ``a/b.md`` becomes ``a/b/``, and a directory's
    ``README.md`` or ``index.md`` is the directory itself. MkDocs treats a ``README.md`` as the
    index page of its directory, which is why ``docs/README.md`` is the site home.
    """
    if target.startswith(("http://", "https://")):
        return target
    path = Path(target)
    rel = path.parent if path.name in ("README.md", "index.md") else path.with_suffix("")
    parts = [part for part in rel.parts if part not in (".", "")]
    return site_url + ("/".join(parts) + "/" if parts else "")


def _description(config: dict) -> str:
    """``site_description`` as a single line.

    It is a folded YAML scalar, so it arrives with its line breaks intact and would otherwise break
    the blockquote the llmstxt.org format asks for.
    """
    return " ".join(str(config.get("site_description", "")).split())


def generate(config_path: Path = DEFAULT_CONFIG) -> str:
    """The ``llms.txt`` text for the nav in ``config_path``."""
    config = yaml.load(Path(config_path).read_text(encoding="utf-8"), Loader=_loader())
    site_url = str(config["site_url"]).rstrip("/") + "/"

    # A nav entry whose value is a string is one page; a list is a section. External entries are
    # kept as bullets rather than given a section of their own -- a "## Changelog" heading over a
    # single off-site link reads as a mistake.
    top_level: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for entry in config["nav"]:
        for title, value in entry.items():
            if isinstance(value, str):
                top_level.append(f"- [{title}]({page_url(site_url, value)})")
            else:
                sections.append(
                    (
                        title,
                        [
                            f"- [{item_title}]({page_url(site_url, item_target)})"
                            for item in value
                            for item_title, item_target in item.items()
                        ],
                    )
                )

    out = [_HEADER.format(description=_description(config)).rstrip("\n"), ""]
    out.append("## Pages")
    out.append("")
    out.extend(top_level)
    for title, entries in sections:
        out.append("")
        out.append(f"## {title}")
        out.append("")
        out.extend(entries)
    return "\n".join(out).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if the file on disk differs from the generated one",
    )
    args = parser.parse_args(argv)

    generated = generate(args.config)
    output = Path(args.output)

    if args.check:
        current = output.read_text(encoding="utf-8") if output.exists() else ""
        if current == generated:
            print(f"{output} is current ({generated.count(chr(10))} lines)")
            return 0
        print(
            f"{output} is stale: regenerate it with "
            f"`python {Path(__file__).resolve().relative_to(REPO_ROOT)}`",
            file=sys.stderr,
        )
        return 1

    output.write_text(generated, encoding="utf-8")
    print(f"wrote {output} ({generated.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

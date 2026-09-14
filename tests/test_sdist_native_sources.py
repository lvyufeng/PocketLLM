"""The sdist must be able to configure CMake.

Installing from PyPI builds the native engine out of the sdist, so every source
path `cpp_engine/CMakeLists.txt` names must be inside the archive that
`MANIFEST.in` describes. Nothing checked the two against each other, and 0.1.1
shipped with 110 `add_executable` calls naming `tools/` and `tests/` files that
`MANIFEST.in` omits. CMake configuration then failed with one "Cannot find source
file" per target, which failed the install for every user, after the Torch CUDA
extensions had already compiled for several minutes.

The fix is an option, `POCKET_BUILD_DEV_TARGETS`, whose default follows whether the
tree holds `tools/` and `tests/`: a git checkout keeps building the developer
tools and tests, an unpacked sdist does not declare them at all. These tests pin
both halves of that -- the guarded targets are allowed to be unshipped but must
exist, and everything outside the guard must be shipped.

`MANIFEST.in` is checked rather than the built archive because that is the input
the release actually controls; the archive itself is verified by the install step
in `docs/PYPI_RELEASE.md`.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CMAKE_LISTS = ROOT / "cpp_engine" / "CMakeLists.txt"
MANIFEST = ROOT / "MANIFEST.in"

# The guard the test splits the file on. Asserted to exist, so renaming the option
# fails this file loudly instead of turning every assertion below into a no-op.
GUARD_OPEN = re.compile(r"^\s*if\(POCKET_BUILD_DEV_TARGETS\)\s*$", re.MULTILINE)
GUARD_CLOSE = re.compile(r"^\s*endif\(\)\s*#\s*POCKET_BUILD_DEV_TARGETS\s*$", re.MULTILINE)

# Directives this file understands. An unrecognised one is a hard failure: a
# `graft` or `global-include` added to MANIFEST.in would otherwise make the
# shipped-set computation silently too small, and the test would complain about
# files that are in fact shipped.
KNOWN_DIRECTIVES = {"include", "recursive-include", "prune", "exclude", "global-exclude"}

TARGET_RE = re.compile(r"^\s*(add_executable|add_library|pybind11_add_module)\(([^)]*)\)", re.MULTILINE)
SOURCE_SUFFIXES = (".cpp", ".cu", ".cuh", ".cc", ".h", ".hpp")

# The only unguarded targets with a literal source file. Every other target takes a
# `${..._SOURCES}` variable, so these two are what catches the parser drifting out
# from under the assertions below.
LITERAL_INSTALL_SOURCES = {"cpp_engine/engine/main.cpp", "cpp_engine/python/bindings.cpp"}


def _split_guard() -> tuple[str, str]:
    """Return (everything outside the dev-target guard, everything inside it).

    The text after the guard belongs to the first value on purpose. A target
    appended below the guard is unguarded too, and dropping it silently would let
    exactly the bug this file exists for walk back in through the end of the file.
    """
    text = CMAKE_LISTS.read_text()
    opens = list(GUARD_OPEN.finditer(text))
    closes = list(GUARD_CLOSE.finditer(text))
    assert len(opens) == 1, f"expected exactly one POCKET_BUILD_DEV_TARGETS guard, found {len(opens)}"
    assert len(closes) == 1, f"expected exactly one guard endif, found {len(closes)}"
    assert opens[0].end() < closes[0].start(), "the guard is closed before it is opened"
    outside = text[: opens[0].start()] + text[closes[0].end() :]
    return outside, text[opens[0].end() : closes[0].start()]


def _sources(block: str) -> set[str]:
    """Source paths named by add_executable/add_library calls in `block`.

    Paths are relative to cpp_engine/, which is where CMakeLists.txt lives, and
    returned as sdist-relative so they can be compared against MANIFEST.in.
    """
    found = set()
    for _kind, arguments in TARGET_RE.findall(block):
        # arguments is "<target> [STATIC] <source>..." on one line, or
        # "<target> STATIC\n  <source>\n  <source>..." across lines.
        for token in arguments.split():
            if "$" in token or token in {"STATIC", "SHARED", "MODULE", "INTERFACE", "OBJECT", "EXCLUDE_FROM_ALL"}:
                continue
            if token.endswith(SOURCE_SUFFIXES):
                found.add((Path("cpp_engine") / token).as_posix())
    return found


def _targets(block: str) -> set[str]:
    found = set()
    for _kind, arguments in TARGET_RE.findall(block):
        tokens = arguments.split()
        if tokens:
            found.add(tokens[0])
    return found


def _shipped(path: str) -> bool:
    """Whether MANIFEST.in would put the repo-relative `path` in the sdist.

    Only the rules are modelled, not a tree walk: the question is always whether a
    specific path CMakeLists.txt names, or a specific file on disk, is selected.
    """
    includes: set[str] = set()
    recursive: list[tuple[str, list[str]]] = []
    excluded: set[str] = set()
    pruned: list[str] = []

    directives = set()
    for line in MANIFEST.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        directive = parts[0]
        directives.add(directive)
        assert directive in KNOWN_DIRECTIVES, (
            f"MANIFEST.in uses {directive!r}, which the shipped-set computation in this test does not "
            f"model. Teach it the directive rather than letting the test pass on an incomplete set."
        )
        if directive == "include":
            includes.update(parts[1:])
        elif directive == "recursive-include":
            recursive.append((parts[1].rstrip("/"), parts[2:]))
        elif directive == "exclude":
            excluded.update(parts[1:])
        elif directive == "prune":
            pruned.append(parts[1].rstrip("/"))
        elif directive == "global-exclude":
            raise AssertionError("global-exclude needs a pattern matcher across the whole tree; add it before use")

    assert directives, "MANIFEST.in is empty, so the sdist would ship nothing"

    if any(path == prune or path.startswith(prune + "/") for prune in pruned):
        return False
    if path in excluded:
        return False
    if path in includes:
        return True
    for directory, patterns in recursive:
        if not path.startswith(directory + "/"):
            continue
        name = path.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            return True
    return False


def test_the_guard_exists() -> None:
    # _split_guard asserts this too; called out separately so a rename reports the
    # reason rather than looking like a target went missing.
    _split_guard()


def test_nothing_is_declared_after_the_guard() -> None:
    """The guard has to be the last thing in the file to describe the file's whole tail."""
    text = CMAKE_LISTS.read_text()
    tail = text[GUARD_CLOSE.search(text).end() :]
    assert not tail.strip(), (
        "there is content after the POCKET_BUILD_DEV_TARGETS guard, where it is not covered by the "
        f"option and would break an sdist configure anyway:\n{tail.strip()[:400]}"
    )


def test_every_unguarded_source_is_shipped_in_the_sdist() -> None:
    """A source outside the guard is one an install compiles, so it must be in the archive."""
    unguarded, _guarded = _split_guard()
    sources = _sources(unguarded)
    assert sources == LITERAL_INSTALL_SOURCES, (
        "the set of literal sources outside the dev-target guard changed to "
        f"{sorted(sources)}; if a new target was added here, confirm its source is shipped "
        "or move it inside the guard, then update LITERAL_INSTALL_SOURCES"
    )

    missing = sorted(path for path in sources if not _shipped(path))
    assert not missing, (
        f"cpp_engine/CMakeLists.txt declares {len(missing)} target source(s) that MANIFEST.in does not "
        f"ship, so CMake configuration fails on an unpacked sdist:\n  "
        + "\n  ".join(missing)
        + "\nAdd them to MANIFEST.in, or move the target inside the POCKET_BUILD_DEV_TARGETS guard."
    )


def test_guarded_sources_exist_in_the_repository() -> None:
    """Skipping the guard is only safe if everything inside it is a real development file."""
    _unguarded, guarded = _split_guard()
    sources = _sources(guarded)
    assert len(sources) >= 50, f"only {len(sources)} guarded sources parsed, the parser has probably drifted"

    absent = sorted(path for path in sources if not (ROOT / path).exists())
    assert not absent, "the dev-target guard names sources that do not exist:\n  " + "\n  ".join(absent)


def test_dev_only_sources_are_not_shipped() -> None:
    """The guard's point is that these stay out of the archive, keeping the sdist small."""
    _unguarded, guarded = _split_guard()
    shipped_dev = sorted(path for path in _sources(guarded) if _shipped(path))
    assert not shipped_dev, (
        "these guarded, development-only sources are in the sdist anyway: they are dead weight in the "
        "archive, which is what the guard exists to avoid:\n  " + "\n  ".join(shipped_dev)
    )


def test_the_library_targets_are_not_guarded() -> None:
    """Guarding what an install actually builds would leave the sdist with nothing to compile."""
    unguarded, guarded = _split_guard()
    required = {"pocket_core", "pocket_runtime", "pocket_cpp_core", "pocketllm_engine", "pocketllm_cpp"}
    missing = sorted(required - _targets(unguarded))
    assert not missing, (
        f"targets an install needs are not declared outside the dev-target guard: {missing}"
    )


def test_the_guard_default_follows_the_tree() -> None:
    """A constant default is wrong either way: ON breaks an sdist install, OFF hides the tests."""
    text = CMAKE_LISTS.read_text()
    assert 'EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools"' in text
    assert 'EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests"' in text
    assert "${POCKET_DEV_TARGETS_DEFAULT}" in text, "the option no longer defaults to the tree-derived value"


def test_manifest_ships_the_engine_sources_the_library_needs() -> None:
    """The static libraries compile from tree files that must reach the archive."""
    for directory in ("backends", "core", "engine", "include", "python", "cmake"):
        files = [
            path.relative_to(ROOT).as_posix()
            for path in sorted((ROOT / "cpp_engine" / directory).rglob("*"))
            if path.is_file()
        ]
        assert files, f"cpp_engine/{directory}/ has no files, so this check tests nothing"
        shipped = [path for path in files if _shipped(path)]
        assert shipped, (
            f"MANIFEST.in ships no file from cpp_engine/{directory}/, so the library cannot compile "
            f"from an unpacked sdist"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

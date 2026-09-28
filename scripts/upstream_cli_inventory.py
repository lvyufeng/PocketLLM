#!/usr/bin/env python3
"""What vLLM and SGLang put on the command line, against what PocketLLM does.

Written because "make our flags look like theirs" is a claim nobody can check by reading a design
document, and because both stacks move: the answer is only true of the commit it was read at. Point
this at two checkouts and it prints the inventory, the flag-by-flag collision check, and the tables
the design document quotes.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/vllm-project/vllm
    cd vllm && git sparse-checkout set vllm/engine
    git clone --depth 1 --filter=blob:none --sparse https://github.com/sgl-project/sglang
    cd sglang && git sparse-checkout set python/sglang/srt/arg_groups

    python scripts/upstream_cli_inventory.py --vllm /tmp/vllm_ref --sglang /tmp/sglang_ref

The two stacks declare their flags in different shapes, and that difference is half of what this
reports:

* vLLM builds them by reflection over config dataclasses, but *registers* them imperatively --
  one ``add_argument_group(title="CacheConfig")`` and a run of ``add_argument`` calls per config
  class. So the group is a literal string in the source and the flag is a literal beside it.
* SGLang makes every flag a field of a namespace class under ``arg_groups/fields/``, annotated
  ``A[str, Arg(help=...)]``, and derives the flag name from the field name
  (``_field_to_cli_name``) unless ``Arg(cli_name=...)`` overrides it. The group is the file.

Nothing here imports either project: both are read as text, so this runs in any environment and
against a commit whose dependencies are not installed.
"""

from __future__ import annotations

import argparse
import ast
import collections
import pathlib
import re
import subprocess
import sys
from dataclasses import fields
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------------------------


def vllm_flags(root: pathlib.Path) -> tuple[dict[str, list[str]], str]:
    """``{group: [flag, ...]}`` for vLLM, and the commit it was read at.

    The group boundaries come from splitting the file on ``add_argument_group(title=...)``: every
    flag literal between one title and the next belongs to that group. A flag added before the
    first group belongs to ``(ungrouped)``, and one added after the last belongs to the last group
    -- both are reported rather than dropped, because a flag outside every group is a flag whose
    ``--help`` section is wherever the last one was.
    """
    source = (root / "vllm" / "engine" / "arg_utils.py").read_text(encoding="utf-8")
    parts = re.split(r'add_argument_group\(\s*\n?\s*title="([^"]+)"', source)
    groups: dict[str, list[str]] = {}
    current = "(ungrouped)"
    groups[current] = list(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9_.\-]+)"', parts[0]))
    for index in range(1, len(parts), 2):
        current = parts[index]
        groups[current] = list(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9_.\-]+)"', parts[index + 1]))
    return {name: flags for name, flags in groups.items() if flags}, commit_of(root)


def sglang_flags(root: pathlib.Path) -> tuple[dict[str, list[str]], str]:
    """``{namespace: [flag, ...]}`` for SGLang, and the commit it was read at.

    A field is a flag when its annotation is ``A[...]``; ``Arg(cli_name=...)`` and ``Arg(aliases=...)``
    are read off the annotation source, which is why the annotation is unparsed back to text rather
    than walked: it is metadata attached to a type, and the metadata is what names the flag.
    """
    fields = root / "python" / "sglang" / "srt" / "arg_groups" / "fields"
    groups: dict[str, list[str]] = {}
    for path in sorted(fields.glob("*.py")):
        if path.name == "__init__.py":
            continue
        found: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            annotation = ast.unparse(node.annotation)
            if not annotation.startswith("A["):
                continue
            named = re.search(r'cli_name=["\']([^"\']+)', annotation)
            found.append(named.group(1) if named else cli_name(node.target.id))
            for alias in re.findall(r'aliases=\[([^\]]*)\]', annotation):
                found.extend(re.findall(r'["\']([^"\']+)["\']', alias))
        if found:
            groups[path.stem] = sorted(set(found))
    return groups, commit_of(root)


# ---------------------------------------------------------------------------------------------
# PocketLLM
# ---------------------------------------------------------------------------------------------


def our_flags() -> tuple[dict[str, list[str]], str]:
    """``{section: [flag, ...]}`` for this repository's ``serve`` command, as it now is.

    Both halves are read off the parser the CLI builds, so these are the flags an operator can type
    rather than a list of ones they ought to be able to: the host's own options in one section, and
    each declared option under the ``--help`` section its declaration names. Nothing is inferred from
    the ``OPTIONS`` lists -- the generation is
    :func:`pocketllm.backends.cli_surface.add_declared_options`' job, and this reads its result.
    """
    groups: dict[str, list[str]] = {"serve (host flags)": _our_host_flags()}
    for section, flags in _our_generated_sections().items():
        groups[f"serve ({section})"] = sorted(flags)
    return groups, commit_of(REPO)


def our_repeats() -> list[tuple[str, list[str], str]]:
    """``(flag, runtimes that declare it, what stands behind it)`` for every repeated flag.

    A name two runtimes both declare is the shape a duplicated flag comes from, but after the merge
    it is also what a shared concept *is* -- SGLang's ``page_size`` is declared once and read by
    several backends. So the interesting half is not that a name repeats; it is whether one
    declaration stands behind it, whether the answers it leaves to a runtime are the only thing that
    differs, and whether anything else does. This is the machine-checkable form of the design
    document's collision table, and the header it prints under is now empty of the other kind of
    repeat -- one name, two flags -- which is what the merge was for.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from pocketllm.backends import shared_options
    from pocketllm.backends.options import BackendOption

    declared = {
        name: [(runtime, option) for runtime, option in readers.items()]
        for name, readers in _declared().items()
        if len(readers) > 1
    }
    shared = {option.name: option for option in shared_options.SHARED}
    found: list[tuple[str, list[str], str]] = []
    for key, readers in sorted(declared.items()):
        reference = shared.get(key) or readers[0][1]
        drifted = sorted(
            field.name
            for field in fields(BackendOption)
            if field.name not in _STATED_PER_RUNTIME
            and any(getattr(o, field.name) != getattr(reference, field.name) for _, o in readers)
        )
        answered = sorted(
            field.name
            for field in fields(BackendOption)
            if field.name in _STATED_PER_RUNTIME
            and any(getattr(o, field.name) != getattr(reference, field.name) for _, o in readers)
        )
        if drifted:
            # Two flags wearing one name -- the collision this whole exercise is about, and the one
            # case a reader has to look at rather than count.
            where = f"{len(readers)} declarations disagreeing on {', '.join(drifted)}"
        elif answered:
            where = f"one declaration (`{key}`), with {' and '.join(answered)} answered per runtime"
        else:
            where = f"one declaration (`{key}`), read as declared"
        found.append((cli_name(key), [name for name, _ in readers], where))
    return found


def our_host_only() -> list[tuple[str, str]]:
    """``(flag, why)`` for every declaration that did not become a flag.

    One of them: the native engine's own ``--prefill-chunk-tokens``, which is the declaration's CLI
    under another spelling. The list had two entries between U2b-2 and U3 -- ``--device`` was the
    other, its name taken by a flag that meant the vendor on one path and the card on another -- and
    the split took the *declaration* away rather than giving it a flag, so this prints one line
    where the design document of that commit printed two.
    """
    import pocketllm.backends.cli_surface as surface

    found: list[tuple[str, str]] = []
    for name, attribute in surface.HOST_FLAGS.items():
        found.append((cli_name(name), f"spelled `--{attribute.replace('_', '-')}` by the host"))
    for name in sorted(surface.NO_FLAG):  # empty since U3; a name here is a gap to explain
        found.append((cli_name(name), "the host spells this one, under a name of its own"))
    return found


def _our_host_flags() -> list[str]:
    """The option strings the ``serve`` parser declares outside the generated sections.

    One spelling per flag -- the one it is named by -- rather than every string argparse registered.
    A ``BooleanOptionalAction`` registers ``--x`` *and* ``--no-x``, and both halves are real on a
    command line; upstream's are equally real and equally invisible to the source reader below,
    which sees the literal ``--x`` and the ``action=`` beside it. Counting ours and not theirs would
    be a comparison of two different things, so this counts declarations.
    """
    generated = {flag for flags in _our_generated_sections().values() for flag in flags}
    return sorted({
        action.option_strings[0]
        for action in _our_serve()._actions
        if action.option_strings
        and action.option_strings[0].startswith("--")
        and action.option_strings[0] not in generated
    })


def _our_generated_sections() -> dict[str, list[str]]:
    """``{--help section: [flag, ...]}`` for the flags the declarations generate."""
    found: dict[str, list[str]] = collections.defaultdict(list)
    for section in _our_serve()._action_groups:
        for action in section._group_actions:
            if getattr(action, "option_strings", None) and _is_generated(action):
                found[section.title].append(action.option_strings[0])
    return found


def _is_generated(action: Any) -> bool:
    import pocketllm.backends.cli_surface as surface

    return action.dest == surface.DECLARED_DEST


def _our_serve() -> Any:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from pocketllm.cli import build_parser

    return build_parser()._subparsers._group_actions[0].choices["serve"]


def _declared() -> dict[str, dict[str, Any]]:
    """``{name: {runtime: declaration}}`` over every runtime that declares options."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import pocketllm.backends.cli_surface as surface

    return surface.declarations()


#: The fields of a shared declaration a runtime answers for itself: its own default, its own
#: resolution, and its own sentence appended to the shared one. Everything else -- the name, the
#: alias, the kind, the group, the accepted values, the bounds -- is the shape the flag has wherever
#: it is read. ``tests/test_declared_options.py`` holds the tree to the same set.
_STATED_PER_RUNTIME = frozenset({"default", "resolution", "help"})


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------


def cli_name(field: str) -> str:
    return "--" + field.replace("_", "-")


def commit_of(root: pathlib.Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%h %ad %s", "--date=short"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "(not a git checkout)"


def spellings(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """``{flag: [group, ...]}`` -- a flag in two groups is one a client cannot place."""
    found: dict[str, list[str]] = collections.defaultdict(list)
    for group, flags in groups.items():
        for flag in flags:
            found[flag].append(group)
    return dict(found)


def report(name: str, groups: dict[str, list[str]], revision: str) -> None:
    every = spellings(groups)
    print(f"## {name}")
    print()
    print(f"read at `{revision}`")
    print()
    total = sum(len(flags) for flags in groups.values())
    print(f"- {len(groups)} group(s), {total} registrations, {len(every)} distinct flags")
    print()
    print("| Group | Flags |")
    print("| --- | ---: |")
    for group, flags in sorted(groups.items(), key=lambda item: -len(item[1])):
        print(f"| `{group}` | {len(flags)} |")
    print()
    shared = {flag: where for flag, where in every.items() if len(where) > 1}
    print(
        "Declared in more than one group: "
        f"{shared if shared else 'none'} "
        "(upstream, a flag in two groups is one a client cannot place)"
    )
    print()
    repeat = [
        (flag, where) for flag, where in every.items()
        # `---x-explicitly-set` is SGLang's own marks, not a family: the triple dash is the point,
        # see the design document. A family here is `--word-rest`.
        if not flag.startswith("---") and flag.count("-") >= 2 and len(where) == 1
    ]
    families = collections.defaultdict(list)
    for flag, _ in repeat:
        families[flag[2:].split("-")[0]].append(flag)
    if families:
        print("Prefixed families (a group of flags sharing a first word after `--`):")
        print()
        for prefix, flags in sorted(families.items(), key=lambda item: -len(item[1])):
            if len(flags) >= 2:
                print(f"- `--{prefix}-*`: {len(flags)} -- {', '.join(sorted(flags)[:4])}...")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm", type=pathlib.Path, help="a vLLM checkout (sparse: vllm/engine)")
    parser.add_argument(
        "--sglang", type=pathlib.Path, help="an SGLang checkout (sparse: python/sglang/srt/arg_groups)"
    )
    parser.add_argument("--ours", action="store_true", help="this repository's serve surface")
    parser.add_argument("--overlap", action="store_true", help="flags two of them both spell")
    args = parser.parse_args(argv)

    if not (args.vllm or args.sglang or args.ours):
        parser.error("name at least one of --vllm, --sglang, --ours")

    stacks: dict[str, dict[str, list[str]]] = {}
    if args.vllm:
        groups, revision = vllm_flags(args.vllm)
        stacks["vLLM"] = groups
        report("vLLM", groups, revision)
    if args.sglang:
        groups, revision = sglang_flags(args.sglang)
        stacks["SGLang"] = groups
        report("SGLang", groups, revision)
    if args.ours:
        groups, revision = our_flags()
        stacks["PocketLLM"] = groups
        report("PocketLLM", groups, revision)
        print("A name more than one runtime declares, and what stands behind it:")
        print()
        for flag, readers, standing in our_repeats():
            print(f"- `{flag}` -- {standing}; declared by {', '.join(readers)}")
        print()
        print("A declaration with no flag of its own:")
        print()
        for flag, why in our_host_only():
            print(f"- `{flag}` -- {why}")
        print()

    if args.overlap and len(stacks) > 1:
        names = list(stacks)
        print("## Flags more than one stack spells the same way")
        print()
        print("| Flag | " + " | ".join(names) + " |")
        print("| --- |" + " --- |" * len(names))
        every = {name: spellings(stacks[name]) for name in names}
        for flag in sorted(set().union(*(set(v) for v in every.values()))):
            where = [name for name in names if flag in every[name]]
            if len(where) > 1:
                print(f"| `{flag}` | " + " | ".join("yes" if name in where else "" for name in names) + " |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

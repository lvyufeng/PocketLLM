#!/usr/bin/env python
"""Same-named methods across the backend adapters, ranked by how much they agree.

`#442` is the slice that collapses the five request-lifecycle adapters, and every slice of it turned
on the same question: which two bodies are the same body? The issue's original figures were estimates
by line count and they were wrong twice -- a "four mechanical methods" plan that reached only one
pair, and a `stream` fold that was a behaviour bug rather than a tidy-up. This is the measurement
that answers the question the same way each time, so a slice starts from a number rather than an
impression.

The method is deliberately blunt:

* one `ast` parse per module, every method of every class;
* the docstring dropped and comment-only lines removed, because the prose around a body is where two
  copies of it differ most and it is not what the fold is about;
* `difflib` over the remaining statement lines, pairwise, for each name with more than one
  definition;
* the **best** pair per name is what is printed.

Three caveats that matter when reading the output, all of them in
`docs/architecture/per_method_duplication_2026_09.md` with the table it produced:

* a ratio is where to look, not a verdict -- two bodies one parameter apart score 0.97 and are not
  duplicates, and two that share a shape but differ in the one line that is the method's point score
  0.82;
* the pair shown is the *closest* pair, so "three definitions at 0.97" says those two agree and says
  nothing about the third;
* a body can be a duplicate of a *name* that is not in this table at all, because a method that was
  folded into a base class has one definition and never appears.

Run from the repository root: ``python scripts/method_duplication.py [name ...]``.
"""

from __future__ import annotations

import ast
import difflib
import itertools
import sys
from pathlib import Path

MODULES = (
    "base",
    "cpp_backend",
    "v41_backend",
    "mimo_backend",
    "xing4_backend",
    "torch_backend",
)
BACKENDS = Path("pocketllm/backends")


def stripped(item: ast.AST, src: str) -> list[str]:
    """The method's own lines, without its docstring or its comments."""
    lines = src.splitlines()[item.lineno - 1 : item.end_lineno]
    first = item.body[0] if item.body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        lines = lines[: first.lineno - item.lineno] + lines[first.end_lineno - item.lineno :]
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def collect() -> dict[str, list[tuple[str, list[str]]]]:
    found: dict[str, list[tuple[str, list[str]]]] = {}
    for module in MODULES:
        src = (BACKENDS / f"{module}.py").read_text()
        for cls in ast.walk(ast.parse(src)):
            if not isinstance(cls, ast.ClassDef):
                continue
            for item in cls.body:
                if isinstance(item, ast.FunctionDef):
                    found.setdefault(item.name, []).append(
                        (f"{module}.{cls.name}", stripped(item, src))
                    )
    return found


def main(argv: list[str]) -> int:
    found = collect()
    for name in argv or sorted(found):
        copies = found.get(name) or []
        if len(copies) < 2:
            continue
        best = None
        for (left_name, left), (right_name, right) in itertools.combinations(copies, 2):
            if not left or not right:
                continue
            ratio = difflib.SequenceMatcher(None, left, right).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, left_name, right_name, len(left), len(right))
        ratio, left_name, right_name, left_len, right_len = best
        print(
            f"{ratio:5.2f} {left_len:3d}/{right_len:<3d}  {name:<22} "
            f"{left_name} ~ {right_name}  [{len(copies)}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
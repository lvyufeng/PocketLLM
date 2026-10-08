# Guides

Task-oriented documentation: how to develop in this tree, how to check a change, and how to release.

| Guide | What it covers |
|---|---|
| [Cutting a release](pypi_release.md) | The single source of truth for publishing `pocketllm` to PyPI, including the Test PyPI dry run |
| [The board fleet](fleet_workflow.md) | The three resident board sessions, the one rule that keeps them from forking the tree, and how a change travels home as a git bundle |
| [Getting started](../getting-started.md) | Install, the four read-only commands, and running the test suite |
| [The kernel ABI](../architecture/kernel_abi_v1.md) | What a backend must implement before it can be selected |
| [Adding a backend](../architecture/devices.md#adding-a-backend) | The entry-point group, and the two obligations the harness holds a backend to |

## Developing in this tree

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # from the repository root
python scripts/check_test_baseline.py
```

Run pytest **from the repository root**: there is no `conftest.py` and no pytest configuration, and
the modules import from the repository root. `tests/README.md` is the suite's own documentation —
what pytest collects and what it does not, the skip policy, the golden fixtures, and what CI runs.

`tests/baseline_failures.txt` is a **set** of known failures, not a count. Diff a run against it with
`scripts/check_test_baseline.py` (`--update` to re-record). It is a set on purpose: three tests fixed
and one broken is a net improvement in a count and a regression in the tree. A test that fails on this
host belongs in the file; one skipped for want of a card or a checkpoint does not — a skip is not a
pass.

**CI runs no tests.** The workflows build the documentation and publish to PyPI, and nothing else, so
the baseline check is a manual step and a gate only where somebody runs it.

## Changing the documentation

Two rules that fail the build rather than embarrassing you later:

- **Regenerate `docs/llms.txt` in the same commit as any nav change:**
  `python scripts/gen_llms_txt.py`. `mkdocs build --strict` fails when it is stale, and that build is
  the only check CI runs on a documentation change.
- **A new page has to be in the nav.** `validation.nav.omitted_files` warns, and `--strict` promotes
  it — so a page that is not listed fails the build rather than shipping unreachable.

New documents go in one of the topic directories under `docs/`, never at the top level of `docs/`,
which holds only the three site entry points. Renaming or moving a file means fixing every inbound
reference in the same commit — including the ones in source comments and test headers, not just other
Markdown.

## Workflow

`main` is protected: every change goes on a `feature/`, `fix/`, `refactor/`, `docs/` or `perf/`
branch and through a pull request, one concern per PR. See the repository's contributor notes for the
commit and PR conventions.
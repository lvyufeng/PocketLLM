"""Where code may not be imported from, one test per surviving layer.

Each rule here is a boundary the tree is expected to hold, written the way the
crate asks for it: the loader, the components, and the models are separate
layers, and an import that crosses the wrong way is a design error rather than
a style one.

The rules once described a `src/` tree with a runtime layer and a MoE
component layer beside it. Those layers went out with the multi-card cut, and
the survivors now sit under `pocketllm/`, so each rule was retargeted at the
package that is actually there instead of being left pointed at a directory
that no longer exists.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "pocketllm"

# Namespaces that were removed with the multi-card cut. They must not be
# imported anywhere; a hit means a module survived that should not have.
REMOVED_IMPORT_PREFIXES = (
    "pocketllm.moe",
    "pocketllm.moe_model",
    "pocketllm.models.moe",
    "pocketllm.runtime",
    "pocketllm.csrc",
    "pocketllm.gguf",
)

# The checkpoint format is a loader concern. Nothing in the loader is allowed
# to reach up into the models or the serving adapters.
LOADER_FORBIDDEN_IMPORT_PREFIXES = (
    "pocketllm.models",
    "pocketllm.backends",
    "pocketllm.server",
    "pocketllm.api",
)

# The GGUF component layer wraps a loader for a model. It may read the loader,
# but it must not pull in a whole model or the runtime that drives one.
COMPONENTS_FORBIDDEN_IMPORT_PREFIXES = (
    "pocketllm.models",
    "pocketllm.backends",
    "pocketllm.server",
)


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _starts_with_any(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path.relative_to(REPO_ROOT)} imports {module}" for path, module in violations)


def test_removed_namespaces_are_not_imported_from_source() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT):
        for module in _imported_modules(path):
            if _starts_with_any(module, REMOVED_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_loader_does_not_depend_on_models_components_or_serving() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT / "loader"):
        for module in _imported_modules(path):
            if _starts_with_any(module, LOADER_FORBIDDEN_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_components_do_not_depend_on_models_or_serving() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT / "components"):
        for module in _imported_modules(path):
            if _starts_with_any(module, COMPONENTS_FORBIDDEN_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)
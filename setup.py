"""Package pocketllm.

Pure Python: the native kernels moved to relic-core, and the C++ engine they
used to sit beside was retired to the relic-engine archive (see
docs/architecture/). What remains is the single-card runtime -- the PyTorch
model implementations, the serving adapters, and the `src` tree they import
from -- so there are no ext_modules here and no CMake configure step.
"""

from setuptools import find_namespace_packages, setup

setup(
    # PocketLLM's own package plus the shared `src` tree it carries. Resolved
    # from the tree rather than hand-listed, so it cannot drift.
    packages=find_namespace_packages(
        include=["pocketllm", "pocketllm.*", "src", "src.*"],
        exclude=["src.csrc", "src.csrc.*", "src.gguf", "src.moe"],
    ),
)
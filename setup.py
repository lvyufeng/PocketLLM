"""Package pocketllm.

Pure Python: the native kernels moved to relic-core, and the C++ engine they
used to sit beside was retired to the relic-engine archive (see
docs/architecture/). What remains is the single-card runtime -- the PyTorch
model implementations, the checkpoint loaders, and the serving adapters -- all
of which now live under the one `pocketllm` package, so there are no
ext_modules here and no CMake configure step.
"""

from setuptools import find_namespace_packages, setup

setup(
    # Exactly one top-level root, `pocketllm`. This used to also claim the
    # generic `src` package, which RelicLLM's wheel claimed too -- two
    # distributions owning one `src/__init__.py`, with install order deciding
    # which tree won. Owning a name of our own is what makes that impossible.
    packages=find_namespace_packages(include=["pocketllm", "pocketllm.*"]),
)
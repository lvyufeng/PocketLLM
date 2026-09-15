# Guides

Task-oriented documentation: how to measure PocketLLM, how to call it, and how to
release it. Start with [Benchmarking and reporting rules](benchmarking.md) before
quoting any performance number from this site — it defines the measurement
conventions the rest of the documentation assumes.

| Guide | What it covers |
| --- | --- |
| [Benchmarking and reporting rules](benchmarking.md) | The prefill/decode split, what every result record must contain, and the hardware and invocation details a comparable number needs. |
| [PocketLLM API and backend guide](pocketllm_api.md) | The single user-facing API over the two execution planes (Torch and C++), backend selection, tensor parallelism, and the batch/scheduler surface. |
| [PyPI release guide](pypi_release.md) | The single source of truth for releasing `pocketllm` to PyPI, including the Test PyPI dry run. |
| [Ascend SoC generations](ascend_soc_generations.md) | Why `910B` and `910B1`–`910B4` are different chips, how to read `Short_SoC_version`, and why they cannot share kernels. |

For install and build instructions, see [Getting started](../getting-started.md).
For per-model runtimes see [Model guides](../models/README.md).

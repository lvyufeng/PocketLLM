"""PocketLLM: run a large model on one accelerator.

One process owns one card.  If the model does not fit, it is quantized further
-- Q4 to Q2 to IQ2 to IQ1 to ternary -- and never split across devices.

This façade is the only module that may import a backend, and even here it does
so lazily: ``import pocketllm`` declares the kernel vocabulary and nothing else.
Neither torch nor a device runtime is pulled in by importing the package, which
is what lets the same wheel install on a CUDA box and a phone.
"""

from __future__ import annotations

from . import kernels

__version__ = "0.2.0.dev0"

__all__ = ["__version__", "LLM", "AsyncLLM", "kernels"]


def __getattr__(name: str):
    # ``LLM`` / ``AsyncLLM`` live in the engine layer, which is where a backend
    # first becomes real; importing them at module scope would import a device
    # runtime on every ``import pocketllm``.  PEP 562 keeps the name available
    # without paying that cost.
    if name in {"LLM", "AsyncLLM"}:
        from pocketllm import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
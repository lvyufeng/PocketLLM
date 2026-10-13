"""Shared discovery for the serving tests that drive the real C engine.

Two modules in this directory -- ``test_native_backend`` (the service path, one engine held open
for the whole module) and ``test_native_batch`` (the batched/ragged path, each test opening the
engines it needs) -- ask the same two questions: *where is a Qwen3-0.6B checkpoint?* and *which
backend does the engine run on?*  They ask them here so the answers cannot drift apart, and so the
``aclInit`` constraint that keeps them in separate modules has one place to be stated:

**A board holds one engine at a time.**  The Ascend runtime refuses a second ``aclInit`` while an
engine is still open -- ``the engine refused ...: aclInit failed (aclError 100002)`` -- so a module
that pools one engine for its service tests cannot also run a test that opens its own.  The pooled
engine lives in ``test_native_backend``; every test that opens its own against a non-default
``max_batch_size`` or ``batch_kv_window`` lives in ``test_native_batch``, and pytest finalizes the
first module's engine before the second module starts.  Opening two and closing them one after the
other is fine; overlapping two is what fails, which is what this split keeps apart.
"""

from __future__ import annotations

import os
import pathlib

import pytest

#: An explicit checkpoint wins.  The fallback searches the directories this repository's hosts keep
#: Qwen3 GGUFs in, so the batched path, the KV-wipe regression test and the G2 gate run on a host
#: that has *a* Qwen3-0.6B -- not only the f16 one the module used to name by absolute path.  G2
#: compares batched against solo on the same file, so it is checkpoint-independent; only the golden
#: fixture is not, and it skips unless its exact f16 (its recorded sha) is present.
_CHECKPOINT_ENV = ("POCKETLLM_SERVING_GGUF", "POCKETLLM_MODEL", "CKPT_PATH")
_CHECKPOINT_DIRS = (
    pathlib.Path("/mnt/data1/models"),
    pathlib.Path.home() / "models",
    pathlib.Path.home() / "models" / "gguf",
)


def _find_checkpoint() -> str:
    for name in _CHECKPOINT_ENV:
        override = os.environ.get(name)
        if override and pathlib.Path(override).is_file():
            return override
    for directory in _CHECKPOINT_DIRS:
        if not directory.is_dir():
            continue
        # The name filter is case-insensitive on purpose: the same checkpoint is spelled
        # `Qwen3-0.6B-Q4_K_M.gguf` on one host and `qwen3-0.6b-q4_k_m.gguf` on another, and a
        # case-sensitive glob made the second host fail the module with "no Qwen3-0.6B GGUF on
        # this host" while the file sat in the directory it names.
        for candidate in sorted(directory.glob("*.gguf")):
            if candidate.name.lower().startswith("qwen3-0.6b"):
                return str(candidate)
    return ""


#: The checkpoint the engine is pointed at, or ``""`` when this host has none.
SERVING_CHECKPOINT = _find_checkpoint()

#: The C backend the gates run the engine on.  Env-first for the same reason the checkpoint is:
#: the host knows what it built and what it is for.  ``cpu`` is the default every build of the
#: library has, so nothing moves on a host that does not set this; a 310B board sets
#: ``POCKETLLM_SERVING_DEVICE=ascend``, because the engine that actually serves there is the
#: ascend one and gates run against the board's three ARM cores would be gating a build nobody
#: ships.  The value is the *backend* name the C core is asked for -- ``cpu``, ``cuda``,
#: ``ascend`` -- the same name ``pocketllm.cli._run_device`` hands ``serve``, not the
#: ``--device`` kind that resolves to it.
SERVING_DEVICE = os.environ.get("POCKETLLM_SERVING_DEVICE", "cpu")


def native_available() -> bool:
    """Whether ``libpocketllm.so`` is built on this host.

    An unbuilt engine is a *skip*: that is a build state of the host, not the condition under
    test, and the conformance suite already fails if a declared op has no reference.
    """
    from pocketllm import native

    return native.is_available()


def require_checkpoint() -> None:
    """Fail -- never skip -- when no checkpoint is on the host.

    These modules are the gate for the batched path, the KV-wipe regression and G2, and the bug
    those gates exist to catch is *silent wrong text* -- so a module that quietly skips when no
    checkpoint is on the host is indistinguishable from the bug passing.  If there is no
    Qwen3-0.6B GGUF the gates cannot run, and that has to be loud.  A module calls this from an
    autouse fixture rather than importing it directory-wide, because the fake-driven serving tests
    next door must still run on a host that has no checkpoint.
    """
    if not SERVING_CHECKPOINT:
        pytest.fail(
            "no Qwen3-0.6B GGUF on this host, so the serving gates cannot run -- and a gate that "
            "cannot run must fail, not skip. Set POCKETLLM_SERVING_GGUF (or POCKETLLM_MODEL / "
            "CKPT_PATH), or place a Qwen3-0.6B*.gguf under /mnt/data1/models or ~/models"
        )

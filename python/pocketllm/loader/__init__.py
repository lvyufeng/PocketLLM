"""Reading a checkpoint off disk, without a device runtime.

The loader turns a ``.gguf`` file into the descriptors and buffers the ABI names.
It depends on ``pocketllm.quant`` and numpy and on nothing else -- in particular
it does not import torch, which is what lets a phone install read a checkpoint
without the training stack.  Where a tensor ends up is the session's business;
:func:`pocketllm.loader.gguf.host_array.upload` is the one boundary function that
moves it, and it takes a session rather than a device.

**This module must stay a real package, not a namespace package.**  An implicit
namespace directory here is silently omitted from the wheel by
``namespaces = false``, which drops the whole loader -- and the mistake only
surfaces when an installed copy tries to open a checkpoint, not at build time.
"""
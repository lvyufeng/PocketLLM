"""A host-resident store of prompt prefixes for Xing4.0-29B-A4B.

A chat loop resends its whole history every turn and the serving loop resets its
cache at the top of each request, so the history is prefilled again each time:
turn twenty pays for turns one to nineteen.  What is stored here is what a
prefill of those tokens left behind -- one `[1, length, 576]` latent a layer,
which is the whole of the cache's state because the absorbed form keeps no
separate value tensor -- keyed by the tokens themselves.  A later request
restores the longest prefix it shares with one already served and forwards only
the remainder.

**It is the whole cache and not a summary of it.**  Nothing is recomputed and
nothing is approximated: a resumed prefill's first new token sees exactly the
rows a cold prefill would have left, which is why the answer is the same one and
why the reused tokens are honest in `usage`.

**The bytes are the cost, and they are 46 KB a token.**  Forty layers of a
512-wide latent and its 64-wide rope key in fp16, so a 4096-token entry is
189 MiB and the default budget holds a handful of conversations.  The store lives
on the host rather than on the card because the card has 3.6 GiB free with this
checkpoint resident and the point of a prefix cache is to be larger than what
fits in one cache: restoring is one H2D of 189 MiB, ~19 ms, against the ~74 s a
4096-token prefill takes at this checkpoint's measured rate.

Stage 5 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch

__all__ = ["LatentPrefixCache", "snapshot", "restore"]

DEFAULT_PREFIX_CACHE_BYTES = 2 << 30
"""Host memory the store may hold, when prefix caching is on.

2 GiB is about 46000 tokens of cache state: four conversations at 8192 tokens,
or one at 32K with room for the next few turns.  It is pageable memory and
deliberately not ``/dev/shm``.
"""


def snapshot(cache: Any, length: int) -> list[torch.Tensor]:
    """The cache's first `length` rows, a layer at a time, on the host.

    A list rather than one stacked tensor because the layers are separate
    allocations and stacking them would copy 189 MiB to make one.
    """
    rows = []
    for layer in cache:
        if isinstance(layer, list):  # a batched run hands a list of per-row caches
            rows.append([entry.latent[:, :length].to("cpu", copy=True) for entry in layer])
        else:
            rows.append(layer.latent[:, :length].to("cpu", copy=True))
    return rows


def restore(cache: Any, rows: Sequence[Any], length: int) -> None:
    """Write `rows` back into `cache` and set its length.

    The length is set after the copy and not before: a cache whose `length` was
    ahead of its rows would have a later append overwrite holes with whatever was
    there, and the failure is a wrong answer rather than a crash.
    """
    for layer, entry in zip(cache, rows):
        if isinstance(entry, list):
            for target, source in zip(layer, entry):
                target.latent[:, :length] = source.to(target.latent.device, non_blocking=False)
                target.length = int(length)
        else:
            layer.latent[:, :length] = entry.to(layer.latent.device, non_blocking=False)
            layer.length = int(length)


class LatentPrefixCache:
    """Longest-prefix lookup over stored prompts, with an LRU byte budget.

    Keyed by the token tuple rather than by a hash of it.  A collision between two
    prompts would restore the wrong state and the answer would be plausible and
    wrong, which is the one failure mode this store cannot have; a tuple of a few
    thousand ints costs 8 bytes a token in the index, against the 46 KB a token
    the state itself costs.
    """

    def __init__(self, *, budget_bytes: int, capacity: int, n_layers: int) -> None:
        self.budget_bytes = int(budget_bytes)
        self.capacity = int(capacity)
        self.n_layers = int(n_layers)
        self._entries: OrderedDict[tuple[int, ...], tuple[int, list[torch.Tensor]]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0

    # -- lookup -------------------------------------------------------------- #

    def longest(self, ids: Sequence[int]) -> int:
        """The longest stored prefix of `ids`, or zero.

        A prefix has to start at the prompt's first token: a store that could
        match in the middle would be a store of *segments*, and the state a middle
        segment needs is not the state a prefill leaves -- the rows a token
        attends to are the ones before it in the same run.
        """
        wanted = [int(token) for token in ids]
        for length in sorted({len(key) for key in self._entries}, reverse=True):
            if length == 0 or length > len(wanted) or length > self.capacity:
                continue
            if tuple(wanted[:length]) in self._entries:
                return length
        return 0

    def materialise(self, ids: Sequence[int], length: int) -> list[torch.Tensor]:
        """The rows for `ids[:length]`, and the entry is now the most recent."""
        key = tuple(int(token) for token in ids[:length])
        rows = self._entries[key][1]
        self._entries.move_to_end(key)
        return rows

    # -- fill ---------------------------------------------------------------- #

    def store(self, ids: Sequence[int], rows: Sequence[Any]) -> None:
        """Keep the state a prefill of `ids` left, evicting until it fits."""
        length = len(ids)
        if length <= 0 or length > self.capacity:
            return
        key = tuple(int(token) for token in ids)
        size = _rows_bytes(rows)
        if size > self.budget_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous[0]
        self._entries[key] = (size, list(rows))
        self._entries.move_to_end(key)
        self._bytes += size
        while self._bytes > self.budget_bytes and len(self._entries) > 1:
            _, (evicted, _) = self._entries.popitem(last=False)
            self._bytes -= evicted

    def note(self, length: int) -> None:
        """One request's outcome, for the counters `/metrics` reads."""
        if length > 0:
            self.hits += 1
            self.reused_tokens += int(length)
        else:
            self.misses += 1

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "reused_tokens": self.reused_tokens,
            "entries": len(self._entries),
            "bytes": self._bytes,
            "budget_bytes": self.budget_bytes,
        }


def _rows_bytes(rows: Sequence[Any]) -> int:
    total = 0
    for entry in rows:
        if isinstance(entry, list):
            total += sum(int(t.numel()) * t.element_size() for t in entry)
        else:
            total += int(entry.numel()) * entry.element_size()
    return total

#!/usr/bin/env python
"""What a chat turn costs when the previous turn's prefix is already on the host.

`generate` resets the cache at the top of every request, so a chat loop that resends its history
forward-passes the history again on every turn. The store makes turn N pay for the user turn only:
the state after `--prefix` tokens is snapshotted to the host, keyed by the tokens that produced it,
and the next request restores it and forwards the remainder.

Four arms, interleaved, on the same tokens and the same cache:

* **cold** -- the whole prompt from zero, which is what every request did before the store existed;
* **warm** -- the stored prefix restored and the remainder forwarded at its own position, which is
  what a second turn is;
* **rechunk** -- the whole prompt again, but cut at `--prefix` with no store in the way. This is the
  null, and it is the arm the other two are read against;
* **store** -- the snapshot and the admission that a cold prefill pays for, which is the price of
  the feature and not a preference.

The three prefill arms' last rows are held against each other, because the saving is only worth
anything if the answer did not move. **warm against rechunk is the store's own contribution**: the
two arms forward the same tokens from the same state, one of them having taken that state out to the
host and back, so a difference there is a snapshot that missed something. **cold against rechunk is
not the store's to answer for**: a resume is a chunk boundary at `--prefix`, which is generally not a
multiple of `--chunk`, and this model's chunked prefill is not bit-exact against a one-shot one --
see `probe_mimo_v2_decode_prefix.py` and
`test_models_mimo_v2_device_attention.py::test_a_chunked_prefill_is_a_one_shot_prefill`, where the
tolerance is 1e-4 on one layer's output rather than on 48 layers' logits. Without the null arm that
number would be read as the store's. The probe reports all three, the row's own peak, and the token
each row would draw, so they are the reader's to judge.

The prompt is real text through the checkpoint's own tokenizer rather than a list of ids, and the
prefix is the first `--prefix` of its tokens, which is the shape a chat turn actually has.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_prefix_cache.py --tokens 4096 --prefix 3072
    torchrun --nproc_per_node=4 tests/probe_mimo_v2_prefix_cache.py --tokens 2048 --prefix 1024

The checkpoint is the default asset path; without it the script exits 0 and says so.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.bank import open_expert_bank  # noqa: E402
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.ep import EpGroup  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from src.models.mimo_v2.prefix_cache import (  # noqa: E402
    PrefixCache,
    geometry_tag,
    restore_rows,
    snapshot_rows,
)

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"

#: Real prose, from this repository: long enough to cut any `--tokens` a run of this box can afford
#: out of, and the kind of text a code assistant is actually sent.
PROMPT_SOURCE = "docs/architecture/mimo_v2_6_flash_design.md"


def prompt_ids(checkpoint: str, tokens: int) -> list[int]:
    """`tokens` real ids, tokenized by the checkpoint's own tokenizer.

    Synthesized ids would be a different model's problem: the router's draw, and therefore the bytes
    a prefill stages, depends on the tokens, and a probe that fed random ids would measure a draw no
    request makes. This is the same discipline the other MiMo probes follow.
    """
    from transformers import AutoTokenizer

    with open(PROMPT_SOURCE, "r", encoding="utf-8") as handle:
        text = handle.read()
    encoder = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    ids = [int(token) for token in encoder(text, add_special_tokens=False)["input_ids"]]
    while len(ids) < tokens:
        ids = ids + ids  # a longer prompt than the document, repeated
    return ids[:tokens]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--tokens", type=int, default=4096, help="the prompt's length")
    parser.add_argument("--prefix", type=int, default=3072, help="tokens stored and reused")
    parser.add_argument("--chunk", type=int, default=2048, help="prefill chunk width")
    parser.add_argument("--slots", type=int, default=2, help="expert arena slots")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=16,
        help="experts a chunk's arena bands in, per rank; the served default",
    )
    parser.add_argument("--rounds", type=int, default=2, help="times round the two arms")
    parser.add_argument("--budget", type=int, default=8, help="host bytes the store may hold")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    ep = EpGroup.from_env()
    world, rank = ep.world, ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")
    tag = f"[r{rank}]"

    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint,
        device=device,
        expert_source=bank,
        ep=ep,
        slots=args.slots,
        chunk_rows=args.chunk_rows,
    )
    torch.cuda.synchronize()

    ids = prompt_ids(args.checkpoint, args.tokens)
    if args.prefix >= len(ids):
        print(f"{tag} a prefix of {args.prefix} is the whole prompt; nothing to continue")
        return 0
    cache = model.cache(len(ids) + 8)
    print(
        f"{tag} world {world}, prompt {len(ids)} tokens, prefix {args.prefix}, chunk {args.chunk}, "
        f"cache {cache.memory_bytes / 2**30:.2f} GiB",
        flush=True,
    )

    # The store, built and tagged the way the adapter builds it.
    store = PrefixCache(
        budget_bytes=args.budget << 30,
        max_seq_len=len(ids) + 8,
        tag=geometry_tag(cache, world, len(ids) + 8),
        head_tokens=0,
    )

    # The store the warm arm reads, and a second one the cold arm's own end anchor goes into. A
    # served cold request does store the prompt's end, but turn 2 of *this* conversation is the arm
    # being measured, and a full-prompt entry in the same store would answer the warm lookup with an
    # exact repeat instead of the prefix under test. The budget is the same, so eviction behaves the
    # same; only which lookup wins changes.
    scratch = PrefixCache(
        budget_bytes=args.budget << 30,
        max_seq_len=len(ids) + 8,
        tag=geometry_tag(cache, world, len(ids) + 8),
        head_tokens=0,
    )

    # The setup every arm's second half needs: the prompt's first `--prefix` tokens, and the entry
    # the store keeps of them. Untimed -- a turn that has already been served is the assumption the
    # warm arm is measuring, and the cold arm pays for its own copy below. The row that entry carries
    # is the one the setup prefill ended on, which is what makes an exact repeat a sample.
    cache.reset()
    prefix_logits = model.prefill(ids[: args.prefix], cache=cache, chunk=args.chunk)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()
    entry = store.store(ids, args.prefix, snapshot_rows(cache), prefix_logits)
    if entry is None:
        print(f"{tag} the store refused a {args.prefix}-token entry at a {args.budget} GiB budget")
        return 0
    print(
        f"{tag} stored {entry.nbytes / 2**20:.1f} MiB for {args.prefix} tokens "
        f"({entry.nbytes / args.prefix:.0f} bytes a token)",
        flush=True,
    )

    def cold() -> tuple[float, float, torch.Tensor]:
        """The whole prompt from zero, and what storing its state afterwards would cost.

        The reset is the first line because it is the first line of a served request and the whole
        reason the store exists: `generate` wipes the cache at the top of the loop, so a cold turn
        really does forward every token of its prompt again. Its end anchor goes to `scratch`; the
        work is the same and it is the same number, but it must not answer the warm arm's lookup.
        """
        cache.reset()
        started = time.perf_counter()
        logits = model.prefill(ids, cache=cache, chunk=args.chunk)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        kept = time.perf_counter()
        saved = snapshot_rows(cache)
        torch.cuda.synchronize()
        scratch.store(ids, len(ids), saved, logits)
        return seconds, time.perf_counter() - kept, logits

    def warm() -> tuple[float, float, torch.Tensor]:
        """The stored prefix back, and the remainder forwarded at its own position.

        The lookup is inside the timer because it is inside the request: a served warm turn hashes
        the prompt's blocks, finds the entry and restores it, and a number that left the first of
        those out would be a number for a store nobody has.
        """
        started = time.perf_counter()
        hit = store.lookup(ids)
        assert hit is not None and hit[1] is entry, "the store did not find what it just stored"
        cache.reset()
        restore_rows(cache, entry.saved)
        torch.cuda.synchronize()
        restored = time.perf_counter() - started
        logits = model.prefill(
            ids[args.prefix :], cache=cache, chunk=args.chunk, start_pos=args.prefix
        )
        torch.cuda.synchronize()
        return (
            time.perf_counter() - started - restored,
            restored,
            logits,
        )

    def rechunk() -> tuple[float, float, torch.Tensor]:
        """The whole prompt from zero, cut at `--prefix` -- the warm arm's arithmetic, no store.

        This is the null. A resume forwards `ids[prefix:]` at `start_pos = prefix` and that is a
        chunk boundary whatever the state came from, so the state this arm's tail reads is the state
        the warm arm's tail reads -- one of them having been out to the host and back. A difference
        between the two is therefore the store's, and a difference between this arm and `cold` is a
        chunk boundary's, which is not.
        """
        cache.reset()
        started = time.perf_counter()
        model.prefill(ids[: args.prefix], cache=cache, chunk=args.chunk)
        logits = model.prefill(
            ids[args.prefix :], cache=cache, chunk=args.chunk, start_pos=args.prefix
        )
        torch.cuda.synchronize()
        return time.perf_counter() - started, 0.0, logits

    def repeat() -> float:
        """An exact repeat: a restore and nothing else, which is what turn N costs when it *is* N-1."""
        started = time.perf_counter()
        cache.reset()
        restore_rows(cache, entry.saved)
        torch.cuda.synchronize()
        return time.perf_counter() - started

    # Warm the kernels on every path once, so the first measured arm is not the one paying for them.
    cold()
    warm()
    rechunk()

    arms: dict[str, list[tuple[float, float]]] = {"cold": [], "warm": [], "rechunk": []}
    stores: list[float] = []
    rests: list[float] = []
    moved: list[tuple[float, float, float]] = []
    for _ in range(args.rounds):
        for name, arm in (("cold", cold), ("warm", warm), ("rechunk", rechunk)):
            prefill, extra, logits = arm()
            arms[name].append((prefill, logits))
            if name == "cold":
                stores.append(extra)
            elif name == "warm":
                rests.append(extra)
            if world > 1:
                torch.distributed.barrier()
    # The three arms' last rows, held against each other: the same prompt, some of them cut at
    # `--prefix`. `warm` against `rechunk` is the store's own contribution and `cold` against
    # `rechunk` is the chunk boundary's, which is the whole reason the null arm is here.
    peaks: list[float] = []
    for index in range(args.rounds):
        whole = arms["cold"][index][1].to(torch.float32)
        part = arms["warm"][index][1].to(torch.float32)
        null = arms["rechunk"][index][1].to(torch.float32)
        peak = float(whole.abs().max())
        peaks.append(peak)
        moved.append(
            (
                float((whole - part).abs().max() / peak),
                float((whole - null).abs().max() / peak),
                float((null - part).abs().max() / peak),
            )
        )

    repeat_seconds = repeat()
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    for index in range(args.rounds):
        cold_seconds = arms["cold"][index][0]
        warm_seconds = arms["warm"][index][0]
        null_seconds = arms["rechunk"][index][0]
        print(
            f"{tag} round {index + 1}: cold {cold_seconds:7.3f} s over {len(ids)} tokens, "
            f"warm {warm_seconds:7.3f} s over {len(ids) - args.prefix}, "
            f"rechunk {null_seconds:7.3f} s (the same {len(ids)} cut at {args.prefix})",
            flush=True,
        )
        against_cold, boundary, store_own = moved[index]
        picks = {
            name: int(torch.argmax(arms[name][index][1].to(torch.float32)))
            for name in ("cold", "warm", "rechunk")
        }
        print(
            f"{tag} round {index + 1}: the row, against a peak of {peaks[index]:.3g} — "
            f"cold-warm {against_cold:.2e}, cold-rechunk {boundary:.2e} (the chunk boundary), "
            f"warm-rechunk {store_own:.2e} (the store)",
            flush=True,
        )
        print(
            f"{tag} round {index + 1}: the token the row would draw — cold {picks['cold']}, "
            f"warm {picks['warm']}, rechunk {picks['rechunk']} "
            f"({'the same' if len(set(picks.values())) == 1 else 'DIFFERENT'})",
            flush=True,
        )
    for name, served in (
        ("cold", len(ids)),
        ("warm", len(ids) - args.prefix),
        ("rechunk", len(ids)),
    ):
        seconds = sum(entry_[0] for entry_ in arms[name]) / len(arms[name])
        print(
            f"{tag} {name}: {seconds:7.3f} s — {served / seconds:7.1f} tok/s over the "
            f"{served} tokens it forward-passed (a rate per arm, not a speedup: the arms forward "
            f"different token counts)",
            flush=True,
        )
    print(
        f"{tag} store cost {sum(stores) / len(stores) * 1e3:7.1f} ms on a cold prefill, "
        f"restore {sum(rests) / len(rests) * 1e3:6.1f} ms, an exact repeat {repeat_seconds * 1e3:.1f} ms",
        flush=True,
    )
    print(f"{tag} store: {store.stats()}", flush=True)
    print(f"{tag} scratch: {scratch.stats()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

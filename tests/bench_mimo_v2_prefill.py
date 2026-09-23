#!/usr/bin/env python
"""What a MiMo-V2.6 prompt costs when it goes through as chunks instead of a token at a time.

`bench_mimo_v2_model.py` measures a *prompt* by feeding it one token at a time, which is the only
shape the single-token expert kernel has, and `bench_mimo_v2_ep.py` measures the same loop with the
experts dealt out over four cards. Both of them report a prefill of a few tokens a second, and
neither of them is a prefill: a prompt fed a row at a time draws its experts a row at a time, so the
cost is the decode step's times the prompt's length.

This script measures the path that exists for the prompt. `prefill` feeds `--chunk` tokens a call
through `moe_multi_token_fp4_forward`, whose arena holds a rank's share of the layer's experts and
whose kernel groups the chunk's drawings by expert, so the bytes a layer moves are the experts a
*chunk* draws -- nearly all of a rank's share -- and the calls a layer makes are the bands that
share is computed in, not the tokens. The two rates are in the same table, on the same prompt and
the same weights, and the ratio between them is what the stage bought.

What the numbers are made of:

* `MiB/token` is the rank's own staging counter divided by the prompt. It falls as the chunk grows
  and stops falling when the chunk draws a rank's entire share, which is where a prefill stops
  being about the copy: the floor is `--band` rows of 12.75 MiB a layer, divided by the chunk.
* `copy ms/token` prices those bytes at the link rate this host actually sustains, so the copy's
  share of the token is a measurement of what the prefill would cost if the copy were all of it.
* `attn ms` and `mlp ms` are the same per-layer event pairs the other two benches use, and the
  split is the opposite of decode's: a chunk's attention is quadratic in its own length while its
  expert copy is spread over the chunk.

And the verification, which is why this is worth running with four ranks: the ranks must agree on
the prefill's last row bit for bit -- they hold the same sum -- and each rank's staged bytes must be
its own share.

`--deal` is the *step's* deal and not the chunk's, and it is the difference between this table and a
served run. A chunk can only go through a deal that partitions *experts* (`id`), so a model built
with `--deal sorted` -- which is what `pocketllm serve` builds -- keeps two modules and routes the
chunks to the `id` one; the prefill rows are identical either way, and the `--decode` rows are not:
under `id` a step runs through the prefill's own module, which the step probe measures at 41 to 53%
slower. `--deal id` builds that single-module model, which is the right thing for a prefill-only run
and the wrong number for a step. The `--decode` steps are warmed (`--warmup`) and the warm-up is
printed on its own, because the window otherwise opens on whatever the prompt's last chunk left
behind; the deal's own price is not read off this column but off the interleaved step probe, and the
two arms here are two processes.

Usage:

    python tests/bench_mimo_v2_prefill.py --tokens 512 --chunk 512
    torchrun --nproc_per_node=4 tests/bench_mimo_v2_prefill.py --tokens 2048 --chunk 2048
    torchrun --nproc_per_node=4 tests/bench_mimo_v2_prefill.py --tokens 8192 --chunk 4096 --band 0
    torchrun --nproc_per_node=4 tests/bench_mimo_v2_prefill.py \
        --tokens 262144 --chunk 2048 --band 16 --floor 0 --decode 2

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
from tests.bench_mimo_v2_model import PhaseTimer, h2d_rate  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"

#: The same eight ids the other two benches feed, repeated out to the prompt's length so all three
#: scripts' token lists compare.
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def prompt_of(tokens: int) -> list[int]:
    return (PROMPT_IDS * (tokens // len(PROMPT_IDS) + 1))[:tokens]


def score_tile_bytes(model, rows: int, block: int = 1024) -> int:
    """What a global layer's score tile would cost at `rows` a chunk, were the chunk its width.

    The tile `blocked_attention` holds is `[kv_heads, groups, step, block]` float32, where `step` is
    what `attention`'s `tile_budget` buys and not what the caller asked for -- so this is the number
    the *tile budget* retires, quoted to say how large it was before. It is not a bound on a chunk:
    the layer concatenates its prefix key and value to the chunk's, and at 256k that pair is larger
    than any tile the loop holds. `block` is `attention`'s own default, which is what the model
    calls it with.

    A windowed layer is not asked: its keys are its window, so the tile is `rows x 128` and never
    the thing that misses. The largest global layer is the one to quote.
    """
    shapes = [model.config.attention(layer) for layer in model.config.global_layer_indices]
    shape = max(shapes, key=lambda item: item.num_kv_heads * item.num_key_value_groups)
    return shape.num_kv_heads * shape.num_key_value_groups * rows * block * 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--tokens", type=int, default=512, help="prompt tokens")
    parser.add_argument(
        "--capacity",
        type=int,
        default=0,
        help="positions the KV cache holds; 0 sizes it to the prompt, which is the default",
    )
    parser.add_argument(
        "--chunk",
        default="512",
        help="tokens a prefill call, one width or several: `512` or `512,1024,4096`",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=0,
        help="experts one kernel call holds; 0 is a rank's whole share in one call",
    )
    parser.add_argument(
        "--floor", type=int, default=64, help="tokens to also feed one at a time; 0 skips it"
    )
    parser.add_argument("--slots", type=int, default=2, help="expert arena slots")
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="scores the block loop's tile may hold; 0 uses the model's own constant",
    )
    parser.add_argument(
        "--decode",
        type=int,
        default=0,
        help="decode steps to run after the prompt is in the cache, timed",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="decode steps run and dropped before `--decode` ones are timed: the first steps after "
             "a prompt are the prompt's own tail, and they are reported on their own",
    )
    parser.add_argument("--no-pin", action="store_true", help="leave the bank pageable")
    parser.add_argument(
        "--deal",
        default="sorted",
        help="the *step's* deal: `sorted` is what `pocketllm serve` builds, and `id` is the "
             "chunk-only configuration, which leaves a step through the prefill's module",
    )
    args = parser.parse_args()
    args.chunk = sorted({int(part) for part in str(args.chunk).split(",") if part.strip()})

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    ep = EpGroup.from_env()
    world, rank = ep.world, ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")
    tag = f"[r{rank}]"

    started = time.perf_counter()
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint,
        device=device,
        expert_source=bank,
        ep=ep,
        slots=args.slots,
        # The deal a *step* takes, which is the one a served run's decode goes through. It is not
        # the deal a chunk can use -- `id` partitions the experts and `sorted` partitions the
        # drawings -- so a world over one with `sorted` here builds the second module too, and the
        # prefill below goes through that one. `--deal id` builds a single module, which is what a
        # prefill-only run wants and is *not* what a step in a served run costs.
        deal=args.deal,
        chunk_rows=args.band,
        pin=not args.no_pin,
        tile_budget=args.tile or None,
    )
    torch.cuda.synchronize()
    # The module a *chunk* runs on, and the one the counters below are read from: a two-module
    # model routes a chunk to `chunk_experts`, so a counter read off `model.experts` reports that
    # the prefill moved nothing.
    experts = model.experts if model.chunk_experts is None else model.chunk_experts
    print(
        f"{tag} world {world}, {len(model.layers)} layers in {time.perf_counter() - started:.1f}s, "
        f"{model.memory_bytes / 2**20:.0f} MiB on the card, "
        f"{experts.arena_bytes / 2**20:.0f} MiB chunk arena, {experts.n_local} experts a rank in "
        f"bands of {experts.chunk_rows} under a `{experts.deal}` chunk deal"
        + (
            f"; a `{model.experts.deal}` step arena of {model.experts.arena_bytes / 2**20:.0f} MiB"
            if model.chunk_experts is not None
            else f", and no separate step module: a step here goes through that"
        ),
        flush=True,
    )
    if model.pin_result is not None:
        print(
            f"{tag} bank pin: rc={model.pin_result.rc} ok={model.pin_result.ok} in "
            f"{model.pin_result.seconds:.1f}s",
            flush=True,
        )

    rate, expert_mib = h2d_rate(bank, model.config.moe_layer_indices[0])
    if world > 1:
        rates = [None] * world
        torch.distributed.all_gather_object(rates, round(rate, 2))
    else:
        rates = [round(rate, 2)]
    print(
        f"{tag} expert pages to the card: {rate:.2f} GiB/s ({expert_mib:.2f} MiB an expert), "
        f"all ranks {rates}",
        flush=True,
    )

    ids = prompt_of(args.tokens)
    # The cache holds the whole prompt's keys and values: nine layers of it at full length and
    # thirty-nine rings of the sliding window. A prefill that does not fit here is a memory
    # question and not a speed one, and this reports where it landed.
    #
    # `--capacity` sizes it apart from the prompt, which is how the *memory state* of a long
    # context is reached without paying for the prefill that gets there: the cache is the
    # allocation that grows with the context and nothing else in the path is, so a 32k prompt
    # against a 256k cache is the same allocator and the same headroom as a 256k one at the
    # moment the first chunk is answered. It is a diagnostic for what the memory state costs
    # and not a substitute for the run.
    cache = model.cache(args.capacity or len(ids) + 8)
    free_before = torch.cuda.mem_get_info()[0]
    print(
        f"{tag} cache: {cache.memory_bytes / 2**30:.2f} GiB over {len(cache)} layers for "
        f"{cache.context_capacity} tokens; {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
        f"allocated, {free_before / 2**30:.2f} GiB free on the card",
        flush=True,
    )

    # A discarded warmup, on a short prompt: it pays the CUDA context, the first staging, the
    # first bandwidth the driver picks, and the collective's first message.
    warm = model.cache(64)
    model.prefill(ids[: min(32, len(ids))], cache=warm, chunk=min(32, args.chunk[0]))
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    rows: list[tuple[int, int, float, int, int, dict[str, float]]] = []

    # The floor: the same prompt fed one row at a time, which is what the expert kernel's
    # single-token path can do. `--floor` caps how many tokens of it are paid for, because the
    # rate is what is being compared and not the prompt.
    floor_logits = None
    # Bound before either pass runs, because a sweep whose every width was refused leaves both
    # unset and the answer check below reads `logits` unconditionally. A run that measured
    # nothing still has to report that it measured nothing.
    logits = None
    if args.floor:
        floor_ids = ids[: min(args.floor, len(ids))]
        cache.reset()
        mark = (experts.staged_experts, experts.staged_bytes)
        if world > 1:
            torch.distributed.barrier()
        started = time.perf_counter()
        logits = None
        for position, token in enumerate(floor_ids):
            logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        rows.append(
            (
                1,
                len(floor_ids),
                seconds,
                experts.staged_experts - mark[0],
                experts.staged_bytes - mark[1],
                {},
            )
        )
        floor_logits = logits.clone()

    # The prefill itself, one pass over the cache a width. The widths share everything else --
    # the same prompt, the same weights, the same lanes -- so the rows of the table are the
    # width and nothing else, which is the whole question: a chunk's bytes are a fixed cost a
    # *call* and not a token, so what a width buys is how many tokens pay them.
    #
    # A row is printed as it is measured and not at the end, because a width can run out of
    # card and a table that never printed is a run with no numbers in it. The block loop's
    # score tile is `groups x rows x block` float32 and a global layer's late key blocks see
    # the whole chunk, so the row count used to be a memory bound and not only a speed one --
    # the tile budget is what retired it, and the OOM handler below quotes the exception
    # rather than blaming the tile for whatever allocation actually missed.
    print()
    print(
        f"{'chunk':>7} {'tokens':>7} {'seconds':>9} {'tok/s':>8} {'calls':>6} {'MiB/token':>10} "
        f"{'copy ms/tok':>12} {'copy share':>11} {'attn ms/tok':>12} {'mlp ms/tok':>11}",
        flush=True,
    )
    timer = PhaseTimer(model)

    def report(row):
        width, tokens, seconds, _, byte_count, phases = row
        copy_ms = byte_count / tokens / 2**30 / rate * 1e3
        token_ms = seconds / tokens * 1e3
        print(
            f"{width:>7} {tokens:>7} {seconds:>9.3f} {tokens / seconds:>8.2f} "
            f"{'--' if width == 1 else -(-tokens // width):>6} "
            f"{byte_count / tokens / 2**20:>10.1f} {copy_ms:>12.1f} "
            f"{copy_ms / token_ms * 100:>10.1f}% "
            f"{phases.get('attn', 0) / tokens:>12.2f} {phases.get('mlp', 0) / tokens:>11.2f}",
            flush=True,
        )

    # The floor first, because it is the column the rest of the table is read against.
    for row in rows:
        report(row)

    for width in args.chunk:
        cache.reset()
        timer.read()
        mark = (experts.staged_experts, experts.staged_bytes)
        if world > 1:
            torch.distributed.barrier()
        started = time.perf_counter()
        try:
            logits = model.prefill(ids, cache=cache, chunk=width)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            # Reported and not raised: the card's ceiling is a result of this bench, and the
            # widths that do fit are worth the table even when a wider one does not. No barrier
            # of its own -- the next width's is the sync, and one rank skipping a barrier the
            # others also skip is exactly what keeps the counts equal.
            #
            # The exception is quoted and the site is printed, because the tile is *one* tensor in
            # a chunk and not the whole of it: at 256k the prefix the layer concatenates is an
            # order of magnitude larger, so a line naming the tile as the cause would be a guess
            # dressed as a finding. The tile budget's own number goes beside it as what a chunk no
            # longer scales, and the last frame says which allocation actually missed.
            site = "<no frame>"
            traced = exc.__traceback__
            while traced is not None:
                site = f"{os.path.basename(traced.tb_frame.f_code.co_filename)}:{traced.tb_lineno}"
                traced = traced.tb_next
            print(
                f"[r{rank}] {width:>7} {'--':>7} did not fit at {site}: "
                f"{str(exc).splitlines()[0]}",
                flush=True,
            )
            print(
                f"[r{rank}]            the tile the block loop holds is bounded by "
                f"`attention`'s tile budget; uncapped by the chunk it would be "
                f"{score_tile_bytes(model, width) / 2**20:.0f} MiB at {width} rows",
                flush=True,
            )
            torch.cuda.empty_cache()
            continue
        seconds = time.perf_counter() - started
        phases = timer.read()
        rows.append(
            (
                width,
                len(ids),
                seconds,
                experts.staged_experts - mark[0],
                experts.staged_bytes - mark[1],
                phases,
            )
        )
        byte_count = rows[-1][4]
        tokens = rows[-1][1]
        copy_ms = byte_count / tokens / 2**30 / rate * 1e3
        token_ms = seconds / tokens * 1e3
        report(rows[-1])
    if args.floor and len(rows) > 1:
        print()
        floor = rows[0]
        for row in rows[1:]:
            print(
                f"[r{rank}] {row[0]} tokens a call: {row[1] / row[2]:.2f} tok/s against "
                f"{floor[1] / floor[2]:.2f} one at a time = "
                f"{row[2] / row[1] and (floor[2] / floor[1]) / (row[2] / row[1]):.1f}x",
                flush=True,
            )

    # The prefill's own answer, which is the thing the collective has to get right: four ranks
    # holding partials of the same rows must land on the same logits, and on the same share of
    # the bytes. Gathered on every rank and printed on one, so a rank that left early hangs
    # rather than silently passing.
    if world > 1:
        import torch.distributed as dist

        if logits is None:
            # Every width was refused, so there is no row to compare and no share to report.
            # The barrier is still owed: the ranks that skipped their blocks all arrive here.
            if rank == 0:
                print("\n[r0] no width fit, so there is no last row to compare", flush=True)
            dist.barrier()
            return 0

        gathered = [torch.empty_like(logits) for _ in range(world)]
        dist.all_gather(gathered, logits.contiguous())
        disagree = float((gathered[1] - gathered[0]).abs().max())
        share = [None] * world
        dist.all_gather_object(share, round(rows[-1][4] / rows[-1][1] / 2**20, 1))
        if rank == 0:
            print(f"\n[r0] ranks agree on the prefill's last row: {not disagree} ({disagree})")
            print(
                f"[r0] MiB a token by rank at the widest chunk: {share}; the deals differ by rank "
                f"only in which experts they are",
                flush=True,
            )
        if args.floor:
            dist.all_gather_object(share, round(rows[0][4] / rows[0][1] / 2**20, 1))
            if rank == 0:
                print(f"[r0] MiB a token by rank, one at a time: {share}", flush=True)
        dist.barrier()

    # What a step at this depth costs, which is the other half of a long context: the prompt in
    # the cache is what every later token reads, and a global layer's read grows with it. The
    # steps are timed the way the decode benches time them, one token a step, and the phases are
    # the same event pairs -- so "prefill is linear in the prompt and decode is linear in the
    # depth" is a measurement here and not a claim.
    if args.decode and rows and rows[-1][0] != 1:
        # No re-prefill: the last width's pass left the whole prompt in the cache, and that cache
        # is what a step at this depth reads.
        drawn = [int(logits.argmax())]
        timer = PhaseTimer(model)
        # And the first of those steps is not one of them. This window starts the instant the
        # prompt's last chunk returns, so what it times first is whatever the prefill left
        # behind -- `bench_mimo_v2_ep.py` and the step probe both discard a first pass for exactly
        # that reason, and this bench was the one that did not -- and the warm-up steps are timed
        # on their own and reported beside the table, because a client waits for them and because
        # they are the only place that tail is visible. They are not a correction to the deal's
        # price, which is a per-step quantity and is the interleaved probe's to state: two runs of
        # this command at 8192 tokens of context read 246.4 ms a token under `sorted` and 274.2
        # under `id` in the first pair and 199.2 against 259.7 in the second, which is the spread
        # this box has between processes.
        if world > 1:
            torch.distributed.barrier()
        started = time.perf_counter()
        for step in range(args.warmup):
            logits = model.step(
                drawn[-1], start_pos=len(ids) + step, cache=cache
            )[-1]
            drawn.append(int(logits.argmax()))
        torch.cuda.synchronize()
        cold_s = time.perf_counter() - started
        timer.read()  # the warm-up's phases are not a token's, and they are timed above
        if world > 1:
            torch.distributed.barrier()
        started = time.perf_counter()
        for step in range(args.warmup, args.warmup + args.decode):
            logits = model.step(
                drawn[-1], start_pos=len(ids) + step, cache=cache
            )[-1]
            drawn.append(int(logits.argmax()))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        phases = timer.read()
        token_ms = seconds / args.decode * 1e3
        print(
            f"\n{tag} decode at {len(ids)} tokens of context: {args.warmup} warm-up steps in "
            f"{cold_s:.3f}s, then {args.decode} in {seconds:.3f}s = {args.decode / seconds:.2f} "
            f"tok/s ({token_ms:.1f} ms a token)",
            flush=True,
        )
        print(
            f"{tag} at that depth: attention {phases['attn'] / args.decode:.1f} ms a token, "
            f"FFN and staging {phases['mlp'] / args.decode:.1f} ms a token, "
            f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated "
            f"({cold_s / max(args.warmup, 1) * 1e3:.1f} ms a warm-up step)",
            flush=True,
        )
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, drawn[:4])
            if rank == 0:
                print(f"[r0] the first tokens at that depth, by rank: {gathered}", flush=True)
        if rank == 0:
            print(f"[r0] drew {drawn}", flush=True)
        if world > 1:
            dist.barrier()

    if rank == 0:
        print(f"[r0] last row draws token {int(logits.argmax())}")
        if floor_logits is not None:
            print(f"[r0] the per-token pass drew {int(floor_logits.argmax())} for the last row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

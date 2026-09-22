"""Generate text from a loaded DeepSeek-V4.1-Flash backbone.

`loader.load_backbone` returns a `LoadedBackbone` that answers one forward at a time and carries the
KV, indexer and Engram-hash state between calls. This is the loop above it: prefill the prompt in one
call, then one call per new token, decoding greedily by default.

The loop is small on purpose, because the cost is not in it. Every token is a full forward over the
host-offload expert path, so a decode step re-reads and re-expands the experts its layer routes to;
on this machine that is **15 to 42 s per generated token**, and 99.7% of it is the fp4-to-bf16
expansion of the 240-odd experts the step misses rather than any read. There is no batching and no
speculative decoding here.

`--expert-device cuda --expert-world 4` is the one lever this loop does expose: it hands the routed
experts to all four cards instead of the host, one `RoutedExperts` swap in `load_backbone` and no
change to the loop. The cards consume the checkpoint's packed fp4 directly, so the expansion above
does not happen at all -- see `device_experts.py` and
`docs/performance/deepseek_v4_1_flash_host_run.md` for what remains.

Command line:

    python -m src.models.deepseek_v4_1.generate --checkpoint /path/to/DeepSeek-V4.1-Flash \
        --prompt "The capital of France is" --max-new-tokens 8
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from .prefix_cache import HASH_CACHE, restore_rows, snapshot_rows

__all__ = ["Generation", "generate", "main"]

# The environment spellings of the two device-path flags, so that the machine remembers the
# configuration and the command stays a command. Everything about the choice is still the host path
# until one of them is set.
EXPERT_DEVICE_ENV = "DEEPSEEK_V41_EXPERT_DEVICE"
EXPERT_WORLD_ENV = "DEEPSEEK_V41_EXPERT_WORLD"


@dataclass
class Generation:
    """What `generate` produced, and why it stopped."""

    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    stopped: str = "length"
    """`eos`, `length` (max_new_tokens reached), or `max_seq_len` (the model's context is full)."""

    cached_tokens: int = 0
    """How many of `prompt_tokens` came out of the prefix store instead of being forward-passed.

    Zero without a store, and zero for a prompt with no stored prefix. Read by the serving layer for
    `usage.prompt_tokens_details` and by a probe for what a repeat actually reused -- the two callers
    disagree about nothing here, because the number is a count of tokens and not a guess at one.
    """

    driver: object | None = None
    """The `graphs.DecodeGraphs` a `graphs=True` run built, so a caller can report what it cost.

    `None` on the eager path and on a graph run that stopped before its first step. The graphs stay
    installed on the model when this returns, so a caller that wants the eager body back calls
    `driver.release()`.
    """

    decode_seconds: float = 0.0
    """Wall time of the decode steps alone: the prompt's forward and, on the graph path, the capture
    pass and the step replayed behind it are all outside it.

    `len(tokens)` steps either way -- both loops run one forward per new token after the first, and
    the first token comes off the prefill -- so this divides by `len(tokens)` for a ms/token that
    the two paths can be compared on. `elapsed / len(tokens)` over the whole call does not: on the
    graph path it carries the capture pass, which is a step that is not one of the tokens.
    """


def _pick(logits: torch.Tensor, temperature: float, top_k: int | None,
          generator: torch.Generator | None) -> int:
    """One token from one row of logits. Greedy at temperature 0, which is the default."""
    if temperature <= 0.0:
        # Ties go to the lowest id, which is what `argmax` does and what a fixed reference would do.
        return int(logits.argmax())
    scores = logits.float() / temperature
    if top_k is not None and 0 < top_k < scores.numel():
        cutoff = torch.topk(scores, top_k).values[-1]
        scores = torch.where(scores < cutoff, torch.full_like(scores, float("-inf")), scores)
    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.inference_mode()
def generate(
    front,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    seed: int | None = None,
    on_token: Callable[[int, torch.Tensor], None] | None = None,
    graphs: bool = False,
    prefill_chunk: int | None = None,
    prefix_cache=None,
) -> Generation:
    """Prefill `prompt_ids`, then decode up to `max_new_tokens` more.

    The prompt goes through as one call rather than one call per token, which is what a prefill is.
    It is not bit-identical to stepping the same tokens through one at a time -- the two orderings
    differ from the second position on, by fp32 reduction order across 40 layers, which
    `docs/performance/deepseek_v4_1_flash_host_run.md` bisects -- so a caller comparing against a
    reference has to pick one and say which.

    `prefill_chunk` splits that one call into calls of `prefill_chunk` tokens, which is what a prompt
    longer than the card can hold in one forward needs: the activations that are linear in the
    sequence length -- the Hyper-Connections mixing and the MTP hidden states -- are per-row, so they
    are the same tensors a chunk at a time and a 21 GiB tensor all at once. It is a *different
    arithmetic*, by the same reduction order the paragraph above is about, and it is not free: the
    layers' continuation bodies walk the chunk rather than running one vectorized expression. `None`
    runs the prompt in one forward.

    `on_token` is called with each new token id and the logits that produced it, before the next
    forward, for a caller that wants to stream. With `temperature > 0` the sampling is seeded by
    `seed` and reproducible only for a fixed torch build and device; greedy decoding is reproducible
    outright.

    `graphs` replays each block from a captured CUDA graph instead of running it, which is
    `_decode_graphs` below and is off by default: the same tokens at the same positions either way,
    and a pool of card memory on the other side of the choice. It needs a card -- the position
    reaches the graphs as a tensor -- and a `max_new_tokens` of at least one, since the recording is
    a decode step and there is nothing to record a decode step for otherwise.

    `prefix_cache` is a `prefix_cache.PrefixCache` the caller keeps across requests, or `None` for no
    reuse. With one, the prompt is looked up first: a stored prefix of it is restored instead of
    forward-passed, and a prompt that *is* one costs no forward at all. The prompt's caches are then
    kept on the host -- keyed by the tokens that produced them, sliced to the rows those tokens wrote
    -- along with a fixed-length head anchor the store was configured with, so that a later
    conversation sharing a rendered header reuses it too. `Generation.cached_tokens` reports how much
    of the prompt was reused. A store is not a correctness boundary: everything it returns is a
    snapshot of a forward this loop already ran, and a miss is a cold prefill.

    Each token comes from the logits this returns, not from `Backbone.forward`'s `output_ids`, which
    the model samples by its own config `temperature` -- a field the released schema does not carry,
    so it is always the 1.0 default, a softmax and a Gumbel draw over the whole vocabulary. That draw
    is not just wasted here, it consumes the global RNG a seeded caller expects to own, so this
    pins the model to greedy for the duration of the call and puts the value back afterwards.
    """
    ids = [int(t) for t in prompt_ids]
    if not ids:
        raise ValueError("prompt_ids is empty, so there is nothing to prefill")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens is {max_new_tokens}, which is negative")

    model = getattr(front, "model", front)
    saved_temperature = getattr(model, "temperature", None)
    if saved_temperature is not None:
        model.temperature = 0.0
    loop = _decode_graphs if graphs and max_new_tokens > 0 else _decode
    try:
        return loop(
            front, ids, max_new_tokens, temperature, top_k, eos_token_id, seed, on_token,
            prefill_chunk, prefix_cache,
        )
    finally:
        if saved_temperature is not None:
            model.temperature = saved_temperature


def _restore(front, model, saved: dict[str, torch.Tensor]) -> None:
    """Put a stored prefix back: the tree's caches, then the Engram hash slice beside them."""
    restore_rows(model, saved)
    front.restore_prefix(saved.get(HASH_CACHE))


def _keep(front, cache, model, ids, length: int, logits: torch.Tensor) -> None:
    """Keep the state the forward that just ran leaves, keyed by the first `length` tokens of `ids`.

    `logits` is that forward's row, which is what `Entry.logits` is for: an entry has to be able to
    answer a request that *is* its prefix without one more forward. `cache.max_seq_len` is the width
    the buffers were built at -- the store is constructed with it and `geometry_tag` keys on it -- and
    it is what tells `snapshot_rows` how many positions a row of a grouped buffer stands for.

    Nothing here fails loudly, and nothing here is a decision: a store that does not take the payload
    -- no store, a length under its floor, a snapshot past its budget -- costs the next request a
    forward it would have paid for anyway.
    """
    if cache is None:
        return
    saved = snapshot_rows(model, length, cache.max_seq_len)
    hashes = front.snapshot_prefix(length)
    if hashes is not None:
        saved[HASH_CACHE] = hashes
    cache.store(ids, length, saved, logits[0])


def _prefill(front, cache, ids, chunk) -> tuple[int, int, torch.Tensor]:
    """Bring the model to the end of `ids` and say what of the prompt it did not have to run.

    Returns `(cached_len, position, logits)`: how much of the prompt came out of the store, where the
    next forward goes, and the row the first new token is picked from. `position` is `len(ids)` in all
    three cases below -- a prompt answered out of the store was forwarded once already, by the request
    that stored it -- and the only caller that reads it is the graph path, whose position has to be a
    tensor by then.

    The store is what tells the three apart. A prompt with no stored prefix is forwarded whole, as it
    always was. A prompt that starts with one is forwarded from the prefix's end, which is a chunk
    boundary and not a different body -- `tests/test_models_deepseek_v4_1_chunked_prefill.py` pins
    every boundary bit-equal to a one-shot prefill. A prompt that *is* a stored prefix is not
    forwarded at all: the row its first new token comes from is the one the anchor kept, and the state
    is still restored, because the decode steps after it read the caches. Forwarding the prompt's last
    token again to recompute that row is the shortcut `prefix_cache` documents as unsound.

    `reset_state` runs before the restore rather than after a hit test: a sliced snapshot leaves the
    rows it does not carry as whatever the last request left there, so the reset is what makes a
    restore a reconstruction. It runs on a miss too, which is where it has always run.

    `Entry.logits[None]` is a row on the host and stays there. The two things that read the row -- the
    sampler and the streaming hook -- are indifferent to where it lives, and a sampler is *not*: a
    `torch.Generator("cpu")` against a card tensor is an error, so the host row is the safer of the
    two on a path where `temperature > 0` is reachable.
    """
    model = getattr(front, "model", front)
    hit = None if cache is None else cache.lookup(ids)
    front.reset_state(1)
    if hit is not None:
        cached_len, entry = hit
        _restore(front, model, entry.saved)
        if cached_len == len(ids):
            return cached_len, cached_len, entry.logits[None]
        _, logits, _ = front(torch.tensor([ids[cached_len:]]), cached_len, chunk=chunk)
        _keep(front, cache, model, ids, len(ids), logits)
        return cached_len, len(ids), logits

    # A cold prompt also takes the head anchor, which is a fixed length rather than a position in this
    # prompt: the snapshot has to be taken at a forward boundary, so the first chunk ends there and
    # the rest is a continuation. That boundary is the whole cost of the anchor, and it is paid on a
    # miss only -- a resume above never runs this branch.
    start = 0
    head = 0 if cache is None else cache.head_tokens
    if head and len(ids) > head:
        _, row, _ = front(torch.tensor([ids[:head]]), 0, chunk=chunk)
        _keep(front, cache, model, ids, head, row)
        start = head
    _, logits, _ = front(torch.tensor([ids[start:]]), start, chunk=chunk)
    _keep(front, cache, model, ids, len(ids), logits)
    return 0, len(ids), logits


def _decode(front, ids, max_new_tokens, temperature, top_k, eos_token_id, seed, on_token, prefill_chunk=None, prefix_cache=None) -> Generation:
    """The loop, with the model's own sampling already taken out of the picture."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    limit = getattr(getattr(front, "model", front), "max_seq_len", None)
    cached_len, position, logits = _prefill(front, prefix_cache, ids, prefill_chunk)
    result = Generation(prompt_tokens=len(ids), cached_tokens=cached_len)

    # The prompt is one forward, or none for a stored prefix, and its last row is the distribution the
    # first new token comes from either way -- so the first new token costs no extra forward.
    # `decode_seconds` starts after it, so splitting the forward into chunks moves work between the two
    # numbers rather than into either of them. A resume's forward is inside `_prefill`, and so is the
    # head chunk a cold prompt pays for its head anchor.
    started = time.perf_counter()
    try:
        while len(result.tokens) < max_new_tokens:
            if limit is not None and position >= limit:
                result.stopped = "max_seq_len"
                return result
            token = _pick(logits[0], temperature, top_k, generator)
            result.tokens.append(token)
            if on_token is not None:
                on_token(token, logits[0])
            if eos_token_id is not None and token == eos_token_id:
                result.stopped = "eos"
                return result
            _, logits, _ = front(torch.tensor([[token]]), position)
            position += 1

        return result
    finally:
        result.decode_seconds = time.perf_counter() - started


def _cache_device(model) -> torch.device:
    """The card a decode graph has to put the position on: the one the attention caches are on.

    Read off a buffer rather than off `torch.cuda.current_device()`. The two agree whenever the
    caller set the device first -- `src/cli/generate_v41.py` does -- but nothing in `load_backbone`
    promises it, and a position tensor built on the wrong card is a device mismatch forty layers
    down instead of a line here. `named_buffers` is what `graphs.snapshot` walks for the same
    reason: the caches are buffers and are the only part of the tree whose location is compiled in.
    """
    for name, buffer in model.named_buffers():
        if name.endswith("window_kv_cache"):
            return buffer.device
    raise RuntimeError(
        "the decode graphs replay the attention layers, and this model has no `window_kv_cache` "
        "buffer for their position to be sized against -- it is not a V4.1 backbone"
    )


def _decode_graphs(
    front, ids, max_new_tokens, temperature, top_k, eos_token_id, seed, on_token, prefill_chunk=None,
    prefix_cache=None,
) -> Generation:
    """`_decode` with every block replayed from a captured graph, split around the expert call.

    The loop is the same loop and the difference is where the position lives. A graph freezes
    whatever a Python value said when it was recorded, so the position here is a `decode_pos.Pos`,
    which carries a 0-dim index tensor for the layers and the integer beside it for this loop --
    the limit, the eos test and the pick all read the number, and the layers read the tensor.

    The recording is a decode step that has to happen anyway. The prompt is forwarded first, so the
    graphs are recorded from the activation after a real prefill and at a position in the sequence
    rather than at zero; the first new token comes off that prefill's logits, exactly as it does in
    `_decode`; the capture pass is then the step that consumes it. `DecodeGraphs.capture_pass`
    rewinds the caches behind that pass -- the bodies it runs write the same slots more than once,
    and on a layer whose compressor can emit the two variants write at positions the other would
    not have -- so its logits describe a cache state that no longer exists. Those logits are
    dropped and the step is run once more, this time by the graphs it just recorded, and it is
    *that* step the loop goes on from. One extra step out of a generation, all of it before the
    first token is picked from a graphed forward.

    A stored prefix changes the position and not the shape of any of that: the prompt is eager here
    whatever it was, eager prefill bodies are the only ones a graph may not hold, and a prefix that
    was answered without a forward leaves the caches a resume would have. The one thing to keep is
    that the store is written before the capture: `capture_pass` moves the caches, and the snapshot
    a store keeps has to be the prefill's.
    """
    from .decode_pos import Pos
    from .graphs import DecodeGraphs

    model = getattr(front, "model", front)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    limit = getattr(model, "max_seq_len", None)
    cached_len, position, logits = _prefill(front, prefix_cache, ids, prefill_chunk)
    result = Generation(prompt_tokens=len(ids), cached_tokens=cached_len)

    # The prefill's last row is the distribution the first new token comes from -- whatever produced
    # it -- so it is picked before anything is captured, and a generation that stops on it never
    # builds a graph at all.
    if limit is not None and position >= limit:
        result.stopped = "max_seq_len"
        return result
    token = _pick(logits[0], temperature, top_k, generator)
    result.tokens.append(token)
    if on_token is not None:
        on_token(token, logits[0])
    if eos_token_id is not None and token == eos_token_id:
        result.stopped = "eos"
        return result

    pos = Pos.device(position, _cache_device(model))
    driver = DecodeGraphs(model)
    result.driver = driver

    def step(token_id: int):
        """One decode forward, at wherever the position currently is."""
        return front(torch.tensor([[token_id]]), pos)

    driver.capture_pass(lambda: step(token))
    started = time.perf_counter()
    try:
        _, logits, _ = step(token)
        pos.advance()

        while len(result.tokens) < max_new_tokens:
            if limit is not None and pos.host >= limit:
                result.stopped = "max_seq_len"
                return result
            token = _pick(logits[0], temperature, top_k, generator)
            result.tokens.append(token)
            if on_token is not None:
                on_token(token, logits[0])
            if eos_token_id is not None and token == eos_token_id:
                result.stopped = "eos"
                return result
            _, logits, _ = step(token)
            pos.advance()

        return result
    except BaseException:
        # The loop can be left by a callback raising -- the serving adapter's cancellation and
        # stop-string paths both unwind from `on_token` -- and a driver no caller received is one
        # nobody can release. Left installed it is not inert: `Block.forward` hands *every* forward
        # to `block.decode_graph`, whose sink the first real pass allocated one row wide, so the
        # next prompt dies on the sink's own `copy_` with "output with shape [1, 1, 4, 5120] doesn't
        # match the broadcast shape [1, 1364, 4, 5120]". A run that unwinds installs nothing.
        driver.release()
        raise
    finally:
        result.decode_seconds = time.perf_counter() - started


def main(argv: Sequence[str] | None = None) -> int:
    """Load a checkpoint, generate from a prompt, and print the text.

    Exit status is 0 when the model loaded and generated something, 2 when the checkpoint or the
    prompt could not be read. A token rate is printed to stderr so that stdout stays the generation.
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(
        prog="python -m src.models.deepseek_v4_1.generate",
        description="Generate text from a released DeepSeek-V4.1-Flash checkpoint, on the host.",
    )
    parser.add_argument("--checkpoint", required=True, help="the released checkpoint directory")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 is greedy, which is the default and the reproducible one")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resident-engram", action="store_true",
                        help="copy the 189.13 GiB of Engram tables into RAM instead of gathering "
                             "them from the shards; roughly 750 s of reading, once")
    parser.add_argument("--expert-cache", type=int, default=None,
                        help="dequantized experts one layer keeps on the host")
    parser.add_argument("--expert-device", default=os.environ.get(EXPERT_DEVICE_ENV), metavar="DEVICE",
                        help="put the routed experts on DEVICE (e.g. cuda) across --expert-world "
                             "cards instead of the host, which never expands fp4 to a dense "
                             "weight; falls "
                             "back to the host path if the build does not work. "
                             f"Default: ${EXPERT_DEVICE_ENV} if set")
    parser.add_argument("--expert-world", type=int, default=None,
                        help="cards to split the routed experts over. Default: "
                             f"${EXPERT_WORLD_ENV} if set, else 1")
    parser.add_argument("--quiet", action="store_true", help="suppress the progress and timing lines")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from .config import load_config
    from .loader import DEFAULT_EXPERT_CACHE, V41Checkpoint, build_hasher, load_backbone

    def say(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    config = load_config(f"{args.checkpoint}/config.json").text
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    layout, hasher = build_hasher(config, args.checkpoint, tokenizer=tokenizer)
    checkpoint = V41Checkpoint(args.checkpoint)

    started = time.perf_counter()
    front = load_backbone(
        config,
        checkpoint,
        hasher=hasher,
        resident_engram=args.resident_engram,
        expert_cache=DEFAULT_EXPERT_CACHE if args.expert_cache is None else args.expert_cache,
        expert_device=args.expert_device,
        expert_world=(
            int(os.environ.get(EXPERT_WORLD_ENV, "1")) if args.expert_world is None
            else args.expert_world
        ),
        progress=say,
    )
    say(f"loaded in {time.perf_counter() - started:.1f} s")
    if args.expert_device is not None:
        # Not `front.model.routed`: `load_backbone` unpacks that dict into the layers, so what a
        # forward reaches is `layer.ffn.routed`, and `DeviceRoutedExperts` is deliberately not an
        # `nn.Module` -- `modules()` would not surface it either. The line reports what the run
        # actually built, because "it fell back to the host" is the failure this flag has, and the
        # world is read off the objects rather than off the flag for the same reason.
        held = [layer.ffn.routed for layer in front.model.layers]
        kinds = sorted({type(r).__name__ for r in held})
        world = getattr(held[0], "world", None)
        where = "" if world is None else f", world {world}"
        # A rank that drives one share of the deal returns a partial the ffn's all-reduce completes;
        # one that drives all of them sums them here. Both are correct and they are different runs,
        # so the line says which one this is.
        ranks = getattr(held[0], "ranks", None)
        if ranks is not None and world is not None and len(ranks) < world:
            where += f", holding rank {ranks[0]} of the deal"
        say(f"routed experts: {', '.join(kinds)} on {len(held)} layers{where}")

    prompt_ids = tokenizer(args.prompt)["input_ids"]
    say(f"prompt {len(prompt_ids)} tokens: {tokenizer.convert_ids_to_tokens(prompt_ids)}")

    started = time.perf_counter()
    result = generate(
        front,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started
    if result.tokens:
        say(f"{len(result.tokens)} tokens in {elapsed:.1f} s "
            f"({elapsed / len(result.tokens):.2f} s/token), stopped on {result.stopped}")

    print(tokenizer.decode(prompt_ids + result.tokens), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

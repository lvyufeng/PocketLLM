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
) -> Generation:
    """Prefill `prompt_ids`, then decode up to `max_new_tokens` more.

    The prompt goes through as one call rather than one call per token, which is what a prefill is.
    It is not bit-identical to stepping the same tokens through one at a time -- the two orderings
    differ from the second position on, by fp32 reduction order across 40 layers, which
    `docs/performance/deepseek_v4_1_flash_host_run.md` bisects -- so a caller comparing against a
    reference has to pick one and say which.

    `on_token` is called with each new token id and the logits that produced it, before the next
    forward, for a caller that wants to stream. With `temperature > 0` the sampling is seeded by
    `seed` and reproducible only for a fixed torch build and device; greedy decoding is reproducible
    outright.

    `graphs` replays each block from a captured CUDA graph instead of running it, which is
    `_decode_graphs` below and is off by default: the same tokens at the same positions either way,
    and a pool of card memory on the other side of the choice. It needs a card -- the position
    reaches the graphs as a tensor -- and a `max_new_tokens` of at least one, since the recording is
    a decode step and there is nothing to record a decode step for otherwise.

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
        return loop(front, ids, max_new_tokens, temperature, top_k, eos_token_id, seed, on_token)
    finally:
        if saved_temperature is not None:
            model.temperature = saved_temperature


def _decode(front, ids, max_new_tokens, temperature, top_k, eos_token_id, seed, on_token) -> Generation:
    """The loop, with the model's own sampling already taken out of the picture."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    front.reset_state(1)
    position = 0
    limit = getattr(getattr(front, "model", front), "max_seq_len", None)
    result = Generation(prompt_tokens=len(ids))

    # The prompt is one forward. Its last row is the distribution the first new token comes from,
    # so the first new token costs no extra forward.
    _, logits, _ = front(torch.tensor([ids]), position)
    position += len(ids)

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


def _decode_graphs(front, ids, max_new_tokens, temperature, top_k, eos_token_id, seed, on_token) -> Generation:
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
    """
    from .decode_pos import Pos
    from .graphs import DecodeGraphs

    model = getattr(front, "model", front)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    front.reset_state(1)
    limit = getattr(model, "max_seq_len", None)
    result = Generation(prompt_tokens=len(ids))

    # The prompt is one eager forward, before any graph exists: prefill branches on the position
    # being zero and is a different body from the one a graph holds, so a layer handed a graph here
    # would record the wrong one.
    _, logits, _ = front(torch.tensor([ids]), 0)

    # The prefill's last row is the distribution the first new token comes from, so it is picked
    # before anything is captured -- and a generation that stops on it never builds a graph at all.
    if limit is not None and len(ids) >= limit:
        result.stopped = "max_seq_len"
        return result
    token = _pick(logits[0], temperature, top_k, generator)
    result.tokens.append(token)
    if on_token is not None:
        on_token(token, logits[0])
    if eos_token_id is not None and token == eos_token_id:
        result.stopped = "eos"
        return result

    pos = Pos.device(len(ids), _cache_device(model))
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
                             "cards instead of the host, which never expands fp4 to bf16; falls "
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

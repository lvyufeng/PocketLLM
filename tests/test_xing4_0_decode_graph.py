"""The bucketed, captured decode step, held against the eager one it replaces.

`src/models/xing4_0/graphs.py` makes three claims and each has a test here rather than a comment:

* **a bucket is free at its end** — a step at the last position of a bucket reads exactly the rows it
  would read unbucketed and masks nothing, so its logits are bit-identical to the eager path's;
* **a replay is the same arithmetic as an eager submission at the same width** — the two arms differ
  in how the ops reach the card and in nothing else, so `torch.equal` is the assertion and not a
  tolerance;
* **the loop is unchanged** — the greedy chain a `DecodeGraphs` produces is the chain the eager path
  produces, token for token, across a bucket boundary.

## The fixture is a truncation of the released checkpoint, and why

`BLOCKS = 4` loads the checkpoint's own first four blocks — its embedding, its attention, its
hyper-connection, and (from `first_k_dense_replace 2`) two dense and two MoE blocks — with the real
released weights and the real kernels. What is being tested is the mechanism: an index tensor written
by a graph, a mask expressed against a device position, and an `N` that a recording can hold. Every
one of those happens 40 times a step in exactly the form it happens 4 times here, so depth is not a
variable in any of these properties, and 4 blocks costs 2.31 GiB and 4.8 s against the full model's
17.84 GiB and 13 s. The claims that *do* depend on the whole checkpoint — the per-step milliseconds,
the pool, and the end-to-end rate — are measured on the full model in
`docs/performance/xing4_0_decode_graph.md` rather than asserted here.

The released tokenizer supplies the prompts, so the token ids and the bucket boundary a run crosses
are real rather than chosen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.models.xing4_0.decode_pos import Pos
from src.models.xing4_0.gguf_model import Xing4_0GGUFModel
from src.models.xing4_0.generate import generate
from src.models.xing4_0.graphs import CAPTURE_WARMUP, MIN_BUCKET, DecodeGraphs, bucket_ladder


CHECKPOINT_DIR_ENV = "POCKETLLM_XING4_DIR"
CHECKPOINT_DIR_DEFAULT = "/mnt/data2"
GGUF_NAME = "Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"
HF_DIR = "Xing4.0-29B-A4B"
CORPUS = Path(__file__).resolve().parent.parent / "docs" / "models" / "qwen3.8-27b-fp8.md"

#: The prefix of the checkpoint these tests load.  See the module docstring.
BLOCKS = 4
#: Two rungs and a tail: 64, 128, and the capacity itself at 256.
CAPACITY = 256


def _dir() -> Path:
    path = Path(__import__("os").environ.get(CHECKPOINT_DIR_ENV, CHECKPOINT_DIR_DEFAULT))
    if not (path / GGUF_NAME).exists() or not (path / HF_DIR / "config.json").exists():
        pytest.skip(f"set {CHECKPOINT_DIR_ENV} or place {path / GGUF_NAME}")
    return path


# --------------------------------------------------------------------------- #
# The ladder: what a step may be recorded at
# --------------------------------------------------------------------------- #


def test_the_ladder_is_the_powers_of_two_from_the_floor() -> None:
    assert bucket_ladder(8192, 64) == [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    assert bucket_ladder(8192) == bucket_ladder(8192, MIN_BUCKET)


def test_the_ladder_ends_at_the_capacity_and_not_at_the_next_power_of_two() -> None:
    """A rung is a slice width, and a slice cannot read past the buffer.

    A ladder that rounded the capacity up would hold a rung the cache is too short for, so the last
    rung is 30000 and not 32768. That tail rung is what a 4096-wide cache contributes at a 32K
    context, and it is the one rung that is not a power of two.
    """
    assert bucket_ladder(30000, 64)[-1] == 30000
    assert bucket_ladder(30000, 64)[-2] == 16384
    assert bucket_ladder(4116, 64)[-1] == 4116


def test_a_power_of_two_capacity_appears_once() -> None:
    """The loop appends while a rung is *below* the capacity, so the capacity is not doubled."""
    assert bucket_ladder(64, 64) == [64]
    assert bucket_ladder(128, 64) == [64, 128]


def test_a_capacity_below_the_floor_is_the_capacity_alone() -> None:
    """A ladder has to hold something, and every rung has to be readable."""
    assert bucket_ladder(32, 64) == [32]
    assert bucket_ladder(1, 64) == [1]


def test_a_cache_of_no_positions_is_refused() -> None:
    with pytest.raises(ValueError):
        bucket_ladder(0)


def test_the_warmup_is_the_measured_one() -> None:
    """Two bodies on a side stream, the value `deepseek_v4_1` measured for the same purpose.

    Not a correctness device — the recorded body is what replay runs — but what the caching
    allocator needs: a capture wants every allocation it will make already free, or the recording
    grows the pool a block at a time.
    """
    assert CAPTURE_WARMUP == 2


# --------------------------------------------------------------------------- #
# The released checkpoint, truncated
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for a capture")
    root = _dir()
    from src.kernels.cuda_loader import load_cuda_kernel

    module = load_cuda_kernel()
    if module is None or not hasattr(module, "gguf_moe_prefill_grouped_forward"):
        pytest.skip("the GGUF kernels are not built for this interpreter")
    device = torch.device("cuda", torch.cuda.current_device())
    return Xing4_0GGUFModel(
        str(root / GGUF_NAME),
        device=str(device),
        config_path=str(root / HF_DIR / "config.json"),
        use_kernel=True,
        block_count=BLOCKS,
    )


@pytest.fixture(scope="module")
def prompt_ids() -> list[int]:
    """Real tokens from a repository document, so a bucket boundary is crossed by prose."""
    transformers = pytest.importorskip("transformers")
    root = _dir()
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(root / HF_DIR), trust_remote_code=True)
    ids = [int(t) for t in tokenizer(CORPUS.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]]
    assert len(ids) > 4096
    return ids


def _cache(model, capacity: int = CAPACITY):
    return model.make_cache(capacity, batch=1)


def _prefill(model, cache, ids: list[int], chunk: int = 64) -> None:
    for offset in range(0, len(ids), chunk):
        model.forward(ids[offset : offset + chunk], cache=cache, start_pos=offset)
    torch.cuda.synchronize()


def _token(ids: list[int]) -> int:
    return int(ids[0])


# --------------------------------------------------------------------------- #
# What a step is recorded at, and when
# --------------------------------------------------------------------------- #


def test_the_bucket_is_the_narrowest_rung_that_holds_the_position(model) -> None:
    """`bucket_for` rounds up, so a step wastes at most a factor of two on its read."""
    holder = DecodeGraphs(model, _cache(model))
    assert holder.ladder == [64, 128, 256]
    assert [holder.bucket_for(n) for n in (1, 64, 65, 128, 129, 256)] == [64, 64, 128, 128, 256, 256]


def test_a_position_past_the_capacity_is_refused(model) -> None:
    holder = DecodeGraphs(model, _cache(model))
    with pytest.raises(ValueError):
        holder.bucket_for(CAPACITY + 1)


def test_a_step_with_no_reserve_still_records_the_rung_it_needs(model, prompt_ids) -> None:
    """The floor of the recording range is the *rung*, not the row count the step needs.

    A ladder has no rung at 101, so a range of `[need, need]` selects nothing, records nothing, and
    leaves the step replaying a rung that does not exist — a `KeyError` at the first step of any run
    that did not call `reserve`. This pins the case, because `reserve` is a hint the loop happens to
    provide and the no-hint path is the one a caller reaches by not reading the docstring.
    """
    cache = _cache(model)
    _prefill(model, cache, prompt_ids[:100], chunk=50)
    holder = DecodeGraphs(model, cache)
    assert holder.reserved == 0

    holder.step(_token(prompt_ids), cache, 100)
    assert holder.recorded == [128], "101 rows need the 128 rung, and it has to be recorded"


def test_reserve_records_every_rung_the_run_reaches_at_the_first_step(model, prompt_ids) -> None:
    """One step's cost, not one step per boundary — which is what `reserve` is for.

    A run of 8 more tokens from position 60 reaches 68, so it needs the 64 rung and the 128 one. With
    no reserve the second would be recorded when a step first crosses it, in the middle of an answer;
    with one, both are recorded at the first step and the run settles immediately.
    """
    cache = _cache(model)
    _prefill(model, cache, prompt_ids[:60])
    holder = DecodeGraphs(model, cache)
    holder.reserve(60 + 8)

    holder.step(_token(prompt_ids), cache, 60)
    assert holder.recorded == [64, 128]
    assert holder.capture_seconds > 0.0 and holder.pool_bytes > 0
    assert set(holder.rung_bytes) == {64, 128}


def test_a_rung_is_recorded_once(model, prompt_ids) -> None:
    """The recordings outlive the step that needed them: a second step in the same bucket is free."""
    cache = _cache(model)
    _prefill(model, cache, prompt_ids[:60])
    holder = DecodeGraphs(model, cache)
    holder.step(_token(prompt_ids), cache, 60)
    captured = holder.capture_seconds

    for position in (61, 62):
        holder.step(_token(prompt_ids), cache, position)
    assert holder.recorded == [64] and len(holder.steps) == 1
    assert holder.capture_seconds == captured, "no second recording was paid for"
    # The replay counter is what makes `replay_millis` an average instead of a cumulative total --
    # the total is a property of the run and the average is a property of the step.
    assert holder.replays == 3 and holder.replay_millis > 0.0


def test_the_step_advances_each_layer_s_length(model, prompt_ids) -> None:
    """The device path cannot move `length` itself, so the loop that knows its position does.

    `KVLatentCache.append` compares an index tensor against the capacity to advance it, and that
    comparison is a host read — the one thing a capture forbids. So `length` is the caller's, and a
    holder that forgot it would leave every layer believing the cache holds nothing.
    """
    cache = _cache(model)
    _prefill(model, cache, prompt_ids[:60])
    holder = DecodeGraphs(model, cache)
    holder.step(_token(prompt_ids), cache, 60)
    assert {layer.length for layer in cache} == {61}


# --------------------------------------------------------------------------- #
# The two parities
# --------------------------------------------------------------------------- #


def test_a_bucket_end_is_bit_identical_to_the_unbucketed_step(model, prompt_ids) -> None:
    """The strongest statement the mechanism admits, and the reason a bucket is legal.

    At position 127 the step needs 128 rows, which *is* the rung: the read is the width the unbucketed
    step would have read and the mask hides nothing. So the two are not close, they are the same
    tensor — and if they were not, the bucket would be changing the model's arithmetic rather than
    only its submission.
    """
    ids = prompt_ids[:127]
    cache = _cache(model)
    _prefill(model, cache, ids)
    assert {layer.length for layer in cache} == {127}

    token = _token(prompt_ids)
    eager = model.forward([token], cache=cache, start_pos=127).clone()
    bucketed = model.forward([token], cache=cache, start_pos=Pos.bucket(127, 128))
    torch.cuda.synchronize()
    assert torch.equal(eager, bucketed)
    assert float((eager - bucketed).abs().max()) == 0.0


def test_the_bucket_moves_the_read_width_and_not_the_answer(model, prompt_ids) -> None:
    """Mid-bucket the read is wider than the position, and the mask is what makes it right.

    **The logits do move, and the size of that is not asserted here.** The rows past the position are
    masked to exactly zero probability, so what is left is float re-association: the softmax's sum and
    the value contraction both reduce over `N`, and a wider `N` groups the same terms differently. It
    is not a function of how much wider — measured identical at 1.27x and 2.55x — and it grows with
    depth: `max |dlogits|` 0.0625 at 2 blocks, 0.414 at 16, 0.789 at 40, against 0.4375 at the released
    checkpoint's 2.00x rung at a 4096-token context. A tolerance would pin whichever cuBLAS kernel the
    fixture's widths happen to select, and this geometry's is one that groups them identically: the
    bound this test used to carry (1e-3) passed *at exactly zero*, which is not evidence of anything.

    So what is asserted is the property a decode loop actually depends on — **the argmax does not
    change** — plus the equal-width control, which is structural and must be exact. Across the
    measurements above, 0 of 224 sampled positions changed the token.

    The control is the bucket *spelling* at a width equal to the position, which is the width the
    unbucketed read uses. If it is not exactly zero then the harness is comparing something other than
    the width and the claim about the argmax means nothing.
    """
    ids = prompt_ids[:100]
    cache = _cache(model)
    _prefill(model, cache, ids, chunk=50)

    token = _token(prompt_ids)
    eager = model.forward([token], cache=cache, start_pos=100).clone()
    worse = model.forward([token], cache=cache, start_pos=Pos.bucket(100, 128))
    equal = model.forward([token], cache=cache, start_pos=Pos.bucket(100, 101))
    torch.cuda.synchronize()
    assert torch.equal(eager, equal), "the same width has to be the same arithmetic"
    assert int(eager.argmax()) == int(worse.argmax())
    assert int(eager.argmax()) == int(equal.argmax())


def test_a_replay_is_bit_identical_to_the_same_width_eager_step(model, prompt_ids) -> None:
    """The capture's own parity: what a graph buys is submission, and nothing else moves.

    Both arms read the same rung, so a difference between them is the recording's and not the
    bucket's — which is exactly why `step_eager` exists beside `step` and why it is the oracle a
    served run's disagreement is settled with.
    """
    cache = _cache(model)
    _prefill(model, cache, prompt_ids[:100], chunk=50)
    holder = DecodeGraphs(model, cache)

    eager = holder.step_eager(_token(prompt_ids), cache, 100).clone()
    replayed = holder.step(_token(prompt_ids), cache, 100).clone()
    torch.cuda.synchronize()
    assert torch.equal(eager, replayed)
    assert holder.recorded == [128]


def test_the_greedy_chain_is_the_same_with_and_without_the_graph(model, prompt_ids) -> None:
    """The loop's own acceptance: same tokens, across a bucket boundary, with the holder in it.

    The prompt is 60 tokens and the run is 8, so it starts in the 64 rung and crosses into the 128
    one mid-answer — a boundary, a recording, and a replayed step all inside the comparison. Both
    arms get their own cache because a holder holds its cache's addresses.
    """
    steps = 8
    prompt = prompt_ids[:60]

    def run(decode_step, cache):
        return generate(
            model,
            prompt,
            max_new_tokens=steps,
            temperature=0.0,
            eos_token_id=None,
            cache=cache,
            chunk=64,
            decode_step=decode_step,
        )

    eager = run(None, _cache(model))
    # The holder holds its cache's addresses, so the arm that uses it is handed that same cache.
    cache = _cache(model)
    holder = DecodeGraphs(model, cache)
    graphed = run(holder, cache)

    assert eager.tokens == graphed.tokens
    assert len(eager.tokens) == steps
    assert holder.recorded == [64, 128], "the run has to have crossed the boundary it was chosen for"


# --------------------------------------------------------------------------- #
# Lifetime
# --------------------------------------------------------------------------- #


def test_a_step_against_a_different_cache_is_refused(model, prompt_ids) -> None:
    """The recordings hold raw addresses, so a step against another cache is a wrong answer.

    Refused rather than trusted because the failure mode is silent: the replay would write into the
    buffer it was recorded against and return logits for a cache nobody is reading.
    """
    cache = _cache(model)
    _prefill(model, cache, prompt_ids[:60])
    holder = DecodeGraphs(model, cache)
    with pytest.raises(ValueError):
        holder.step(_token(prompt_ids), _cache(model), 60)


def test_release_gives_the_memory_back_and_the_holder_records_again(model, prompt_ids) -> None:
    """A release ends a run; it does not spend the holder.

    The pool handle is dropped because it *is* spent — a capture into a freed pool aborts rather than
    failing, and this is the same rule that makes a fresh handle per recording round necessary — but
    the holder itself is re-recordable, and this pins the round trip: record, release, record again,
    and get the same logits. Nothing in a serving process depends on the refusal, and a refusal would
    be a failure mode invented for no reader.
    """
    cache = _cache(model)
    _prefill(model, cache, prompt_ids[:60])
    holder = DecodeGraphs(model, cache)

    first = holder.step(_token(prompt_ids), cache, 60).clone()
    assert holder.recorded == [64]

    holder.release()
    assert holder.recorded == [] and holder.pool is None
    assert holder.pool_bytes > 0, "the cost of the released recordings is still what it was"

    again = holder.step(_token(prompt_ids), cache, 60)
    torch.cuda.synchronize()
    assert holder.recorded == [64]
    assert torch.equal(first, again)

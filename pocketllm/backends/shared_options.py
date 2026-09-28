"""The options more than one runtime reads, declared once.

Four concepts here were declared once per runtime that reads them, and the declarations had drifted:
the same key meant ``None`` (unset) on one runtime and a value on another, the help text said
different things about the same lever, and ``expert_deal`` on one runtime was ``deal`` on the other
-- so a launch script had to know which runtime it was going to get. Read at their tip on
2026-09-28, vLLM has 246 flags and SGLang 526 and **neither has one flag spelled twice**; we had
four. ``docs/architecture/cli_surface_design_2026_09.md`` is the comparison this list is the first
step of.

A shared declaration is the *shape* -- the name, the type, the accepted values, the ``--help``
section and the sentence describing it -- and a runtime that reads it takes it as it is or states
where it differs, in one word:

```python
replace(PREFILL_CHUNK, resolution="the loader's own width")     # same flag; this runtime computes it
replace(EXPERT_DEAL, default="sorted")                          # same flag; this runtime's own default
```

The difference is then *data* rather than two ``_Options`` classes agreeing by hand, which is what
lets a command line be generated from these lists (U2b-2) and what lets a test hold every reader to
one shape (``tests/test_declared_options.py``).

The naming rule is upstream's, not ours: a flag keeps an unprefixed name when the concept is shared
and takes a family prefix when the concept belongs to one family -- SGLang has ``--dsv4-attn-backend``
beside the general ``--attention-backend`` it keeps, and eleven ``--cuda-graph-*`` flags. So
``expert_deal`` is family-prefixed because the concept is the expert arena's and *shared* because
v41 and mimo both have one; ``device`` and ``prefill_chunk`` are shared and unprefixed; and nothing
here is named after a runtime, because "which runtime reads this" is what ``group`` and ``readers``
answer.
"""

from __future__ import annotations

from .options import BackendOption, Group, Kind

#: ``device`` used to be here, and U3 took it out rather than splitting it in place.
#:
#: It was declared shared because three runtimes each read a *card* under it, and each read it
#: differently: v41 as the card its dense tree sits on and again as the *base* card the loader's
#: rank offset is applied to, mimo as the card it drives, xing4 as a fallback. Splitting it left two
#: concepts with one meaning each -- ``--device`` is the platform, ``--device-ids`` is the list of
#: cards -- and a declaration is what a *runtime's own* lever gets. A flag whose meaning is the same
#: on every runtime, including the native one that has no options list at all, is a fact about the
#: launch rather than about the runtime, so both live in :class:`pocketllm.api.EngineArgs` beside
#: ``tensor_parallel_size``, which is the same kind of thing. Nothing reads a card out of
#: ``backend_options`` any more: ``--backend-option device=...`` is refused as an undeclared key.

#: How wide one prefill forward is.
#:
#: The three runtimes answer this differently, and that is the point of declaring it once: v41 falls
#: back to ``--prefill-chunk-tokens`` and then to the loader's own width, mimo's is a measured
#: constant, and Xing4's is derived from the card's free memory and the context. One flag, three
#: answers -- which is what SGLang does with ``page_size`` (256 on CUDA, 128 on NPU) and why it has
#: no ``--dsv4-page-size``.
PREFILL_CHUNK = BackendOption(
    "prefill_chunk",
    Kind.INTEGER,
    None,
    "tokens one prefill forward takes",
    group=Group.PREFILL,
    minimum=1,
    readers=("v41", "mimo", "xing4"),
)

#: Host memory a rank's prefix store may hold.
#:
#: One concept, two constants: 4 GiB on v41 and mimo, 2 GiB on Xing4, which is a smaller checkpoint
#: with a store sized for its own prompts. Each runtime declares its own default, so the flag has
#: one spelling and the answer is still the runtime's.
PREFIX_CACHE_BYTES = BackendOption(
    "prefix_cache_bytes",
    Kind.BYTES,
    None,
    "host memory a rank's prefix store may hold",
    group=Group.PREFIX_CACHE,
    readers=("v41", "mimo", "xing4"),
)

#: The fixed-length anchor a prefill also stores, or 0 for the prompt's end alone.
#:
#: The one shared option both readers take exactly as declared: a 1024-token head anchor exists for
#: the reason ``docs/architecture/v41_prefix_cache.md`` gives, and it is the same reason on MiMo's
#: store, so there is no runtime difference here to carry.
PREFIX_CACHE_HEAD_TOKENS = BackendOption(
    "prefix_cache_head_tokens",
    Kind.INTEGER,
    1024,
    "the fixed-length anchor a prefill also stores, or 0 for the prompt's end alone",
    group=Group.PREFIX_CACHE,
    minimum=0,
    readers=("v41", "mimo"),
)

#: Which deal divides the routed experts across ranks.
#:
#: ``sorted`` balances the experts across ranks, ``id`` leaves them in checkpoint order. v41 leaves
#: it to the loader (unset); mimo defaults to ``sorted``, and an explicit ``null`` asks the
#: process's ``POCKETLLM_MIMO_EXPERT_DEAL`` instead. Two answers for one option is the case the
#: declarations exist to make visible: before this the lever was ``expert_deal`` on one runtime and
#: ``deal`` on the other.
EXPERT_DEAL = BackendOption(
    "expert_deal",
    Kind.STRING,
    None,
    "which deal divides the experts: `sorted` balances them across ranks, `id` leaves them in "
    "checkpoint order",
    group=Group.EXPERT,
    aliases=("deal",),
    # ``sorted`` first because it is the deal the served deployment runs and the one the sentence
    # above describes at length; the refusal quotes this order, so it is the order an operator meets
    # the two values in.
    choices=("sorted", "id"),
    readers=("v41", "mimo"),
)

#: Every shared declaration, in the order a reader should meet them.
SHARED: tuple[BackendOption, ...] = (
    PREFILL_CHUNK,
    PREFIX_CACHE_BYTES,
    PREFIX_CACHE_HEAD_TOKENS,
    EXPERT_DEAL,
)

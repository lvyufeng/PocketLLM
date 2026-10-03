"""Qwen3, as a graph of ABI ops.

This is the architecture the *C* engine implements in ``src/model/qwen3.cpp``,
rebuilt here as a :class:`~pocketllm.architectures.ir.ModelSpec` so the Python
engine can run it.  The two are the same model written twice against different
interfaces, and this file is the second one: the sequence below is read line by
line from that forward pass, not from a description of Qwen3 in general, because
the C engine is the only authority in this tree on what this checkpoint computes.

Three details are Qwen3's and each is easy to get subtly wrong:

* **QK-norm is per head, over ``head_dim``.**  The query projection is
  ``n_head * head_dim`` wide and the norm weight is ``head_dim`` long, so the
  normalization reduces over the last ``head_dim`` of a ``(tokens * heads,
  head_dim)`` view -- not over the projection row.  Normalizing the row instead
  reduces over sixteen times as many values and produces a different, still
  finite, still plausible model.  That is why ``reshape`` is here at all.
* **Norm runs before RoPE**, on both q and k.
* **RoPE is split-half (NeoX)**, pairing ``i`` with ``i + d/2``.

The op order per layer, exactly as ``Qwen3Model::forward`` runs it::

    embed -> rms_norm(attn_norm)
          -> gemm(wq) -> reshape -> rms_norm(q_norm)
          -> gemm(wk) -> reshape -> rms_norm(k_norm)
          -> gemm(wv) -> reshape
          -> rope(q), rope(k)
          -> cache_append(k), cache_append(v)
          -> attention -> reshape -> gemm(wo) -> add(residual)
          -> rms_norm(ffn_norm)
          -> gemm(gate), gemm(up) -> silu_mul -> gemm(down) -> add(residual)

and then, once::

    rms_norm(output_norm) -> gemm(lm_head) -> logits

**This is a decode-step graph: ``rows`` tokens per call, one by default.**  The C
engine's ``forward`` takes a whole prompt at once and produces only the last
row's logits; this graph cannot, and the reason is the ABI rather than the model.
``reshape`` needs a literal target shape, so the ``(rows * heads, head_dim) ->
(rows, heads, head_dim)`` split has to know ``rows`` at build time; there is no
dynamic-shape reshape and no ``slice``.  ``Qwen3Config.rows`` therefore fixes the
call's width, and prefilling a prompt means calling the graph once per token --
which is what the C engine does for every position but the first anyway.  An
AOT backend that compiles the graph would want exactly this split: a prefill
artifact at one width and a decode artifact at ``rows=1``.

**Only the last position's logits are wanted.**  The C engine slices the final
row out of ``x_`` and projects that alone; here all ``rows`` rows are projected
because there is no ``slice`` to take the last one.  At ``rows=1`` -- the path a
decode loop takes -- the two are the same computation.  At a wider ``rows`` the
earlier rows are computed and discarded, which is a real cost this version
accepts knowingly rather than hides.

**The KV cache is unrolled over the layer axis.**  Each layer gets its own graph
inputs ``blk.<i>.k_cache`` / ``blk.<i>.v_cache`` of shape ``(capacity, kv_heads,
head_dim)``, which is how the C engine addresses it: ``forward`` offsets into one
allocation per layer slab.  The alternative -- one ``(layers, capacity, kv_heads,
head_dim)`` input -- cannot work, because ``attention`` reads a 3-D cache and
there is no ``slice`` to take a layer out of a 4-D one.  Unrolling costs one
input pair per layer and buys the ability to express the model at all, with no
new op.

The layer axis is then necessarily *unrolled in the body* too: with per-layer
cache inputs the recurrent value changes per layer, so the builder cannot be a
loop over one value.  The graph is ``layers`` unrolled copies of the body, which
is honest for a fixed layer count and would need a real loop IR for one without.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.tensor import TensorDesc

from .cache import CacheLayout, CachePlan
from .ir import GraphBuilder, ModelSpec

__all__ = ["Qwen3Config", "build"]


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Qwen3's hyperparameters, defaulted to the 0.6B checkpoint.

    The field names are the C engine's and the defaults are what
    ``Qwen3Model::load`` reads from a 0.6B GGUF.  Reading both from the same
    checkpoint's metadata is what keeps them from drifting: a ``Qwen3Config``
    built by hand and a config read from a file are two answers, and only the
    file's is the model.

    ``context`` and ``rows`` are the two fields that are **not** properties of
    the checkpoint.  ``context`` is a deployment choice -- the same 0.6B can
    serve a 4K chat and its full 40960 -- and ``rows`` is the graph's fixed call
    width, which the ABI's literal reshape forces (see the module docstring).
    """

    hidden: int = 2048
    layers: int = 28
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    ff: int = 6144
    vocab: int = 151936
    #: ``qwen3.attention.layer_norm_rms_epsilon``.
    rms_eps: float = 1e-6
    #: ``qwen3.rope.freq_base``.  Qwen3's RoPE base is 1e6 and the C engine's
    #: *fallback* is 1e4; they are not interchangeable, so this is a field.
    rope_theta: float = 1_000_000.0
    #: ``qwen3.context_length``, the cap the cache is sized to.
    context: int = 40960
    #: How many tokens one call takes.  One is a decode step.
    rows: int = 1
    #: Whether the head shares the embedding table.  The C engine ties when the
    #: checkpoint carries no ``output.weight``; 0.6B carries one, so this is
    #: ``False`` and the default config is the checkpoint's.
    tie_embeddings: bool = False
    dtype: DType = DType.F32

    def __post_init__(self) -> None:
        for name in ("hidden", "layers", "heads", "kv_heads", "head_dim", "ff", "vocab", "context", "rows"):
            if getattr(self, name) <= 0:
                raise ValueError(f"qwen3: {name} must be positive, got {getattr(self, name)}")
        if self.heads % self.kv_heads:
            raise ValueError(
                f"qwen3: heads ({self.heads}) must be a multiple of kv_heads ({self.kv_heads})"
            )
        if self.head_dim % 2:
            raise ValueError(f"qwen3: head_dim must be even for the rotary split, got {self.head_dim}")
        if self.rows > self.context:
            raise ValueError(f"qwen3: rows ({self.rows}) cannot exceed the context ({self.context})")

    @property
    def q_width(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim


def build(config: Qwen3Config | None = None) -> ModelSpec:
    """Build the Qwen3 graph, verified, with its cache plan filled in.

    The weights are named as GGUF names them, which is how the C engine binds
    them: a loader that reads ``blk.3.attn_q.weight`` from a file needs the same
    spelling here, and a rename is a binding failure rather than a rebuild.
    """
    config = config or Qwen3Config()
    cfg = config
    hidden, ff = cfg.hidden, cfg.ff
    heads, kv_heads, head_dim = cfg.heads, cfg.kv_heads, cfg.head_dim
    rows, dtype = cfg.rows, cfg.dtype

    b = GraphBuilder("qwen3")
    tokens = b.input("tokens", TensorDesc((rows,), DType.I32))
    # The positions of these rows, and the rotary tables they index.  All three
    # are bound per call rather than carried as weights: the position advances
    # every step even though the tables do not.
    positions = b.input("positions", TensorDesc((rows,), DType.I32))
    cos = b.input("rope_cos", TensorDesc((cfg.context, head_dim // 2), dtype))
    sin = b.input("rope_sin", TensorDesc((cfg.context, head_dim // 2), dtype))

    table = b.weight(
        "token_embd.weight",
        TensorDesc((cfg.vocab, hidden), dtype),
        role="token embedding (also the head when tied)",
    )
    lm_head = (
        table
        if cfg.tie_embeddings
        else b.weight("output.weight", TensorDesc((cfg.vocab, hidden), dtype), role="output projection")
    )
    output_norm = b.weight(
        "output_norm.weight", TensorDesc((hidden,), dtype), role="final rms norm", quantizable=False
    )

    #: One layout per (layer, k/v): the cache is unrolled, so the plan is too.
    cache = CachePlan(default_capacity=cfg.context)
    cache_values: list[str] = []

    h = b.one("embedding", tokens, table, outputs="embed", tag="embed")

    for il in range(cfg.layers):
        base = f"blk.{il}."
        head_shape = (rows, heads, head_dim)
        kv_head_shape = (rows, kv_heads, head_dim)

        attn_norm = b.weight(base + "attn_norm.weight", TensorDesc((hidden,), dtype), role="attention rms norm", quantizable=False)
        ffn_norm = b.weight(base + "ffn_norm.weight", TensorDesc((hidden,), dtype), role="ffn rms norm", quantizable=False)
        q_norm = b.weight(base + "attn_q_norm.weight", TensorDesc((head_dim,), dtype), role="per-head query rms norm", quantizable=False)
        k_norm = b.weight(base + "attn_k_norm.weight", TensorDesc((head_dim,), dtype), role="per-head key rms norm", quantizable=False)
        wq = b.weight(base + "attn_q.weight", TensorDesc((cfg.q_width, hidden), dtype), role="query projection")
        wk = b.weight(base + "attn_k.weight", TensorDesc((cfg.kv_width, hidden), dtype), role="key projection")
        wv = b.weight(base + "attn_v.weight", TensorDesc((cfg.kv_width, hidden), dtype), role="value projection")
        wo = b.weight(base + "attn_output.weight", TensorDesc((hidden, cfg.q_width), dtype), role="attention output projection")
        w_gate = b.weight(base + "ffn_gate.weight", TensorDesc((ff, hidden), dtype), role="ffn gate projection")
        w_up = b.weight(base + "ffn_up.weight", TensorDesc((ff, hidden), dtype), role="ffn up projection")
        w_down = b.weight(base + "ffn_down.weight", TensorDesc((hidden, ff), dtype), role="ffn down projection")

        k_cache_name = base + "k_cache"
        v_cache_name = base + "v_cache"
        k_cache = b.input(k_cache_name, TensorDesc((cfg.context, kv_heads, head_dim), dtype))
        v_cache = b.input(v_cache_name, TensorDesc((cfg.context, kv_heads, head_dim), dtype))
        cache = cache.add(
            CacheLayout(name=k_cache_name, layers=1, kv_heads=kv_heads, head_dim=head_dim, dtype=dtype)
        ).add(
            CacheLayout(name=v_cache_name, layers=1, kv_heads=kv_heads, head_dim=head_dim, dtype=dtype)
        )
        cache_values.extend((k_cache_name, v_cache_name))

        normed = b.one("rms_norm", h, attn_norm, outputs=base + "normed", attrs={"eps": cfg.rms_eps}, tag=base + "attn.norm")
        q2 = b.one("gemm", normed, wq, outputs=base + "q2", tag=base + "attn.q")
        k2 = b.one("gemm", normed, wk, outputs=base + "k2", tag=base + "attn.k")
        v2 = b.one("gemm", normed, wv, outputs=base + "v2", tag=base + "attn.v")

        # To heads, normalize each head over its own `head_dim`, and back to
        # (rows, heads, head_dim) for the rope.
        #
        # Four reshapes rather than two, and the reason is the schema: `rms_norm`
        # is declared over 2-D `(tokens, d)`, which is what makes its weight a
        # 1-D vector over the *last* axis, and the C engine's
        # `rms_norm(q_, q_norm, q_, n * n_head_, head_dim_, eps)` is exactly that
        # 2-D view.  Flattening `(rows, heads*head_dim)` to `(rows*heads,
        # head_dim)` puts one head on each row, which is the view the norm wants
        # and the one that makes the reduction `head_dim` and not `heads*head_dim`.
        q_flat = b.one("reshape", q2, outputs=base + "q_flat", attrs={"shape": (rows * heads, head_dim)}, tag=base + "attn.q.flat")
        k_flat = b.one("reshape", k2, outputs=base + "k_flat", attrs={"shape": (rows * kv_heads, head_dim)}, tag=base + "attn.k.flat")
        v3 = b.one("reshape", v2, outputs=base + "vh", attrs={"shape": kv_head_shape}, tag=base + "attn.v.heads")

        q_flat = b.one("rms_norm", q_flat, q_norm, outputs=base + "q_flat_norm", attrs={"eps": cfg.rms_eps}, tag=base + "attn.q_norm")
        k_flat = b.one("rms_norm", k_flat, k_norm, outputs=base + "k_flat_norm", attrs={"eps": cfg.rms_eps}, tag=base + "attn.k_norm")

        q3 = b.one("reshape", q_flat, outputs=base + "qh", attrs={"shape": head_shape}, tag=base + "attn.q.heads")
        k3 = b.one("reshape", k_flat, outputs=base + "kh", attrs={"shape": kv_head_shape}, tag=base + "attn.k.heads")

        q3 = b.one("rope", q3, positions, cos, sin, outputs=base + "q_rot", attrs={"layout": "split", "theta_base": cfg.rope_theta}, tag=base + "attn.q_rope")
        k3 = b.one("rope", k3, positions, cos, sin, outputs=base + "k_rot", attrs={"layout": "split", "theta_base": cfg.rope_theta}, tag=base + "attn.k_rope")

        # The cache holds the *rotated* key -- a score is not a function of the
        # unrotated one -- which is why the writes follow the rope, not the norm.
        updated_k = b.one("cache_append", k_cache, k3, positions, outputs=base + "k_cache_out", tag=base + "attn.k_cache")
        updated_v = b.one("cache_append", v_cache, v3, positions, outputs=base + "v_cache_out", tag=base + "attn.v_cache")

        attn = b.one(
            "attention",
            q3,
            updated_k,
            updated_v,
            positions,
            outputs=base + "attn",
            attrs={
                "softmax_scale": 1.0 / math.sqrt(head_dim),
                "causal": True,
                "num_kv_heads": kv_heads,
            },
            tag=base + "attn.scores",
        )
        attn2 = b.one("reshape", attn, outputs=base + "attn2", attrs={"shape": (rows, cfg.q_width)}, tag=base + "attn.merge")
        h = b.one("add", h, b.one("gemm", attn2, wo, outputs=base + "o", tag=base + "attn.o"), outputs=base + "resid1", tag=base + "attn.residual")

        normed2 = b.one("rms_norm", h, ffn_norm, outputs=base + "ffn_normed", attrs={"eps": cfg.rms_eps}, tag=base + "ffn.norm")
        gate = b.one("gemm", normed2, w_gate, outputs=base + "gate", tag=base + "ffn.gate")
        up = b.one("gemm", normed2, w_up, outputs=base + "up", tag=base + "ffn.up")
        act = b.one("silu_mul", gate, up, outputs=base + "act", tag=base + "ffn.act")
        down = b.one("gemm", act, w_down, outputs=base + "down", tag=base + "ffn.down")
        h = b.one("add", h, down, outputs=base + "resid2", tag=base + "ffn.residual")

    final = b.one("rms_norm", h, output_norm, outputs="final_norm", attrs={"eps": cfg.rms_eps}, tag="output_norm")
    logits = b.one("gemm", final, lm_head, outputs="logits", tag="lm_head")
    b.output(logits)

    for name in cache_values:
        b.mark_cache(name)

    spec = b.build()
    spec.cache = cache
    return spec
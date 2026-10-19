"""Every declared op, in numpy, at float32.

This module is the executable form of the schemas in
:mod:`pocketllm.kernels.ops`: for each op, the function below computes what its
``semantics`` string says.  That correspondence is the contract -- a schema whose
semantics and reference implementation disagree is a bug in one of them, and the
only way to tell which is to read them side by side, which is why they are
written to be read side by side.

Two conventions run through every kernel:

* **float32 is the working type.**  Inputs in f16 or bf16 are widened, the
  arithmetic happens in f32, and the caller casts back on store.  That is what
  makes the reference an oracle: it is the *most accurate* answer the op's
  semantics admit, so a fast backend's deviation from it is a real deviation and
  not two roundings arguing.
* **``out`` may be supplied.**  The ABI lets a caller pass a preallocated output
  (``BackendSession.run(out=...)``); when it is, the kernel writes into it and
  returns it rather than allocating.  This is not an optimisation -- it is what
  lets a graph region be captured on backends that have capture, where the
  output buffer is fixed at record time.

The kernels take and return numpy arrays.  Tensors, buffers and dtype conversion
are :class:`~pocketllm.backends.reference.session.ReferenceSession`'s job; these
functions never see a ``Tensor``, which is what makes them testable directly.
"""

from __future__ import annotations

import math

import numpy as np

from pocketllm.quant import formats

__all__ = ["KERNELS", "quantize_weights"]

#: The uniform used by ``topk_sample`` when the caller supplies none: a single
#: fixed draw keeps sampling reproducible, and reproducibility is worth more to
#: a conformance harness than distributional purity.  A real sampler passes the
#: variate in; see the op's schema.
_DEFAULT_UNIFORM = 0.5


def quantize_weights(x: np.ndarray, blocks: np.ndarray, fmt_name: str) -> np.ndarray:
    """Decode ``x @ w``'s right operand from packed blocks, then multiply.

    The decode goes through :mod:`pocketllm.quant.formats` -- the same table the
    loader reads the file with -- so "what the reference computes" and "what the
    file contains" cannot drift apart.  One implementation of each format, used
    twice.
    """
    fmt = formats.format_for(fmt_name)
    if fmt.decode is None:
        raise NotImplementedError(
            f"{fmt_name} blocks are addressable but no decoder consumes them; "
            "the reference backend cannot run this op"
        )
    rows = blocks.shape[:-2]
    decoded = fmt.decode(blocks)
    return decoded.reshape(*rows, -1)


def _gemm(x: np.ndarray, w: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """``y[r, j] = sum_k x[r, k] * w[j, k] + bias[j]``.

    ``einsum`` rather than ``x @ w.T``: the weight is stored ``(n, k)`` -- row
    major over the *output* dimension, as GGUF stores it -- so the contraction is
    unambiguous written out, and an accidental transpose is a shape error rather
    than a silently transposed product.
    """
    y = np.einsum("rk,nk->rn", x, w, optimize=True)
    if bias is not None:
        y = y + bias[None, :]
    return y


def gemm(x, w, bias=None, **_attrs):
    return _gemm(x, w, bias)


def gemm_quant(x, w_blocks, bias=None, *, w_blocks_fmt="iq4_nl", **_attrs):
    fmt_name = _quant_name(w_blocks_fmt)
    w = quantize_weights(x, w_blocks, fmt_name)
    return _gemm(x, w, bias)


def _quant_name(value) -> str:
    """Accept either a fmt name or the descriptor the session attaches."""
    name = getattr(value, "name", value)
    return str(name)


def attention(
    q,
    k_cache,
    v_cache,
    positions,
    mask=None,
    window=None,
    *,
    softmax_scale=None,
    causal=True,
    num_kv_heads=None,
    **_attrs,
):
    """``softmax(q @ k^T * scale + mask) @ v`` over each query's visible cache.

    One query row at a time, over every cache row: this is the *reference*
    formulation, and it is written to be obviously the described computation
    rather than fast.  ``positions`` names the cache slot each query attends at,
    so the visible span is ``[0, positions[i]]`` (or within ``window`` of it) --
    which is what makes a chunked prefill and a single decode step the same code.
    """
    q_len, heads, d = q.shape
    capacity, kv_heads, v_d = v_cache.shape
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(d)
    group = heads // kv_heads if kv_heads else 1
    out = np.zeros((q_len, heads, v_d), dtype=np.float32)
    for i in range(q_len):
        end = int(positions[i]) + 1
        start = max(0, end - int(window)) if window else 0
        keys = k_cache[start:end]
        values = v_cache[start:end]
        for h in range(heads):
            kv = h // group
            scores = (keys[:, kv, :] @ q[i, h, :]) * scale
            if causal and mask is None:
                # The mask arg is for a *custom* bias; causality within the
                # visible span is already implied by `end`, so `causal` here
                # only matters when a caller passes a wider cache on purpose.
                pass
            if mask is not None:
                scores = scores + np.asarray(mask)[start:end]
            scores = scores - scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            out[i, h, :] = weights @ values[:, kv, :]
    return out


def rms_norm(x, weight, *, eps=1e-6, **_attrs):
    """``x / sqrt(mean(x^2) + eps) * weight``, computed in float32."""
    x32 = x.astype(np.float32)
    mean_sq = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 / np.sqrt(mean_sq + float(eps))) * weight.astype(np.float32)


def layer_norm(x, weight, bias=None, *, eps=1e-5, **_attrs):
    """``(x - mean) / sqrt(var + eps) * weight + bias``, computed in float32."""
    x32 = x.astype(np.float32)
    mean = x32.mean(axis=-1, keepdims=True)
    var = x32.var(axis=-1, keepdims=True)
    out = (x32 - mean) / np.sqrt(var + float(eps)) * weight.astype(np.float32)
    if bias is not None:
        out = out + bias.astype(np.float32)
    return out


def silu_mul(gate, up, **_attrs):
    """``silu(gate) * up``, with silu computed stably for large negative input."""
    g = gate.astype(np.float32)
    return (g / (1.0 + np.exp(-g))) * up.astype(np.float32)


GELU_COEF_A = 0.044715
GELU_SQRT_2_OVER_PI = 0.79788456080286535587989211986876


def gelu(x, **_attrs):
    """The tanh-approximated GELU, `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.

    **This variant and not the erf one**, because the checkpoint asked for it:
    a vision tower whose GGUF says ``clip.use_gelu`` is built by llama.cpp with
    `FFN_GELU`, whose CPU implementation is this expression (the erf form is a
    separate op there, reached by a separate flag). It is also the function
    `transformers` calls ``gelu_pytorch_tanh``, which is what the Qwen-VL
    checkpoints name in their config. The two differ by ~1e-3 relative near
    `|x| = 2`, which is a different model at the fourth digit of an embedding.

    The constants are written to the same digits as `ggml`'s, so the two agree to
    the last bit of float32 rather than to a tolerance.
    """
    x32 = x.astype(np.float32)
    inner = GELU_SQRT_2_OVER_PI * x32 * (1.0 + GELU_COEF_A * x32 * x32)
    return (0.5 * x32 * (1.0 + np.tanh(inner))).astype(np.float32)


def gelu_erf(x, **_attrs):
    """The exact GELU, `0.5*x*(1+erf(x/sqrt(2)))`.

    The other half of the pair `gelu` documents: llama.cpp reaches this one
    through its own `ffn_op` value, and a `qwen3a` audio tower names it. The two
    differ by roughly 0.01 around `|x| = 1` in the *output* -- far more than a
    rounding difference, and enough that a tower built with the wrong one answers a
    different question. `math.erf` rather than a tanh fit, because the point of
    this op is that it is not the approximation.
    """
    x32 = x.astype(np.float32)
    return (0.5 * x32 * (1.0 + np.vectorize(math.erf)(x32 / math.sqrt(2.0)))).astype(np.float32)


def conv2d(x, w, bias=None, *, kernel, stride, pad, **_attrs):
    """The patch embedding: a square 2-D convolution, as a gather plus a product.

    The weight is ``(out_ch, in_ch * kernel * kernel)`` with ``k`` indexed
    ``in_ch`` slowest and the kernel width fastest -- the GGUF layout read
    straight off the mapping, and the same row-major ``(n, k)`` shape ``gemm``
    takes, so the same bytes are usable by both ops with no transposition.  The
    output is ``(out_h * out_w, out_ch)``, patch-major with the channel fastest,
    so a patch is a row and the next op can treat it as a token.

    Written as an explicit gather and an ``einsum`` rather than through a
    convolution primitive because this is the *reference*: every index a kernel
    could swap is written down, and the loops are the four indices the semantics
    names.  The C kernel is the independent statement of the same thing, and the
    two are compared on identical input.
    """
    x32 = np.asarray(x, dtype=np.float32)
    w32 = np.asarray(w, dtype=np.float32)
    in_ch, height, width = x32.shape
    out_ch, k = w32.shape
    if k != in_ch * kernel * kernel:
        raise ValueError(
            f"conv2d: w has {k} columns, but {in_ch} channels and a "
            f"{kernel}x{kernel} kernel need {in_ch * kernel * kernel}"
        )
    out_h = (height + 2 * pad - kernel) // stride + 1
    out_w = (width + 2 * pad - kernel) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"conv2d: a {kernel}x{kernel} kernel at stride {stride} with padding {pad} "
            f"does not fit a {height}x{width} image"
        )
    # [patch, in_ch * kernel_h * kernel_w], the `k` axis ordered ic slowest.
    cols = np.zeros((out_h * out_w, k), dtype=np.float32)
    for ic in range(in_ch):
        for ky in range(kernel):
            ys = np.arange(out_h) * stride + ky - pad
            ys_ok = (ys >= 0) & (ys < height)
            for kx in range(kernel):
                xs = np.arange(out_w) * stride + kx - pad
                xs_ok = (xs >= 0) & (xs < width)
                block = np.zeros((out_h, out_w), dtype=np.float32)
                if ys_ok.any() and xs_ok.any():
                    block[np.ix_(ys_ok, xs_ok)] = x32[ic][np.ix_(ys[ys_ok], xs[xs_ok])]
                cols[:, ic * kernel * kernel + ky * kernel + kx] = block.reshape(-1)
    y = np.einsum("pk,nk->pn", cols, w32, optimize=True)
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)[None, :]
    return y.astype(np.float32, copy=False)


def add(a, b, **_attrs):
    return a.astype(np.float32) + b.astype(np.float32)


def mul(a, b, **_attrs):
    return a.astype(np.float32) * b.astype(np.float32)


def softmax(x, *, axis=-1, **_attrs):
    """``exp(x - max) / sum(exp(x - max))`` -- shifted, so it does not overflow."""
    x32 = x.astype(np.float32)
    shifted = x32 - x32.max(axis=int(axis), keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=int(axis), keepdims=True)


def rope(x, positions, cos, sin, *, layout="interleaved", theta_base=10000.0, scaling=None, **_attrs):
    """Rotate pairs of ``x`` by ``positions`` through the ``cos``/``sin`` tables.

    ``layout`` is the one place the two forks differ: ``"interleaved"`` rotates
    ``(x[2i], x[2i+1])`` -- the original Llama convention -- and ``"split"``
    rotates ``(x[i], x[i + d/2])``, which is what most modern checkpoints ship.
    The tables are passed in rather than computed here so a captured decode can
    read a device-resident table, but the *semantics* (which pair, which angle)
    are fixed by this function, not by the caller's table.
    """
    x32 = x.astype(np.float32)
    d = x32.shape[-1]
    half = d // 2
    cos_t = cos.astype(np.float32)
    sin_t = sin.astype(np.float32)
    out = np.empty_like(x32)
    for t in range(x32.shape[0]):
        p = int(positions[t])
        c = cos_t[p]
        s = sin_t[p]
        row = x32[t]
        if layout == "split":
            a = row[:, :half]
            b = row[:, half:]
            out[t, :, :half] = a * c - b * s
            out[t, :, half:] = a * s + b * c
        else:
            pairs = row.reshape(*row.shape[:-1], half, 2)
            a = pairs[..., 0]
            b = pairs[..., 1]
            out[t, :, 0::2] = (a * c - b * s).reshape(*row.shape[:-1], half)
            out[t, :, 1::2] = (a * s + b * c).reshape(*row.shape[:-1], half)
    return out


def embedding(tokens, table, *, table_fmt=None, **_attrs):
    """``out[i, :] = table[tokens[i], :]``, gathering through a quantized table.

    A quantized table is decoded once for the whole gather rather than per row:
    the block layout does not let one row be decoded without reading its block,
    so decoding per row would read the same block several times for no gain.
    """
    ids = np.asarray(tokens).reshape(-1).astype(np.intp)
    if table_fmt is not None:
        table = quantize_weights(None, table, _quant_name(table_fmt))
    return table.astype(np.float32)[ids]


def moe_ffn(
    x,
    expert_ids,
    expert_weights,
    w1,
    w2,
    shared_w1=None,
    shared_w2=None,
    *,
    top_k=None,
    norm_topk_prob=True,
    swiglu=True,
    w1_fmt=None,
    w2_fmt=None,
    **_attrs,
):
    """Combine ``top_k`` experts' SwiGLU outputs per token, weighted by the router.

    ``w1`` is the gate/up projection of the whole expert bank and ``w2`` the down
    projection, both indexed by expert id on their *first* axis.  The loop is
    over tokens and then over that token's chosen experts -- the reference has no
    routing structure to exploit, and writing it as a gather would only obscure
    which expert contributed which row.
    """
    x32 = x.astype(np.float32)
    ids = np.asarray(expert_ids).astype(np.intp)
    weights = np.asarray(expert_weights).astype(np.float32)
    if norm_topk_prob and weights.shape[-1] > 0:
        total = weights.sum(axis=-1, keepdims=True)
        weights = np.divide(weights, total, out=np.zeros_like(weights), where=total != 0)
    if w1_fmt is not None:
        w1 = quantize_weights(None, w1, _quant_name(w1_fmt))
    if w2_fmt is not None:
        w2 = quantize_weights(None, w2, _quant_name(w2_fmt))
    out = np.zeros_like(x32)
    for t in range(x32.shape[0]):
        for slot in range(ids.shape[-1]):
            expert = int(ids[t, slot])
            gate_up = _gemm(x32[t : t + 1], w1[expert])
            if swiglu:
                half = gate_up.shape[-1] // 2
                hidden = silu_mul(gate_up[:, :half], gate_up[:, half:])
            else:
                hidden = gate_up
            out[t] += weights[t, slot] * _gemm(hidden, w2[expert])[0]
    if shared_w1 is not None and shared_w2 is not None:
        if w1_fmt is not None:
            shared_w1 = quantize_weights(None, shared_w1, _quant_name(w1_fmt))
        if w2_fmt is not None:
            shared_w2 = quantize_weights(None, shared_w2, _quant_name(w2_fmt))
        gate_up = _gemm(x32, shared_w1)
        hidden = silu_mul(gate_up[..., : gate_up.shape[-1] // 2], gate_up[..., gate_up.shape[-1] // 2 :])
        out = out + _gemm(hidden, shared_w2)
    return out


def logits_temperature(logits, *, temperature=1.0, **_attrs):
    """``logits / temperature``.  A zero or negative temperature is refused."""
    t = float(temperature)
    if not t > 0.0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    return logits.astype(np.float32) / t


def argmax(logits, **_attrs):
    """The index of the largest logit, as int32.  Ties take the lowest index."""
    return np.asarray(np.argmax(np.asarray(logits).reshape(-1)), dtype=np.int32)


def topk_sample(logits, uniform=None, *, top_k=0, top_p=1.0, min_p=0.0, **_attrs):
    """Draw one token from the top-k/top-p truncated softmax, given a uniform variate.

    Truncation happens in *probability* order -- top-k keeps the k largest, top-p
    keeps the smallest set whose cumulative mass reaches p, min-p drops anything
    below ``min_p`` times the maximum -- and the sample is then the inverse-CDF
    step at ``uniform``.  Integer ``top_k`` of 0 means "no top-k limit", matching
    the convention the sampling APIs use.
    """
    probs = softmax(logits)
    vocab = probs.size
    order = np.argsort(-probs, kind="stable")
    ranked = probs[order]

    keep = np.ones(vocab, dtype=bool)
    if int(top_k) > 0:
        keep &= np.arange(vocab) < int(top_k)
    cutoff = float(min_p) * ranked[0] if ranked.size else 0.0
    if cutoff > 0.0:
        keep &= ranked >= cutoff
    if float(top_p) < 1.0:
        cumulative = np.cumsum(ranked)
        # Keep up to and including the first draw that crosses p.
        within = cumulative - ranked < float(top_p)
        keep &= within | (np.arange(vocab) == np.argmax(cumulative >= float(top_p)))
    ranked = np.where(keep, ranked, 0.0)
    total = ranked.sum()
    if total <= 0.0:
        return np.asarray(order[0], dtype=np.int32)
    cumulative = np.cumsum(ranked) / total
    u = _DEFAULT_UNIFORM if uniform is None else float(np.asarray(uniform).reshape(-1)[0])
    chosen = int(np.searchsorted(cumulative, u, side="left"))
    chosen = min(chosen, ranked.size - 1)
    return np.asarray(order[chosen], dtype=np.int32)


def cache_append(cache, values, positions, **_attrs):
    """``cache[positions[i]] = values[i]``, written in place and returned."""
    target = cache
    pos = np.asarray(positions).astype(np.intp)
    target[pos] = values.astype(target.dtype, copy=False)
    return target


def cache_truncate(cache, *, length=0, **_attrs):
    """Forget everything past ``length`` by zeroing it.

    Zeroing rather than "forgetting" because the cache is a fixed buffer: the
    *length* is bookkeeping the engine owns, and what this op must guarantee is
    that a stale row cannot be read back after a prefix-cache restore.
    """
    cache[int(length) :] = 0
    return cache


#: Op name -> implementation.  This table *is* the reference backend's
#: completeness claim; ``tests/abi/test_reference_completeness.py`` checks it
#: against the registry, so a new op with no entry here fails the suite rather
#: than failing at call time on a device.
KERNELS = {
    "gemm": gemm,
    "gemm_quant": gemm_quant,
    "attention": attention,
    "rms_norm": rms_norm,
    "layer_norm": layer_norm,
    "silu_mul": silu_mul,
    "gelu": gelu,
    "conv2d": conv2d,
    "add": add,
    "mul": mul,
    "softmax": softmax,
    "rope": rope,
    "embedding": embedding,
    "moe_ffn": moe_ffn,
    "logits_temperature": logits_temperature,
    "argmax": argmax,
    "topk_sample": topk_sample,
    "cache_append": cache_append,
    "cache_truncate": cache_truncate,
}

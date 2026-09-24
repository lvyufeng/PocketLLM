#!/usr/bin/env python
"""Where a decode step's 1.2 ms of attention goes, statement by statement.

`probe_mimo_v2_host_phases.py` wraps `MimoV2DeviceAttention.forward` and reports 57 to 59 ms a
token for it -- 48 calls at about 1.2 ms each. That is not a cost the arithmetic explains: a decode
step is one row against a span of at most `FOLD_KEYS` keys, a few tens of thousands of elements,
and a 4096-wide elementwise add on this box costs 10 us of host time. Forty-five of those is 450 us,
so about 700 us a layer is something other than the kernels: a stall, a collective, or a
`perf_counter`'s worth of work nobody has attributed yet.

So this probe attributes it. `decode_output` is re-implemented with a `perf_counter` between every
statement -- a copy, which is the risk of the probe, and what makes the copy readable is that its
own total is printed next to the shipped method's, measured in the same window. If the two are not
within a few percent the body has drifted and the split below is not read.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_attention_path.py --steps 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.bank import open_expert_bank  # noqa: E402
from src.models.mimo_v2.device_attention import (  # noqa: E402
    AttentionStats,
    MimoV2DeviceAttention,
    rope_rows,
)
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.ep import EpGroup  # noqa: E402
from src.models.mimo_v2.layers import split_fused_qkv  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]

#: `(name, what it covers)`, in the order one call pays them.
REGIONS = (
    "cos/sin",
    "qkv",
    "split",
    "rope q",
    "rope k",
    "value scale",
    "cast",
    "unsqueeze",
    "append",
    "window",
    "scores",
    "softmax",
    "out",
    "view",
    "gather",
    "o_proj",
    "ALL",
)


def instrument(*, with_gather: bool = True) -> dict[str, list[float]]:
    """Point every attention module at a re-implementation of its decode path, and time it."""
    tally: dict[str, list[float]] = {name: [] for name in REGIONS}
    shipped_decode = MimoV2DeviceAttention.decode_output
    shipped_forward = MimoV2DeviceAttention.forward

    def decode_output(self, flat, *, start_pos, cache):  # noqa: ANN001
        began_all = time.perf_counter()
        shape = self.shape
        heads, kv_heads = shape.num_q_heads, shape.num_kv_heads
        head_dim, v_head_dim = shape.head_dim, shape.v_head_dim
        groups = heads // kv_heads

        now = time.perf_counter()
        cos = self._rope_table[0][start_pos : start_pos + 1]
        sin = self._rope_table[1][start_pos : start_pos + 1]
        took = time.perf_counter()
        tally["cos/sin"].append(took - now)

        qkv = torch.nn.functional.linear(flat, self.qkv_proj)
        now = time.perf_counter()
        tally["qkv"].append(now - took)

        if self._qkv_order is None:
            query, key, value = split_fused_qkv(qkv, shape, shape.qkv_row_layout)
        else:
            query, key, value = qkv.index_select(-1, self._qkv_order).split(
                [shape.q_size, shape.k_size, shape.v_size], dim=-1
            )
        took = time.perf_counter()
        tally["split"].append(took - now)

        query = rope_rows(query.view(heads, head_dim), cos, sin, shape.rope_dim)
        now = time.perf_counter()
        tally["rope q"].append(now - took)

        key = rope_rows(key.view(kv_heads, head_dim), cos, sin, shape.rope_dim)
        took = time.perf_counter()
        tally["rope k"].append(took - now)

        value = value.view(kv_heads, v_head_dim)
        if shape.value_scale is not None:
            value = value * shape.value_scale
        now = time.perf_counter()
        tally["value scale"].append(now - took)

        if cache.dtype != key.dtype:
            key, value = key.to(cache.dtype), value.to(cache.dtype)
        else:
            key, value = key.to(self.dtype), value.to(self.dtype)
        took = time.perf_counter()
        tally["cast"].append(took - now)

        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        now = time.perf_counter()
        tally["unsqueeze"].append(now - took)

        span = cache.append_and_span(self.layer_idx, key, value, start_pos=start_pos)
        if span is not None:
            all_key, all_value = span
            prefix_len = start_pos
        else:
            prefix_key, prefix_value, prefix_len = cache.prefix(self.layer_idx, start_pos)
            if prefix_len:
                all_key = torch.cat([prefix_key, key], dim=1)
                all_value = torch.cat([prefix_value, value], dim=1)
            else:
                all_key, all_value = key, value
            cache.append(self.layer_idx, key, value)
        took = time.perf_counter()
        tally["append"].append(took - now)

        if shape.sliding_window is not None:
            lower = max(0, prefix_len - int(shape.sliding_window) + 1)
            if lower:
                all_key = all_key[:, lower:]
                all_value = all_value[:, lower:]
        now = time.perf_counter()
        tally["window"].append(now - took)

        scores = (
            torch.matmul(
                query.to(torch.float32).view(kv_heads, groups, 1, head_dim),
                all_key.to(torch.float32).transpose(1, 2).unsqueeze(1),
            )
            * shape.scaling
        )
        took = time.perf_counter()
        tally["scores"].append(took - now)

        column = (
            None
            if self.sink is None
            else self.sink.view(kv_heads, groups, 1, 1).to(torch.float32)
        )
        running_max = (
            scores.amax(dim=-1, keepdim=True)
            if column is None
            else torch.maximum(scores.amax(dim=-1, keepdim=True), column)
        )
        probabilities = torch.exp(scores - running_max)
        denominator = probabilities.sum(dim=-1, keepdim=True)
        if column is not None:
            denominator = denominator + torch.exp(column - running_max)
        probabilities = probabilities / denominator
        now = time.perf_counter()
        tally["softmax"].append(now - took)

        out = torch.matmul(probabilities, all_value.to(torch.float32).unsqueeze(1))
        took = time.perf_counter()
        tally["out"].append(took - now)

        keys = all_key.shape[1]
        self.last_stats = AttentionStats("decode", 1, heads * keys, heads * keys)
        answer = out.view(1, shape.o_in).to(query.dtype)
        now = time.perf_counter()
        tally["view"].append(now - took)
        tally["ALL"].append(now - began_all)
        return answer, qkv

    def forward(self, hidden_states, *, start_pos=0, positions=None, cache=None):  # noqa: ANN001
        pre_o, qkv = self.attention_output(
            hidden_states, start_pos=start_pos, positions=positions, cache=cache
        )
        began = time.perf_counter()
        if self.gather is not None:
            pre_o = self.gather(pre_o)
        took = time.perf_counter()
        tally["gather"].append(took - began)
        post_o = torch.nn.functional.linear(pre_o.to(self.o_proj.dtype), self.o_proj)
        tally["o_proj"].append(time.perf_counter() - took)
        return {"qkv_raw": qkv, "attn_out_pre_o": pre_o, "attn_out_post_o": post_o}

    MimoV2DeviceAttention.decode_output = decode_output
    if with_gather:
        MimoV2DeviceAttention.forward = forward
    del shipped_decode, shipped_forward
    return tally


def arm(model, cache, position, steps, warmup, logits):
    step_position = position
    for _ in range(warmup):
        logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
        step_position += 1
    torch.cuda.synchronize()
    host = tail = 0.0
    for _ in range(steps):
        began = time.perf_counter()
        logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
        queued = time.perf_counter()
        torch.cuda.synchronize()
        host += (queued - began) * 1e3
        tail += (time.perf_counter() - queued) * 1e3
        step_position += 1
    return host / steps, tail / steps, step_position, logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--resident-rows", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    ep = EpGroup.from_env()
    world, rank = ep.world, ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint, device=device, expert_source=bank, ep=ep, resident_rows=args.resident_rows
    )
    torch.cuda.synchronize()

    cache = model.cache(max(args.prompt, 8) + 2 * (args.warmup + args.steps) + 16)
    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
    model.greedy(prompt_ids, max_tokens=1, cache=cache)
    cache.reset()
    logits = None
    for position, token in enumerate(prompt_ids):
        logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
    position = len(prompt_ids)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    host, tail, position, logits = arm(model, cache, position, args.steps, args.warmup, logits)
    shipped_ms = host + tail
    print(f"[r{rank}] shipped: {shipped_ms:7.1f} ms a token", flush=True)

    tally = instrument()
    host, tail, position, logits = arm(model, cache, position, args.steps, args.warmup, logits)
    copied_ms = host + tail
    print(
        f"[r{rank}] instrumented: {copied_ms:7.1f} ms a token ({copied_ms - shipped_ms:+.1f})",
        flush=True,
    )

    seen = args.steps + args.warmup
    layers = sum(1 for layer in model.layers)
    for name in REGIONS:
        got = tally[name]
        if not got:
            continue
        per_step = sum(got) / seen * 1e3
        print(
            f"[r{rank}]   {name:12s} {per_step:7.1f} ms a token  {len(got) / seen:5.1f} calls  "
            f"max {max(got) * 1e3:7.3f} ms",
            flush=True,
        )
    print(f"[r{rank}] {layers} layers", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

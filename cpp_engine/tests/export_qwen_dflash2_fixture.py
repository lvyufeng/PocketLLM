#!/usr/bin/env python3
"""Export real-checkpoint DFlash2 stage references from captured target taps.

The C++ parity executable first writes native Qwen target taps. This script uses
those exact activations with upstream DFlash2 weights, avoiding a false mismatch
between the native FP8 target and a separately loaded Python target. It emits:

* an upstream BF16 reference using the checkpoint's native weight precision;
* an SM75 reference whose weights/activations follow the C++ FP16 boundaries and
  whose residual/down/MLP-finish tensors remain FP32 to avoid overflow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from qwen_dflash2_tensor_file import TensorFile, read_tensor_file, write_tensor_file


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)
    return tensors


def load_target_weight(checkpoint: Path, tensor_name: str) -> torch.Tensor:
    with open(checkpoint / "model.safetensors.index.json", encoding="utf-8") as input_file:
        index = json.load(input_file)
    shard_name = index["weight_map"][tensor_name]
    with safe_open(checkpoint / shard_name, framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name)


def f16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float16)


def f32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float32)


def direct_rmsnorm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    variance = hidden.float().square().mean(dim=-1, keepdim=True)
    normalized = hidden.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(output_dtype)


def linear(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
    mode: str,
) -> torch.Tensor:
    if mode == "sm75":
        # SM75 cuBLAS consumes FP16 inputs/weights with FP32 accumulation. CPU
        # PyTorch cannot execute FP16 GEMM with that contract, so accumulate an
        # FP32 product from values already rounded to FP16, then round the output.
        return F.linear(hidden.float(), weight.float()).to(output_dtype)
    return F.linear(hidden, weight).to(output_dtype)


def grouped_dynamic_conv(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    rows, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(rows, groups, group_size).float()
    dynamic = dynamic.reshape(rows, base.shape[0], groups, 1).float()
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(
            blocks[:-offset], (0, 0, 0, 0, offset, 0)
        )
        kernel = base[offset].reshape(1, groups, group_size).float()
        output = output + values * (kernel + dynamic[:, offset])
    return output.reshape(rows, hidden_size).to(output_dtype)


def apply_rope(
    hidden: torch.Tensor,
    rows: int,
    heads: int,
    head_dim: int,
    start_position: int,
    theta: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    values = hidden.reshape(rows, heads, head_dim).float()
    half = head_dim // 2
    frequency = theta ** (
        -2.0 * torch.arange(half, dtype=torch.float32) / float(head_dim)
    )
    positions = torch.arange(
        start_position, start_position + rows, dtype=torch.float32
    )
    angles = positions[:, None] * frequency[None]
    cosine = torch.cos(angles)[:, None]
    sine = torch.sin(angles)[:, None]
    left = values[..., :half]
    right = values[..., half:]
    rotated = torch.cat(
        [left * cosine - right * sine, right * cosine + left * sine], dim=-1
    )
    return rotated.reshape(rows, heads * head_dim).to(output_dtype)


def noncausal_attention(
    query: torch.Tensor,
    context_k: torch.Tensor,
    context_v: torch.Tensor,
    block_k: torch.Tensor,
    block_v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    context_len: int,
    sliding_window: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    rows = query.shape[0]
    q = query.reshape(rows, q_heads, head_dim).float()
    ck = context_k.reshape(context_len, kv_heads, head_dim).float()
    cv = context_v.reshape(context_len, kv_heads, head_dim).float()
    bk = block_k.reshape(rows, kv_heads, head_dim).float()
    bv = block_v.reshape(rows, kv_heads, head_dim).float()
    group = q_heads // kv_heads
    output = torch.empty_like(q)
    scale = head_dim**-0.5
    context_positions = torch.arange(context_len)
    block_positions = context_len + torch.arange(rows)
    key_positions = torch.cat([context_positions, block_positions])
    for row in range(rows):
        query_position = context_len + row
        visible = (query_position - key_positions).abs() < sliding_window
        for head in range(q_heads):
            kv_head = head // group
            keys = torch.cat([ck[:, kv_head], bk[:, kv_head]], dim=0)[visible]
            values = torch.cat([cv[:, kv_head], bv[:, kv_head]], dim=0)[visible]
            scores = (keys * q[row, head]).sum(-1) * scale
            output[row, head] = torch.softmax(scores, dim=-1) @ values
    return output.reshape(rows, q_heads * head_dim).to(output_dtype)


def numpy(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy().astype(dtype, copy=False)


class Recorder:
    def __init__(self) -> None:
        self.tensors: OrderedDict[str, np.ndarray] = OrderedDict()

    def half(self, name: str, tensor: torch.Tensor) -> None:
        self.tensors[name] = numpy(tensor, np.dtype("<f2"))

    def float(self, name: str, tensor: torch.Tensor) -> None:
        self.tensors[name] = numpy(tensor, np.dtype("<f4"))

    def integer(self, name: str, tensor: torch.Tensor | np.ndarray) -> None:
        self.tensors[name] = np.asarray(tensor, dtype="<i4")


class DFlashReference:
    def __init__(
        self,
        config: dict,
        weights: dict[str, torch.Tensor],
        embedding: torch.Tensor,
        lm_head: torch.Tensor,
        mode: str,
        vocab_start: int,
    ) -> None:
        draft = config["dflash_config"]
        self.config = config
        self.weights = weights
        self.embedding = embedding
        self.lm_head = lm_head
        self.residual_dtype = torch.float32
        self.mode = mode
        self.vocab_start = vocab_start
        self.rows = int(draft["block_size"])
        self.hidden = int(config["hidden_size"])
        self.intermediate = int(config["intermediate_size"])
        self.q_heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.q_dim = self.q_heads * self.head_dim
        self.kv_dim = self.kv_heads * self.head_dim
        self.group_size = int(draft["conv_group_size"])
        self.groups = self.hidden // self.group_size
        self.kernel_size = int(draft["conv_kernel_size"])
        self.dynamic_taps = self.kernel_size * self.groups
        self.dynamic_stride = 2 * self.dynamic_taps
        self.top_k = int(draft["selector_top_k"])
        self.selector_rank = int(draft["selector_rank"])
        self.mask_token = int(draft["mask_token_id"])
        self.eps = float(config["rms_norm_eps"])
        rope = config.get("rope_parameters", {})
        self.theta = float(rope.get("rope_theta", config.get("rope_theta", 1e7)))
        self.sliding_window = int(config["sliding_window"])
        if mode == "sm75":
            self.weights = {name: f16(value) for name, value in weights.items()}
            self.embedding = f16(embedding)
            self.lm_head = f16(lm_head)
            self.activation_dtype = torch.float16
            self.residual_dtype = torch.float32
        elif mode == "bf16":
            self.activation_dtype = torch.bfloat16
            self.residual_dtype = torch.bfloat16
            self.weights = {name: value.to(torch.bfloat16) for name, value in weights.items()}
            self.embedding = embedding.to(torch.bfloat16)
            self.lm_head = lm_head.to(torch.bfloat16)
        else:
            raise ValueError(f"unknown reference mode: {mode}")

    def weight(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def build(
        self,
        target_taps: torch.Tensor,
        position_offset: int,
        anchor_token: int,
    ) -> Recorder:
        recorder = Recorder()
        context_rows = target_taps.shape[0]
        target_taps = target_taps.to(self.activation_dtype)
        recorder.half("target_taps", target_taps)
        projected = linear(
            target_taps, self.weight("fc.weight"), self.activation_dtype, self.mode
        )
        recorder.half("context.projected", projected)
        normalized_context = direct_rmsnorm(
            projected,
            self.weight("hidden_norm.weight"),
            self.eps,
            self.activation_dtype,
        )
        recorder.half("context.normalized", normalized_context)

        context_keys: list[torch.Tensor] = []
        context_values: list[torch.Tensor] = []
        for layer_index in range(int(self.config["num_hidden_layers"])):
            prefix = f"layers.{layer_index}.self_attn."
            record_prefix = f"layer.{layer_index}.context."
            k_projection = linear(
                normalized_context,
                self.weight(prefix + "k_proj.weight"),
                self.activation_dtype,
                self.mode,
            )
            recorder.half(record_prefix + "k_projection", k_projection)
            value = linear(
                normalized_context,
                self.weight(prefix + "v_proj.weight"),
                self.activation_dtype,
                self.mode,
            )
            recorder.half(record_prefix + "v", value)
            k_norm = direct_rmsnorm(
                k_projection.reshape(context_rows, self.kv_heads, self.head_dim),
                self.weight(prefix + "k_norm.weight"),
                self.eps,
                self.activation_dtype,
            ).reshape(context_rows, self.kv_dim)
            recorder.half(record_prefix + "k_norm", k_norm)
            k_rope = apply_rope(
                k_norm,
                context_rows,
                self.kv_heads,
                self.head_dim,
                position_offset,
                self.theta,
                self.activation_dtype,
            )
            recorder.half(record_prefix + "k_rope", k_rope)
            context_keys.append(k_rope)
            context_values.append(value)

        tokens = torch.full((self.rows,), self.mask_token, dtype=torch.long)
        tokens[0] = anchor_token
        recorder.integer("noise.tokens", tokens.numpy())
        residual = F.embedding(tokens, self.embedding).to(self.activation_dtype)
        recorder.half("noise.embedding", residual)
        residual = residual.to(self.residual_dtype)
        recorder.float("residual.initial", residual)

        for layer_index in range(int(self.config["num_hidden_layers"])):
            layer = f"layers.{layer_index}."
            attn = layer + "self_attn."
            record = f"layer.{layer_index}."
            normalized = direct_rmsnorm(
                residual,
                self.weight(layer + "input_layernorm.weight"),
                self.eps,
                self.activation_dtype,
            )
            recorder.half(record + "input_norm", normalized)
            dynamic = linear(
                normalized,
                self.weight(layer + "attention_conv.kernel_projection.weight"),
                self.activation_dtype,
                self.mode,
            )
            recorder.half(record + "attention.dynamic", dynamic)
            prepared = grouped_dynamic_conv(
                normalized,
                dynamic[:, : self.dynamic_taps],
                self.weight(layer + "attention_conv.base_kernel")[0],
                self.group_size,
                self.activation_dtype,
            )
            recorder.half(record + "attention.prepare", prepared)
            q_projection = linear(
                prepared, self.weight(attn + "q_proj.weight"), self.activation_dtype, self.mode
            )
            recorder.half(record + "attention.q_projection", q_projection)
            k_projection = linear(
                prepared, self.weight(attn + "k_proj.weight"), self.activation_dtype, self.mode
            )
            recorder.half(record + "attention.k_projection", k_projection)
            value = linear(
                prepared, self.weight(attn + "v_proj.weight"), self.activation_dtype, self.mode
            )
            recorder.half(record + "attention.v", value)
            q_norm = direct_rmsnorm(
                q_projection.reshape(self.rows, self.q_heads, self.head_dim),
                self.weight(attn + "q_norm.weight"),
                self.eps,
                self.activation_dtype,
            ).reshape(self.rows, self.q_dim)
            recorder.half(record + "attention.q_norm", q_norm)
            k_norm = direct_rmsnorm(
                k_projection.reshape(self.rows, self.kv_heads, self.head_dim),
                self.weight(attn + "k_norm.weight"),
                self.eps,
                self.activation_dtype,
            ).reshape(self.rows, self.kv_dim)
            recorder.half(record + "attention.k_norm", k_norm)
            q_rope = apply_rope(
                q_norm,
                self.rows,
                self.q_heads,
                self.head_dim,
                context_rows,
                self.theta,
                self.activation_dtype,
            )
            k_rope = apply_rope(
                k_norm,
                self.rows,
                self.kv_heads,
                self.head_dim,
                context_rows,
                self.theta,
                self.activation_dtype,
            )
            recorder.half(record + "attention.q_rope", q_rope)
            recorder.half(record + "attention.k_rope", k_rope)
            attention = noncausal_attention(
                q_rope,
                context_keys[layer_index],
                context_values[layer_index],
                k_rope,
                value,
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                context_rows,
                self.sliding_window,
                self.activation_dtype,
            )
            recorder.half(record + "attention.output", attention)
            output = linear(
                attention, self.weight(attn + "o_proj.weight"), self.activation_dtype, self.mode
            )
            recorder.half(record + "attention.o_projection", output)
            attention_finish = grouped_dynamic_conv(
                output,
                dynamic[:, self.dynamic_taps :],
                self.weight(layer + "attention_conv.base_kernel")[1],
                self.group_size,
                self.activation_dtype,
            )
            recorder.half(record + "attention.finish", attention_finish)
            branch = attention_finish if self.mode == "bf16" else attention_finish.float()
            recorder.float(record + "attention.branch_f32", branch)
            residual = residual + branch
            recorder.float(record + "attention.residual", residual)
            normalized = direct_rmsnorm(
                residual,
                self.weight(layer + "post_attention_layernorm.weight"),
                self.eps,
                self.activation_dtype,
            )
            recorder.half(record + "post_norm", normalized)
            dynamic = linear(
                normalized,
                self.weight(layer + "mlp_conv.kernel_projection.weight"),
                self.activation_dtype,
                self.mode,
            )
            recorder.half(record + "mlp.dynamic", dynamic)
            prepared = grouped_dynamic_conv(
                normalized,
                dynamic[:, : self.dynamic_taps],
                self.weight(layer + "mlp_conv.base_kernel")[0],
                self.group_size,
                self.activation_dtype,
            )
            recorder.half(record + "mlp.prepare", prepared)
            gate = linear(
                prepared,
                self.weight(layer + "mlp.gate_proj.weight"),
                self.activation_dtype,
                self.mode,
            )
            recorder.half(record + "mlp.gate", gate)
            up = linear(
                prepared,
                self.weight(layer + "mlp.up_proj.weight"),
                self.activation_dtype,
                self.mode,
            )
            recorder.half(record + "mlp.up", up)
            swiglu = (F.silu(gate.float()) * up.float()).to(self.activation_dtype)
            recorder.half(record + "mlp.swiglu", swiglu)
            down = linear(
                swiglu,
                self.weight(layer + "mlp.down_proj.weight"),
                torch.float32,
                self.mode,
            )
            recorder.float(record + "mlp.down", down)
            finish = grouped_dynamic_conv(
                down,
                dynamic[:, self.dynamic_taps :],
                self.weight(layer + "mlp_conv.base_kernel")[1],
                self.group_size,
                torch.float32,
            )
            recorder.float(record + "mlp.finish", finish)
            residual = residual + finish
            recorder.float(record + "residual", residual)

        final_norm = direct_rmsnorm(
            residual,
            self.weight("norm.weight"),
            self.eps,
            self.activation_dtype,
        )
        recorder.float("final.residual", residual)
        recorder.half("final.norm", final_norm)
        draft_hidden = final_norm[1:]
        logits = linear(draft_hidden, self.lm_head, torch.float32, self.mode)
        recorder.float("logits.local", logits)
        local_logits, local_tokens = torch.topk(
            logits, self.top_k, dim=-1, sorted=True
        )
        local_tokens = local_tokens + self.vocab_start
        recorder.integer("topk.local.tokens", local_tokens.numpy())
        recorder.float("topk.local.logits", local_logits)
        if self.lm_head.shape[0] != int(self.config["vocab_size"]):
            return recorder
        recorder.integer("topk.global.tokens", local_tokens.numpy())
        recorder.float("topk.global.logits", local_logits)

        selector_hidden = linear(
            draft_hidden,
            self.weight("candidate_selector.hidden_projection.weight"),
            torch.float32,
            self.mode,
        )
        predecessor_table = self.weight("candidate_selector.predecessor_codebook")
        successor_table = self.weight("candidate_selector.successor_codebook")
        predecessor = anchor_token
        path: list[int] = []
        for row in range(self.rows - 1):
            candidates = local_tokens[row]
            scores = local_logits[row].float() + (
                predecessor_table[predecessor].float()
                * selector_hidden[row].float()
                * successor_table[candidates].float()
            ).sum(-1)
            best_value = scores.max()
            best_tokens = candidates[scores == best_value]
            predecessor = int(best_tokens.min())
            path.append(predecessor)
        recorder.integer("selector.path", np.asarray(path, dtype="<i4"))
        return recorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-checkpoint", required=True)
    parser.add_argument("--target-checkpoint", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--bf16-out", required=True)
    parser.add_argument("--sm75-out", required=True)
    parser.add_argument("--tp-world", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    args = parser.parse_args()
    if args.tp_world <= 0 or not 0 <= args.tp_rank < args.tp_world:
        parser.error("invalid TP rank/world")

    draft_checkpoint = Path(args.draft_checkpoint)
    target_checkpoint = Path(args.target_checkpoint)
    capture = read_tensor_file(args.capture)
    with open(draft_checkpoint / "config.json", encoding="utf-8") as input_file:
        config = json.load(input_file)
    weights = load_safetensors(draft_checkpoint / "model.safetensors")
    embedding = load_target_weight(
        target_checkpoint, "model.language_model.embed_tokens.weight"
    )
    lm_head = load_target_weight(target_checkpoint, "lm_head.weight")
    vocab_per_rank = lm_head.shape[0] // args.tp_world
    vocab_start = args.tp_rank * vocab_per_rank
    lm_head = lm_head[vocab_start : vocab_start + vocab_per_rank]
    target_taps = torch.from_numpy(capture.tensors["target_taps"].copy())

    for mode, output_path in (("bf16", args.bf16_out), ("sm75", args.sm75_out)):
        model = DFlashReference(
            config, weights, embedding, lm_head, mode, vocab_start
        )
        recorder = model.build(
            target_taps, capture.position_offset, capture.anchor_token
        )
        write_tensor_file(
            output_path,
            TensorFile(
                capture.position_offset,
                capture.anchor_token,
                recorder.tensors,
            ),
        )
        print(f"[fixture] mode={mode} stages={len(recorder.tensors)} wrote={output_path}")


if __name__ == "__main__":
    main()

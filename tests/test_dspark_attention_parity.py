"""Numerical parity check: cpp_engine's DSparkAttention vs the PyTorch reference.

Everything before this test only proved that DSpark weights load and that the
C++ compiles. Attention is where the draft can be silently wrong: its KV comes
from the main model's hidden rather than its own input, it reads a ring cache,
and it applies an inverse rope on the way out. Each of those produces
finite, plausible-looking numbers when implemented wrong.

The test drives both sides with the same random x / main_x / ring cache and the
same real stage weights, then compares outputs:

    python tests/test_dspark_attention_parity.py /mnt/data3/DeepSeek-V4-Flash-0731

Run under the `deepseek` conda env (the FP8 dequant path needs its torch build).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.models.deepseek_v4.runtime as rt  # noqa: E402
from src.models.deepseek_v4.dspark import DSparkAttention  # noqa: E402
from src.models.deepseek_v4.runtime import ModelArgs  # noqa: E402

MAGIC = 0x44534B41  # "DSKA"
CPP_BIN = os.path.join(os.path.dirname(__file__), "..", "cpp_engine", "build",
                       "tests", "test_dspark_attention_parity")


def build_args(cfg: dict) -> ModelArgs:
    """ModelArgs for a single uncompressed DSpark stage."""
    n_layers = cfg["num_hidden_layers"]
    args = ModelArgs()
    args.max_batch_size = 1
    args.max_seq_len = 8192
    args.dtype = "fp8"
    args.scale_fmt = "ue8m0"
    args.scale_dtype = "fp8"
    args.vocab_size = cfg["vocab_size"]
    args.dim = cfg["hidden_size"]
    args.n_layers = n_layers
    args.n_heads = cfg["num_attention_heads"]
    args.head_dim = cfg["head_dim"]
    args.rope_head_dim = cfg["qk_rope_head_dim"]
    args.q_lora_rank = cfg["q_lora_rank"]
    args.o_lora_rank = cfg["o_lora_rank"]
    args.o_groups = cfg["o_groups"]
    args.window_size = cfg["sliding_window"]
    args.norm_eps = cfg["rms_norm_eps"]
    args.rope_theta = cfg["rope_theta"]
    args.compress_rope_theta = cfg["compress_rope_theta"]
    args.original_seq_len = 0
    args.hc_mult = cfg["hc_mult"]
    args.hc_eps = cfg["hc_eps"]
    args.dspark_block_size = cfg["dspark_block_size"]
    args.dspark_noise_token_id = cfg["dspark_noise_token_id"]
    args.dspark_markov_rank = cfg["dspark_markov_rank"]
    args.dspark_target_layer_ids = tuple(cfg["dspark_target_layer_ids"])
    # The stage is layer n_layers + stage_id and is never a compressed layer.
    args.compress_ratios = [0] * (n_layers + 8)
    return args


def load_stage_attention(ckpt_dir: str, stage_id: int, args: ModelArgs, device):
    """Build a DSparkAttention and fill it from the checkpoint's mtp.N.attn.*

    Mirrors the two conversions the real loader does, because getting either
    wrong yields finite but wrong numbers:
      - wo_a's module parameter is bf16 while the checkpoint stores FP8 +
        blockwise scale, so it must be dequantized (and its [128,128] blocks
        un-shuffled) rather than copied raw.
      - attn_sink / the norms are stored in their target dtype already.
    """
    from safetensors import safe_open
    from src.kernels.ops import soft_fp8_blockfp8_weight_dequant

    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"mtp.{stage_id}.attn."
    wanted = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not wanted:
        raise SystemExit(f"no tensors under {prefix} in the index")

    # Globals the module reads at construction time.
    rt.default_dtype = torch.float8_e4m3fn
    rt.scale_fmt = "ue8m0"
    rt.scale_dtype = torch.float8_e8m0fnu
    rt.tp_world_size = 1
    rt.tp_rank = 0

    torch.set_default_dtype(torch.bfloat16)
    attn = DSparkAttention(args.n_layers + stage_id, args).to(device)

    tensors = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in wanted.items():
        by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
            for name in names:
                tensors[name[len(prefix):]] = f.get_tensor(name).to(device)

    state = attn.state_dict()
    with torch.no_grad():
        for key, target in state.items():
            if key in ("kv_cache", "freqs_cis"):
                continue
            # The module names FP8 scales "<lin>.weight.scale"; the checkpoint
            # names them "<lin>.scale".
            src = key[:-len(".weight.scale")] + ".scale" if key.endswith(".weight.scale") else key
            if src not in tensors:
                raise SystemExit(f"checkpoint is missing {src} (for {key})")
            t = tensors[src]

            if key == "wo_a.weight" and t.dtype == torch.float8_e4m3fn and target.dtype != torch.float8_e4m3fn:
                t = soft_fp8_blockfp8_weight_dequant(t, tensors["wo_a.scale"])
                if tuple(t.shape) != tuple(target.shape):
                    t = t.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).flatten(2, 3).flatten(0, 1)
            if tuple(t.shape) != tuple(target.shape):
                raise SystemExit(
                    f"shape mismatch {key}: module {tuple(target.shape)} ckpt {tuple(t.shape)}")
            target.copy_(t.to(target.dtype))

    attn.load_state_dict(state, strict=False)
    return attn


def fp32_gold(ckpt_dir: str, stage_id: int, args: ModelArgs, x, main_x, kv_cache,
              start_pos: int, device, act_quant_kv: bool = False):
    """Dequantize every stage weight to fp32 and run the attention in plain
    torch, with no activation quantization anywhere.

    Needed because the two implementations quantize differently -- the reference
    act_quants activations to FP8 before each GEMM and keeps wo_a in bf16, while
    cpp_engine keeps wo_a in FP8 and only act_quants the KV. A raw allclose
    between them cannot distinguish "different rounding" from "wrong math", so
    both are measured against this instead.

    act_quant_kv=True reproduces the one activation quantization cpp_engine does
    do (the KV nope prefix), which turns the comparison into a sharp test of the
    control flow -- indices, ring slot, rope positions -- rather than a fuzzy
    one dominated by FP8 weight noise.
    """
    from safetensors import safe_open
    from src.kernels.ops import soft_fp8_blockfp8_weight_dequant

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"mtp.{stage_id}.attn."
    raw = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith(prefix):
            by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
            for name in names:
                raw[name[len(prefix):]] = f.get_tensor(name).to(device)

    def w(stem):
        return soft_fp8_blockfp8_weight_dequant(raw[stem + ".weight"], raw[stem + ".scale"]).float()

    wq_a, wq_b, wkv = w("wq_a"), w("wq_b"), w("wkv")
    wo_a, wo_b = w("wo_a"), w("wo_b")
    q_norm_g = raw["q_norm.weight"].float()
    kv_norm_g = raw["kv_norm.weight"].float()
    sink = raw["attn_sink"].float()

    heads, hd, rd = args.n_heads, args.head_dim, args.rope_head_dim
    win, bsz = args.window_size, args.dspark_block_size
    eps = args.norm_eps
    theta = args.rope_theta

    def rms(t, gamma=None, e=eps):
        y = t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + e)
        return y * gamma if gamma is not None else y

    def rope(t, pos, inverse=False):
        """Rotate the trailing rd dims of t in place-ish.

        t is [N, hd]; pos is [N] of absolute positions. Matches
        head_rmsnorm_rope_rows_kernel: angle = pos / theta**(pair/rd), applied
        to (even, odd) pairs of the last rd elements.
        """
        out = t.clone()
        pair = torch.arange(0, rd, 2, dtype=torch.float32, device=t.device)
        ang = pos.view(-1, 1).float() * theta ** (-pair / rd)   # [N, rd/2]
        c, s = torch.cos(ang), torch.sin(ang)
        if inverse:
            s = -s
        seg = out[:, hd - rd:]
        a, b = seg[:, 0::2].clone(), seg[:, 1::2].clone()
        seg[:, 0::2] = a * c - b * s
        seg[:, 1::2] = a * s + b * c
        return out

    def act_quant_nope(t):
        """cpp_engine's fp8_act_quant_dequant on the leading hd-rd columns.

        Matches fp8_act_quant_dequant_rows_strided_kernel exactly, which is not
        a true e4m3 grid: scale = exp2(round(log2(max(amax,1e-4)/448))), then
        q = clamp(rint(v/scale), -448, 448) * scale -- integer rounding, not
        FP8 mantissa rounding.
        """
        if not act_quant_kv:
            return t
        out = t.clone()
        n = hd - rd
        nope = out[:, :n].reshape(t.size(0), n // 64, 64)
        amax = nope.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        scale = torch.exp2(torch.round(torch.log2(amax / 448.0)))
        q = torch.round(nope / scale).clamp(-448.0, 448.0)
        out[:, :n] = (q * scale).reshape(t.size(0), n)
        return out

    x = x.to(device).float()
    main_x = main_x.to(device).float()
    cache = kv_cache.to(device).float().clone()
    draft_pos = torch.arange(start_pos + 1, start_pos + 1 + bsz, dtype=torch.float32, device=device)
    # One position per (token, head) row, in the [bsz, heads] row order.
    q_pos = draft_pos.view(bsz, 1).expand(bsz, heads).reshape(-1)

    # Q: wq_a -> q_norm -> wq_b -> per-head rmsnorm -> rope
    q = rms(x @ wq_a.T, q_norm_g) @ wq_b.T
    q = rms(q.reshape(bsz * heads, hd), e=eps)
    q = rope(q, q_pos).view(bsz, heads, hd)

    # Main KV -> ring slot start_pos % win
    mkv = rms(main_x.view(1, -1) @ wkv.T, kv_norm_g)
    mkv = act_quant_nope(rope(mkv, torch.tensor([float(start_pos)], device=device)))
    cache[start_pos % win] = mkv.view(hd)

    # Draft KV, never cached
    dkv = rms(x @ wkv.T, kv_norm_g)
    dkv = act_quant_nope(rope(dkv, draft_pos))

    kv = torch.cat([cache, dkv], dim=0)                       # [win+bsz, hd]
    committed = min(win, start_pos + 1)
    idx = torch.cat([torch.arange(committed, device=device),
                     win + torch.arange(bsz, device=device)])
    keys = kv[idx]                                            # [topk, hd]

    logits = torch.einsum("thd,kd->thk", q, keys) * (hd ** -0.5)   # [bsz, heads, topk]
    logits = torch.cat([logits, sink.view(1, heads, 1).expand(bsz, heads, 1)], dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., :-1]
    o = torch.einsum("thk,kd->thd", probs, keys)              # [bsz, heads, hd]
    o = rope(o.reshape(bsz * heads, hd), q_pos, inverse=True).view(bsz, heads, hd)

    groups, rank = args.o_groups, args.o_lora_rank
    gdim = heads * hd // groups
    o = o.reshape(bsz, groups, gdim)
    wo_a_g = wo_a.view(groups, rank, gdim)
    mid = torch.einsum("bgd,grd->bgr", o, wo_a_g).reshape(bsz, groups * rank)
    return (mid @ wo_b.T).cpu()


def run_case(a, args, cpp_bin, device, stage, start_pos, seed):
    """One (stage, start_pos) case. Returns (pytorch_err, cpp_err) vs fp32 gold."""
    torch.manual_seed(seed)
    bsz, dim, win, hd = args.dspark_block_size, args.dim, args.window_size, args.head_dim

    # Same inputs on both sides. Scale matches post-RMSNorm activations.
    x = torch.randn(bsz, dim, dtype=torch.float32, device="cpu")
    main_x = torch.randn(dim, dtype=torch.float32, device="cpu")
    kv_cache = torch.randn(win, hd, dtype=torch.float32, device="cpu") * 0.1

    # --- C++ side ---
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.bin")
        out_path = os.path.join(tmp, "out.bin")
        with open(in_path, "wb") as f:
            f.write(struct.pack("<7i", MAGIC, stage, start_pos, bsz, dim, win, hd))
            f.write(x.numpy().astype("<f4").tobytes())
            f.write(main_x.numpy().astype("<f4").tobytes())
            f.write(kv_cache.numpy().astype("<f4").tobytes())
        proc = subprocess.run([cpp_bin, a.ckpt_dir, in_path, out_path],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"cpp parity binary failed ({proc.returncode})")
        cpp_out = torch.from_numpy(np.fromfile(out_path, dtype="<f4")).view(bsz, dim)

    # --- PyTorch reference ---
    attn = load_stage_attention(a.ckpt_dir, stage, args, device)
    with torch.inference_mode():
        attn.kv_cache[:1].copy_(kv_cache.to(device).unsqueeze(0).to(attn.kv_cache.dtype))
        ref = attn(x.to(device).unsqueeze(0).to(torch.bfloat16), start_pos,
                   main_x.to(device).view(1, 1, dim).to(torch.bfloat16))
    ref = ref.squeeze(0).float().cpu()
    del attn
    torch.cuda.empty_cache()

    # --- fp32 gold ---
    with torch.inference_mode():
        gold = fp32_gold(a.ckpt_dir, stage, args, x, main_x, kv_cache,
                         start_pos, device).float()
        # Same gold, but reproducing the one activation quantization cpp_engine
        # performs. Weight quantization noise dominates the plain comparison;
        # this one isolates whether the control flow matches.
        gold_q = fp32_gold(a.ckpt_dir, stage, args, x, main_x, kv_cache,
                           start_pos, device, act_quant_kv=True).float()

    def report(name, got, target):
        err = ((got - target).norm() / target.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten().unsqueeze(0), target.flatten().unsqueeze(0)).item()
        print(f"    {name:16s} mean|x|={got.abs().mean():.5f} "
              f"rel_l2={err:.5f} cos={cos:.6f}")
        return err

    print(f"  stage={stage} start_pos={start_pos}  gold mean|x|={gold.abs().mean():.5f}")
    ref_err = report("pytorch/gold", ref, gold)
    cpp_err = report("cpp/gold", cpp_out, gold)
    cpp_q_err = report("cpp/gold+actq", cpp_out, gold_q)
    return ref_err, cpp_err, cpp_q_err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2])
    # 1 exercises the barely-filled window, 64 a partial one, 200 a wrapped ring
    # (200 % 128 = 72), which is where an off-by-one in the ring index shows up.
    ap.add_argument("--start-pos", type=int, nargs="+", default=[1, 64, 200])
    ap.add_argument("--seed", type=int, default=0)
    # The primary bar is cpp vs the gold that reproduces cpp's own activation
    # quantization: if the control flow (key indices, ring slot, rope positions)
    # matches, only FP8 *weight* rounding is left, which lands near 1e-2. The
    # pytorch/gold number is printed alongside as the noise reference, not as a
    # target -- the two implementations quantize differently on purpose.
    ap.add_argument("--max-rel", type=float, default=0.03)
    a = ap.parse_args()

    cpp_bin = os.path.abspath(CPP_BIN)
    if not os.path.exists(cpp_bin):
        raise SystemExit(f"build cpp_engine first: missing {cpp_bin}")

    with open(os.path.join(a.ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    args = build_args(cfg)

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    # get_dspark_topk_idxs builds its index with a bare torch.arange, so the
    # reference only works with cuda as the default device -- same as the real
    # generation entrypoints do.
    torch.set_default_device("cuda")

    failures = []
    for stage in a.stages:
        for start_pos in a.start_pos:
            ref_err, cpp_err, cpp_q_err = run_case(
                a, args, cpp_bin, device, stage, start_pos, a.seed)
            if cpp_q_err > a.max_rel:
                failures.append(
                    f"stage={stage} start_pos={start_pos}: cpp/gold+actq "
                    f"{cpp_q_err:.5f} > {a.max_rel}")
                print(f"    -> FAIL (bar {a.max_rel})")
            else:
                print(f"    -> ok (bar {a.max_rel})")

    print()
    if failures:
        for f in failures:
            print("FAIL " + f)
        print(f"\nFAIL ({len(failures)} case{'' if len(failures) == 1 else 's'})")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

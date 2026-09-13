"""
GGUF Q2 端到端精度验证：
1. 单步 decode forward
2. 逐层记录 hidden_states
3. 对比 cpp_engine 输出
4. 分析误差累积
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.gguf.reader import GGUFReader
from src.gguf.tensor_reader import GGUFTensorDataReader, get_iq2xxs_signed_grid_tensor
from src.kernels.cuda_loader import load_cuda_kernel


def load_gguf_config(gguf_path: str):
    """读取 GGUF 模型配置"""
    with GGUFReader(gguf_path) as reader:
        metadata = {}
        for i in range(reader.metadata_count()):
            key, value = reader.read_metadata_kv(i)
            metadata[key] = value
    
    config = {
        "n_layers": metadata.get("deepseek.block_count", 43),
        "dim": metadata.get("deepseek.embedding_length", 4096),
        "n_heads": metadata.get("deepseek.attention.head_count", 32),
        "n_kv_heads": metadata.get("deepseek.attention.head_count_kv", 32),
        "vocab_size": metadata.get("deepseek.vocab_size", 100352),
        "rope_theta": metadata.get("deepseek.rope.freq_base", 10000.0),
    }
    return config


def dequant_iq2_xxs_block(block: torch.Tensor, grid: torch.Tensor, in_dim: int) -> torch.Tensor:
    """
    IQ2_XXS dequant reference (简化版，仅用于理解)
    实际应该用 cuda kernel
    """
    # 这里省略具体实现，使用 cuda kernel
    raise NotImplementedError("Use cuda kernel for actual dequant")


def single_step_decode_pytorch(
    gguf_path: str,
    token: int,
    position: int,
    device: torch.device,
) -> dict:
    """
    PyTorch Q2 reference 单步 decode
    返回 per-layer hidden states 和 final logits
    """
    config = load_gguf_config(gguf_path)
    cuda_mod = load_cuda_kernel()
    
    # TODO: 完整实现
    # 1. embed token
    # 2. 逐层 forward (attn + MoE)
    # 3. 记录每层的 post-attn, post-moe hidden_states
    # 4. final norm + head
    
    results = {
        "config": config,
        "token": token,
        "position": position,
        "layer_outputs": {},  # {layer_id: {"post_attn": ..., "post_moe": ...}}
        "final_logits": None,
        "top_token": None,
    }
    
    print(f"[PyTorch Reference] Not implemented yet")
    return results


def load_cpp_engine_dump(dump_path: str) -> dict:
    """加载 cpp_engine 的 per-layer dump"""
    # TODO: cpp_engine 需要添加 dump 功能
    raise NotImplementedError("cpp_engine per-layer dump not implemented")


def compare_precision(pytorch_result: dict, cpp_result: dict):
    """对比 PyTorch vs cpp_engine 的逐层精度"""
    print("\n=== Precision Comparison ===")
    
    for layer_id in pytorch_result["layer_outputs"]:
        py_out = pytorch_result["layer_outputs"][layer_id]
        cpp_out = cpp_result["layer_outputs"].get(layer_id)
        
        if cpp_out is None:
            print(f"Layer {layer_id}: cpp_engine output missing")
            continue
        
        # 对比 post-attn
        if "post_attn" in py_out and "post_attn" in cpp_out:
            diff = np.abs(py_out["post_attn"] - cpp_out["post_attn"])
            print(f"Layer {layer_id} post-attn: max={diff.max():.6f}, mean={diff.mean():.6f}")
        
        # 对比 post-moe
        if "post_moe" in py_out and "post_moe" in cpp_out:
            diff = np.abs(py_out["post_moe"] - cpp_out["post_moe"])
            print(f"Layer {layer_id} post-moe:  max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    # 对比 final logits
    if pytorch_result["final_logits"] is not None and cpp_result["final_logits"] is not None:
        diff = np.abs(pytorch_result["final_logits"] - cpp_result["final_logits"])
        print(f"\nFinal logits: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    # 对比 top token
    py_top = pytorch_result["top_token"]
    cpp_top = cpp_result["top_token"]
    match = "✓" if py_top == cpp_top else "✗"
    print(f"Top token: PyTorch={py_top}, cpp_engine={cpp_top} {match}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--token", type=int, default=100, help="Input token")
    parser.add_argument("--position", type=int, default=10, help="Position")
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    parser.add_argument("--cpp-dump", help="cpp_engine per-layer dump JSON")
    args = parser.parse_args()
    
    device = torch.device("cuda", args.device)
    
    print(f"=== GGUF Q2 Precision Validation ===")
    print(f"Model: {args.model}")
    print(f"Token: {args.token}, Position: {args.position}")
    
    # 运行 PyTorch reference
    print("\n[1/2] Running PyTorch Q2 reference...")
    pytorch_result = single_step_decode_pytorch(args.model, args.token, args.position, device)
    
    # 加载 cpp_engine 结果
    if args.cpp_dump:
        print("\n[2/2] Loading cpp_engine dump...")
        cpp_result = load_cpp_engine_dump(args.cpp_dump)
        
        # 对比精度
        compare_precision(pytorch_result, cpp_result)
    else:
        print("\nSkip cpp_engine comparison (no --cpp-dump)")
    
    print("\n=== Current Status ===")
    print("✗ PyTorch Q2 reference - NOT IMPLEMENTED")
    print("✗ cpp_engine per-layer dump - NOT IMPLEMENTED")
    print("\nNext steps:")
    print("1. Implement PyTorch Q2 single-step decode with per-layer recording")
    print("2. Add per-layer dump to cpp_engine (--dump-layers flag)")
    print("3. Automated precision report generation")


if __name__ == "__main__":
    main()

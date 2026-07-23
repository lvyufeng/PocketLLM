"""GLM-5.2 (glm-dsa) GGUF text-in / text-out greedy generation entrypoint.

A single fused command: each rank encodes the text prompt to token ids (a
cheap, deterministic CPU step, so no broadcast is needed), then the shared
model-agnostic driver ``run_gguf_generation`` loads the TP-sharded model and
greedily decodes. Rank 0 decodes the generated ids back to text and prints
``generated_text=...``; other ranks stay silent.

Run under torchrun, e.g.::

    torchrun --nproc_per_node=4 -m src.cli.generate_glm \\
        --gguf-path /path/to/GLM-5.2-GGUF/UD-Q2_K_XL \\
        --prompt "你好" --chat --max-new-tokens 32 --prewarm
"""

from __future__ import annotations

import argparse

import torch

from src.encoding.glm_dsa import decode_glm_dsa_ids, encode_glm_dsa_prompt
from src.runtime.generation import run_gguf_generation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GLM-5.2 GGUF text-in/text-out greedy generation (TP)")
    parser.add_argument("--gguf-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True, help="text prompt to generate from")
    parser.add_argument("--chat", action="store_true", help="wrap --prompt in compact GLM chat framing")
    parser.add_argument("--thinking", action="store_true", help="with --chat, append <think> before generation")
    parser.add_argument("--system-prompt", type=str, default=None, help="optional system prompt (used with --chat)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--layers", type=int, default=0, help="debug: limit layer count; 0 means full model")
    parser.add_argument("--gpu-memory-gib", type=float, default=22.0)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--skip-special", action="store_true", help="skip special tokens when decoding output")
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help="read GGUF shards into the OS page cache before load (helps on HDD)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    prompt_tokens, prompt_text, _metadata = encode_glm_dsa_prompt(
        args.gguf_path,
        args.prompt,
        chat=bool(args.chat),
        thinking=bool(args.thinking),
        system_prompt=args.system_prompt,
    )

    result = run_gguf_generation(
        args.gguf_path,
        prompt_tokens=prompt_tokens,
        max_new_tokens=int(args.max_new_tokens),
        architecture="glm-dsa",
        dtype=dtype,
        n_layers=None if int(args.layers) <= 0 else int(args.layers),
        gpu_memory_gib=float(args.gpu_memory_gib),
        prewarm=bool(args.prewarm),
    )

    # run_gguf_generation returns the ids+stats on rank 0 and None elsewhere,
    # so only rank 0 reaches the decode/print below.
    if result is not None:
        generated_ids, _stats = result
        text = decode_glm_dsa_ids(args.gguf_path, generated_ids, skip_special=bool(args.skip_special))
        print("prompt_text=" + repr(prompt_text), flush=True)
        print("generated_text=" + repr(text), flush=True)


if __name__ == "__main__":
    main()

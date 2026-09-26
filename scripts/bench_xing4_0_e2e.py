"""One-card prefill and decode rates for Xing4.0-29B-A4B, on real text.

The prompt is real prose out of this repository's own documentation, repeated to
the length being measured and then tokenized -- not random ids and not synthetic
tensors.  That matters here more than usual: the routed MoE's draws are a
function of the activations, so a prompt made of noise routes differently from
one made of English and its experts are not the ones a served request would use.

Each length is measured in one process, serially, and the model is loaded once.
The numbers a report wants are prefill tokens a second (the prompt's own pass)
and decode tokens a second (the steps after it), stated at the context they were
taken at -- a decode rate without its prompt length is not a comparable number.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.xing4_0.gguf_model import Xing4_0GGUFModel

DEFAULT_GGUF = "/mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"
DEFAULT_RELEASE = "/mnt/data2/Xing4.0-29B-A4B"
CORPUS = Path(__file__).resolve().parents[1] / "docs" / "models" / "qwen3.8-27b-fp8.md"


def build_prompt(tokenizer, tokens: int) -> list[int]:
    """Real prose, repeated and truncated to exactly `tokens` ids."""
    text = CORPUS.read_text(encoding="utf-8")
    ids: list[int] = []
    while len(ids) < tokens:
        ids.extend(int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids[:tokens]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--lengths", default="1024,4096,16384")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--no-kernel", action="store_true")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    sys.path.insert(0, args.release)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.release, trust_remote_code=True)

    from src.models.xing4_0.generate import generate

    started = time.perf_counter()
    model = Xing4_0GGUFModel(
        args.gguf,
        device=args.device,
        config_path=str(Path(args.release) / "config.json"),
        use_kernel=not args.no_kernel,
    )
    torch.cuda.synchronize()
    load = time.perf_counter() - started
    free, total = torch.cuda.mem_get_info(args.device)
    print(
        f"loaded {model.nbytes / 2**30:.2f} GiB in {load:.1f} s on {args.device}; "
        f"{free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB"
    )

    cache = model.make_cache(args.max_model_len, batch=1)
    print(f"cache {sum(int(l.latent.numel()) * l.latent.element_size() for l in cache) / 2**30:.2f} GiB "
          f"over {args.max_model_len} positions")
    print()

    for length in (int(item) for item in args.lengths.split(",")):
        ids = build_prompt(tokenizer, length)
        # One full run per length, so the rates below are of a settled card and a
        # warm allocator rather than of the first call in the process.
        result = generate(
            model,
            ids,
            max_new_tokens=args.decode_steps + 1,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            cache=cache,
            chunk=args.chunk,
        )
        prefill = result.prefill_seconds
        decode = result.decode_seconds / max(1, len(result.tokens) - 1)
        print(
            f"{length:6d} tokens:  prefill {length / prefill:8.2f} tok/s ({prefill * 1000:8.0f} ms)   "
            f"decode {1 / decode:6.2f} tok/s ({decode * 1000:7.1f} ms/token)   "
            f"ttft {result.ttft_seconds * 1000:.0f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

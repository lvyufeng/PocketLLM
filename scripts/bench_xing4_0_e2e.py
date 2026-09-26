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
from pocketllm.backends.xing4_backend import (
    PREFILL_DEVICE_RESERVE,
    PREFILL_SCORE_BUDGET,
    _chunk_for,
    _prefill_peak_bytes,
)

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
    ap.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="prefill chunk; 0 derives it from the context the way the server does",
    )
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

    cache = model.make_cache(args.max_model_len + args.decode_steps + 8, batch=1)
    print(f"cache {sum(int(l.latent.numel()) * l.latent.element_size() for l in cache) / 2**30:.2f} GiB "
          f"over {args.max_model_len} positions")
    # The room the *adapter* would compute, which is the device's own free memory
    # after the model and the cache are on it minus its reserve.  Read here rather
    # than taken from the adapter because this script holds the device directly.
    free = int(torch.cuda.mem_get_info(torch.device(args.device).index)[0])
    room = max(PREFILL_SCORE_BUDGET // 2, free - PREFILL_DEVICE_RESERVE)
    print(f"room above the cache {room / 2**20:.0f} MiB")
    print()

    for length in (int(item) for item in args.lengths.split(",")):
        ids = build_prompt(tokenizer, length)
        # The chunk the *server* would pick for this context and this card, not
        # one the caller likes: the score path is 8 bytes an element and at 32K a
        # 2048-wide chunk is 2.0 GiB of a card that has 2.6 GiB spare.  Measured
        # at any other width this row is a rate for a configuration nobody runs.
        chunk = args.chunk or _chunk_for(args.max_model_len, int(model.params.n_heads), room)
        # One full run per length, so the rates below are of a settled card and a
        # warm allocator rather than of the first call in the process.
        result = generate(
            model,
            ids,
            max_new_tokens=args.decode_steps + 1,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            cache=cache,
            chunk=chunk,
        )
        prefill = result.prefill_seconds
        steps = max(1, len(result.tokens) - 1)
        decode = result.decode_seconds / steps
        # The first step after a prefill is a one-off: the allocator settles a
        # request-sized working set, measured at 0.6 s after a narrow chunk and
        # 3.0 s after a wide one, against a steady ~0.18 s.  Both numbers are
        # reported, because a client pays the first one and a rate wants neither
        # hidden nor averaged into the other.
        steady = result.steady_step_seconds
        print(
            f"{length:6d} tokens:  prefill {length / prefill:8.2f} tok/s ({prefill * 1000:8.0f} ms)   "
            f"decode {1 / decode:6.2f} tok/s ({decode * 1000:7.1f} ms/token)   "
            f"steady {1 / steady:6.2f} tok/s ({steady * 1000:6.1f} ms)   "
            f"first step {result.first_step_seconds * 1000:6.0f} ms   ttft {result.ttft_seconds * 1000:.0f} ms   "
            f"chunk {chunk} (peak {_prefill_peak_bytes(chunk, int(model.params.n_heads), args.max_model_len) / 2**20:.0f} MiB)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

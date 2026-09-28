"""The DeepSeek-V4 PyTorch serving runtime: payload in, completion out.

What is here is the half of serving that is about *this* model and format -- how a request becomes a
payload, how the executor's tokens become text, how a streamed answer is split, and the worker loop a
non-zero rank runs. What is *not* here is the HTTP surface or the request lifecycle: those are
`pocketllm/server/openai.py` and `pocketllm/backends/`, and they are shared with every other runtime.

That split is why this module has no `BaseHTTPRequestHandler` in it any more. It used to, and it used
to have a `main()` that started a server of its own -- a second OpenAI surface, in a repository that
already had one, serving a subset of the models the first one serves. See
[#447](https://github.com/lvyufeng/PocketLLM/issues/447).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import timedelta
from typing import Any


def _set_best_env_defaults() -> None:
    os.environ.setdefault("DEEPSEEK_PD_PHASE_AUTO_SELECT", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS", "3")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_ARENA", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_BUCKETED_GEMM", "1")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_BUCKET_EXPERTS", "16")
    os.environ.setdefault("DEEPSEEK_GPU_PREFILL_MOE_CHUNK_TOKENS", "2048")
    os.environ.setdefault("DEEPSEEK_GPU_MOE_DECODE_ACTIVE", "1")
    os.environ.setdefault("DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH", "0")
    os.environ.setdefault("DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K", "10")
    os.environ.setdefault("DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT", "2")
    os.environ.setdefault("DEEPSEEK_INT8_IMPL", "cuda_ext")
    os.environ.setdefault("DEEPSEEK_MOE_ASYNC_ALLREDUCE", "1")
    os.environ.setdefault("DEEPSEEK_SHARED_EXPERT_PAIR_INT8_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_PD_SHARED_EXPERT_FP16", "1")
    os.environ.setdefault("DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_PREFILL_SPARSE_ATTN_HEADPAIR_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_FUSED_C4_INDEXER_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_C4_TOPK_TILE_MERGE_CUDA", "0")
    os.environ.setdefault("DEEPSEEK_HC_PRE_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_HC_POST_CUDA", "1")
    os.environ.setdefault("DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD", "0")
    os.environ.setdefault("DEEPSEEK_CPU_TOPK_PERSISTENT", "1")
    os.environ.setdefault("DEEPSEEK_PD_DECODE_OMP_THREADS", "12")
    os.environ.setdefault("DEEPSEEK_FUSED_ATTN_PREFUSE", "1")
    for module in ("WQ_A", "WQ_B", "WKV", "WO_B", "INDEXER_WQ_B"):
        os.environ.setdefault(f"DEEPSEEK_PD_PREFILL_{module}_INT8", "1")
        os.environ.setdefault(f"DEEPSEEK_PD_DECODE_{module}_INT8", "1")
    os.environ.setdefault("DEEPSEEK_PD_PREFILL_WO_A_INT8", "0")
    os.environ.setdefault("DEEPSEEK_PD_DECODE_WO_A_INT8", "1")
    os.environ.setdefault("DEEPSEEK_WO_A_FP16", "1")
    os.environ.setdefault("DEEPSEEK_CPU_MOE_EXTERNAL_SERVER", "0")
    os.environ.setdefault("DEEPSEEK_CPU_MOE_INPROC_SERVER", "0")
    os.environ.setdefault("DEEPSEEK_CPU_MOE_SHARED_WEIGHTS", "0")


_set_best_env_defaults()

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.components.moe.shared_weights import SharedCPUMoEWeightArena
from src.models.deepseek_v4.generation import (
    _bind_shared_cpu_moe_weights,
    _cpu_affinity_for_rank,
    _enable_numa_interleave,
    generate,
    generate_stream,
    load_model,
)
from src.models.deepseek_v4.runtime import ModelArgs, Transformer
from src.runtime.pd_scheduler import PDExecutionFacade, PDScheduler

current_dir = os.path.dirname(os.path.abspath(__file__))
from pocketllm import protocol  # noqa: E402
from src.encoding.deepseek_v4 import dsml_token, encode_messages, eos_token, parse_message_from_completion_text  # noqa: E402


DEFAULT_MODEL_ID = "deepseek-v4-flash-w8a8"


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes"}


def _setup_cpu_runtime(routed_experts_device: str, local_rank: int, world_size: int) -> None:
    if routed_experts_device != "cpu":
        torch.set_num_threads(8)
        return
    omp_threads_env = os.getenv("DEEPSEEK_CPU_OMP_THREADS")
    omp_threads = int(omp_threads_env) if omp_threads_env else None
    use_affinity = os.getenv("DEEPSEEK_CPU_AFFINITY", "1").lower() not in {"0", "false", "no"}
    rank0_server = _env_enabled("DEEPSEEK_CPU_MOE_RANK0_SERVER")
    inproc_server = _env_enabled("DEEPSEEK_CPU_MOE_INPROC_SERVER")
    centralized_cpu_server = rank0_server or inproc_server
    server_omp_threads_env = os.getenv("DEEPSEEK_CPU_MOE_SERVER_OMP_THREADS")
    nonserver_omp_threads = int(os.getenv("DEEPSEEK_CPU_MOE_NONSERVER_OMP_THREADS", "1"))
    affinity_cpus = None if (centralized_cpu_server and local_rank == 0) else (_cpu_affinity_for_rank(local_rank, world_size) if use_affinity else None)
    if affinity_cpus is not None:
        os.sched_setaffinity(0, affinity_cpus)
        cpu_threads = omp_threads or max(len(affinity_cpus), 1)
    else:
        if centralized_cpu_server and local_rank == 0:
            cpu_threads = omp_threads or int(server_omp_threads_env or "22")
        elif inproc_server:
            cpu_threads = omp_threads or max(nonserver_omp_threads, 1)
        else:
            cpu_threads = omp_threads or max((os.cpu_count() or 1) // world_size, 1)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    if centralized_cpu_server and local_rank == 0:
        os.environ.setdefault("OMP_PROC_BIND", "spread")
    else:
        os.environ.setdefault("OMP_PROC_BIND", "close")
    import src.components.moe.cpu_backend as cpu_routed_backend

    cpu_routed_backend.configure_cpu_routed_runtime(omp_threads=cpu_threads)
    torch.set_num_threads(1)


def _init_runtime(args):
    if args.ckpt_format != "gguf" and not (args.ckpt_format == "auto" and args.ckpt_path.endswith(".gguf")):
        os.environ.setdefault("DEEPSEEK_SHARED_EXPERT_INT8", "1")
    if args.partition_policy == "baseline_4gpu" and args.pd_mode != "scheduler":
        raise ValueError("partition_policy=baseline_4gpu requires --pd-mode scheduler")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if args.partition_policy == "layer_pp_4gpu" and world_size not in {2, 4}:
        raise ValueError("partition_policy=layer_pp_4gpu requires torchrun with WORLD_SIZE=2 or 4")
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(days=7))
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    _setup_cpu_runtime(args.routed_experts_device, local_rank, world_size)
    torch.manual_seed(33377335)

    with open(args.config) as f:
        config_data = json.load(f)
    config_data["routed_experts_device"] = args.routed_experts_device
    config_data["partition_policy"] = args.partition_policy
    # The 0731 checkpoint's mtp.* tensors belong to DSpark, not the legacy
    # MTPBlock. Serving does not attach DSpark, so build the main model only.
    config_data["n_mtp_layers"] = 0
    model_args = ModelArgs(**config_data)
    serving_max_batch_size = max(1, int(os.getenv("DEEPSEEK_SERVING_MAX_RUNNING_REQUESTS", "1")))
    model_args.max_batch_size = serving_max_batch_size
    if args.max_model_len:
        model_args.max_seq_len = int(args.max_model_len)

    shared_cpu_moe_arena = None
    init_start = time.perf_counter()
    with torch.device("cuda"):
        model = Transformer(model_args)
    model.max_batch_size = serving_max_batch_size
    if args.routed_experts_device == "cpu" and SharedCPUMoEWeightArena.enabled():
        if _env_enabled("DEEPSEEK_CPU_MOE_SHARED_WEIGHT_NUMA_INTERLEAVE"):
            _enable_numa_interleave()
        shared_root_dir = SharedCPUMoEWeightArena.root_dir_from_env()
        if not shared_root_dir:
            raise RuntimeError("DEEPSEEK_CPU_MOE_SHARED_WEIGHTS=1 requires DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR")
        shared_cpu_moe_arena = _bind_shared_cpu_moe_weights(model, shared_root_dir, model_args, world_size, rank)
    tokenizer_path = args.tokenizer_path or args.ckpt_path
    if args.ckpt_format == "gguf" or (args.ckpt_format == "auto" and args.ckpt_path.endswith(".gguf")):
        if not args.tokenizer_path:
            raise ValueError("GGUF checkpoints require --tokenizer-path to point to a tokenizer directory")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"init time: {time.perf_counter() - init_start:.3f}s", flush=True)

    print("load model", flush=True)
    load_start = time.perf_counter()
    load_model(model, args.ckpt_path, world_size, rank, args.ckpt_format)
    if args.routed_experts_device == "cpu":
        model.prepare_cpu_expert_int8()
        if shared_cpu_moe_arena is not None:
            shared_cpu_moe_arena.mark_ready()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"load time: {time.perf_counter() - load_start:.3f}s", flush=True)
    torch.set_default_device("cuda")

    control_group = dist.new_group(backend="gloo", timeout=timedelta(days=7)) if world_size > 1 else None
    scheduler = PDScheduler() if args.pd_mode == "scheduler" else None
    executor = PDExecutionFacade.from_env(generate, generate_stream, scheduler)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "executor": executor,
        "model_id": args.model,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "control_group": control_group,
        "shared_cpu_moe_arena": shared_cpu_moe_arena,
    }


def _openai_error(message: str, error_type: str = "invalid_request_error", param: str | None = None, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


# Request normalization lives in the backend-neutral protocol package so the
# unified PocketLLM server shares one implementation with this legacy server.
_normalize_content = protocol.normalize_content
_tool_names = protocol.tool_names
_tool_choice_instruction = protocol.tool_choice_instruction
_prepare_messages = protocol.prepare_messages
_thinking_config = protocol.thinking_config


def _strip_special_eos(text: str) -> str:
    if text.endswith(eos_token):
        return text[: -len(eos_token)]
    return text


def _timing_metrics(prefill_time: float, decode_time: float, prefill_tokens: int, decode_tokens: int) -> dict[str, float | int]:
    total_time = prefill_time + decode_time
    total_tokens = prefill_tokens + decode_tokens
    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "ttft": prefill_time,
        "ttft_ms": prefill_time * 1000.0,
        "tpot": decode_time / max(decode_tokens, 1),
        "tpot_ms": decode_time * 1000.0 / max(decode_tokens, 1),
        "prefill_tokens_per_second": prefill_tokens / max(prefill_time, 1e-9),
        "decode_tokens_per_second": decode_tokens / max(decode_time, 1e-9),
        "throughput_tokens_per_second": total_tokens / max(total_time, 1e-9),
        "total_tokens_per_second": total_tokens / max(total_time, 1e-9),
    }


_normalize_tool_calls = protocol.normalize_tool_calls
_stop_strings = protocol.stop_strings


def _apply_stop_to_result(result: dict[str, Any], stop: Any) -> None:
    content, stopped = protocol.apply_stop_to_text(result.get("content") or "", stop)
    if stopped:
        result["content"] = content
        result["finish_reason"] = "stop"


def _format_logprobs(tokenizer, rows: list[dict] | None) -> tuple[list[str], list[float], list[list[dict[str, Any]]]]:
    token_texts: list[str] = []
    token_logprobs: list[float] = []
    top_rows: list[list[dict[str, Any]]] = []
    for row in rows or []:
        token_id = int(row["token_id"])
        text = tokenizer.decode([token_id])
        token_texts.append(text)
        token_logprobs.append(float(row["logprob"]))
        top_rows.append([
            {"token": tokenizer.decode([int(candidate["token_id"])]), "logprob": float(candidate["logprob"])}
            for candidate in row.get("top_logprobs", [])
        ])
    return token_texts, token_logprobs, top_rows


def _format_completion_result(tokenizer, thinking_mode: str, prompt_ids: list[int], completion_ids: list[int], prefill_time: float, decode_time: float, prefill_tokens: int, decode_tokens: int, stop: Any = None, max_tokens: int | None = None, logprobs: list[dict] | None = None) -> dict[str, Any]:
    completion_text = tokenizer.decode(completion_ids)
    try:
        assistant_msg = parse_message_from_completion_text(completion_text, thinking_mode)
    except Exception:
        assistant_msg = {
            "role": "assistant",
            "content": _strip_special_eos(completion_text),
            "reasoning_content": "",
            "tool_calls": [],
        }
    content = assistant_msg.get("content") or ""
    reasoning = assistant_msg.get("reasoning_content") or ""
    tool_calls = _normalize_tool_calls(assistant_msg.get("tool_calls") or [])
    finish_reason = "tool_calls" if tool_calls else "stop"
    if max_tokens is not None and len(completion_ids) >= int(max_tokens) and not tool_calls:
        finish_reason = "length"
    result = {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "timings": _timing_metrics(prefill_time, decode_time, prefill_tokens, decode_tokens),
    }
    if logprobs is not None:
        token_texts, token_logprobs, top_rows = _format_logprobs(tokenizer, logprobs)
        result["token_texts"] = token_texts
        result["token_logprobs"] = token_logprobs
        result["top_logprobs"] = top_rows
    _apply_stop_to_result(result, stop)
    return result


def _run_payload(runtime: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    torch.cuda.set_device(runtime["local_rank"])
    torch.set_default_device("cuda")
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    executor = runtime["executor"]
    batched_prompt_ids = payload.get("_prompt_ids_batch")
    if batched_prompt_ids is not None:
        prompt_ids_list = [[int(t) for t in prompt_ids] for prompt_ids in batched_prompt_ids]
        thinking_modes = payload.get("_thinking_mode_batch") or [payload["thinking_mode"]] * len(prompt_ids_list)
        max_tokens_list = [int(v) for v in (payload.get("_max_tokens_batch") or [])]
        if not max_tokens_list:
            max_tokens_list = [int(payload.get("max_tokens") or 512)] * len(prompt_ids_list)
        if len(set(max_tokens_list)) != 1:
            raise RuntimeError("batched non-stream requests must use the same max_tokens in the first serving engine version")
        max_tokens = max_tokens_list[0]
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        generation_options = payload.get("generation_options") or {}
        completion_result = executor.run(
            model,
            prompt_ids_list,
            max_tokens,
            tokenizer.eos_token_id,
            temperature,
            generation_options=generation_options,
            prefix_snapshot_hint=payload.get("_prefix_snapshot_hint"),
        )
        if len(completion_result) == 6:
            completion_tokens, prefill_time, decode_time, prefill_tokens, decode_tokens, batch_logprobs = completion_result
        else:
            completion_tokens, prefill_time, decode_time, prefill_tokens, decode_tokens = completion_result
            batch_logprobs = [None] * len(prompt_ids_list)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return [
            _format_completion_result(
                tokenizer,
                str(thinking_modes[i] or "chat"),
                prompt_ids_list[i],
                completion_tokens[i],
                prefill_time,
                decode_time,
                prefill_tokens,
                decode_tokens,
                stop=payload.get("stop"),
                max_tokens=max_tokens,
                logprobs=batch_logprobs[i] if batch_logprobs else None,
            )
            for i in range(len(prompt_ids_list))
        ]

    thinking_mode = payload["thinking_mode"]
    prompt_ids = payload.get("_prompt_ids")
    if prompt_ids is None:
        messages = payload["messages"]
        reasoning_effort = payload.get("reasoning_effort")
        prompt_text = encode_messages(messages, thinking_mode=thinking_mode, reasoning_effort=reasoning_effort)
        prompt_ids = tokenizer.encode(prompt_text)
    else:
        prompt_ids = [int(t) for t in prompt_ids]
    max_tokens = int(payload.get("max_tokens") or 512)
    temperature = float(payload.get("temperature", 0.0) or 0.0)
    n = int(payload.get("n", 1) or 1)
    prompt_batch = [prompt_ids for _ in range(n)]
    generation_options = payload.get("generation_options") or {}
    completion_result = executor.run(
        model,
        prompt_batch,
        max_tokens,
        tokenizer.eos_token_id,
        temperature,
        generation_options=generation_options,
        prefix_snapshot_hint=payload.get("_prefix_snapshot_hint"),
    )
    if len(completion_result) == 6:
        completion_tokens, prefill_time, decode_time, prefill_tokens, decode_tokens, batch_logprobs = completion_result
    else:
        completion_tokens, prefill_time, decode_time, prefill_tokens, decode_tokens = completion_result
        batch_logprobs = [None] * n
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if n == 1:
        completion_ids = completion_tokens[0]
        return _format_completion_result(
            tokenizer,
            thinking_mode,
            prompt_ids,
            completion_ids,
            prefill_time,
            decode_time,
            prefill_tokens,
            decode_tokens,
            stop=payload.get("stop"),
            max_tokens=max_tokens,
            logprobs=batch_logprobs[0] if batch_logprobs else None,
        )
    return [
        _format_completion_result(
            tokenizer,
            thinking_mode,
            prompt_ids,
            completion_tokens[i],
            prefill_time,
            decode_time,
            prefill_tokens,
            decode_tokens,
            stop=payload.get("stop"),
            max_tokens=max_tokens,
            logprobs=batch_logprobs[i] if batch_logprobs else None,
        )
        for i in range(n)
    ]


def _int_param(body: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = body.get(key, default)
    if value is None:
        return None
    return int(value)


def _float_param(body: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = body.get(key, default)
    if value is None:
        return None
    return float(value)


def _make_payload(body: dict[str, Any]) -> dict[str, Any]:
    thinking_mode, reasoning_effort = _thinking_config(body)
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 512))
    n = int(body.get("n", 1) or 1)
    stream = bool(body.get("stream", False))
    if n < 1:
        raise ValueError("n must be >= 1")
    if stream and n != 1:
        raise ValueError("streaming currently supports n=1")
    if stream and body.get("logprobs"):
        raise ValueError("streaming logprobs are not supported")
    return {
        "op": "chat_completion",
        "request_id": f"chatcmpl-{uuid.uuid4().hex}",
        "messages": _prepare_messages(body),
        "max_tokens": int(max_tokens or 512),
        "temperature": float(body.get("temperature", 0.0) or 0.0),
        "top_p": _float_param(body, "top_p"),
        "top_k": _int_param(body, "top_k"),
        "min_p": _float_param(body, "min_p"),
        "frequency_penalty": _float_param(body, "frequency_penalty", 0.0),
        "presence_penalty": _float_param(body, "presence_penalty", 0.0),
        "repetition_penalty": _float_param(body, "repetition_penalty", 1.0),
        "seed": _int_param(body, "seed"),
        "stop": body.get("stop"),
        "logprobs": bool(body.get("logprobs", False)),
        "top_logprobs": _int_param(body, "top_logprobs"),
        "n": n,
        "generation_options": {
            "top_p": _float_param(body, "top_p"),
            "top_k": _int_param(body, "top_k"),
            "min_p": _float_param(body, "min_p"),
            "frequency_penalty": _float_param(body, "frequency_penalty", 0.0),
            "presence_penalty": _float_param(body, "presence_penalty", 0.0),
            "repetition_penalty": _float_param(body, "repetition_penalty", 1.0),
            "seed": _int_param(body, "seed"),
            "logprobs": bool(body.get("logprobs", False)),
            "top_logprobs": _int_param(body, "top_logprobs"),
        },
        "tool_choice": body.get("tool_choice"),
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
        "user": body.get("user"),
        "metadata": body.get("metadata"),
        "thinking_mode": thinking_mode,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
        "stream_options": body.get("stream_options") or {},
    }


def _run_payload_stream(runtime: dict[str, Any], payload: dict[str, Any]):
    torch.cuda.set_device(runtime["local_rank"])
    torch.set_default_device("cuda")
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    executor = runtime["executor"]
    prompt_ids = payload.get("_prompt_ids")
    if prompt_ids is None:
        prompt_text = encode_messages(
            payload["messages"],
            thinking_mode=payload["thinking_mode"],
            reasoning_effort=payload.get("reasoning_effort"),
        )
        prompt_ids = tokenizer.encode(prompt_text)
    else:
        prompt_ids = [int(t) for t in prompt_ids]
    max_tokens = int(payload.get("max_tokens") or 512)
    temperature = float(payload.get("temperature", 0.0) or 0.0)
    events = executor.stream(
        model,
        [prompt_ids],
        max_tokens,
        tokenizer.eos_token_id,
        temperature,
        generation_options=payload.get("generation_options") or {},
        prefix_snapshot_hint=payload.get("_prefix_snapshot_hint"),
    )
    for event in events:
        if event.get("type") == "done":
            event = dict(event)
            event["prompt_tokens"] = len(prompt_ids)
        yield event


class _StreamingDecoder:
    _TOOL_CALLS_START = f"\n\n<{dsml_token}tool_calls"

    def __init__(self, tokenizer, thinking_mode: str):
        self.tokenizer = tokenizer
        self.thinking_mode = thinking_mode
        self.token_ids: list[int] = []
        self.emitted_text = ""
        self.content_emitted = ""
        self.reasoning_emitted = ""
        self._content_truncated = False
        self._content_visible_len: int | None = None

    def _truncate_for_tool_calls(self, content_text: str) -> str:
        if self._content_truncated:
            return content_text[: self._content_visible_len or 0]
        idx = content_text.find(self._TOOL_CALLS_START)
        if idx >= 0:
            self._content_truncated = True
            self._content_visible_len = idx
            return content_text[:idx]
        marker = self._TOOL_CALLS_START
        max_overlap = min(len(content_text), len(marker) - 1)
        for n in range(max_overlap, 0, -1):
            if content_text.endswith(marker[:n]):
                return content_text[: len(content_text) - n]
        return content_text

    def append(self, token_ids: list[int]) -> list[dict[str, str]]:
        self.token_ids.extend(int(t) for t in token_ids)
        decoded = _strip_special_eos(self.tokenizer.decode(self.token_ids)).rstrip("�")
        if len(decoded) < len(self.emitted_text):
            return []
        self.emitted_text = decoded
        if self.thinking_mode != "thinking":
            visible = self._truncate_for_tool_calls(decoded)
            if len(visible) <= len(self.content_emitted):
                return []
            delta_text = visible[len(self.content_emitted):]
            self.content_emitted = visible
            return [{"content": delta_text}]
        outputs: list[dict[str, str]] = []
        marker = "</think>"
        marker_idx = decoded.find(marker)
        if marker_idx < 0:
            new_reasoning = decoded[len(self.reasoning_emitted):]
            if new_reasoning:
                self.reasoning_emitted = decoded
                outputs.append({"reasoning_content": new_reasoning})
            return outputs
        reasoning_text = decoded[:marker_idx]
        content_text = decoded[marker_idx + len(marker):]
        if len(reasoning_text) > len(self.reasoning_emitted):
            outputs.append({"reasoning_content": reasoning_text[len(self.reasoning_emitted):]})
            self.reasoning_emitted = reasoning_text
        visible_content = self._truncate_for_tool_calls(content_text)
        if len(visible_content) > len(self.content_emitted):
            outputs.append({"content": visible_content[len(self.content_emitted):]})
            self.content_emitted = visible_content
        return outputs

    def final_message(self, thinking_mode: str) -> dict[str, Any]:
        text = self.tokenizer.decode(self.token_ids)
        try:
            return parse_message_from_completion_text(text, thinking_mode)
        except Exception:
            return {
                "role": "assistant",
                "content": _strip_special_eos(text),
                "reasoning_content": "",
                "tool_calls": [],
            }


def _broadcast_payload(payload: dict[str, Any], runtime: dict[str, Any]) -> None:
    if runtime["world_size"] > 1:
        dist.broadcast_object_list([payload], src=0, group=runtime["control_group"])


def _logprobs_payload(result: dict[str, Any], top_logprobs: int | None) -> dict[str, Any] | None:
    token_logprobs = result.get("token_logprobs")
    token_texts = result.get("token_texts")
    top_candidates = result.get("top_logprobs")
    if not token_logprobs or not token_texts:
        return None
    content = []
    for idx, text in enumerate(token_texts):
        entry = {"token": text, "logprob": float(token_logprobs[idx]), "bytes": list(text.encode("utf-8"))}
        if top_candidates and idx < len(top_candidates) and top_candidates[idx]:
            entry["top_logprobs"] = [
                {
                    "token": cand.get("token", ""),
                    "logprob": float(cand.get("logprob", 0.0)),
                    "bytes": list(str(cand.get("token", "")).encode("utf-8")),
                }
                for cand in top_candidates[idx][: max(int(top_logprobs or 0), 0)]
            ]
        content.append(entry)
    return {"content": content}


def _completion_choice(result: dict[str, Any], index: int, top_logprobs: int | None = None) -> dict[str, Any]:
    message = {"role": "assistant", "content": result.get("content", "")}
    if result.get("reasoning_content"):
        message["reasoning_content"] = result["reasoning_content"]
    if result.get("tool_calls"):
        message["tool_calls"] = result["tool_calls"]
    choice = {
        "index": index,
        "message": message,
        "finish_reason": result.get("finish_reason", "stop"),
    }
    logprobs = _logprobs_payload(result, top_logprobs)
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return choice


def _completion_response(runtime: dict[str, Any], payload: dict[str, Any], result: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    results = result if isinstance(result, list) else [result]
    prompt_tokens = int(results[0].get("prompt_tokens", 0)) if results else 0
    completion_tokens = sum(int(item.get("completion_tokens", 0)) for item in results)
    return {
        "id": payload["request_id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": runtime["model_id"],
        "choices": [_completion_choice(item, idx, payload.get("top_logprobs")) for idx, item in enumerate(results)],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "deepseek_timings": results[0].get("timings", {}) if results else {},
    }


def _chunk_payload(runtime: dict[str, Any], payload: dict[str, Any], delta: dict[str, Any], finish_reason: str | None = None, index: int = 0) -> dict[str, Any]:
    choice = {"index": index, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {
        "id": payload["request_id"],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": runtime["model_id"],
        "choices": [choice],
    }


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sse_line(obj: Any) -> bytes:
    if obj == "[DONE]":
        return b"data: [DONE]\n\n"
    return b"data: " + _json_bytes(obj) + b"\n\n"


def _worker_loop(runtime: dict[str, Any]) -> None:
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0, group=runtime["control_group"])
        payload = box[0]
        if not isinstance(payload, dict):
            continue
        if payload.get("op") == "shutdown":
            break
        if payload.get("op") in {"chat_completion", "chat_completion_batch"}:
            if payload.get("stream"):
                for _event in _run_payload_stream(runtime, payload):
                    pass
            else:
                _run_payload(runtime, payload)



# Phase 3.5: Automatic Tensor Parallelism - Implementation Results

## Date
2026-09-03

## Objective
Make CppBackend tensor parallelism launch automatically like vLLM/SGLang, without requiring manual process management or special launch scripts.

**User Requirement**: "我不希望有什么特殊启动流程，你应该把cpp engine后端的启动方式和pytorch后端的启动方式统一，并且要符合vllm/sglang的使用习惯"

## Implementation

### Design
Following vLLM/SGLang conventions, users should be able to write:
```python
from pocketllm import EngineArgs, LLM

args = EngineArgs(
    model="/path/to/model",
    backend="cpp",
    tensor_parallel_size=4
)
llm = LLM(args)  # Automatically spawns 4 processes
```

### Key Changes

#### 1. `pocketllm/backends/factory.py`
- Added automatic supervision detection in `create_backend()`:
  - Detects when TP > 1, rank == 0, and no external NCCL ID provided
  - Creates `TensorParallelSupervisor` to spawn worker processes (ranks 1-3)
  - Main process runs as rank 0
  - Supervisor manages NCCL rendezvous file automatically

- Worker script generation (`_worker_script()`):
  - Reads configuration from environment variables set by supervisor
  - Each worker process creates its own `CppBackend` instance
  - Worker rank mapping: supervisor rank N → actual TP rank N+1
  - Workers call `backend.run_worker()` to enter service loop

#### 2. `pocketllm/backends/cpp_backend.py`
- Added cleanup logic in `close()` method:
  - Stops supervisor if owned by this backend
  - Removes temporary NCCL ID file
  - Prevents process/file leaks

#### 3. Module Installation
- Fixed stale module issue: `pocketllm_cpp.so` in site-packages was outdated
- Solution: copied latest build to site-packages directly
- Verified `run_worker_loop` symbol is present in loaded module

### Verification

#### Basic Functionality Test
```python
# 4 GPU processes spawn automatically
llm = LLM(EngineArgs(model="...", backend="cpp", tensor_parallel_size=4))
result = llm.generate([[1,2,3,4,5]], sampling_params={'max_tokens': 8})
# ✓ Success: generated 8 tokens
llm.close()
# ✓ All processes cleaned up
```

Output:
```
[tp rank 0] POCKETLLM_RANK_READY rank=1
[tp rank 1] POCKETLLM_RANK_READY rank=2
[tp rank 2] POCKETLLM_RANK_READY rank=3
✓ LLM created!
✓ Generated 8 tokens: [151644, 8948, 374, 264, 1296, 4320, 624, 358]
✓ Done!
```

All 4 GPU processes confirmed via `nvidia-smi`.

#### Concurrent Throughput Baseline
**Model**: Qwen3.8-27B-FP8  
**Configuration**: TP=4, prompt=16 tokens, max_tokens=32, temperature=0.0  
**Backend**: CppBackend with `enable_batching=False` (Phase 3.5 baseline)

| Concurrent Requests | Serial Time | Concurrent Time | Speedup |
|---------------------|-------------|-----------------|---------|
| 2                   | 1.493s      | 1.346s          | 1.11×   |
| 4                   | 2.696s      | 2.706s          | 1.00×   |
| 8                   | 5.437s      | 5.482s          | 0.99×   |

**Analysis**:
- Minimal speedup (≤1.11×) is expected because:
  - Backend has no batch scheduler (Phase 3.4 pending)
  - Requests are serialized: "serialized compatibility session"
  - Multiple threads calling `generate()` just queue requests

**Comparison to Targets**:
- Target: 2 concurrent ≥1.7×, 4 concurrent ≥3.0×, 8 concurrent ≥4.5×
- Current: 1.11×/1.00×/0.99× — significantly below target
- **Root cause**: Lack of batching support, not TP launch mechanism

### Issues Encountered and Resolved

#### Issue 1: Only rank 0 process spawned
**Symptom**: Only one GPU process, stuck waiting for NCCL rendezvous  
**Root cause**: Initial implementation used `world_size - 1`, spawning only 2 workers instead of 3  
**Fix**: Corrected to spawn exactly 3 workers (ranks 1-3), main process runs rank 0

#### Issue 2: Module missing `run_worker_loop`
**Symptom**: Worker processes failed with "native engine does not expose run_worker_loop"  
**Root cause**: Python was loading stale `pocketllm_cpp.so` from site-packages (Sept 2 build) instead of latest (Sept 3)  
**Fix**: Copied latest `cpp_engine/build-python/pocketllm_cpp.so` to site-packages

#### Issue 3: NCCL ID path mismatch
**Symptom**: All processes failed with "CmdChannel: connect failed"  
**Root cause**: `factory.py` created `/tmp/pocketllm_nccl_*.txt`, but supervisor created its own `/tmp/pocketllm-tp-*/nccl_id`  
**Fix**: Let supervisor manage NCCL ID file creation; main process reads path from supervisor

## Current Status

### ✅ Completed
- Automatic TP process spawning for CppBackend
- Unified API with PyTorch backend (same `EngineArgs` interface)
- Follows vLLM/SGLang usage conventions
- Proper cleanup of supervisor and temp files
- Basic correctness verification

### ⏳ Pending (Next Phase)
- Phase 3.4: Implement `QwenBatchScheduler` in C++
- Phase 3.5: Enable batch mode in CppBackend
- Phase 3.6: Concurrent throughput optimization to meet targets (1.7×/3.0×/4.5×)

## Performance Baseline

**Single-request latency** (from previous test):
- Prompt: 16 tokens
- Generated: 32 tokens  
- Latency: ~0.67s per request

**Concurrent throughput** (no batching):
- 2 concurrent: 1.11× (below 1.7× target)
- 4 concurrent: 1.00× (below 3.0× target)
- 8 concurrent: 0.99× (below 4.5× target)

These numbers establish the baseline before batch scheduler implementation.

## Conclusion

**Primary objective achieved**: CppBackend now launches tensor-parallel processes automatically, matching the user experience of vLLM/SGLang. Users no longer need manual process management or launcher scripts.

**Performance targets not yet met**: Concurrent throughput improvements require batch scheduler (Phase 3.4-3.5), which is the next implementation step.

**User requirement satisfied**: "不希望有什么特殊启动流程" — launch process is now unified with PyTorch backend and follows vLLM/SGLang conventions.

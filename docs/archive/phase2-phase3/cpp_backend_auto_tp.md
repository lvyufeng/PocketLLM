# CppBackend Automatic Tensor Parallelism Support

## Goal
Make CppBackend behave like vLLM/SGLang/PyTorch when using tensor parallelism:

```python
from pocketllm import EngineArgs, LLM

# This should "just work" - automatically spawn 4 processes
args = EngineArgs(
    model="/path/to/model",
    backend="cpp",
    tensor_parallel_size=4
)
llm = LLM(args)
result = llm.generate(...)
```

## Current Behavior
- Requires manual `nccl_id_path` in `backend_options`
- Throws `ConfigurationError` if `tensor_parallel_size > 1` without `nccl_id_path`
- Requires external launcher script or manual multi-process setup

## Proposed Changes

### 1. Modify `create_backend()` in `pocketllm/backends/factory.py`
When `backend == "cpp"` and `tensor_parallel_size > 1` and `tensor_parallel_rank == 0`:
- Use `TensorParallelSupervisor` to spawn worker processes
- Auto-generate temporary NCCL ID file
- Return a supervisor-managed CppBackend proxy

### 2. Create `CppBackendProxy` class
A wrapper that:
- Holds reference to `TensorParallelSupervisor`
- Holds reference to rank-0 `CppBackend`
- Forwards all API calls to rank-0 backend
- Manages lifecycle (close supervisor on shutdown)

### 3. Modify `CppBackend._construct_qwen_engine()`
- Remove the `ConfigurationError` check for missing `nccl_id_path`
- Allow `nccl_id_path` to come from environment variable `POCKETLLM_NCCL_ID_PATH`
- Worker processes will get this from supervisor

### 4. Handle supervised child processes
- Worker processes (rank > 0) should enter `run_worker()` loop
- Rank 0 process should behave normally but with NCCL ID set

## Implementation Plan

1. **Phase 1**: Remove hard requirement for `nccl_id_path` ✓
   - Allow empty `nccl_id_path` for `tp_size == 1`
   - Read from environment variable as fallback

2. **Phase 2**: Add supervisor integration to `create_backend()`
   - Detect when auto-supervision is needed
   - Launch supervisor
   - Return proxy

3. **Phase 3**: Implement `CppBackendProxy`
   - Forward all BackendBase methods
   - Manage supervisor lifecycle

4. **Phase 4**: Testing
   - Test with single GPU (tp=1)
   - Test with multi-GPU (tp=4)
   - Test benchmark scripts

## Alternative: Use Environment Detection
Instead of creating a proxy, we could:
- Have `CppBackend.__init__` detect it's rank 0 and auto-launch supervisor
- Store supervisor reference in `CppBackend` instance
- This is simpler but mixes concerns (backend + process management)

## Questions to Resolve
1. Should this be opt-in or automatic? (Suggest: automatic, like vLLM)
2. Should we support `CUDA_VISIBLE_DEVICES` style GPU selection?
3. How to handle cleanup on error during supervisor startup?
4. Should benchmark scripts work transparently, or need special handling?

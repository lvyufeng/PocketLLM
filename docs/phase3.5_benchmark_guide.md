# Phase 3.5 Performance Validation

This directory contains benchmark scripts for validating the QwenBatchScheduler performance.

## Quick Start

Run all validation tests:

```bash
python scripts/run_phase3.5_validation.py /path/to/qwen3.5/checkpoint
```

Results will be saved to `phase3.5_results/` directory.

## Individual Benchmarks

### 1. Single-Request Latency (Regression Test)

Tests that batch mode doesn't introduce significant overhead for single requests.

**Target**: ≤1.05× baseline (serial mode)

```bash
python scripts/bench_single_request_latency.py /path/to/checkpoint
```

**Options**:
- `--max-tokens N` - Max tokens per request (default: 128)
- `--warmup-runs N` - Warmup iterations (default: 3)
- `--test-runs N` - Test iterations (default: 10)
- `--prompt-length N` - Prompt length in tokens (default: 32)

### 2. Concurrent Request Throughput

Tests throughput improvement with multiple concurrent requests.

**Targets**:
- 2 concurrent: ≥1.7× baseline
- 4 concurrent: ≥3.0× baseline
- 8 concurrent: ≥4.5× baseline

```bash
python scripts/bench_concurrent_throughput.py /path/to/checkpoint
```

**Options**:
- `--num-concurrent 2 4 8` - Concurrency levels to test (default: 2 4 8)
- `--max-tokens N` - Max tokens per request (default: 128)
- `--warmup-runs N` - Warmup iterations (default: 1)
- `--test-runs N` - Test iterations per concurrency (default: 3)
- `--prompt-length N` - Prompt length in tokens (default: 32)

## Understanding Results

### Single-Request Latency

The test compares serial mode (baseline) vs batch mode for single requests:

```
Comparison
============================================================
Average Latency:
  Serial: 0.523s
  Batch:  0.531s
  Ratio:  1.02× (+1.5%)

✅ PASS: Batch mode latency is 1.02× baseline (≤1.05×)
```

**Pass criteria**: Batch mode ≤1.05× serial mode latency

### Concurrent Throughput

The test measures throughput (requests/sec) for different concurrency levels:

```
2 Concurrent Requests:
  Serial throughput: 1.91 req/s
  Batch throughput:  3.45 req/s
  Improvement:       1.81× (target: ≥1.7×)
  ✅ PASS

4 Concurrent Requests:
  Serial throughput: 1.87 req/s
  Batch throughput:  6.12 req/s
  Improvement:       3.27× (target: ≥3.0×)
  ✅ PASS
```

**Pass criteria**: Batch mode throughput meets or exceeds target improvement

## Expected Performance

Based on Phase 3.1-3.3 batch decode kernel capabilities:

| Metric | Mode | Expected |
|--------|------|----------|
| Single request latency | Batch | ≤1.05× serial |
| 2 concurrent throughput | Batch | ≥1.7× serial |
| 4 concurrent throughput | Batch | ≥3.0× serial |
| 8 concurrent throughput | Batch | ≥4.5× serial |

## Troubleshooting

### "Native module does not expose QwenBatchScheduler"

The Python module was not built with batch scheduler support. Rebuild:

```bash
cd cpp_engine/build
cmake .. -DPOCKET_BACKEND=cuda -DPOCKET_BUILD_PYTHON=ON \
         -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make pocketllm_cpp -j8
```

### "enable_batching=True but scheduler not available"

The native module is old. Rebuild cpp_engine with latest code.

### Low throughput improvement

Possible causes:
1. **Prefill bottleneck**: Current `batch_prefill()` is sequential (Phase 3.2 limitation)
2. **Small batch size**: Increase `max_batch_size` in backend_options
3. **GPU underutilization**: Check with `nvidia-smi` during benchmark

### High single-request overhead

If batch mode latency > 1.05× serial:
1. Check scheduler thread sleep interval (1ms in `schedule_loop()`)
2. Profile with nvprof to identify hotspots
3. Verify slot allocation is not blocking

## Performance Analysis

### Viewing Detailed Logs

All benchmark output is saved to `phase3.5_results/`:

```bash
# View single-request latency results
cat phase3.5_results/single_latency_*.txt

# View concurrent throughput results
cat phase3.5_results/concurrent_throughput_*.txt

# View JSON summary
cat phase3.5_results/results_*.json
```

### Comparing Runs

```bash
# Run multiple times with different configs
python scripts/bench_concurrent_throughput.py /path/to/checkpoint \
    --num-concurrent 2 4 8 16

# Compare results
diff phase3.5_results/concurrent_throughput_*.txt
```

## Next Steps

After Phase 3.5 validation passes:

1. **Document results** in `docs/phase3.5_performance_report.md`
2. **Update README.md** with actual benchmark numbers
3. **Create Phase 3.5 completion PR**

If validation fails:
1. Profile bottlenecks with nvprof/nsight
2. Optimize scheduler thread overhead
3. Consider prefill batching improvements (future work)

/* Measure `MatmulCubeCustom`'s throughput as a function of m, at a fixed
 * (n, k) weight shape.
 *
 * The point is one question: is the ~18 GFLOP/s a decode step runs at an m=1
 * *shape* limit -- the cube's M dimension sitting idle -- or a hard device
 * limit?  If throughput rises with m, batching is a lever; if it is flat, the
 * 310B's rate is the device's and the perf question is closed.
 *
 * It drives the same public `gemm_quant` the engine uses, on synthetic q4_K
 * weights, so it measures the shipped path and not a reimplementation.  The
 * weight plane is built once (the backend caches it), so only m varies.
 *
 *     ./pocketllm-mscale --device ascend --n 2560 --k 2560 --rep 20
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernel/backend.h"
#include "quant/blocks.h"
#include "runtime/status.h"

namespace {

/* Deterministic synthetic q4_K blocks.  The values are irrelevant to the timing
 * -- only the shape and the data type are -- so this is a plain LCG rather than
 * anything that has to decrypt a real checkpoint. */
std::vector<uint8_t> make_q4k(int64_t n, int64_t k) {
  const int64_t block_bytes = pocketllm::quant::kQ4KBlockBytes; /* 144 */
  const int64_t per_row = k / pocketllm::quant::kBlockWeights;
  std::vector<uint8_t> bytes(static_cast<std::size_t>(n * per_row * block_bytes));
  uint32_t state = 0x12345678u;
  for (uint8_t &b : bytes) {
    state = state * 1664525u + 1013904223u;
    b = static_cast<uint8_t>(state >> 24);
  }
  /* A finite fp16 `d`/`dmin` in the first four bytes of each block keeps the
   * decode well-defined rather than an inf/nan. */
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t c = 0; c < per_row; ++c) {
      uint8_t *block = bytes.data() + (r * per_row + c) * block_bytes;
      block[0] = 0x00; block[1] = 0x24; /* fp16 ~= 0.03125 */
      block[2] = 0x00; block[3] = 0x1c; /* fp16 ~= 0.015625 */
    }
  }
  return bytes;
}

const char *arg(int argc, char **argv, const char *name, const char *fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      return argv[i + 1];
    }
  }
  return fallback;
}

}  // namespace

int main(int argc, char **argv) {
  const std::string device = arg(argc, argv, "--device", "ascend");
  const int64_t n = std::atoll(arg(argc, argv, "--n", "2560"));
  const int64_t k = std::atoll(arg(argc, argv, "--k", "2560"));
  const int64_t rep = std::atoll(arg(argc, argv, "--rep", "20"));
  const int64_t type_id = pocketllm::quant::kGgmlQ4K;

  std::fprintf(stderr, "device=%s n=%lld k=%lld rep=%lld\n", device.c_str(),
               static_cast<long long>(n), static_cast<long long>(k),
               static_cast<long long>(rep));

  auto backend = pocketllm::kernel::make_backend(device);
  const std::vector<uint8_t> blocks = make_q4k(n, k);
  pocketllm::kernel::DeviceBuffer dblocks = backend->allocate(static_cast<int64_t>(blocks.size()));
  backend->copy_to_device(dblocks, blocks.data(), static_cast<int64_t>(blocks.size()));

  /* Every m's activation and output, allocated once so the timing loop does no
   * allocating of its own. */
  const int64_t max_m = 32;
  std::vector<float> x(static_cast<std::size_t>(max_m * k), 0.01F);
  pocketllm::kernel::DeviceBuffer dx = backend->allocate(max_m * k * 4);
  backend->copy_to_device(dx, x.data(), max_m * k * 4);
  pocketllm::kernel::DeviceBuffer dout = backend->allocate(max_m * n * 4);

  std::printf("m,rep,ms_total,ms_per_call,gflops\n");
  for (int64_t m : {int64_t{1}, int64_t{2}, int64_t{4}, int64_t{8}, int64_t{16}, int64_t{32}}) {
    /* Warm up: the first gemm_quant at this (n,k,type) builds and caches the
     * fp16 plane, which is a one-time cost and not what is being measured. */
    backend->gemm_quant(dx, dblocks, pocketllm::kernel::DeviceBuffer{}, dout, m, n, k, type_id, false);
    backend->synchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t r = 0; r < rep; ++r) {
      backend->gemm_quant(dx, dblocks, pocketllm::kernel::DeviceBuffer{}, dout, m, n, k, type_id, false);
    }
    backend->synchronize();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double flop = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                        static_cast<double>(k) * static_cast<double>(rep);
    const double gflops = flop / (ms / 1000.0) / 1e9;
    std::printf("%lld,%lld,%.2f,%.3f,%.2f\n", static_cast<long long>(m),
                static_cast<long long>(rep), ms, ms / static_cast<double>(rep), gflops);
  }

  /* A second shape the engine actually runs: the head (n = 151936, chunked) and
   * the ffn (n = 9728).  These say whether the rate is shape-dependent. */
  std::printf("#\n");
  for (int64_t n2 : {int64_t{9728}, int64_t{151936}}) {
    const std::vector<uint8_t> b2 = make_q4k(n2, k);
    pocketllm::kernel::DeviceBuffer db2 = backend->allocate(static_cast<int64_t>(b2.size()));
    backend->copy_to_device(db2, b2.data(), static_cast<int64_t>(b2.size()));
    pocketllm::kernel::DeviceBuffer do2 = backend->allocate(max_m * n2 * 4);
    for (int64_t m : {int64_t{1}, int64_t{8}, int64_t{32}}) {
      backend->gemm_quant(dx, db2, pocketllm::kernel::DeviceBuffer{}, do2, m, n2, k, type_id, false);
      backend->synchronize();
      const auto t0 = std::chrono::steady_clock::now();
      for (int64_t r = 0; r < rep; ++r) {
        backend->gemm_quant(dx, db2, pocketllm::kernel::DeviceBuffer{}, do2, m, n2, k, type_id, false);
      }
      backend->synchronize();
      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      const double gflops = 2.0 * m * n2 * k * rep / (ms / 1000.0) / 1e9;
      std::printf("%lld,%lld,%.2f,%.3f,%.2f,n=%lld\n", static_cast<long long>(m),
                  static_cast<long long>(rep), ms, ms / static_cast<double>(rep), gflops,
                  static_cast<long long>(n2));
    }
    backend->release(db2);
    backend->release(do2);
  }

  backend->release(dblocks);
  backend->release(dx);
  backend->release(dout);
  return 0;
}
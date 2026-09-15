// Peak device-memory bandwidth probe.
//
// The decode target is a question about a physical limit: a decode step re-reads
// every resident weight, so the per-token floor is (weight bytes) / (achievable
// HBM read rate). The GEMM bench reports an effective rate for a one-row GEMV,
// which mixes kernel efficiency into the number; this measures the rate on its
// own so the two can be told apart.
//
// The instrument is a device-to-device copy, which moves one byte in and one byte
// out per element and is therefore the standard streaming number. A read-only
// rate is not directly expressible through the memcpy API, so the copy figure is
// reported as-is and used as the upper bound it is: a kernel that reads without
// writing cannot beat it.
//
//   ./tests/bench_qwen_ascend_bandwidth [--device N] [--iters 20]

#include "device_runtime.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Timing {
    double seconds = 0.0;
    double gbs = 0.0;
};

// Copy `bytes` from `src` to `dst` `iters` times and report the two-way rate.
Timing time_copy(uint8_t* dst, const uint8_t* src, size_t bytes, int iters) {
    if (!pocket::memcpy_d2d(dst, src, bytes)) return {};
    if (!pocket::device_synchronize()) return {};
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!pocket::memcpy_d2d(dst, src, bytes)) return {};
    }
    if (!pocket::device_synchronize()) return {};
    const auto stop = std::chrono::steady_clock::now();
    Timing timing;
    timing.seconds =
        std::chrono::duration<double>(stop - start).count() / iters;
    timing.gbs = 2.0 * static_cast<double>(bytes) / timing.seconds / 1e9;
    return timing;
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int iters = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (!pocket::device_runtime_available()) {
        std::printf("[SKIP] no device runtime available\n");
        return 0;
    }
    if (!pocket::device_set(device)) {
        std::printf("[SKIP] device_set failed for device %d\n", device);
        return 0;
    }

    // 256 MB per buffer: 8x L2 (32 MB on this part), so the copy is not served
    // out of cache. 4 MB is included as the counter-example -- it fits in L2 and
    // reports the cache rate rather than the memory rate.
    const std::vector<size_t> sizes = {4ull << 20, 64ull << 20, 256ull << 20};
    for (size_t bytes : sizes) {
        uint8_t* src = nullptr;
        uint8_t* dst = nullptr;
        if (!pocket::device_malloc_into(src, bytes) ||
            !pocket::device_malloc_into(dst, bytes)) {
            std::printf("size=%.0fMB [SKIP] device_malloc failed\n",
                        static_cast<double>(bytes) / 1e6);
            pocket::device_free(src);
            pocket::device_free(dst);
            continue;
        }
        if (!pocket::device_memset(src, 1, bytes)) return 1;
        // The first pass pays page mapping; time the second onwards.
        const Timing timing = time_copy(dst, src, bytes, iters);
        std::printf("d2d-copy size=%4.0fMB bytes=%zu seconds=%.6f gbs=%.1f\n",
                    static_cast<double>(bytes) / 1e6, bytes, timing.seconds,
                    timing.gbs);
        std::fflush(stdout);
        pocket::device_free(dst);
        pocket::device_free(src);
    }
    return 0;
}

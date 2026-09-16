// Device-memory bandwidth probes.
//
// The decode target is a question about a physical limit: a decode step re-reads
// every resident weight, so the per-token floor is (weight bytes) / (achievable
// HBM read rate). Two instruments measure that limit, and they disagree by 40x
// because they measure different things — which is the whole point of running
// them side by side rather than quoting either alone.
//
//   d2d-copy   `memcpy_d2d` over N bytes. One byte in and one byte out per
//              element, so the reported rate is a two-way total (the helper
//              doubles the byte count to make that explicit).
//   hbm-read   `qwen_hbm_read_probe`: MTE2 streams 64x256 FP16 tiles and the
//              kernel touches one element per tile. Reads only, no Vector or
//              Cube op anywhere, so the figure is the memory system's read rate
//              with no kernel efficiency mixed in.
//
// The copy is the instrument that was here first, and its number is the one not
// to use as the bound. A device-to-device copy pays for a write and a
// read-modify-write the read path never issues, so at 8.7 GB/s it is far *under*
// what the GEMV already achieves — it cannot bound a read-only kernel from
// above, only from below. The read probe is the upper bound: a kernel that reads
// the same bytes and also computes cannot beat a kernel that only reads them.
//
// Both are reported at 4 MB / 64 MB / 256 MB. 4 MB fits in the 32 MB L2 and
// reports the cache rate rather than the memory rate, for either instrument.
//
//   ./tests/bench_qwen_ascend_bandwidth [--device N] [--iters 20]

#include "device_runtime.hpp"
#include "qwen_hbm_probe_geometry.hpp"
#include "qwen_ops.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// One 64x256 FP16 tile, the unit `qwen_hbm_read_probe` counts in.
constexpr size_t kProbeTileBytes = 64u * 256u * 2u;

// The probe fills its source with a single byte pattern, so the value it sums
// has to be whatever half word that pattern happens to spell. 0x3C3C is half
// 1084 * 2^-10 = 1.05859375 -- exact, and large enough that the per-core sum
// stays well clear of fp16 subnormals over a few hundred tiles.
constexpr int kProbeFillByte = 0x3C;
constexpr double kProbeFillHalf = 1.05859375;

struct Timing {
    double seconds = 0.0;
    double gbs = 0.0;
};

// half bits to float, for checking the probe's sink on the host.
double half_to_double(uint16_t bits) {
    const int sign = (bits >> 15) & 1;
    const int exponent = (bits >> 10) & 0x1f;
    const int mantissa = bits & 0x3ff;
    double value;
    if (exponent == 0) {
        value = mantissa * 5.960464477539063e-08;  // 2^-24
    } else if (exponent == 31) {
        value = mantissa ? 0.0 : 1.0;  // inf/nan collapsed; the probe cannot make one
    } else {
        value = (1024 + mantissa) * std::pow(2.0, exponent - 25);
    }
    return sign ? -value : value;
}

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

// Stream `tiles` tiles out of `src` `iters` times and report the one-way read
// rate. The kernels are launched back to back; the single synchronize covers the
// whole batch, the same way time_copy times its loop.
Timing time_read(uint16_t* sink, const uint16_t* src, int tiles, int iters) {
    if (!pocket::qwen_hbm_read_probe(src, sink, tiles, nullptr)) return {};
    if (!pocket::device_synchronize()) return {};
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!pocket::qwen_hbm_read_probe(src, sink, tiles, nullptr)) return {};
    }
    if (!pocket::device_synchronize()) return {};
    const auto stop = std::chrono::steady_clock::now();
    Timing timing;
    timing.seconds =
        std::chrono::duration<double>(stop - start).count() / iters;
    const double bytes = static_cast<double>(tiles) * kProbeTileBytes;
    timing.gbs = bytes / timing.seconds / 1e9;
    return timing;
}

// Run the probe once more and sum its per-core sinks, as the check on the timed
// figure above. Every tile is read by exactly one core and contributes one
// element to that core's sum, so the total is pinned at tiles * kProbeFillHalf.
// A shortfall means a core skipped part of its span -- which is the one way the
// timed number could come out high for a reason that has nothing to do with
// bandwidth. Returns a negative value when the probe itself failed.
double probe_sink_sum(uint16_t* sink, const uint16_t* src, int tiles) {
    constexpr int kSinks = 30;  // one per core; must match the launcher's grid
    if (!pocket::qwen_hbm_read_probe(src, sink, tiles, nullptr)) return -1.0;
    if (!pocket::device_synchronize()) return -1.0;
    uint16_t host_sink[kSinks] = {};
    if (!pocket::memcpy_d2h(host_sink, sink, sizeof(host_sink))) return -1.0;
    double sum = 0.0;
    for (int i = 0; i < kSinks; ++i) sum += half_to_double(host_sink[i]);
    return sum;
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

    // 1 GB: 32x L2 (32 MB on this part), so the read probe is measuring HBM and
    // not a cache. The smaller points are there as the counter-example, and the
    // read figures rise as they shrink because the L2 starts carrying them.
    const std::vector<size_t> sizes = {4ull << 20, 64ull << 20, 256ull << 20,
                                       1ull << 30};
    for (size_t bytes : sizes) {
        const double mib = static_cast<double>(bytes) / 1e6;

        uint8_t* src = nullptr;
        uint8_t* dst = nullptr;
        if (!pocket::device_malloc_into(src, bytes) ||
            !pocket::device_malloc_into(dst, bytes)) {
            std::printf("size=%.0fMB [SKIP] device_malloc failed\n", mib);
            pocket::device_free(src);
            pocket::device_free(dst);
            continue;
        }
        if (!pocket::device_memset(src, 1, bytes)) return 1;
        const Timing copy = time_copy(dst, src, bytes, iters);
        std::printf("d2d-copy size=%4.0fMB bytes=%zu seconds=%.6f gbs=%6.1f\n",
                    mib, bytes, copy.seconds, copy.gbs);
        std::fflush(stdout);
        pocket::device_free(dst);
        pocket::device_free(src);

        // Whole tiles only: the probe partitions tiles across cores, and a
        // partial tile at the end would be read by no one.
        const int tiles = static_cast<int>(bytes / kProbeTileBytes);
        if (tiles <= 0) continue;
        uint16_t* read_src = nullptr;
        uint16_t* sink = nullptr;
        if (!pocket::device_malloc_into(read_src, bytes) ||
            !pocket::device_malloc_into(sink, 30 * sizeof(uint16_t))) {
            std::printf("size=%.0fMB [SKIP] read-probe malloc failed\n", mib);
            pocket::device_free(read_src);
            pocket::device_free(sink);
            continue;
        }
        if (!pocket::device_memset(read_src, kProbeFillByte, bytes)) return 1;
        const Timing read = time_read(sink, read_src, tiles, iters);
        const double summed = probe_sink_sum(sink, read_src, tiles);
        const double expected = static_cast<double>(tiles) * kProbeFillHalf *
                                pocket::kHbmProbeSamples;
        const double err = summed < 0.0 ? -1.0
                                        : std::fabs(summed - expected) / expected;
        // fp16 storage of the per-core sums is the only slack in the check, and
        // it is a few parts in 1e-3; a skipped tile shows up as a whole tile's
        // worth, orders of magnitude larger.
        const char* verdict = err < 0.0 ? "FAILED" : (err < 5e-3 ? "ok" : "SHORT");
        std::printf(
            "hbm-read size=%4.0fMB bytes=%zu tiles=%d seconds=%.6f gbs=%6.1f "
            "tiles_sum=%.3f expect=%.3f rel_err=%.2e check=%s\n",
            mib, static_cast<size_t>(tiles) * kProbeTileBytes, tiles,
            read.seconds, read.gbs, summed, expected, err, verdict);
        std::fflush(stdout);
        pocket::device_free(sink);
        pocket::device_free(read_src);
    }
    return 0;
}

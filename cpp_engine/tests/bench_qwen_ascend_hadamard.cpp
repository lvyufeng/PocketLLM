// What the incoherence rotation costs, at the geometry the checkpoint uses.
//
// The rotation is on the per-token path of every projection whose weights were
// quantized in the rotated frame, so its cost is a layer-budget item and not a
// one-off. The kernel's own header says so and leaves the measurement open: three
// of the ten butterfly levels pair elements 4, 8 and 16 bytes apart, which the
// vector unit cannot address, so they run on the scalar unit one 8-element group at
// a time. That stage is the one that could dominate, and the block sweep below is
// what says whether it does.
//
// The two halves of the model:
//
//   scalar   12 butterflies per aligned group of eight, i.e. a fixed 1.5 adds per
//            element whatever `block` is -- levels 1, 2 and 4 never leave the
//            group they start in.
//   vector   the remaining log2(block) - 3 levels, two vector instructions per
//            pair of runs, so the per-element count grows with log2(block) but
//            each instruction covers a whole 32-byte block.
//
// If the scalar stage dominates, the per-element cost falls as `block` grows (a
// smaller fraction of the levels is scalar); if the vector levels dominate it
// rises. `--block-sweep` is that comparison, reported per element so the
// different widths stay comparable.
//
// What it measured (910A, 30 cores, 2026-09-26):
//
//   The scalar stage does not dominate. Cost per element falls from 2.83 ns at
//   block 16 to 0.89 ns at block 4096, monotonically, so the kernel is not
//   paying a fixed per-group scalar tax -- the growth is in the vector levels,
//   which is the cheap side. Two points bracket a plane of the checkpoint's
//   geometry (block 1024, 5120 wide): one row is 29.4 us and 112 rows is 505.0 us,
//   i.e. 5.7 ns/element alone against 0.88 ns/element batched. The single row
//   reaches only 5 of the 30 cores -- five runs of 1024 features -- so at decode
//   widths the number is latency, not throughput, and it is the batching that
//   amortizes it. The 112-row rate is 9.1 GB/s of the 8 bytes per element the
//   kernel moves, which is where a memory-bound elementwise pass on this part
//   sits.
//
//   A launch that is waited on costs ~63 us more than one that is not (92.0 us
//   against 29.4 us at one row), and an empty `device_synchronize` is 0.3 us, so
//   that gap is the cost of starting a kernel on an idle device and not the wait.
//   The engine launches back to back, so `stream` is the figure that applies to
//   it; `sync` is what a caller that blocks per op pays, and it is three times
//   the device time at decode widths.
//
// The same three timings every Ascend op is measured with, for the same reason: a
// host-bound op and a device-bound op can look alike at one number.
//
//   enqueue  N launches back to back, one synchronize at the end
//   stream   the same loop with its drain divided back in, i.e. the device's rate
//   sync     one launch and one synchronize per op
//
// The traffic the GB/s column is figured on is x read, y written, and the FP32
// sign vector re-read once per row: `rows * width * (2 + 2 + 4)` bytes. The signs
// are a third of it and are the same 20 KiB of +-1 for every row, which the report
// prints rather than hides -- a `+-1` fits an FP16 exactly, so that third is the
// one part of the traffic that could be halved by changing the table's dtype.
//
//   ./tests/bench_qwen_ascend_hadamard [--device N] [--iters 200] [--block-sweep]

#include "device_runtime.hpp"
#include "qwen_ascend_ops.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Qwen3.8-27B's hidden size and the checkpoint's rotation block: five runs of
// 1024 per row, which is also the shape that decides how many of the 30 AI cores
// a single decode row can reach.
constexpr int kHidden = 5120;
constexpr int kBlock = 1024;

// The largest plane measured, so one allocation serves every point.
constexpr int kMaxRows = 112;

uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | half);
}

struct Timing {
    double enqueue_seconds = 0.0;
    double stream_seconds = 0.0;
    double sync_seconds = 0.0;
};

// `enqueue` runs the body `iters` times with no synchronize between them, so it is
// the host's own cost when the host runs ahead of the device. The drain at the end
// is not optional: without it the next block's one synchronize waits for this
// block's backlog as well, and the two figures below come out doubled. `stream` is
// the same loop with its drain divided back in, which is the device's rate when
// launches are queued behind one another -- what the engine pays. `sync` waits
// after every launch, which is what a caller that blocks per op pays, and the gap
// between the two is the cost of starting a kernel on an idle device. The program
// prints an empty `device_synchronize` before the planes so that gap can be read
// as the launch rather than the wait.
template <typename Op>
Timing time_op(Op op, int iters) {
    Timing timing;
    if (!op()) throw std::runtime_error("warmup launch failed");
    if (!pocket::device_synchronize()) throw std::runtime_error("warmup sync failed");

    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!op()) throw std::runtime_error("launch failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        timing.enqueue_seconds =
            std::chrono::duration<double>(stop - start).count() / iters;
        if (!pocket::device_synchronize()) throw std::runtime_error("sync failed");
    }
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!op()) throw std::runtime_error("launch failed");
        }
        if (!pocket::device_synchronize()) throw std::runtime_error("sync failed");
        const auto stop = std::chrono::steady_clock::now();
        timing.stream_seconds =
            std::chrono::duration<double>(stop - start).count() / iters;
    }
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!op()) throw std::runtime_error("launch failed");
            if (!pocket::device_synchronize()) throw std::runtime_error("sync failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        timing.sync_seconds =
            std::chrono::duration<double>(stop - start).count() / iters;
    }
    return timing;
}

// `sync` is host plus device plus the synchronize itself, so it is what a caller
// that waits per op pays; `stream` is what the device costs when nobody waits,
// which is how the engine calls this. Reporting both is the difference between a
// kernel that is slow and a wait that is.
void report(const char* name, double elements, double bytes, const Timing& timing) {
    const double stream_gbs =
        timing.stream_seconds > 0.0 ? bytes / timing.stream_seconds / 1e9 : 0.0;
    const double ns_per_element =
        elements > 0.0 ? timing.stream_seconds * 1e9 / elements : 0.0;
    std::printf("%-30s enqueue=%7.1f us  stream=%8.1f us  sync=%8.1f us  "
                "gbs=%6.1f  ns/element=%8.3f\n",
                name, timing.enqueue_seconds * 1e6, timing.stream_seconds * 1e6,
                timing.sync_seconds * 1e6, stream_gbs, ns_per_element);
    std::fflush(stdout);
}

struct Point {
    const char* name;
    int rows;
    int width;
    int block;
};

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int iters = 200;
    bool block_sweep = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--block-sweep") {
            block_sweep = true;
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

    const size_t max_count = static_cast<size_t>(kMaxRows) * kHidden;
    uint16_t* d_x = nullptr;
    uint16_t* d_y = nullptr;
    float* d_signs = nullptr;
    if (!pocket::device_malloc_into(d_x, max_count * sizeof(uint16_t)) ||
        !pocket::device_malloc_into(d_y, max_count * sizeof(uint16_t)) ||
        !pocket::device_malloc_into(d_signs, kHidden * sizeof(float))) {
        std::printf("[FAIL] device_malloc failed\n");
        return 1;
    }
    struct Free {
        uint16_t* x;
        uint16_t* y;
        float* s;
        ~Free() {
            pocket::device_free(x);
            pocket::device_free(y);
            pocket::device_free(s);
        }
    } free_guard{d_x, d_y, d_signs};

    // Small integers, so a rotation that is wrong is wrong by a visible amount if
    // anyone later wants to check a point by hand; the numbers here are timings
    // either way.
    std::vector<uint16_t> host_x(max_count);
    for (size_t i = 0; i < max_count; ++i) {
        host_x[i] = float_to_half(static_cast<float>(static_cast<int>(i % 33) - 16));
    }
    std::vector<float> host_signs(kHidden);
    for (int i = 0; i < kHidden; ++i) host_signs[static_cast<size_t>(i)] = (i & 1) ? -1.0f : 1.0f;

    if (!pocket::memcpy_h2d(d_x, host_x.data(), max_count * sizeof(uint16_t)) ||
        !pocket::memcpy_h2d(d_signs, host_signs.data(), kHidden * sizeof(float))) {
        std::printf("[FAIL] upload failed\n");
        return 1;
    }

    // What the wait itself costs, so the gap between `stream` and `sync` below can
    // be read as the cost of starting a kernel on an idle device rather than as the
    // cost of waiting.
    {
        if (!pocket::device_synchronize()) {
            std::printf("[FAIL] sync failed\n");
            return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!pocket::device_synchronize()) {
                std::printf("[FAIL] sync failed\n");
                return 1;
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        std::printf("an empty device_synchronize: %.1f us\n",
                    std::chrono::duration<double>(stop - start).count() / iters * 1e6);
    }

    // The planes the checkpoint's own shapes produce: one decode row through the
    // serving ladder's batch widths. Each is a separate launch, so a plane whose
    // runs do not fill the 30 cores shows it here.
    const Point planes[] = {
        {"fwd 1 row", 1, kHidden, kBlock},     {"inv 1 row", 1, kHidden, kBlock},
        {"fwd 2 rows", 2, kHidden, kBlock},    {"inv 2 rows", 2, kHidden, kBlock},
        {"fwd 4 rows", 4, kHidden, kBlock},    {"inv 4 rows", 4, kHidden, kBlock},
        {"fwd 8 rows", 8, kHidden, kBlock},    {"inv 8 rows", 8, kHidden, kBlock},
        {"fwd 16 rows", 16, kHidden, kBlock},  {"inv 16 rows", 16, kHidden, kBlock},
        {"fwd 32 rows", 32, kHidden, kBlock},  {"inv 32 rows", 32, kHidden, kBlock},
        {"fwd 112 rows", 112, kHidden, kBlock}, {"inv 112 rows", 112, kHidden, kBlock},
    };
    std::printf("one row of %d features, the checkpoint's block of %d\n", kHidden, kBlock);
    for (const Point& point : planes) {
        const bool forward = std::strncmp(point.name, "fwd", 3) == 0;
        const double elements = static_cast<double>(point.rows) * point.width;
        const double bytes = elements * 8.0;
        const Timing timing = time_op(
            [&] {
                return forward ? pocket::qwen_hadamard_forward_f16_ascend(
                                     d_x, d_signs, d_y, point.rows, point.width,
                                     point.block, nullptr)
                               : pocket::qwen_hadamard_inverse_f16_ascend(
                                     d_x, d_signs, d_y, point.rows, point.width,
                                     point.block, nullptr);
            },
            iters);
        report(point.name, elements, bytes, timing);
    }

    // One direction only: the two differ by one elementwise multiply and the
    // planes above show that costs nothing measurable, so the sweep does not need
    // both.
    //
    // Every point here is sized so the run count reaches the 30 AI cores, because
    // otherwise the sweep would be measuring the occupancy it just changed rather
    // than the block size: with one row, `block` 1024 is 5 runs and 5 cores and
    // `block` 4096 is 1 run and 1 core, which is a 6x difference in parallelism
    // reported as a difference in geometry. The rows go up instead, so the variable
    // is the block and nothing else. The first point is the exception -- one run on
    // one core, which is what a launch with no work in it costs.
    //
    // 5120 does not divide by 2048, so the last two points are a 4096-wide plane.
    // Everything is reported per element, which is what makes them comparable.
    if (block_sweep) {
        const Point blocks[] = {
            {"block 16, 1 run", 1, 16, 16},     {"block 16, 320 runs", 1, 5120, 16},
            {"block 32, 160 runs", 1, 5120, 32}, {"block 64, 80 runs", 1, 5120, 64},
            {"block 128, 40 runs", 1, 5120, 128}, {"block 256, 40 runs", 2, 5120, 256},
            {"block 512, 30 runs", 3, 5120, 512}, {"block 1024, 30 runs", 6, 5120, 1024},
            {"block 2048, 30 runs", 15, 4096, 2048}, {"block 4096, 30 runs", 30, 4096, 4096},
        };
        std::printf("\nblock sweep, forward, 30 cores busy\n");
        for (const Point& point : blocks) {
            const double elements = static_cast<double>(point.rows) * point.width;
            const double bytes = elements * 8.0;
            const Timing timing = time_op(
                [&] {
                    return pocket::qwen_hadamard_forward_f16_ascend(
                        d_x, d_signs, d_y, point.rows, point.width, point.block,
                        nullptr);
                },
                iters);
            report(point.name, elements, bytes, timing);
        }
    }

    return 0;
}

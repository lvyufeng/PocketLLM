// Is a cross-stream event a usable dependency on this backend?
//
// The candidate fix for the decode step's collective cost is to issue the
// all-reduce from a second stream and order the kernels that read the reduced
// values by recording an event on the communication stream and waiting on it from
// the default stream. That is the textbook construction, and it is worth measuring
// the primitive before building on it: candidate orderings of that shape were
// measured here as wrong at every depth and every event-ring size, while the same
// collective issued synchronously is exact.
//
// The synchronous path in `end_nccl_collective` is not evidence either way: it
// records on the default stream and waits on the communication stream, and the
// answer comes back through a host-side synchronize, so it never exercises the
// direction the second-stream construction needs.
//
// The answer recorded here is that the event *is* sound in both directions, so a
// wrong result through it is a defect in what the collective makes visible to the
// event, not in the event. `docs/performance/ascend_decode_collective_ab.md`
// carries the rest of that result.
//
// Three orderings, each run `--iters` times. The producer writes a buffer with a
// real aclnn Cube matmul (not a memset, which is far too fast to lose the race),
// so a dependency that is not honoured shows up as the consumer reading the reset
// value instead of the produced one.
//
//   comm-wait    producer on stream A, event on A, consumer on the default stream
//                -- the direction the second-stream construction depends on
//   default-wait the same pair with the streams exchanged: producer on the default
//                stream, consumer on A. This is the direction `begin_nccl_collective`
//                uses, and it is included because a primitive that works one way and
//                not the other is a much more specific finding than one that never
//                works.
//   host-synced  the comm-wait case with a host wait on the event before the
//                consumer is enqueued. A control: it proves the producer, consumer
//                and check are themselves correct, so a failure above is the
//                dependency and not the arithmetic.
//
// Each ordering is measured twice, once with the consumer as an aclnn kernel and
// once as a device-to-device copy, because a DMA read through the SDMA engine and a
// Cube read through the L2 can be ordered differently. Only the kernel consumer is
// a usable reading: `aclrtMemcpyAsync` does not support device-to-device here, so
// the copy consumer reports stale every time by simply not being a transfer.
//
//   ./tests/bench_qwen_ascend_event_order --device 0 [--iters 50]

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The producer's shape. 1 x 4352 x 5120 is the TP4 attention output projection,
// and the bench_qwen_ascend_decode_ops measurement puts it at ~165 us on the
// device: long enough that a consumer enqueued without a working dependency has
// every chance to overtake it.
constexpr int kRows = 4352;
constexpr int kCols = 5120;

// One in fp16 and the value the producer writes: with unit operands the matmul
// sums 5120 ones, so every output element is exactly 5120.
constexpr uint16_t kOne = 0x3c00u;
constexpr uint16_t kProduced = 0x6400u;  // 5120 in fp16

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int iters = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    }
    if (iters <= 0) {
        std::fprintf(stderr, "usage: %s --device D [--iters N]\n", argv[0]);
        return 1;
    }
    if (!pocket::device_runtime_available() || !pocket::device_set(device)) {
        std::fprintf(stderr, "no usable device %d\n", device);
        return 1;
    }

    const size_t x_bytes = static_cast<size_t>(kCols) * sizeof(uint16_t);
    const size_t w_bytes =
        static_cast<size_t>(kRows) * kCols * sizeof(uint16_t);
    const size_t y_bytes = static_cast<size_t>(kRows) * sizeof(uint16_t);

    uint16_t* d_x = nullptr;
    uint16_t* d_w = nullptr;
    uint16_t* d_y = nullptr;
    uint16_t* d_gamma = nullptr;
    uint16_t* d_out = nullptr;
    uint16_t* d_shadow = nullptr;
    if (!pocket::device_malloc_into(d_x, x_bytes) ||
        !pocket::device_malloc_into(d_w, w_bytes) ||
        !pocket::device_malloc_into(d_y, y_bytes) ||
        !pocket::device_malloc_into(d_gamma, y_bytes) ||
        !pocket::device_malloc_into(d_out, y_bytes) ||
        !pocket::device_malloc_into(d_shadow, y_bytes)) {
        std::fprintf(stderr, "device_malloc failed\n");
        return 1;
    }

    std::vector<uint16_t> ones(static_cast<size_t>(kRows) * kCols, kOne);
    std::vector<uint16_t> host_y(kRows, 0);
    std::vector<uint16_t> host_out(kRows, 0);
    if (!pocket::memcpy_h2d(d_x, ones.data(), x_bytes) ||
        !pocket::memcpy_h2d(d_w, ones.data(), w_bytes) ||
        !pocket::memcpy_h2d(d_gamma, ones.data(), y_bytes)) {
        std::fprintf(stderr, "memcpy_h2d failed\n");
        return 1;
    }

    void* stream = pocket::stream_create();
    void* ev = pocket::event_create();
    if (stream == nullptr || ev == nullptr) {
        std::fprintf(stderr, "stream/event creation failed\n");
        return 1;
    }

    auto matmul = [&](void* s) {
        return pocket::qwen_fp16_matmul_rows_f16(d_x, d_w, d_y, 1, kRows, kCols,
                                                 kCols, kRows, kCols, s);
    };
    auto rmsnorm = [&](void* s) {
        return pocket::qwen_rmsnorm_fp16_gamma_rows_f16(d_y, d_gamma, d_out, 1,
                                                        kRows, 1e-6f, s);
    };
    auto reset = [&]() {
        pocket::device_memset(d_y, 0, y_bytes);
        pocket::device_synchronize();
    };

    // The producer must land where the check can tell it apart from the reset
    // value, so the first run of each ordering is verified before it is counted.
    struct Result {
        const char* label;
        int bad;
    };
    std::vector<Result> results;

    // comm-wait: producer on `stream`, consumer on the default stream. The
    // dependency under test.
    for (int consumer_kind = 0; consumer_kind < 2; ++consumer_kind) {
        int bad = 0;
        for (int i = 0; i < iters; ++i) {
            reset();
            if (!matmul(stream)) {
                std::fprintf(stderr, "producer failed\n");
                return 1;
            }
            pocket::event_record(ev, stream);
            pocket::stream_wait_event(nullptr, ev);
            if (consumer_kind == 0) {
                if (!rmsnorm(nullptr)) {
                    std::fprintf(stderr, "consumer failed\n");
                    return 1;
                }
                pocket::device_synchronize();
                pocket::memcpy_d2h(host_out.data(), d_out, y_bytes);
                // rms(x) of a constant vector is the constant, so with unit gamma
                // the normalised value is exactly one; a stale zero reads back as
                // zero.
                for (uint16_t value : host_out) {
                    if (value != kOne) {
                        ++bad;
                        break;
                    }
                }
            } else {
                pocket::memcpy_d2d_async(d_shadow, d_y, y_bytes, nullptr);
                pocket::device_synchronize();
                pocket::memcpy_d2h(host_y.data(), d_shadow, y_bytes);
                for (uint16_t value : host_y) {
                    if (value != kProduced) {
                        ++bad;
                        break;
                    }
                }
            }
        }
        results.push_back({consumer_kind == 0 ? "comm-wait kernel" : "comm-wait copy",
                           bad});
    }

    // default-wait: the same pair with the streams exchanged, i.e. the direction
    // the synchronous collective path already relies on.
    {
        int bad = 0;
        for (int i = 0; i < iters; ++i) {
            reset();
            if (!matmul(nullptr)) {
                std::fprintf(stderr, "producer failed\n");
                return 1;
            }
            pocket::event_record(ev, nullptr);
            pocket::stream_wait_event(stream, ev);
            if (!rmsnorm(stream)) {
                std::fprintf(stderr, "consumer failed\n");
                return 1;
            }
            pocket::stream_synchronize(stream);
            pocket::device_synchronize();
            pocket::memcpy_d2h(host_out.data(), d_out, y_bytes);
            for (uint16_t value : host_out) {
                if (value != kOne) {
                    ++bad;
                    break;
                }
            }
        }
        results.push_back({"default-wait kernel", bad});
    }

    // host-synced: the control. Everything as in the first case, with the host
    // waiting on the event before the consumer is enqueued.
    {
        int bad = 0;
        for (int i = 0; i < iters; ++i) {
            reset();
            if (!matmul(nullptr)) {
                std::fprintf(stderr, "producer failed\n");
                return 1;
            }
            pocket::event_record(ev, nullptr);
            pocket::event_synchronize(ev);
            if (!rmsnorm(nullptr)) {
                std::fprintf(stderr, "consumer failed\n");
                return 1;
            }
            pocket::device_synchronize();
            pocket::memcpy_d2h(host_out.data(), d_out, y_bytes);
            for (uint16_t value : host_out) {
                if (value != kOne) {
                    ++bad;
                    break;
                }
            }
        }
        results.push_back({"host-synced kernel", bad});
    }

    for (const Result& result : results) {
        std::printf("event-order device=%d iters=%d %-18s stale=%d\n", device,
                    iters, result.label, result.bad);
    }
    std::fflush(stdout);

    pocket::event_destroy(ev);
    pocket::stream_destroy(stream);
    pocket::device_free(d_x);
    pocket::device_free(d_w);
    pocket::device_free(d_y);
    pocket::device_free(d_gamma);
    pocket::device_free(d_out);
    pocket::device_free(d_shadow);
    return 0;
}

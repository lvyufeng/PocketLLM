// Prices the TP all-reduce at the shapes one decode step actually issues, and
// separates the two halves of its cost.
//
// The engine runs every collective the same way: an event on the compute stream
// makes the communication stream wait for the producer, the HCCL call is issued
// there, and the host then blocks on the communication stream before returning.
// The block is deliberate -- CANN 9.0 on first-generation 910 can return from a
// default-stream wait before an HCCL in-place output is visible -- but it means
// the per-call cost is "latency of the collective" rather than "time to enqueue
// it", so it is worth knowing which of the two the decode loop is paying for.
//
//   sync        the nullptr-stream path: drain the default stream, reduce, sync
//   comm        the production path: event bracketing plus a per-call sync
//   comm-nosync the same launches with a single sync at the end, i.e. the floor
//               this hardware can reach if the per-call sync is removed
//
// One process per rank, as HCCL requires here (see tp_comm.cpp).
//
//   ./tests/bench_qwen_ascend_allreduce --nccl-id-path P \
//       --tp-world 4 --tp-rank 0 --device 0 [--iters 200]

#include "device_runtime.hpp"
#include "tp_comm.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The shapes a decode step issues: the hidden state once per token, and the
// per-layer projection outputs (one row of hidden, in fp16).
const int kCounts[] = {5120, 5120 * 8, 5120 * 64, 5120 * 512, 5120 * 4096};

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

int main(int argc, char** argv) {
    int world = 4;
    int rank = 0;
    int device = 0;
    int iters = 200;
    std::string id_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tp-world" && i + 1 < argc) world = std::atoi(argv[++i]);
        else if (arg == "--tp-rank" && i + 1 < argc) rank = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (arg == "--nccl-id-path" && i + 1 < argc) id_path = argv[++i];
    }
    if (id_path.empty() || world <= 1 || iters <= 0) {
        std::fprintf(stderr,
                     "usage: bench_qwen_ascend_allreduce --nccl-id-path P "
                     "--tp-world N --tp-rank R [--device D] [--iters N]\n");
        return 1;
    }
    if (!pocket::tp_comm_available()) {
        std::fprintf(stderr, "this build has no HCCL support\n");
        return 1;
    }
    if (!pocket::device_runtime_available() || !pocket::device_set(device)) {
        std::fprintf(stderr, "no usable device %d\n", device);
        return 1;
    }

    void* comm_stream = pocket::stream_create();
    void* ready = pocket::event_create();
    if (comm_stream == nullptr || ready == nullptr) {
        std::fprintf(stderr, "stream/event creation failed\n");
        return 1;
    }

    // The collective wrapper re-binds the device on every call so that a thread
    // which moved between devices cannot issue a collective under a stale
    // context. That is two ACL entry points per collective; if their cost is not
    // negligible against the collective itself, the fix belongs in the wrapper
    // rather than in HCCL.
    {
        const int binds = 2000;
        const double started = now_ms();
        for (int i = 0; i < binds; ++i) pocket::device_set(device);
        const double per_bind_us = (now_ms() - started) * 1000.0 / binds;
        std::printf("acl_device_set rank=%d per_call=%.4f us\n", rank, per_bind_us);
        std::fflush(stdout);
    }

    // The floor every device operation on this platform pays, collective or not.
    // If a bare memset plus a synchronize costs as much as the collective, the
    // per-call number above is the platform's, not the fabric's, and the fix is
    // in how the engine batches work rather than in the collective at all.
    {
        void* tiny = pocket::device_malloc(16);
        void* page = pocket::device_malloc(10 * 1024);
        struct Probe {
            const char* label;
            void* buffer;
            size_t bytes;
        };
        const Probe probes[] = {
            {"sync-only            ", nullptr, 0},
            {"memset-4b+sync       ", tiny, 4},
            {"memset-10kb+sync     ", page, 10 * 1024},
        };
        for (const Probe& probe : probes) {
            const int reps = 200;
            const double started = now_ms();
            for (int i = 0; i < reps; ++i) {
                if (probe.buffer != nullptr) {
                    pocket::device_memset(probe.buffer, 0, probe.bytes);
                }
                pocket::device_synchronize();
            }
            std::printf("device_op rank=%d %s per_call=%.4f ms\n", rank, probe.label,
                        (now_ms() - started) / reps);
            std::fflush(stdout);
        }
        pocket::device_free(tiny);
        pocket::device_free(page);
    }

    for (const int count : kCounts) {
        const size_t bytes = static_cast<size_t>(count) * sizeof(uint16_t);
        uint16_t* buffer = nullptr;
        if (!pocket::device_malloc_into(buffer, bytes)) {
            std::fprintf(stderr, "device_malloc failed for %zu bytes\n", bytes);
            return 1;
        }
        std::vector<uint16_t> host(static_cast<size_t>(count), 0);
        for (uint16_t& value : host) value = 0x3c00u;  // 1.0 in fp16
        if (!pocket::memcpy_h2d(buffer, host.data(), bytes)) {
            std::fprintf(stderr, "memcpy_h2d failed\n");
            return 1;
        }

        // The first collective through a fresh communicator pays connection
        // setup -- measured at ~9.7 s on this machine -- so it must not land in
        // a timed region.
        for (int i = 0; i < 5; ++i) {
            pocket::tp_all_reduce_sum_f16_inplace(world, rank, device,
                                                  id_path.c_str(), buffer, count,
                                                  nullptr);
        }
        if (!pocket::device_synchronize()) {
            std::fprintf(stderr, "warmup synchronize failed\n");
            return 1;
        }

        double started = now_ms();
        for (int i = 0; i < iters; ++i) {
            pocket::tp_all_reduce_sum_f16_inplace(world, rank, device,
                                                  id_path.c_str(), buffer, count,
                                                  nullptr);
        }
        const double sync_ms = (now_ms() - started) / iters;

        started = now_ms();
        for (int i = 0; i < iters; ++i) {
            pocket::event_record(ready, nullptr);
            pocket::stream_wait_event(comm_stream, ready);
            pocket::tp_all_reduce_sum_f16_inplace(world, rank, device,
                                                  id_path.c_str(), buffer, count,
                                                  comm_stream);
            pocket::stream_synchronize(comm_stream);
        }
        const double comm_ms = (now_ms() - started) / iters;

        started = now_ms();
        for (int i = 0; i < iters; ++i) {
            pocket::event_record(ready, nullptr);
            pocket::stream_wait_event(comm_stream, ready);
            pocket::tp_all_reduce_sum_f16_inplace(world, rank, device,
                                                  id_path.c_str(), buffer, count,
                                                  comm_stream);
        }
        // Host time only, taken before the drain below. Each HcclAllReduce is
        // asynchronous on this backend, so this separates "the host cannot issue
        // collectives fast enough" from "the device takes this long to run one",
        // which are fixed by entirely different changes.
        const double host_ms = (now_ms() - started) / iters;
        pocket::stream_synchronize(comm_stream);
        pocket::device_synchronize();
        const double nosync_ms = (now_ms() - started) / iters;

        std::printf(
            "allreduce rank=%d bytes=%-9zu sync=%8.4f ms  comm=%8.4f ms  "
            "comm-nosync=%8.4f ms  host-enqueue=%8.4f ms  gbps(sync)=%6.1f\n",
            rank, bytes, sync_ms, comm_ms, nosync_ms, host_ms,
            bytes * 8.0 * (world - 1) / (sync_ms * 1e-3) / 1e9);
        std::fflush(stdout);
        pocket::device_free(buffer);
    }

    // The production path issues a collective through an event pair: record on
    // the compute stream, wait on the communication stream, reduce. If the pair
    // alone costs what the whole sequence costs, then the per-call figure is a
    // property of the event bracketing rather than of HCCL, and the two are fixed
    // in different places. Run last, so the communicator is already warm and the
    // ~9.7 s first-collective setup cannot land in a timed region.
    {
        const int probe_iters = 200;
        uint16_t* dummy = nullptr;
        if (pocket::device_malloc_into(dummy, 5120 * sizeof(uint16_t))) {
            for (int i = 0; i < 5; ++i) {
                pocket::tp_all_reduce_sum_f16_inplace(world, rank, device,
                                                     id_path.c_str(), dummy,
                                                     5120, nullptr);
            }
            pocket::device_synchronize();
            for (int rep = 0; rep < 2; ++rep) {
                double started = now_ms();
                for (int i = 0; i < probe_iters; ++i) {
                    pocket::event_record(ready, nullptr);
                    pocket::stream_wait_event(comm_stream, ready);
                }
                const double pair_ms = (now_ms() - started) / probe_iters;
                pocket::stream_synchronize(comm_stream);
                pocket::device_synchronize();

                started = now_ms();
                for (int i = 0; i < probe_iters; ++i) {
                    pocket::tp_all_reduce_sum_f16_inplace(world, rank, device,
                                                          id_path.c_str(), dummy,
                                                          5120, comm_stream);
                }
                const double bare_ms = (now_ms() - started) / probe_iters;
                pocket::stream_synchronize(comm_stream);
                pocket::device_synchronize();

                std::printf(
                    "ar_probe rank=%d event-pair-enqueue=%8.4f ms  "
                    "ar-bare-enqueue=%8.4f ms\n",
                    rank, pair_ms, bare_ms);
                std::fflush(stdout);
            }
            pocket::device_free(dummy);
        }
    }

    pocket::event_destroy(ready);
    pocket::stream_destroy(comm_stream);
    return 0;
}

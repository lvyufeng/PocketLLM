// Prices the ACL primitives a hand-written TP collective would be built from,
// and answers whether they exist on this stack at all.
//
// The decode all-reduce A/B (docs/performance/ascend_decode_collective_ab.md)
// leaves one number standing: a bare HcclAllReduce costs ~0.37 ms of *host* time
// per call regardless of message size, against a platform floor of ~20 us for an
// aclnn launch and 2.6 us for a bare aclrtSynchronizeStream. A decode step issues
// 129 of them, so the HCCL API call alone is most of the step.
//
// Every environment knob and every HcclCommConfig expansion mode that could move
// it has been measured and does not, which leaves replacing the collective. That
// is only worth starting if the pieces exist:
//
//   1. Can device 0 address device k's memory at all -- aclrtDeviceCanAccessPeer,
//      and if so does aclrtDeviceEnablePeerAccess succeed?
//   2. Does a cross-device copy work, synchronously and asynchronously, and does
//      the result actually land?
//   3. What does one aclrtMemcpyAsync cost on the host, i.e. what is the unit
//      price a ring or butterfly exchange would pay per hop?
//   4. Does aclrtMemcpyBatchAsync issue several copies for the price of one call?
//      A butterfly all-reduce over four ranks is two rounds of one exchange, so
//      the batch entry point is the only thing that could beat HcclAllReduce's
//      fixed cost rather than merely match it.
//   5. What do the cross-rank sync primitives cost on the host?
//
// This is a single-process multi-device probe. The engine runs one process per
// rank, where a peer's memory additionally has to be imported
// (aclrtIpcMemGetExportKey / aclrtIpcMemImportByKey); that is the next probe and
// is pointless if this one fails first.
//
//   ./tests/bench_qwen_ascend_peer_copy [--bytes 10240] [--iters 200]

#include "device_runtime.hpp"
#include "qwen_ascend_ops.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void line(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void line(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
    std::fflush(stdout);
}

// Raw codes are printed as well as named, because the name list is a set of
// `#define`s in acl/error_codes/rt_error_codes.h that acl_base_rt.h does not pull
// in. Only the three this probe can provoke are translated; anything else prints
// its number alone.
const char* acl_error_note(aclError err) {
    switch (static_cast<int>(err)) {
        case 0: return " (ACL_SUCCESS)";
        case 107000: return " (RT_PARAM_INVALID)";
        case 107012: return " (RT_THREAD_SUBSCRIBE)";
        case 207000: return " (RT_FEATURE_NOT_SUPPORT)";
        case 507899: return " (RT_DRV_INTERNAL_ERROR)";
        default: return "";
    }
}

aclrtMemLocation device_location(uint32_t id) {
    aclrtMemLocation location{};
    location.id = id;
    location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    return location;
}

// fp16 bit patterns for the small integers the prototype sums. Every value it
// uses is below 16, so the encoding is exact and a table is the whole function;
// scaling 0x3c00 by an integer would not be, since the exponent moves.
uint16_t fp16_small_int(unsigned value) {
    static const uint16_t kBits[17] = {
        0x0000, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800,
        0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80, 0x4c00};
    return kBits[value < 17 ? value : 0];
}

}  // namespace

int main(int argc, char** argv) {
    size_t bytes = 10 * 1024;
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bytes" && i + 1 < argc) bytes = std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    }
    if (!pocket::device_runtime_available()) {
        std::fprintf(stderr, "no usable device runtime\n");
        return 1;
    }
    const int devices = pocket::device_count();
    line("peer_probe devices=%d bytes=%zu iters=%d\n", devices, bytes, iters);
    if (devices < 2) {
        std::fprintf(stderr, "peer discovery needs at least two devices\n");
        return 1;
    }
    if (!pocket::device_set(0)) {
        std::fprintf(stderr, "device_set(0) failed\n");
        return 1;
    }

    // 1. Does the topology expose peer access at all?
    std::vector<bool> peer(static_cast<size_t>(devices), false);
    for (int k = 1; k < devices; ++k) {
        int32_t can_access = 0;
        const aclError can_err = aclrtDeviceCanAccessPeer(&can_access, 0, k);
        const aclError enable_err =
            can_access != 0 ? aclrtDeviceEnablePeerAccess(k, 0) : ACL_SUCCESS;
        peer[static_cast<size_t>(k)] = can_access != 0 && enable_err == ACL_SUCCESS;
        line("peer_probe can_access peer=%d err=%d(%s) can=%d enable_err=%d(%s)\n", k,
             static_cast<int>(can_err), acl_error_note(can_err), can_access,
             static_cast<int>(enable_err), acl_error_note(enable_err));
    }

    int destination = -1;
    for (int k = 1; k < devices; ++k) {
        if (peer[static_cast<size_t>(k)]) {
            destination = k;
            break;
        }
    }
    if (destination < 0) {
        line("peer_probe result=no_peer_access; a hand-written collective would have "
             "no way to touch a peer's memory\n");
        return 0;
    }
    line("peer_probe selected_source=0 selected_destination=%d\n", destination);

    void* source = nullptr;
    void* target = nullptr;
    if (!pocket::device_set(0) || !pocket::device_malloc_into(source, bytes)) {
        std::fprintf(stderr, "source allocation failed\n");
        return 1;
    }
    if (!pocket::device_set(destination) || !pocket::device_malloc_into(target, bytes)) {
        std::fprintf(stderr, "target allocation failed\n");
        return 1;
    }

    std::vector<uint16_t> host(static_cast<size_t>(bytes) / sizeof(uint16_t));
    for (uint16_t& value : host) value = 0x3c00u;  // 1.0 in fp16
    if (!pocket::device_set(0) ||
        !pocket::memcpy_h2d(source, host.data(), bytes)) {
        std::fprintf(stderr, "source fill failed\n");
        return 1;
    }

    // 2. A synchronous cross-device copy, and whether the bytes land.
    {
        const aclError err = aclrtMemcpy(target, bytes, source, bytes,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE);
        const bool sync_ok = pocket::device_synchronize();
        std::vector<uint16_t> readback(host.size(), 0);
        const bool read_ok = pocket::device_set(destination) &&
                             pocket::memcpy_d2h(readback.data(), target, bytes);
        const bool same = read_ok && std::memcmp(readback.data(), host.data(), bytes) == 0;
        line("peer_probe sync_d2d err=%d(%s) sync_ok=%d readback_ok=%d data_match=%d\n",
             static_cast<int>(err), acl_error_note(err), sync_ok ? 1 : 0, read_ok ? 1 : 0,
             same ? 1 : 0);
        (void)pocket::device_set(0);
    }

    void* stream = pocket::stream_create();
    if (stream == nullptr) {
        std::fprintf(stderr, "stream creation failed\n");
        return 1;
    }

    // 3. The same copy asynchronously, which is what a hand-written exchange would
    // actually use, plus its per-call host price.
    {
        std::memset(host.data(), 0, bytes);
        const aclError err = aclrtMemcpyAsync(target, bytes, source, bytes,
                                              ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        pocket::stream_synchronize(stream);
        std::vector<uint16_t> readback(host.size(), 0);
        const bool read_ok = pocket::device_set(destination) &&
                             pocket::memcpy_d2h(readback.data(), target, bytes);
        bool same = read_ok;
        for (size_t i = 0; same && i < readback.size(); ++i) {
            same = readback[i] == 0x3c00u;
        }
        line("peer_probe async_d2d err=%d(%s) readback_ok=%d data_match=%d\n",
             static_cast<int>(err), acl_error_note(err), read_ok ? 1 : 0, same ? 1 : 0);
        (void)pocket::device_set(0);
    }

    // Host price of one async cross-device copy, in a long queue so the number is
    // the API call and not the copy.
    {
        double started = now_ms();
        for (int i = 0; i < iters; ++i) {
            aclrtMemcpyAsync(target, bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                             stream);
        }
        const double enqueue_ms = (now_ms() - started) / iters;
        pocket::stream_synchronize(stream);
        line("peer_probe cross_copy_enqueue_ms=%.4f\n", enqueue_ms);
    }

    // The same call with both ends on device 0: separates "this ACL entry point is
    // expensive" from "cross-device routing is expensive".
    {
        void* local = nullptr;
        if (pocket::device_malloc_into(local, bytes)) {
            double started = now_ms();
            for (int i = 0; i < iters; ++i) {
                aclrtMemcpyAsync(local, bytes, source, bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            }
            const double enqueue_ms = (now_ms() - started) / iters;
            pocket::stream_synchronize(stream);
            line("peer_probe local_copy_enqueue_ms=%.4f\n", enqueue_ms);
            pocket::device_free(local);
        }
    }

    // 4. Four copies for one host call. A butterfly all-reduce over four ranks is
    // two rounds of one exchange, so this is the entry point that could beat the
    // fixed HCCL cost rather than merely match it. It answers
    // RT_FEATURE_NOT_SUPPORT on this CANN/910 pair, so the prototype below pays
    // one host call per hop instead.
    {
        const size_t chunk = bytes / 4;
        std::vector<void*> dsts(4);
        std::vector<size_t> dest_maxs(4, chunk);
        std::vector<void*> srcs(4);
        std::vector<size_t> sizes(4, chunk);
        for (int i = 0; i < 4; ++i) {
            dsts[static_cast<size_t>(i)] =
                static_cast<uint8_t*>(target) + static_cast<size_t>(i) * chunk;
            srcs[static_cast<size_t>(i)] =
                static_cast<uint8_t*>(source) + static_cast<size_t>(i) * chunk;
        }
        aclrtMemcpyBatchAttr attr{};
        attr.dstLoc = device_location(static_cast<uint32_t>(destination));
        attr.srcLoc = device_location(0);
        size_t attr_index = 0;
        size_t fail_index = SIZE_MAX;
        const aclError err = aclrtMemcpyBatchAsync(
            dsts.data(), dest_maxs.data(), srcs.data(), sizes.data(), 4, &attr,
            &attr_index, 1, &fail_index, stream);
        double started = now_ms();
        for (int i = 0; i < iters; ++i) {
            aclrtMemcpyBatchAsync(dsts.data(), dest_maxs.data(), srcs.data(), sizes.data(),
                                  4, &attr, &attr_index, 1, &fail_index, stream);
        }
        const double enqueue_ms = (now_ms() - started) / iters;
        pocket::stream_synchronize(stream);
        line("peer_probe batch4_enqueue_ms=%.4f err=%d(%s) fail_index=%zu\n",
             enqueue_ms, static_cast<int>(err), acl_error_note(err), fail_index);
    }

    // 5. The ordering primitive an exchange needs between rounds. Host price only;
    // stream-ordered record/wait is what makes a hand-written ring correct.
    {
        aclrtNotify notify = nullptr;
        const aclError create_err = aclrtCreateNotify(&notify, 0);
        if (create_err == ACL_SUCCESS) {
            double started = now_ms();
            for (int i = 0; i < iters; ++i) {
                aclrtRecordNotify(notify, stream);
                aclrtWaitAndResetNotify(notify, stream, 0);
            }
            const double pair_ms = (now_ms() - started) / iters;
            pocket::stream_synchronize(stream);
            uint32_t notify_id = 0;
            const aclError id_err = aclrtGetNotifyId(notify, &notify_id);
            line("peer_probe notify_record_wait_ms=%.4f notify_id_err=%d(%s) id=%u\n",
                 pair_ms, static_cast<int>(id_err), acl_error_note(id_err), notify_id);
            aclrtDestroyNotify(notify);
        } else {
            line("peer_probe notify_create_err=%d(%s)\n", static_cast<int>(create_err),
                 acl_error_note(create_err));
        }
    }

    // 6. A prototype all-reduce over the four devices the TP4 engine uses, built
    // out of the pieces priced above. The construction is
    //
    //   every rank pushes its partial into every peer's slot for it,
    //   then every rank sums the slots locally:
    //
    //   rank r:  for k != r:  copy  local[r] -> slot[r] on device k
    //   barrier
    //   rank r:  out[r] = local[r] + sum over k != r of slot[k] on device r
    //
    // Only the sum needs a device kernel, and the engine already has one --
    // qwen_add_inplace_f16_ascend, which is the same aclnnInplaceAdd the layer
    // itself uses. The price this reports is the unit price the decode step pays
    // 129 times, against the 0.4430 ms HcclAllReduce it would replace.
    //
    // The barrier is the part a one-process-per-rank implementation gets from
    // aclrtRecordNotify / aclrtWaitAndResetNotify over an exported notify key
    // (aclrtNotifyGetExportKey / aclrtNotifyImportByKey). Four ranks in one
    // process cannot express that, so it is emulated by synchronising the four
    // streams and priced as its own term.
    {
        const int world = devices < 4 ? devices : 4;
        const int count = static_cast<int>(bytes / sizeof(uint16_t));
        std::vector<void*> rstream(static_cast<size_t>(world), nullptr);
        std::vector<uint16_t*> local(static_cast<size_t>(world), nullptr);
        std::vector<uint16_t*> out(static_cast<size_t>(world), nullptr);
        std::vector<std::vector<uint16_t*>> slot(
            static_cast<size_t>(world),
            std::vector<uint16_t*>(static_cast<size_t>(world), nullptr));

        bool allocated = true;
        for (int r = 0; r < world && allocated; ++r) {
            pocket::device_set(r);
            rstream[static_cast<size_t>(r)] = pocket::stream_create();
            allocated = rstream[static_cast<size_t>(r)] != nullptr &&
                        pocket::device_malloc_into(local[static_cast<size_t>(r)],
                                                   bytes) &&
                        pocket::device_malloc_into(out[static_cast<size_t>(r)], bytes);
            for (int k = 0; k < world && allocated; ++k) {
                allocated = pocket::device_malloc_into(slot[static_cast<size_t>(r)]
                                                           [static_cast<size_t>(k)],
                                                       bytes);
            }
        }

        // Peer access is per context, so every rank has to enable it towards every
        // other; section 1 only did that from device 0.
        int pairs_enabled = 0;
        int pairs_seen = 0;
        for (int r = 0; r < world; ++r) {
            pocket::device_set(r);
            for (int k = 0; k < world; ++k) {
                if (k == r) continue;
                ++pairs_seen;
                int32_t can = 0;
                if (aclrtDeviceCanAccessPeer(&can, r, k) == ACL_SUCCESS && can != 0 &&
                    aclrtDeviceEnablePeerAccess(k, 0) == ACL_SUCCESS) {
                    ++pairs_enabled;
                }
            }
        }
        if (!allocated || pairs_enabled != pairs_seen) {
            line("peer_probe proto_alloc_ok=%d peer_pairs=%d/%d skipped\n",
                 allocated ? 1 : 0, pairs_enabled, pairs_seen);
        } else {
            std::vector<uint16_t> fill(static_cast<size_t>(count), 0);
            // Distinguishable per rank so a slot mix-up changes the sum.
            for (int r = 0; r < world; ++r) {
                std::fill(fill.begin(), fill.end(),
                          fp16_small_int(static_cast<unsigned>(r + 1)));
                pocket::device_set(r);
                pocket::memcpy_h2d(local[static_cast<size_t>(r)], fill.data(), bytes);
            }

            double push_ms = 0.0;
            double local_ms = 0.0;
            double reduce_ms = 0.0;
            double barrier_ms = 0.0;
            for (int it = 0; it < iters; ++it) {
                const double t0 = now_ms();
                for (int r = 0; r < world; ++r) {
                    pocket::device_set(r);
                    for (int k = 0; k < world; ++k) {
                        if (k == r) continue;
                        aclrtMemcpyAsync(slot[static_cast<size_t>(k)]
                                             [static_cast<size_t>(r)],
                                         bytes, local[static_cast<size_t>(r)], bytes,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE,
                                         rstream[static_cast<size_t>(r)]);
                    }
                }
                const double t1 = now_ms();
                for (int r = 0; r < world; ++r) {
                    pocket::stream_synchronize(rstream[static_cast<size_t>(r)]);
                }
                const double t2 = now_ms();
                for (int r = 0; r < world; ++r) {
                    pocket::device_set(r);
                    pocket::memcpy_d2d_async(out[static_cast<size_t>(r)],
                                             local[static_cast<size_t>(r)], bytes,
                                             rstream[static_cast<size_t>(r)]);
                    for (int k = 0; k < world; ++k) {
                        if (k == r) continue;
                        pocket::qwen_add_inplace_f16_ascend(
                            out[static_cast<size_t>(r)],
                            slot[static_cast<size_t>(r)][static_cast<size_t>(k)], count,
                            rstream[static_cast<size_t>(r)]);
                    }
                }
                const double t3 = now_ms();
                for (int r = 0; r < world; ++r) {
                    pocket::stream_synchronize(rstream[static_cast<size_t>(r)]);
                }
                push_ms += t1 - t0;
                barrier_ms += (t2 - t1) + (now_ms() - t3);
                local_ms += t3 - t2;
            }
            pocket::device_synchronize();

            // The loop pays two device_set calls per rank per iteration, which a
            // one-process-per-rank implementation pays once at startup. Priced and
            // subtracted rather than argued about.
            const int switches = 2 * world;
            double switch_ms = 0.0;
            for (int i = 0; i < switches * iters; ++i) {
                const double t0 = now_ms();
                pocket::device_set(i % world);
                switch_ms += now_ms() - t0;
            }
            pocket::device_set(0);

            std::vector<uint16_t> readback(static_cast<size_t>(count), 0);
            const uint16_t expected = fp16_small_int(
                static_cast<unsigned>(world * (world + 1) / 2));
            bool correct = true;
            for (int r = 0; r < world; ++r) {
                if (!pocket::device_set(r) ||
                    !pocket::memcpy_d2h(readback.data(), out[static_cast<size_t>(r)],
                                        bytes)) {
                    correct = false;
                    break;
                }
                for (int i = 0; i < count; ++i) {
                    if (readback[static_cast<size_t>(i)] != expected) {
                        correct = false;
                        break;
                    }
                }
                if (!correct) break;
            }
            pocket::device_set(0);

            // Price the reduce kernel on its own, so a slow collective can be read
            // as "the sum is expensive" or "the exchange is expensive". The third
            // variant switches device context between calls, which is what the
            // single-process prototype above does per rank; a one-process-per-rank
            // deployment never does it.
            double add_enqueue_ms = 0.0;
            double add_sync_ms = 0.0;
            double add_switch_ms = 0.0;
            {
                pocket::device_set(0);
                const uint16_t* src = slot[0][1];
                uint16_t* dst = out[0];
                double t0 = now_ms();
                for (int i = 0; i < iters; ++i) {
                    pocket::qwen_add_inplace_f16_ascend(dst, src, count,
                                                        rstream[0]);
                }
                add_enqueue_ms = (now_ms() - t0) / iters;
                pocket::stream_synchronize(rstream[0]);
                t0 = now_ms();
                for (int i = 0; i < iters; ++i) {
                    pocket::qwen_add_inplace_f16_ascend(dst, src, count,
                                                        rstream[0]);
                    pocket::stream_synchronize(rstream[0]);
                }
                add_sync_ms = (now_ms() - t0) / iters;
                pocket::device_set(0);
                t0 = now_ms();
                for (int i = 0; i < iters; ++i) {
                    pocket::device_set(i % world);
                    pocket::qwen_add_inplace_f16_ascend(dst, src, count,
                                                        rstream[0]);
                }
                add_switch_ms = (now_ms() - t0) / iters;
                pocket::device_set(0);
            }

            const double total_ms =
                (push_ms + local_ms + reduce_ms + barrier_ms) / iters;
            const double per_rank_ms = (total_ms - switch_ms / iters) / world;
            line("peer_probe proto_allreduce world=%d total_ms=%.4f push_ms=%.4f "
                 "local_ms=%.4f reduce_ms=%.4f barrier_ms=%.4f device_set_ms=%.4f "
                 "per_rank_ms=%.4f correct=%d\n",
                 world, total_ms, push_ms / iters, local_ms / iters,
                 reduce_ms / iters, barrier_ms / iters, switch_ms / iters,
                 per_rank_ms, correct ? 1 : 0);
            line("peer_probe add_launch count=%d enqueue_ms=%.4f sync_each_ms=%.4f "
                 "switching_ms=%.4f\n",
                 count, add_enqueue_ms, add_sync_ms, add_switch_ms);
        }

        for (int r = 0; r < world; ++r) {
            pocket::device_set(r);
            for (int k = 0; k < world; ++k) {
                pocket::device_free(slot[static_cast<size_t>(r)][static_cast<size_t>(k)]);
            }
            pocket::device_free(local[static_cast<size_t>(r)]);
            pocket::device_free(out[static_cast<size_t>(r)]);
            pocket::stream_destroy(rstream[static_cast<size_t>(r)]);
        }
    }

    pocket::stream_destroy(stream);
    pocket::device_set(destination);
    pocket::device_free(target);
    pocket::device_set(0);
    pocket::device_free(source);
    return 0;
}

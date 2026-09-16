// Answers whether the peer-copy primitives bench_qwen_ascend_peer_copy prices can
// be used from the layout the engine actually runs: one process per rank.
//
// That probe works in a single process, where a peer's address is an ordinary
// pointer. The engine launches one process per TP rank (HcclCommInitAll cannot be
// used on this stack; see docs/performance/ascend_decode_collective_ab.md), so a
// hand-written collective has to reach a peer's memory across a process boundary
// first. AscendCL provides exactly the pair CUDA does for that:
// aclrtIpcMemGetExportKey / aclrtIpcMemImportByKey, with an
// ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS flag to make the imported region
// directly addressable, and the matching notify pair
// (aclrtNotifyGetExportKey / aclrtNotifyImportByKey) for the ordering that an
// exchange needs between rounds.
//
// So this probe runs the four pieces a hand-written all-reduce is made of, across
// four processes, and reports what each one does here:
//
//   1. export a device buffer and find it from another process,
//   2. push into a peer's imported buffer with aclrtMemcpyAsync and have the bytes
//      land where they were sent,
//   3. record a notify in one process and wait on it in another, which is the only
//      thing that can order the rounds without a host rendezvous per step,
//   4. the host price of that whole arrive/wait exchange.
//
// Keys are exchanged through files in --dir; that is rendezvous, not the
// collective, and it is not timed.
//
//   for r in 0 1 2 3; do
//     ./tests/bench_qwen_ascend_ipc_exchange --world 4 --rank $r --device $r \
//         --dir /tmp/ipc_probe & done; wait

#include "device_runtime.hpp"

#include <acl/acl.h>

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// AscendCL does not publish a key length. HCCL's own keys are a few hundred bytes
// of text, so this is generous; the call fails loudly rather than truncating if it
// ever is not enough.
constexpr size_t kKeyLen = 4096;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int rank = 0;
int world = 1;

void line(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void line(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::printf("ipc_probe rank=%d ", rank);
    std::vprintf(fmt, args);
    std::printf("\n");
    std::fflush(stdout);
    va_end(args);
}

std::string path_for(const std::string& dir, const std::string& kind, int index) {
    return dir + "/" + kind + "." + std::to_string(index) + ".key";
}

// Receive slots are keyed by (owner, slot), so the kind carries the owner. One
// rank's slot for rank r and its slot for rank j are different buffers, and a
// peer has to be able to name the one it wants.
std::string slot_kind(int owner) { return "recv." + std::to_string(owner); }

// Writes `key` where the other ranks will look for it, via a rename so no reader
// ever sees a half-written file.
bool publish(const std::string& dir, const std::string& kind, int owner, const char* key) {
    const std::string final_path = path_for(dir, kind, owner);
    const std::string temp_path = final_path + "." + std::to_string(rank) + ".tmp";
    std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(key, static_cast<std::streamsize>(std::strlen(key) + 1));
    out.close();
    return std::rename(temp_path.c_str(), final_path.c_str()) == 0;
}

bool wait_for_path(const std::string& path, double deadline) {
    while (now_ms() < deadline) {
        std::ifstream in(path, std::ios::binary);
        if (in.good()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

// Rendezvous only; never inside the timed region.
bool wait_for_all(const std::string& dir, const std::string& kind) {
    const double deadline = now_ms() + 120000.0;
    for (int k = 0; k < world; ++k) {
        if (!wait_for_path(path_for(dir, kind, k), deadline)) return false;
    }
    return true;
}

// Every (owner, slot) pair, since an import can need any of them.
bool wait_for_slots(const std::string& dir) {
    const double deadline = now_ms() + 120000.0;
    for (int owner = 0; owner < world; ++owner) {
        for (int slot = 0; slot < world; ++slot) {
            if (!wait_for_path(path_for(dir, slot_kind(owner), slot), deadline)) {
                return false;
            }
        }
    }
    return true;
}

bool read_key(const std::string& dir, const std::string& kind, int owner, char* key) {
    std::ifstream in(path_for(dir, kind, owner), std::ios::binary);
    if (!in) return false;
    in.read(key, static_cast<std::streamsize>(kKeyLen));
    key[kKeyLen - 1] = '\0';
    return in.gcount() > 1;
}

// AscendCL gates the import with a PID whitelist: by default an exported region
// encodes the exporter's pid and only that pid may open it, which is exactly the
// opposite of what a cross-process collective needs. Two escape hatches exist and
// this probe can be told to use either, so which one works here is measured rather
// than assumed:
//
//   --no-pid-check   export with ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION
//   --set-import-pid call aclrtIpcMemSetImportPid on the peer's key with our own pid
//                    before importing it
enum class PidMode { Default, NoPidCheck, SetImportPid };

const char* acl_error_note(aclError err) {
    switch (static_cast<int>(err)) {
        case 0: return " (ACL_SUCCESS)";
        case 107012: return " (RT_PARAM_INVALID)";
        case 207000: return " (RT_FEATURE_NOT_SUPPORT)";
        case 507899: return " (RT_DRV_INTERNAL_ERROR)";
        default: return "";
    }
}

}  // namespace

int main(int argc, char** argv) {
    size_t bytes = 10 * 1024;
    int iters = 200;
    int device = -1;
    std::string dir;
    PidMode pid_mode = PidMode::Default;
    // The wait is the expensive half of the round, and the fact that it is imported
    // from another process is the obvious suspect. These two arms separate "waiting
    // on a notify at all costs this" from "waiting on a *remote* notify does":
    // --self-wait waits on this rank's own notify, whose signal is already ordered
    // behind the pushes enqueued ahead of it, so it prices the call with no peer
    // involved. --no-wait drops the waits and free-runs, which prices the pushes.
    bool use_wait = true;
    bool self_wait = false;
    // A ring barrier needs one remote wait per round instead of world-1, so this is
    // the arm that separates "a remote wait costs this" from "waiting on all of them
    // costs this". It is not a correct barrier on its own; it prices one.
    bool wait_one = false;
    // The notify import flag is the other half of that question: ENABLE_PEER_ACCESS
    // is what the memory import needs, but a notify may take a slower path when it
    // is opened for peer access than when it is opened plainly.
    bool notify_peer_access = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bytes" && i + 1 < argc) bytes = std::strtoul(argv[++i], nullptr, 10);
        else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (arg == "--rank" && i + 1 < argc) rank = std::atoi(argv[++i]);
        else if (arg == "--world" && i + 1 < argc) world = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (arg == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (arg == "--no-pid-check") pid_mode = PidMode::NoPidCheck;
        else if (arg == "--set-import-pid") pid_mode = PidMode::SetImportPid;
        else if (arg == "--no-wait") use_wait = false;
        else if (arg == "--self-wait") self_wait = true;
        else if (arg == "--wait-one") wait_one = true;
        else if (arg == "--notify-no-peer-access") notify_peer_access = false;
    }
    if (device < 0) device = rank;
    if (dir.empty() || world < 2 || rank < 0 || rank >= world) {
        std::fprintf(stderr,
                     "usage: %s --world N --rank R [--device D] --dir DIR "
                     "[--bytes 10240] [--iters 200] [--no-pid-check|--set-import-pid]\n",
                     argv[0]);
        return 2;
    }
    if (!pocket::device_runtime_available() || !pocket::device_set(device)) {
        std::fprintf(stderr, "device %d unusable\n", device);
        return 1;
    }
    line("start bytes=%zu iters=%d device=%d", bytes, iters, device);

    // Peer access is per context and has to be enabled towards every other rank
    // before an imported region is addressable.
    int pairs = 0;
    for (int k = 0; k < world; ++k) {
        if (k == rank) continue;
        int32_t can = 0;
        if (aclrtDeviceCanAccessPeer(&can, device, k) == ACL_SUCCESS && can != 0 &&
            aclrtDeviceEnablePeerAccess(k, 0) == ACL_SUCCESS) {
            ++pairs;
        }
    }

    uint16_t* local = nullptr;
    if (!pocket::device_malloc_into(local, bytes)) {
        line("result=alloc_failed");
        return 1;
    }
    // One receive slot per rank, all on this process's own device; rank k writes
    // into slot k of every peer.
    std::vector<uint16_t*> recv(static_cast<size_t>(world), nullptr);
    for (int k = 0; k < world; ++k) {
        if (!pocket::device_malloc_into(recv[static_cast<size_t>(k)], bytes)) {
            line("result=alloc_failed");
            return 1;
        }
    }

    void* stream = pocket::stream_create();
    if (stream == nullptr) {
        line("result=stream_failed");
        return 1;
    }

    // 1. Export every one of this rank's receive slots, and import the one slot
    // from each peer that this rank's contribution belongs in.
    //
    // The slot a rank imports has to be named for this rank, not for the peer: rank
    // j holds one slot per source rank, and the buffer rank r pushes into there is
    // rank j's slot r. Indexing the peer's slots by the peer instead collides every
    // source onto one buffer and leaves the others untouched, which is what an
    // earlier revision of this probe did.
    const uint64_t export_flags =
        pid_mode == PidMode::NoPidCheck
            ? ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION
            : ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT;
    int exported = 0;
    aclError first_export_err = ACL_SUCCESS;
    for (int k = 0; k < world; ++k) {
        char key[kKeyLen] = {0};
        const aclError err = aclrtIpcMemGetExportKey(recv[static_cast<size_t>(k)], bytes,
                                                     key, kKeyLen, export_flags);
        if (err == ACL_SUCCESS && publish(dir, slot_kind(rank), k, key)) ++exported;
        else if (first_export_err == ACL_SUCCESS) first_export_err = err;
    }
    line("export ok=%d/%d err=%d%s peer_pairs=%d/%d pid_mode=%d", exported, world,
         static_cast<int>(first_export_err), acl_error_note(first_export_err), pairs,
         world - 1, static_cast<int>(pid_mode));
    if (exported != world) {
        line("result=export_failed");
        return 1;
    }
    if (!wait_for_slots(dir)) {
        line("result=rendezvous_timeout");
        return 1;
    }

    // remote[j] is rank j's slot for *this* rank, so pushing into it puts this
    // rank's contribution where rank j expects to find it.
    std::vector<uint16_t*> remote(static_cast<size_t>(world), nullptr);
    std::vector<std::string> remote_keys(static_cast<size_t>(world));
    int imported = 0;
    aclError first_import_err = ACL_SUCCESS;
    aclError first_pid_err = ACL_SUCCESS;
    for (int j = 0; j < world; ++j) {
        if (j == rank) continue;
        char peer_key[kKeyLen] = {0};
        if (!read_key(dir, slot_kind(j), rank, peer_key)) continue;
        if (pid_mode == PidMode::SetImportPid) {
            int32_t self_pid = static_cast<int32_t>(::getpid());
            const aclError perr = aclrtIpcMemSetImportPid(peer_key, &self_pid, 1);
            if (perr != ACL_SUCCESS && first_pid_err == ACL_SUCCESS) first_pid_err = perr;
        }
        void* ptr = nullptr;
        const aclError err = aclrtIpcMemImportByKey(
            &ptr, peer_key, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
        if (err == ACL_SUCCESS && ptr != nullptr) {
            remote[static_cast<size_t>(j)] = static_cast<uint16_t*>(ptr);
            remote_keys[static_cast<size_t>(j)] = peer_key;
            ++imported;
        } else if (first_import_err == ACL_SUCCESS) {
            first_import_err = err;
        }
    }
    line("import ok=%d/%d err=%d%s set_pid_err=%d%s", imported, world - 1,
         static_cast<int>(first_import_err), acl_error_note(first_import_err),
         static_cast<int>(first_pid_err), acl_error_note(first_pid_err));
    if (imported != world - 1) {
        line("result=import_failed");
        return 1;
    }

    // 2. An exported notify per rank, imported by everyone else. This is the
    // ordering primitive: without it, nothing tells rank k that rank j's bytes
    // have landed in k's slot.
    aclrtNotify own_notify = nullptr;
    const aclError notify_create_err = aclrtCreateNotify(&own_notify, 0);
    bool notify_ready = false;
    std::vector<aclrtNotify> peer_notify(static_cast<size_t>(world), nullptr);
    if (notify_create_err == ACL_SUCCESS) {
        char nkey[kKeyLen] = {0};
        const aclError nerr = aclrtNotifyGetExportKey(
            own_notify, nkey, kKeyLen,
            pid_mode == PidMode::NoPidCheck
                ? ACL_RT_NOTIFY_EXPORT_FLAG_DISABLE_PID_VALIDATION
                : ACL_RT_NOTIFY_EXPORT_FLAG_DEFAULT);
        notify_ready = nerr == ACL_SUCCESS && publish(dir, "notify", rank, nkey);
        line("notify_export err=%d%s ok=%d", static_cast<int>(nerr), acl_error_note(nerr),
             notify_ready ? 1 : 0);
    } else {
        line("notify_create err=%d%s", static_cast<int>(notify_create_err),
             acl_error_note(notify_create_err));
    }
    int notify_imported = 0;
    if (notify_ready && wait_for_all(dir, "notify")) {
        for (int k = 0; k < world; ++k) {
            if (k == rank) continue;
            char nkey[kKeyLen] = {0};
            if (!read_key(dir, "notify", k, nkey)) continue;
            aclrtNotify imported_notify = nullptr;
            if (aclrtNotifyImportByKey(&imported_notify, nkey,
                                       notify_peer_access
                                           ? ACL_RT_NOTIFY_IMPORT_FLAG_ENABLE_PEER_ACCESS
                                           : ACL_RT_NOTIFY_IMPORT_FLAG_DEFAULT) ==
                    ACL_SUCCESS &&
                imported_notify != nullptr) {
                peer_notify[static_cast<size_t>(k)] = imported_notify;
                ++notify_imported;
            }
        }
    }
    line("notify_import ok=%d/%d", notify_imported, world - 1);

    // 3. Fill this rank's contribution and push it into every peer's slot for this
    // rank. Distinguishable per rank so a slot mix-up changes a readback.
    {
        std::vector<uint16_t> fill(static_cast<size_t>(bytes) / sizeof(uint16_t),
                                   static_cast<uint16_t>(0x3c00u + 0x400u * rank));
        if (!pocket::memcpy_h2d(local, fill.data(), bytes)) {
            line("result=fill_failed");
            return 1;
        }
    }

    const bool barrier_available = notify_imported == world - 1;
    double push_ms = 0.0;
    double wait_ms = 0.0;
    double total_ms = 0.0;
    aclError first_copy_err = ACL_SUCCESS;
    int copy_calls = 0;
    for (int it = 0; it < iters; ++it) {
        const double t0 = now_ms();
        for (int j = 0; j < world; ++j) {
            if (j == rank) continue;
            // The return value is the whole point of the readback below: a rejected
            // cross-process copy is silent otherwise, and the slots simply stay
            // whatever they were.
            const aclError cerr = aclrtMemcpyAsync(
                remote[static_cast<size_t>(j)], bytes, local, bytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            ++copy_calls;
            if (cerr != ACL_SUCCESS && first_copy_err == ACL_SUCCESS) first_copy_err = cerr;
        }
        if (barrier_available) aclrtRecordNotify(own_notify, stream);
        const double t1 = now_ms();
        if (barrier_available && use_wait) {
            if (self_wait) {
                aclrtWaitAndResetNotify(own_notify, stream, 60000);
            } else if (wait_one) {
                const int k = (rank + 1) % world;
                aclrtWaitAndResetNotify(peer_notify[static_cast<size_t>(k)], stream, 60000);
            } else {
                for (int k = 0; k < world; ++k) {
                    if (k == rank) continue;
                    aclrtWaitAndResetNotify(peer_notify[static_cast<size_t>(k)], stream,
                                            60000);
                }
            }
        }
        const double t2 = now_ms();
        pocket::stream_synchronize(stream);
        const double t3 = now_ms();
        push_ms += t1 - t0;
        wait_ms += t2 - t1;
        total_ms += t3 - t0;
    }
    pocket::device_synchronize();

    // total_ms against push+wait says whether the notify wait blocks the host here
    // or only enqueues: a wait that returns immediately leaves a gap between the
    // two, and a wait that polls shows up inside wait_ms.
    line("exchange world=%d bytes=%zu push_ms=%.4f wait_ms=%.4f total_ms=%.4f barrier=%d "
         "mode=%s",
         world, bytes, push_ms / iters, wait_ms / iters, total_ms / iters,
         barrier_available ? 1 : 0,
         !use_wait ? "no-wait"
                   : (self_wait ? "self-wait" : (wait_one ? "wait-one" : "peer-wait")));
    line("push copy_calls=%d err=%d%s local_ptr=%p remote_ptr=%p", copy_calls,
         static_cast<int>(first_copy_err), acl_error_note(first_copy_err), (void*)local,
         world > 1 ? (void*)remote[static_cast<size_t>((rank + 1) % world)] : nullptr);

    // 4. Does the data land where it was sent? Two reads, because "the write went
    // elsewhere" and "the write has not become visible to the peer yet" look
    // identical from one sample: our own slot carries the push, and the peer's
    // buffer as this process maps it carries the same push.
    //
    // A host rendezvous and a settle delay come first, so this is not measuring a
    // race between ranks that are still in the loop.
    publish(dir, "done", rank, "1");
    wait_for_all(dir, "done");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::vector<uint16_t> readback(static_cast<size_t>(bytes) / sizeof(uint16_t), 0);
    int slots_ok = 0;
    std::string slot_report;
    for (int k = 0; k < world; ++k) {
        if (!pocket::memcpy_d2h(readback.data(), recv[static_cast<size_t>(k)], bytes)) {
            break;
        }
        // Slot k carries rank k's contribution. This rank's own slot is the one
        // nobody pushes into -- the local term of the all-reduce stays in the
        // reduce, so that buffer is expected to be untouched, not to hold us.
        const uint16_t expected = k == rank ? 0u : static_cast<uint16_t>(0x3c00u + 0x400u * k);
        bool ok = true;
        for (uint16_t value : readback) {
            if (value != expected) {
                ok = false;
                break;
            }
        }
        if (ok) ++slots_ok;
        char buf[72];
        std::snprintf(buf, sizeof(buf), " slot%d=0x%04x(want 0x%04x)", k, readback[0],
                      static_cast<unsigned>(expected));
        slot_report += buf;
    }
    line("result=%s slots_ok=%d/%d first_value=0x%04x",
         slots_ok == world ? "ok" : "mismatch", slots_ok, world, readback[0]);
    line("slots%s", slot_report.c_str());

    // Same bytes read through the imported pointer, one hop out: rank r's own
    // contribution as rank k's memory. If the slots above are stale but these are
    // right, the mapping is sound and only the visibility is in question.
    for (int k = 0; k < world; ++k) {
        if (k == rank) continue;
        if (!pocket::memcpy_d2h(readback.data(), remote[static_cast<size_t>(k)], bytes)) {
            continue;
        }
        int good = 0;
        for (uint16_t value : readback) {
            if (value == static_cast<uint16_t>(0x3c00u + 0x400u * rank)) ++good;
        }
        line("peer_view k=%d expect_own=0x%04x matched=%d/%zu first=0x%04x", k,
             static_cast<unsigned>(0x3c00u + 0x400u * rank), good, readback.size(),
             readback[0]);
    }

    for (int k = 0; k < world; ++k) {
        if (k != rank && !remote_keys[static_cast<size_t>(k)].empty()) {
            aclrtIpcMemClose(remote_keys[static_cast<size_t>(k)].c_str());
        }
    }
    if (own_notify != nullptr) aclrtDestroyNotify(own_notify);
    pocket::stream_destroy(stream);
    for (int k = 0; k < world; ++k) pocket::device_free(recv[static_cast<size_t>(k)]);
    pocket::device_free(local);
    return slots_ok == world ? 0 : 1;
}

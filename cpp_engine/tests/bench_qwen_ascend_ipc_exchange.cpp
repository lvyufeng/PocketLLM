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
//   4. the same arrival signal carried by a cross-process *event* instead, which the
//      stream can wait on (aclrtStreamWaitEvent) rather than the host
//      (aclrtWaitAndResetNotify) -- the difference between a wait the host pays and
//      one only the device pays,
//   5. the host price of that whole arrive/wait exchange.
//
// Findings, in the order they were measured:
//
//   * the copies cross the process boundary and the arithmetic comes out right;
//   * `aclrtWaitAndResetNotify` returns 107000 RT_PARAM_INVALID for a notify imported
//     from another process, so a notify can be exported and imported but not waited
//     on -- which is why the timing of those arms is a rejected call, not a barrier;
//   * neither `aclrtCreateEventExWithFlag` nor `aclrtCreateEventWithFlag` accepts
//     ACL_EVENT_IPC here (207000 and 107000 respectively), so there is no
//     stream-side wait across processes either;
//   * `--poll-wait` is what is left: the arrival signal is the payload itself. Each
//     rank stamps the first element with a per-round counter, pushes, then polls its
//     own receive slots with a small D2H read until every peer's stamp for this round
//     is there. It needs no notify and no event, and it *is* a barrier -- `--skew-us`
//     shows the wait growing with the injected skew, on the ranks that arrived early
//     and not on the ones that arrived late;
//   * a single receive buffer per (peer, source) pair is not enough to carry that
//     scheme, and the failure is a stall rather than a wrong number. If a peer
//     reaches round N+1 before this rank has looked for round N, its push overwrites
//     the very stamp being waited for, and the poll can never match -- it burns its
//     whole deadline and then resynchronises one round late. `--poll-single-set`
//     selects that layout and measures the stall; the default is two buffer sets
//     used on alternate rounds, which is what makes the write land in the other set
//     while a look is outstanding.
//
// Keys are exchanged through files in --dir; that is rendezvous, not the
// collective, and it is not timed.
//
//   for r in 0 1 2 3; do
//     ./tests/bench_qwen_ascend_ipc_exchange --world 4 --rank $r --device $r \
//         --dir /tmp/ipc_probe & done; wait

#include "device_runtime.hpp"
#include "qwen_ascend_ops.hpp"

#include <acl/acl.h>

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// Decodes an fp16 bit pattern exactly. Every value this probe sums is a small
// integer or a sum of them, all of which fp16 represents exactly, so a decoder is
// enough to check the reduce -- and it is needed rather than a table because the
// per-rank ladder below is not the integers 1..world.
double fp16_exact(uint16_t bits) {
    const int exp = (bits >> 10) & 0x1f;
    const int frac = bits & 0x3ff;
    if (exp == 0) return std::ldexp(static_cast<double>(frac), -24);
    if (exp == 31) return 0.0;
    return std::ldexp(1.0 + static_cast<double>(frac) / 1024.0, exp - 15);
}

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
bool wait_for_slots(const std::string& dir, int slots_per_owner) {
    const double deadline = now_ms() + 120000.0;
    for (int owner = 0; owner < world; ++owner) {
        for (int slot = 0; slot < slots_per_owner; ++slot) {
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
        case 107000: return " (RT_PARAM_INVALID)";
        case 107012: return " (RT_THREAD_SUBSCRIBE)";
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
    // --ipc-event replaces the host-side notify wait with a cross-process *event*,
    // which is the one shape that can avoid the wait entirely: a notify can only be
    // waited on from the host (aclrtWaitAndResetNotify), while an event opened from
    // another process can be waited on by the stream (aclrtStreamWaitEvent). Same
    // arrive/wait semantics, but the host only enqueues.
    bool use_event = false;
    // --wait-primed times the remote waits with the signals already recorded, which
    // separates "the signal takes 0.23 ms to arrive" from "a remote wait is charged
    // 0.23 ms whether or not it has to wait".
    bool wait_primed = false;
    // --poll-wait is the barrier that is left when neither a notify nor an event can
    // be waited on across processes: the arrival signal is the *data itself*, and the
    // host polls its own slot until the bytes it is waiting for have appeared. It
    // costs a D2H copy per look, so the number it produces is the cross-device
    // visibility latency rather than an API price.
    bool poll_wait = false;
    // --poll-single-set gives the stamp scheme one buffer per (peer, source) pair
    // instead of two. That is the natural first cut and it is wrong under load: a
    // peer that reaches round N+1 before this rank has looked for round N overwrites
    // the stamp being waited for, so the poll runs to its deadline instead of
    // matching. Keeping the flag lets the stall be measured rather than argued about.
    bool poll_single_set = false;
    // --skew-us makes the ranks arrive at the barrier at different times, which is the
    // only way to tell a real barrier from a measurement riding on all four ranks
    // happening to run in lockstep. Rank r sleeps r*us before its pushes each round,
    // so if the barrier absorbs skew, rank 0's wait_ms grows by about (world-1)*us
    // and the last rank's does not. Keep it well under one round: a rank that gets a
    // whole round ahead would overwrite a slot its peer has not read yet, which is
    // the buffer-reuse hazard the stamp scheme turns into a hang rather than garbage.
    int skew_us = 0;
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
        else if (arg == "--ipc-event") use_event = true;
        else if (arg == "--wait-primed") wait_primed = true;
        else if (arg == "--poll-wait") poll_wait = true;
        else if (arg == "--poll-single-set") poll_single_set = true;
        else if (arg == "--skew-us" && i + 1 < argc) skew_us = std::atoi(argv[++i]);
    }
    if (device < 0) device = rank;
    if (dir.empty() || world < 2 || rank < 0 || rank >= world) {
        std::fprintf(stderr,
                     "usage: %s --world N --rank R [--device D] --dir DIR "
                     "[--bytes 10240] [--iters 200] [--no-pid-check|--set-import-pid] "
                     "[--ipc-event|--no-wait|--self-wait|--wait-one]\n",
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

    // How many buffer sets the exchange rotates through. Two is what the poll
    // barrier needs and one is what the notify arms have always used; every other
    // arm keeps `sets == 1` so its numbers stay comparable with the ones already
    // recorded for it.
    const int sets = (poll_wait && !poll_single_set) ? 2 : 1;

    uint16_t* local = nullptr;
    if (!pocket::device_malloc_into(local, bytes)) {
        line("result=alloc_failed");
        return 1;
    }
    // One receive slot per (set, rank), all on this process's own device; rank k
    // writes into slot k of every peer.
    std::vector<uint16_t*> recv(static_cast<size_t>(sets * world), nullptr);
    for (int s = 0; s < sets * world; ++s) {
        if (!pocket::device_malloc_into(recv[static_cast<size_t>(s)], bytes)) {
            line("result=alloc_failed");
            return 1;
        }
    }

    // The reduced result. Only the --poll-wait arm uses it, but it is allocated
    // unconditionally so that the arms differ in what they measure rather than in
    // what they set up.
    uint16_t* out = nullptr;
    if (!pocket::device_malloc_into(out, bytes)) {
        line("result=alloc_failed");
        return 1;
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
    for (int s = 0; s < sets * world; ++s) {
        char key[kKeyLen] = {0};
        const aclError err = aclrtIpcMemGetExportKey(recv[static_cast<size_t>(s)], bytes,
                                                     key, kKeyLen, export_flags);
        if (err == ACL_SUCCESS && publish(dir, slot_kind(rank), s, key)) ++exported;
        else if (first_export_err == ACL_SUCCESS) first_export_err = err;
    }
    line("export ok=%d/%d err=%d%s peer_pairs=%d/%d pid_mode=%d", exported, sets * world,
         static_cast<int>(first_export_err), acl_error_note(first_export_err), pairs,
         world - 1, static_cast<int>(pid_mode));
    if (exported != sets * world) {
        line("result=export_failed");
        return 1;
    }
    if (!wait_for_slots(dir, sets * world)) {
        line("result=rendezvous_timeout");
        return 1;
    }

    // remote[j] is rank j's slot for *this* rank, so pushing into it puts this
    // rank's contribution where rank j expects to find it. With two sets there is a
    // remote buffer per set, and a round uses the set its parity selects.
    std::vector<uint16_t*> remote(static_cast<size_t>(sets * world), nullptr);
    std::vector<std::string> remote_keys(static_cast<size_t>(sets * world));
    int imported = 0;
    aclError first_import_err = ACL_SUCCESS;
    aclError first_pid_err = ACL_SUCCESS;
    for (int p = 0; p < sets; ++p) {
        for (int j = 0; j < world; ++j) {
            if (j == rank) continue;
            char peer_key[kKeyLen] = {0};
            if (!read_key(dir, slot_kind(j), p * world + rank, peer_key)) continue;
            if (pid_mode == PidMode::SetImportPid) {
                int32_t self_pid = static_cast<int32_t>(::getpid());
                const aclError perr = aclrtIpcMemSetImportPid(peer_key, &self_pid, 1);
                if (perr != ACL_SUCCESS && first_pid_err == ACL_SUCCESS) first_pid_err = perr;
            }
            void* ptr = nullptr;
            const aclError err = aclrtIpcMemImportByKey(
                &ptr, peer_key, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
            if (err == ACL_SUCCESS && ptr != nullptr) {
                remote[static_cast<size_t>(p * world + j)] = static_cast<uint16_t*>(ptr);
                remote_keys[static_cast<size_t>(p * world + j)] = peer_key;
                ++imported;
            } else if (first_import_err == ACL_SUCCESS) {
                first_import_err = err;
            }
        }
    }
    line("import ok=%d/%d err=%d%s set_pid_err=%d%s", imported, sets * (world - 1),
         static_cast<int>(first_import_err), acl_error_note(first_import_err),
         static_cast<int>(first_pid_err), acl_error_note(first_pid_err));
    if (imported != sets * (world - 1)) {
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
    if (notify_imported > 0) {
        // An imported notify that cannot be waited on is an interesting result, so
        // the handle is checked for validity separately from its usability: a valid
        // id on both sides means the import produced a real notify and only the wait
        // is refused.
        uint32_t own_id = 0;
        uint32_t peer_id = 0;
        const aclError own_err = aclrtGetNotifyId(own_notify, &own_id);
        aclrtNotify probe = nullptr;
        for (int k = 0; k < world && probe == nullptr; ++k) {
            if (k != rank) probe = peer_notify[static_cast<size_t>(k)];
        }
        const aclError peer_err = aclrtGetNotifyId(probe, &peer_id);
        line("notify_id own=%u err=%d%s peer=%u err=%d%s", own_id,
             static_cast<int>(own_err), acl_error_note(own_err), peer_id,
             static_cast<int>(peer_err), acl_error_note(peer_err));
    }

    // 2b. The same arrive/wait, carried by an event instead of a notify. The handle
    // is a fixed 64 opaque bytes with no string form, so it travels hex-encoded
    // through the same rendezvous files.
    aclrtEvent own_event = nullptr;
    std::vector<aclrtEvent> peer_event(static_cast<size_t>(world), nullptr);
    int event_imported = 0;
    aclError first_event_err = ACL_SUCCESS;
    if (use_event) {
        // Which of the two entry points accepts ACL_EVENT_IPC is itself worth
        // measuring rather than assuming: the header documents the flag and the
        // handle, and the entry point that rejects it says so with 207000 rather
        // than with anything that names the flag.
        const aclError cerr = aclrtCreateEventExWithFlag(&own_event, ACL_EVENT_IPC);
        line("event_create_ex err=%d%s", static_cast<int>(cerr), acl_error_note(cerr));
        if (cerr != ACL_SUCCESS || own_event == nullptr) {
            const aclError werr = aclrtCreateEventWithFlag(&own_event, ACL_EVENT_IPC);
            line("event_create err=%d%s", static_cast<int>(werr), acl_error_note(werr));
            if (werr != ACL_SUCCESS || own_event == nullptr) own_event = nullptr;
        }
        if (own_event == nullptr) {
            line("event_unavailable");
        } else {
            aclrtIpcEventHandle handle;
            const aclError herr = aclrtIpcGetEventHandle(own_event, &handle);
            bool published = false;
            if (herr == ACL_SUCCESS) {
                char hex[2 * ACL_IPC_EVENT_HANDLE_SIZE + 1] = {0};
                const unsigned char* raw = reinterpret_cast<const unsigned char*>(handle.reserved);
                for (size_t i = 0; i < ACL_IPC_EVENT_HANDLE_SIZE; ++i) {
                    static const char* digits = "0123456789abcdef";
                    hex[2 * i] = digits[raw[i] >> 4];
                    hex[2 * i + 1] = digits[raw[i] & 0xf];
                }
                published = publish(dir, "event", rank, hex);
            }
            line("event_export err=%d%s ok=%d", static_cast<int>(herr), acl_error_note(herr),
                 published ? 1 : 0);
            if (!published) own_event = nullptr;
        }
    }
    if (own_event != nullptr && wait_for_all(dir, "event")) {
        for (int k = 0; k < world; ++k) {
            if (k == rank) continue;
            std::ifstream in(path_for(dir, "event", k), std::ios::binary);
            if (!in) continue;
            char hex[2 * ACL_IPC_EVENT_HANDLE_SIZE + 1] = {0};
            in.read(hex, 2 * ACL_IPC_EVENT_HANDLE_SIZE);
            if (in.gcount() < static_cast<std::streamsize>(2 * ACL_IPC_EVENT_HANDLE_SIZE)) {
                continue;
            }
            aclrtIpcEventHandle handle;
            std::memset(handle.reserved, 0, sizeof(handle.reserved));
            for (size_t i = 0; i < ACL_IPC_EVENT_HANDLE_SIZE; ++i) {
                auto nibble = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    return -1;
                };
                const int hi = nibble(hex[2 * i]);
                const int lo = nibble(hex[2 * i + 1]);
                if (hi < 0 || lo < 0) break;
                reinterpret_cast<unsigned char*>(handle.reserved)[i] =
                    static_cast<unsigned char>((hi << 4) | lo);
            }
            aclrtEvent imported = nullptr;
            const aclError oerr = aclrtIpcOpenEventHandle(handle, &imported);
            if (oerr == ACL_SUCCESS && imported != nullptr) {
                peer_event[static_cast<size_t>(k)] = imported;
                ++event_imported;
            } else if (first_event_err == ACL_SUCCESS) {
                first_event_err = oerr;
            }
        }
    }
    if (use_event) {
        line("event_import ok=%d/%d err=%d%s", event_imported, world - 1,
             static_cast<int>(first_event_err), acl_error_note(first_event_err));
    }

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
    double reduce_ms = 0.0;
    double total_ms = 0.0;
    aclError first_copy_err = ACL_SUCCESS;
    aclError first_wait_err = ACL_SUCCESS;
    int copy_calls = 0;
    long poll_looks = 0;
    int poll_timeouts = 0;
    int poll_stall_iters = 0;
    bool poll_capped = false;
    // The exchange line divides by what actually ran, which is not `iters` once the
    // stall cap has cut the loop short.
    long long rounds = 0;
    // A poll that cannot match is a desynchronised round, and it repeats: once a rank
    // is a round out of step every subsequent round can miss too, at one deadline per
    // iteration. A long deadline combined with a long run is therefore a hang rather
    // than a measurement -- an earlier revision of this arm spent over twelve minutes
    // on 5000 iterations this way. Both bounds are here so the arm reports the stall
    // instead of disappearing into it.
    constexpr double kPollDeadlineMs = 1000.0;
    constexpr int kPollStallLimit = 20;
    int bucket_looks = 0;
    constexpr int kPollBuckets = 4;
    double poll_bucket_ms[kPollBuckets] = {0.0, 0.0, 0.0, 0.0};
    long poll_bucket_looks[kPollBuckets] = {0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        if (skew_us > 0 && rank > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(rank * skew_us));
        }
        const double t0 = now_ms();
        // Which pair of buffers this round uses. Alternating them is what keeps a
        // peer that runs a round ahead from overwriting the stamp this rank is
        // still waiting on: the write goes to the set this round is not reading,
        // and by the time the peer comes back around to this set the rank has
        // finished with it. Every other arm keeps set 0.
        const int set = sets == 2 ? (it & 1) : 0;
        // --poll-wait carries its arrival signal in the payload: each rank stamps the
        // first element with a value that changes every round, and a rank polls its
        // own receive slots until they hold the stamp the peer owning that slot just
        // pushed. A fixed sentinel will not do -- after the first round every slot
        // already holds it, so the poll would return before the data arrived and the
        // arm would measure nothing. The stamp ranges for rank k are
        // [0x3c00 + 0x400k, +iters), which are disjoint as long as iters < 0x400.
        const uint16_t own_stamp = static_cast<uint16_t>(0x3c00u + 0x400u * rank + it);
        if (poll_wait && !pocket::memcpy_h2d(local, &own_stamp, sizeof(own_stamp))) {
            line("result=poll_stamp_failed");
            break;
        }
        for (int j = 0; j < world; ++j) {
            if (j == rank) continue;
            // The return value is the whole point of the readback below: a rejected
            // cross-process copy is silent otherwise, and the slots simply stay
            // whatever they were.
            const aclError cerr = aclrtMemcpyAsync(
                remote[static_cast<size_t>(set * world + j)], bytes, local, bytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            ++copy_calls;
            if (cerr != ACL_SUCCESS && first_copy_err == ACL_SUCCESS) first_copy_err = cerr;
        }
        if (barrier_available) aclrtRecordNotify(own_notify, stream);
        const double t1 = now_ms();
        double t_reduce = t1;
        if (poll_wait) {
            // No notify and no event: the arrival test is the data. Each look is a
            // small synchronous D2H read of this rank's own slot, so the cost per
            // round is (visibility latency of the peer's push) + (looks x read
            // price), and the read price is the platform's, not the fabric's.
            const double poll_start = now_ms();
            bool stalled = false;
            for (int k = 0; k < world; ++k) {
                if (k == rank) continue;
                const uint16_t want = static_cast<uint16_t>(0x3c00u + 0x400u * k + it);
                uint16_t got = 0;
                while (got != want) {
                    pocket::memcpy_d2h(&got, recv[static_cast<size_t>(set * world + k)],
                                       sizeof(got));
                    ++poll_looks;
                    ++bucket_looks;
                    if (now_ms() - poll_start > kPollDeadlineMs) {
                        ++poll_timeouts;
                        stalled = true;
                        break;
                    }
                }
            }
            if (stalled) {
                ++poll_stall_iters;
                // Capping is what keeps a broken layout from turning this bench into
                // a hang. The arm already knows it stalled, so paying another 5000
                // deadlines to average them changes nothing.
                if (poll_stall_iters > kPollStallLimit) {
                    poll_capped = true;
                    break;
                }
            }
            t_reduce = now_ms();
            // The reduce half, which is what makes the arm a complete all-reduce and
            // not only an exchange: out = local + sum over peers of recv[k]. It sits
            // inside the timed window on purpose -- the whole point of comparing
            // against HcclAllReduce is a full result in a full buffer -- but it is
            // accounted separately, because it is the one term the hand-written
            // version adds that the exchange arms do not have.
            aclrtMemcpyAsync(out, bytes, local, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            for (int k = 0; k < world; ++k) {
                if (k == rank) continue;
                pocket::qwen_add_inplace_f16_ascend(out,
                                                    recv[static_cast<size_t>(set * world + k)],
                                                    bytes / sizeof(uint16_t), stream);
            }
        } else if (own_event != nullptr && event_imported == world - 1) {
            // The record has to sit behind this rank's pushes, which is what makes
            // it an arrival signal rather than a bare timestamp.
            aclrtRecordEvent(own_event, stream);
            for (int k = 0; k < world; ++k) {
                if (k == rank) continue;
                aclrtStreamWaitEvent(stream, peer_event[static_cast<size_t>(k)]);
            }
        } else if (barrier_available && use_wait) {
            if (self_wait) {
                const aclError werr = aclrtWaitAndResetNotify(own_notify, stream, 60000);
                if (werr != ACL_SUCCESS && first_wait_err == ACL_SUCCESS) first_wait_err = werr;
            } else if (wait_one) {
                const int k = (rank + 1) % world;
                const aclError werr =
                    aclrtWaitAndResetNotify(peer_notify[static_cast<size_t>(k)], stream, 60000);
                if (werr != ACL_SUCCESS && first_wait_err == ACL_SUCCESS) first_wait_err = werr;
            } else {
                for (int k = 0; k < world; ++k) {
                    if (k == rank) continue;
                    const aclError werr =
                        aclrtWaitAndResetNotify(peer_notify[static_cast<size_t>(k)], stream,
                                                60000);
                    if (werr != ACL_SUCCESS && first_wait_err == ACL_SUCCESS) {
                        first_wait_err = werr;
                    }
                }
            }
        }
        const double t2 = now_ms();
        pocket::stream_synchronize(stream);
        const double t3 = now_ms();
        push_ms += t1 - t0;
        wait_ms += t_reduce - t1;
        reduce_ms += t2 - t_reduce;
        total_ms += t3 - t0;
        ++rounds;
        // The poll arm does not start at its steady state: with nothing else in the
        // loop the barrier has to find its own equilibrium, and whether the cost is
        // flat or ramps over a long run decides whether a single averaged figure
        // means anything. Four buckets, each a quarter of the run.
        if (poll_wait) {
            const size_t bucket = static_cast<size_t>(
                (static_cast<long long>(it) * kPollBuckets) / (iters > 0 ? iters : 1));
            if (bucket < kPollBuckets) {
                poll_bucket_ms[bucket] += (t_reduce - t1);
                poll_bucket_looks[bucket] += bucket_looks;
                bucket_looks = 0;
            }
        }
    }
    pocket::device_synchronize();

    if (poll_wait) {
        // The stamps were an arrival signal, not payload. Put the canonical value
        // back and republish it so the slot check below still means in this arm what
        // it means in the others, rather than tripping over a leftover counter. Set 0
        // is the one both the reduce check and the slot check read.
        std::vector<uint16_t> canonical(static_cast<size_t>(bytes) / sizeof(uint16_t),
                                        static_cast<uint16_t>(0x3c00u + 0x400u * rank));
        if (pocket::memcpy_h2d(local, canonical.data(), bytes)) {
            for (int j = 0; j < world; ++j) {
                if (j == rank) continue;
                aclrtMemcpyAsync(remote[static_cast<size_t>(j)], bytes, local, bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            }
        }
        pocket::stream_synchronize(stream);

        // One reduce round outside the timed window, to check that the arm produced
        // the sum and not merely something. The per-rank ladder 0x3c00 + 0x400*rank
        // strides the exponent field as well as the mantissa, so it is the powers of
        // two 1, 2, 4, 8 rather than the integers 1..world -- which is what makes a
        // slot mix-up change the sum by a factor instead of by a digit, and what
        // makes the expected value 2^world - 1 rather than world*(world+1)/2. Every
        // value involved is exact in fp16. The rendezvous first is not part of the
        // collective: a peer still in its timed loop would leave a stamp in the slot
        // this rank is about to sum.
        publish(dir, "reduce", rank, "1");
        wait_for_all(dir, "reduce");
        for (int j = 0; j < world; ++j) {
            if (j == rank) continue;
            aclrtMemcpyAsync(remote[static_cast<size_t>(j)], bytes, local, bytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }
        aclrtMemcpyAsync(out, bytes, local, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        for (int k = 0; k < world; ++k) {
            if (k == rank) continue;
            pocket::qwen_add_inplace_f16_ascend(out, recv[static_cast<size_t>(k)],
                                                bytes / sizeof(uint16_t), stream);
        }
        pocket::stream_synchronize(stream);
        std::vector<uint16_t> sum_readback(static_cast<size_t>(bytes) / sizeof(uint16_t), 0);
        pocket::memcpy_d2h(sum_readback.data(), out, bytes);
        size_t sum_ok = 0;
        const double want_sum = std::ldexp(1.0, world) - 1.0;
        for (uint16_t value : sum_readback) {
            if (fp16_exact(value) != want_sum) break;
            ++sum_ok;
        }
        line("reduce sum_ok=%zu/%zu first=0x%04x (%.1f) want=%.1f", sum_ok,
             sum_readback.size(), sum_readback[0], fp16_exact(sum_readback[0]), want_sum);
    }

    // total_ms against push+wait says whether the notify wait blocks the host here
    // or only enqueues: a wait that returns immediately leaves a gap between the
    // two, and a wait that polls shows up inside wait_ms. Under --ipc-event there is
    // no host-side wait to attribute, so the whole cross-process dependency has to
    // land in total_ms -- which is the point of the arm.
    const double denom = rounds > 0 ? static_cast<double>(rounds) : 1.0;
    line("exchange world=%d bytes=%zu push_ms=%.4f wait_ms=%.4f reduce_ms=%.4f "
         "total_ms=%.4f rounds=%lld wait_err=%d%s "
         "barrier=%d mode=%s",
         world, bytes, push_ms / denom, wait_ms / denom, reduce_ms / denom,
         total_ms / denom, rounds,
         static_cast<int>(first_wait_err), acl_error_note(first_wait_err),
         barrier_available ? 1 : 0,
         poll_wait ? (sets == 2 ? "poll-wait" : "poll-single-set")
                   : (own_event != nullptr && event_imported == world - 1
                          ? "ipc-event"
                          : (!use_wait ? "no-wait"
                                       : (self_wait ? "self-wait"
                                                    : (wait_one ? "wait-one" : "peer-wait")))));
    if (poll_wait) {
        line("poll looks=%ld over %lld rounds avg_looks=%.2f timeouts=%d stalled_rounds=%d "
             "capped=%d",
             poll_looks, rounds,
             static_cast<double>(poll_looks) / (denom * (world - 1)), poll_timeouts,
             poll_stall_iters, poll_capped ? 1 : 0);
        const double per_bucket = denom / kPollBuckets;
        for (int b = 0; b < kPollBuckets; ++b) {
            line("poll_bucket %d of %d wait_ms=%.4f avg_looks=%.2f", b + 1, kPollBuckets,
                 poll_bucket_ms[b] / per_bucket,
                 static_cast<double>(poll_bucket_looks[b]) / (per_bucket * (world - 1)));
        }
    }
    line("push copy_calls=%d err=%d%s local_ptr=%p remote_ptr=%p", copy_calls,
         static_cast<int>(first_copy_err), acl_error_note(first_copy_err), (void*)local,
         world > 1 ? (void*)remote[static_cast<size_t>((rank + 1) % world)] : nullptr);

    // 3b. The number the remaining design turns on. A remote wait costs 0.23 ms in
    // the loop above, and that figure is only useful if it is known whether the wait
    // is *waiting* -- i.e. the signal genuinely takes that long to cross -- or
    // whether it is a fixed price the call charges for being remote, with the signal
    // already in place. A device-side barrier only beats the notify path in the
    // second case, because a kernel spinning on a flag pays the crossing and not the
    // API.
    //
    // So: publish the arrival signals first, hold every rank at a rendezvous until
    // they are all recorded and settled, and then time exactly the waits.
    if (wait_primed && barrier_available) {
        // The rendezvous has to come *before* the signal, not after: a rank that
        // records early while a straggler is still in the main loop has its count
        // consumed by that straggler's last wait, and the primed wait then has
        // nothing pending. Waiting here first means no rank records until every rank
        // has left the loop.
        publish(dir, "primed", rank, "1");
        wait_for_all(dir, "primed");
        for (int j = 0; j < world; ++j) {
            if (j != rank) {
                aclrtMemcpyAsync(remote[static_cast<size_t>(j)], bytes, local, bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            }
        }
        aclrtRecordNotify(own_notify, stream);
        const bool psync = pocket::stream_synchronize(stream);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const double p0 = now_ms();
        int primed_waits = 0;
        aclError first_primed_err = ACL_SUCCESS;
        for (int k = 0; k < world; ++k) {
            if (k == rank) continue;
            const aclError werr =
                aclrtWaitAndResetNotify(peer_notify[static_cast<size_t>(k)], stream, 60000);
            if (werr == ACL_SUCCESS) {
                ++primed_waits;
            } else if (first_primed_err == ACL_SUCCESS) {
                first_primed_err = werr;
            }
        }
        const double p1 = now_ms();
        pocket::device_synchronize();
        line("primed_wait waits_ok=%d/%d sync_ok=%d wait_err=%d%s total_ms=%.4f "
             "per_wait_ms=%.4f",
             primed_waits, world - 1, psync ? 1 : 0, static_cast<int>(first_primed_err),
             acl_error_note(first_primed_err), p1 - p0,
             primed_waits > 0 ? (p1 - p0) / primed_waits : 0.0);
    }

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

    for (int s = 0; s < sets * world; ++s) {
        if (s % world != rank && !remote_keys[static_cast<size_t>(s)].empty()) {
            aclrtIpcMemClose(remote_keys[static_cast<size_t>(s)].c_str());
        }
    }
    if (own_notify != nullptr) aclrtDestroyNotify(own_notify);
    if (own_event != nullptr && event_imported == world - 1) aclrtDestroyEvent(own_event);
    pocket::stream_destroy(stream);
    for (int s = 0; s < sets * world; ++s) pocket::device_free(recv[static_cast<size_t>(s)]);
    pocket::device_free(out);
    pocket::device_free(local);
    return slots_ok == world ? 0 : 1;
}

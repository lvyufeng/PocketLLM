// Hand-written cross-process all-reduce. See ipc_allreduce.hpp for what it is and
// why each piece is shaped the way it is; the measurements behind it are in
// docs/performance/ascend_single_request_tps.md §5.

#include "ipc_allreduce.hpp"

#include "device_runtime.hpp"
#include "qwen_ascend_ops.hpp"

#include <acl/acl.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace pocket {
namespace {

// AscendCL does not publish a key length. HCCL's own keys are a few hundred bytes
// of text, so this is generous for the IPC keys too; a key longer than this would
// fail the import loudly rather than being truncated silently.
constexpr size_t kKeyLen = 4096;

// Two receive buffers per (peer, source) pair, used on alternate rounds. One is
// not enough: a peer that reaches round N+1 before this rank has looked for round
// N overwrites what is being waited for, and the poll can then never match. The
// probe measured 21 stalls in 5000 rounds with one buffer and none in 20000 with
// two, so this is a correctness requirement and not a tuning choice.
constexpr int kSets = 2;

// The stamp a rank sends is read out of a device-side table rather than written by
// the host, so that nothing in the loop depends on a host buffer staying alive
// until an asynchronous copy has run. The receiver computes the same value
// arithmetically from its own round counter, which is what the modulo here has to
// keep consistent between the two.
constexpr long long kStampTable = 4096;

// Spacing between two stamps in a peer's stamp array, in uint16 words, so 32 is 64
// bytes. A stamp is two bytes and four processes write into every peer's array
// concurrently, one slot each; packed at their natural spacing all four land inside
// a single 32-byte block, and a cross-device write into a block another process is
// also writing can lose one of the two -- the 32-byte-disjoint hazard this backend
// has already been bitten by for scalar GM stores. It is the one structural
// difference between this barrier and the probe that validates it: the probe carries
// its arrival signal inside the payload, where every write is a large aligned copy
// of its own and no two processes share a block. Padding each stamp out to a full
// 64-byte block gives every writer its own.
constexpr int kStampStride = 32;

// Worlds above this would make the per-rank stamp ranges collide after the modulo
// above. The 910B host this backend targets has eight cards.
constexpr int kMaxWorld = 8;

// The largest plane the hand-written path accepts, in FP16 elements.
//
// It is a conservative default, not a measured bound. The sweep that set it
// (docs/performance/ascend_single_request_tps.md 5.5.4) compared planes of 5120,
// 40960, 81920 and 163840 against a shipped HCCL path believed to reproduce the
// batched pass every time; that control has since been measured failing the same
// gate 3 times in 36, and against it the large-plane arms are 7 of 16 against 9 of
// 41 below the ceiling, which is a direction and not a separation (Fisher two-tailed
// p = 0.11). What the ceiling has going for it is that the evidence points the same
// way at every size above it, that the plane it exists to admit -- one row's
// 5120-element decode plane, 129 times a step -- is two orders of magnitude inside
// it, and that raising it would change the shipped default's behaviour on no
// evidence. A caller that batched today, a 16-row decode plane (81920) or a prefill
// plane, stays on HCCL, which is the path it was on before this file existed.
// `POCKET_ASCEND_IPC_ALLREDUCE_MAX_ELEMENTS` overrides it for measurement.
constexpr size_t kDefaultMaxElements = 40960;

// How long a receiver waits for one round's stamps before declaring the round
// lost. A rows=1 decode step is ~100 ms, so this is hundreds of steps of slack: it
// is there to turn a dead or desynchronised peer into an error rather than a
// hang, not to bound normal skew.
constexpr double kDefaultDeadlineMs = 30000.0;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Where the collective's host time goes. `tp_all_reduce` in the engine's host
// profile is wall time on the issuing thread, which for this path is the sum of
// the three terms below -- but only in aggregate and only from outside the call.
// Splitting them is what says whether the barrier is waiting on peers or paying
// for its own host-side calls, and those two point at different fixes.
bool stats_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("POCKET_ASCEND_IPC_ALLREDUCE_STATS");
        return value != nullptr && *value != '\0' && std::atoi(value) != 0;
    }();
    return enabled;
}

// Diagnostic arms for a timing experiment, not for a run. They exist to price the
// barrier's host wait and are documented with the numbers they produced in
// docs/performance/ascend_single_request_tps.md 5.5.3-5.5.4, and in the header;
// SKIP and NOPOLL produce wrong results, and every switch here announces itself
// through `warn_diagnostics`.
//
// `POCKET_ASCEND_IPC_ALLREDUCE_NOPOLL` skips the arrival poll entirely, so the
// reduce runs on whatever the peer slot happened to hold. It bounds what a
// barrier with no host wait at all would be worth.
bool nopoll_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("POCKET_ASCEND_IPC_ALLREDUCE_NOPOLL");
        return value != nullptr && *value != '\0' && std::atoi(value) != 0;
    }();
    return enabled;
}

// `POCKET_ASCEND_IPC_ALLREDUCE_SKIP` returns success without issuing anything --
// no push, no poll, no reduce. It prices the layer with the collectives removed
// altogether, which is the only way to separate "the barrier is expensive" from
// "the layer is expensive and the barrier only sits on top of it". The ranks stop
// being coupled at all, so this is a bound on a hypothetical, not a mode.
bool skip_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("POCKET_ASCEND_IPC_ALLREDUCE_SKIP");
        return value != nullptr && *value != '\0' && std::atoi(value) != 0;
    }();
    return enabled;
}

// `POCKET_ASCEND_IPC_ALLREDUCE_POLL_SLEEP_US` pauses between poll attempts and is a
// diagnostic, not a tuning knob: it separates two explanations for a poll that costs
// more than the plane it is waiting on. Either the peer really is that far behind --
// in which case sleeping only makes the wait longer -- or the poll's own 8-byte
// blocking reads over the IPC fabric are contending with the pushes it is waiting
// for, in which case backing off shortens the wait. Zero, the default, polls flat out.
int poll_sleep_us() {
    static const int us = [] {
        const char* value = std::getenv("POCKET_ASCEND_IPC_ALLREDUCE_POLL_SLEEP_US");
        if (value == nullptr || *value == '\0') return 0;
        const int parsed = std::atoi(value);
        return parsed > 0 ? parsed : 0;
    }();
    return us;
}

// `POCKET_ASCEND_IPC_ALLREDUCE_SETTLE_US` pauses between the poll finding every
// peer's stamp and the reduce reading the plane that stamp announces. It is a
// diagnostic, not a tuning knob, and it exists to test one hypothesis about a
// non-reproducibility that the engine sees and the isolated probe does not: a
// peer's stamp is pushed behind its plane on one stream, so a stamp that has
// landed means the plane before it has landed -- but only if two cross-device
// D2D copies on one stream really do complete in order. If they do not, the
// reduce reads a plane that is still arriving and the sum is wrong by however
// much of the copy had not landed. Sleeping here cannot repair that; it only
// makes the window empty, which is what separates "the arrival signal is early"
// from every other explanation. See
// docs/performance/ascend_single_request_tps.md 5.5.4.
int settle_us() {
    static const int us = [] {
        const char* value = std::getenv("POCKET_ASCEND_IPC_ALLREDUCE_SETTLE_US");
        if (value == nullptr || *value == '\0') return 0;
        const int parsed = std::atoi(value);
        return parsed > 0 ? parsed : 0;
    }();
    return us;
}

// `POCKET_ASCEND_IPC_ALLREDUCE_RSTREAM` moves this collective off the caller's
// stream and onto a stream of its own when the caller passed none, which is what
// the engine does at all 129 decode sites: `all_reduce_half` omits the stream
// precisely because this path is supposed to run on the caller's. Tp_comm's HCCL
// branch does the opposite -- `resolve_stream` substitutes a private stream for a
// null one -- and the two therefore differ in one more way than "hand-written
// versus HCCL": this path issues its cross-device copies on the null stream and
// HCCL issues its collective on a real one.
//
// Everything measured so far says the copies are ordered correctly (the drain in
// the poll below covers this rank's own pushes, and a peer's stamp cannot be
// observed before the plane before it has executed), but "ordered" and "handled
// the same way by the runtime" are not the same claim, and a null stream is the
// one argument this path passes that the validating probe never does -- the probe
// always creates a stream. This switch is how that is tested: with it on, the
// collective runs exactly as it does now except that `acl_stream` is a real
// stream, drained into and out of, which is slower and correct by construction.
// See docs/performance/ascend_single_request_tps.md 5.5.4.
bool rstream_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("POCKET_ASCEND_IPC_ALLREDUCE_RSTREAM");
        return value != nullptr && *value != '\0' && std::atoi(value) != 0;
    }();
    return enabled;
}

// There was a `POCKET_ASCEND_IPC_ALLREDUCE_ASYNCPOLL` here, which read the arrival
// stamps with `aclrtMemcpyAsync` into pinned host memory on a stream of this
// collective's own instead of `pocket::memcpy_d2h`, so that a poll iteration drained
// a 256-byte copy rather than the *default* stream -- which is where the collective
// is issued, so draining it means waiting out the whole layer queued behind it. The
// reasoning was that this drain is dead time: it cannot order the write being waited
// for, because the write belongs to another process. It was built and measured
// against this read, interleaved, and it is a **10.7% loss**: 8 of 8 pairs slower,
// 76.73 -> 85.96 ms, 13.035 -> 11.634 TPS, at 129 calls a step. The drain is not
// dead time -- it *is* the wait. `POCKET_ASCEND_IPC_ALLREDUCE_STATS` shows the
// blocking read finishing in 1.0-1.31 iterations at 0.285 ms of `wait_ms` per call,
// because by the time it has drained the layer every peer's stamp is already there;
// the async read returns before they are and has to be repeated, 2.97-3.19
// iterations at 0.322 ms of `wait_ms` per call, so the loop spins at 0.105 ms an
// iteration where a single blocking iteration costs 0.238 -- and 3 x 0.105 is more
// than 1.2 x 0.238. See docs/performance/ascend_single_request_tps.md 5.5.3.

// A run with one of the switches above set is not producing the measurement it
// looks like it is, so the announcement is unconditional: no number from such a
// run can be quoted without the caveat travelling with it. They stay in the tree
// because they are how the barrier's cost is split and its arrival signal is
// tested in docs/performance/ascend_single_request_tps.md 5.5.3 and 5.5.4. Which
// caveat depends on which one, and the two classes are not the same: SKIP issues
// nothing and NOPOLL reduces whatever the peer slot happened to hold, so those two
// do not compute the all-reduce at all; POLL_SLEEP_US, SETTLE_US and RSTREAM only
// move host time around, and a run of any of them has meaningful tokens and a step
// time that is a bound. Once per process, before the first collective.
void warn_diagnostics() {
    static const bool done = [] {
        const bool wrong = nopoll_enabled() || skip_enabled();
        if (wrong) {
            std::fprintf(stderr,
                         "[ipc_allreduce] WARNING: a wrong-result diagnostic is on "
                         "(NOPOLL/SKIP). This process does not compute the "
                         "all-reduce, so its tokens are meaningless and its step "
                         "time is a bound, not a measurement.\n");
            std::fflush(stderr);
        }
        if (poll_sleep_us() > 0) {
            std::fprintf(stderr,
                         "[ipc_allreduce] WARNING: POLL_SLEEP_US is on. This process "
                         "computes the all-reduce but paces its arrival poll with a "
                         "%d us sleep, so its tokens are meaningful and its step time "
                         "is a bound, not a step time.\n",
                         poll_sleep_us());
            std::fflush(stderr);
        }
        if (settle_us() > 0) {
            std::fprintf(stderr,
                         "[ipc_allreduce] WARNING: SETTLE_US is on. This process "
                         "computes the all-reduce but pays %d us per call for it, so "
                         "its tokens are meaningful and its step time is not a step "
                         "time.\n",
                         settle_us());
            std::fflush(stderr);
        }
        if (rstream_enabled()) {
            std::fprintf(stderr,
                         "[ipc_allreduce] WARNING: RSTREAM is on. The collective runs "
                         "on a stream of its own with a drain on either side, so its "
                         "tokens are meaningful and its step time is a bound, not a "
                         "step time.\n");
            std::fflush(stderr);
        }
        return true;
    }();
    (void)done;
}

struct Stats {
    long long calls = 0;
    double push_ms = 0.0;
    double wait_ms = 0.0;
    double reduce_ms = 0.0;
    long long poll_iters = 0;
    // Cumulative values as of the last report, so the line printed every
    // kReportEvery calls is the average over that interval rather than over the
    // whole process. A run's first calls include prefill and op warm-up, and a
    // whole-run mean would carry them into every later line.
    long long reported_calls = 0;
    double reported_push_ms = 0.0;
    double reported_wait_ms = 0.0;
    double reported_reduce_ms = 0.0;
    double reported_prologue_ms = 0.0;
    long long reported_poll_iters = 0;
    double prologue_ms = 0.0;
};

constexpr long long kReportEvery = 32;

Stats& stats() {
    static Stats state;
    return state;
}

void report_stats() {
    Stats& s = stats();
    const long long calls = s.calls - s.reported_calls;
    if (calls <= 0) return;
    const double n = static_cast<double>(calls);
    const double push = (s.push_ms - s.reported_push_ms) / n;
    const double wait = (s.wait_ms - s.reported_wait_ms) / n;
    const double reduce = (s.reduce_ms - s.reported_reduce_ms) / n;
    const double prologue = (s.prologue_ms - s.reported_prologue_ms) / n;
    const double iters =
        static_cast<double>(s.poll_iters - s.reported_poll_iters) / n;
    std::fprintf(stderr,
                 "[ipc_allreduce] calls=%lld..%lld prologue=%.4f push=%.4f wait=%.4f "
                 "reduce=%.4f total=%.4f ms poll_iters=%.2f\n",
                 s.reported_calls, s.calls, prologue, push, wait, reduce,
                 prologue + push + wait + reduce, iters);
    std::fflush(stderr);
    s.reported_calls = s.calls;
    s.reported_push_ms = s.push_ms;
    s.reported_wait_ms = s.wait_ms;
    s.reported_reduce_ms = s.reduce_ms;
    s.reported_prologue_ms = s.prologue_ms;
    s.reported_poll_iters = s.poll_iters;
}

bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::atoi(value) != 0;
}

void check_acl(aclError err, const char* what) {
    if (err != ACL_SUCCESS) {
        throw std::runtime_error(std::string("Ascend IPC all-reduce: ") + what +
                                 " failed with ACL error " +
                                 std::to_string(static_cast<int>(err)));
    }
}

// One stream per device for `rstream_enabled`. Same construction as tp_comm's
// `internal_stream`, kept local rather than exported because this is the only
// caller outside that translation unit and the two have opposite defaults.
aclrtStream private_stream(int device) {
    static std::unordered_map<int, aclrtStream> streams;
    auto it = streams.find(device);
    if (it != streams.end()) return it->second;
    aclrtStream stream = nullptr;
    check_acl(aclrtCreateStream(&stream), "aclrtCreateStream for IPC collective");
    streams.emplace(device, stream);
    return stream;
}

// The stamp a rank sends at round `round`, and the value every peer expects from
// it. Odd, so it can never be mistaken for the zeroed slot a rank starts with, and
// distinct per rank, so a slot that received the wrong rank's data fails the poll
// instead of passing it.
uint16_t stamp_value(long long round, int who, int world) {
    const long long step = ((round % kStampTable) * world + who) & 0x7fff;
    return static_cast<uint16_t>(step * 2 + 1);
}

// Slot index of the stamp array in the per-rank export set. The payload slots take
// 0 .. kSets*world-1 and one index past them names the stamp array, so both kinds
// of region share one naming scheme.
int stamp_slot(int world) { return kSets * world; }

std::string key_path(const std::string& id_path, int owner, int slot) {
    return id_path + ".ipc." + std::to_string(owner) + "." + std::to_string(slot) +
           ".key";
}

// Write where the peers will look, through a rename, so no reader ever sees a
// half-written key and blocks forever on it.
void publish_key(const std::string& path, int rank, const char* key) {
    const std::string tmp = path + "." + std::to_string(rank) + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("Ascend IPC all-reduce: cannot open " + tmp);
        }
        out.write(key, static_cast<std::streamsize>(std::strlen(key) + 1));
        if (!out) {
            throw std::runtime_error("Ascend IPC all-reduce: cannot write " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("Ascend IPC all-reduce: cannot publish " + path);
    }
}

bool read_key(const std::string& path, char* key) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.read(key, static_cast<std::streamsize>(kKeyLen));
    key[kKeyLen - 1] = '\0';
    return in.gcount() > 1;
}

// The rendezvous is not part of the collective and is not timed, so a file poll is
// the right instrument here even though it would be far too slow inside a step.
// The wait is as generous as the HCCL id file's, because ranks reach their first
// collective only after loading a 13 GB checkpoint off the same disk and that
// skews by minutes.
bool wait_for_key(const std::string& path, double deadline) {
    while (now_ms() < deadline) {
        std::ifstream in(path, std::ios::binary);
        if (in.good()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

double rendezvous_deadline_ms() {
    // Reuse the HCCL id wait: it is the same kind of wait, over the same peers,
    // on the same run. 6000 attempts at 100 ms is the CUDA path's ten minutes.
    int attempts = 6000;
    if (const char* env = std::getenv("POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS")) {
        const int value = std::atoi(env);
        if (value > 0) attempts = value;
    }
    return static_cast<double>(attempts) * 100.0;
}

double poll_deadline_ms() {
    if (const char* env = std::getenv("POCKET_ASCEND_IPC_ALLREDUCE_DEADLINE_MS")) {
        const long value = std::strtol(env, nullptr, 10);
        if (value > 0) return static_cast<double>(value);
    }
    return kDefaultDeadlineMs;
}

size_t max_elements() {
    if (const char* env = std::getenv("POCKET_ASCEND_IPC_ALLREDUCE_MAX_ELEMENTS")) {
        const long value = std::strtol(env, nullptr, 10);
        if (value > 0) return static_cast<size_t>(value);
    }
    return kDefaultMaxElements;
}

// Import one peer's region by the key it published. Every import in this file goes
// through here, so the error text and the flag choice are in one place.
uint16_t* import_region(const std::string& id_path, int owner, int slot) {
    char key[kKeyLen] = {0};
    if (!read_key(key_path(id_path, owner, slot), key)) {
        throw std::runtime_error("Ascend IPC all-reduce: cannot read the key rank " +
                                 std::to_string(owner) + " published for slot " +
                                 std::to_string(slot));
    }
    void* ptr = nullptr;
    const aclError err = aclrtIpcMemImportByKey(
        &ptr, key, ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
    if (err != ACL_SUCCESS || ptr == nullptr) {
        throw std::runtime_error(
            "Ascend IPC all-reduce: import of slot " + std::to_string(slot) +
            " from rank " + std::to_string(owner) + " failed with ACL error " +
            std::to_string(static_cast<int>(err)));
    }
    return static_cast<uint16_t*>(ptr);
}

void export_region(void* ptr, size_t bytes, const std::string& id_path, int rank,
                   int slot) {
    char key[kKeyLen] = {0};
    // The import gate is a PID whitelist, which is exactly the opposite of what a
    // cross-process collective needs, so the export has to opt out of it -- the
    // importer-side aclrtIpcMemSetImportPid the header suggests as the alternative
    // returns 507899 itself.
    const aclError err = aclrtIpcMemGetExportKey(
        ptr, bytes, key, kKeyLen, ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION);
    if (err != ACL_SUCCESS) {
        throw std::runtime_error("Ascend IPC all-reduce: export of slot " +
                                 std::to_string(slot) +
                                 " failed with ACL error " +
                                 std::to_string(static_cast<int>(err)));
    }
    publish_key(key_path(id_path, rank, slot), rank, key);
}

// One rank's worth of the exchange. Everything here is process-lifetime: the
// imported keys are bound to this process's context and closing them from a static
// destructor would race ACL's own teardown, which is a crash with no useful
// report. The key files live in the run directory the launcher creates and removes.
struct IpcState {
    int world = 0;
    int rank = 0;
    int device = 0;
    size_t capacity = 0;    // FP16 elements per plane
    size_t slot_bytes = 0;  // capacity elements
    size_t stamp_bytes = 0; // kSets * world stamps, each kStampStride words apart
    std::vector<uint16_t*> recv;   // kSets * world planes this rank receives into
    std::vector<uint16_t*> remote; // kSets * world peers' planes for this rank
    uint16_t* stamp_recv = nullptr;      // kSets * world * kStampStride, this rank polls it
    std::vector<uint16_t*> stamp_remote; // world, peers' stamp arrays
    uint16_t* stamps = nullptr;          // kStampTable entries
    // world * kStampStride, the poll's landing buffer. Pinned because it is a
    // device-to-host copy's destination, which is the shape `aclrtMallocHost` exists
    // for; it was a plain vector until the asynchronous poll was tried, and it stays
    // pinned because that is what the 76.73 ms blocking arm above was measured with.
    // Process-lifetime like everything else here, so it is never freed.
    uint16_t* seen = nullptr;
    long long round = 0;
};

// Held by pointer so a reference handed out by state_for stays valid when another
// group is added. A process runs one TP group, but nothing here should depend on
// that.
std::unordered_map<std::string, std::unique_ptr<IpcState>>& ipc_states() {
    static std::unordered_map<std::string, std::unique_ptr<IpcState>> states;
    return states;
}

std::unique_ptr<IpcState> initialize(const std::string& id_path, int world, int rank,
                                     int device, size_t capacity) {
    if (!device_set(device)) {
        throw std::runtime_error(
            "Ascend IPC all-reduce: device_set failed on device " +
            std::to_string(device));
    }
    IpcState state;
    state.world = world;
    state.rank = rank;
    state.device = device;
    state.capacity = capacity;
    state.slot_bytes = capacity * sizeof(uint16_t);
    state.stamp_bytes = static_cast<size_t>(kSets * world) * kStampStride *
                        sizeof(uint16_t);

    state.recv.assign(static_cast<size_t>(kSets * world), nullptr);
    state.remote.assign(static_cast<size_t>(kSets * world), nullptr);
    state.stamp_remote.assign(static_cast<size_t>(world), nullptr);
    if (!host_alloc_pinned_into(
            state.seen, static_cast<size_t>(world) * kStampStride * sizeof(uint16_t))) {
        throw std::runtime_error(
            "Ascend IPC all-reduce: cannot allocate the poll's landing buffer");
    }
    // `aclrtMallocHost` hands back whatever the pages held. The loop writes before
    // it reads, so this is belt-and-braces rather than a requirement -- but the
    // buffer used to be a zeroed vector and there is no reason for the difference
    // to be discoverable later.
    std::memset(state.seen, 0,
                static_cast<size_t>(world) * kStampStride * sizeof(uint16_t));

    // Slot (set, source) is the buffer this rank owns for source `source` in round
    // parity `set`: `source` pushes into it, this rank polls and reduces out of it.
    // Naming it per source rather than per peer is the whole addressing scheme --
    // indexing by the peer instead collides every source onto one buffer.
    for (int s = 0; s < kSets * world; ++s) {
        if (!device_malloc_into(state.recv[static_cast<size_t>(s)], state.slot_bytes)) {
            throw std::runtime_error(
                "Ascend IPC all-reduce: cannot allocate receive slot " +
                std::to_string(s));
        }
    }
    if (!device_malloc_into(state.stamp_recv, state.stamp_bytes)) {
        throw std::runtime_error(
            "Ascend IPC all-reduce: cannot allocate the stamp array");
    }
    // Zeroed, so a stamp can never be matched before the peer that owns it has
    // pushed anything. Combined with odd stamps this makes round 0 as safe as any
    // other. The payload slots need no clearing: a plane is only read once its
    // stamp has been seen.
    if (!device_memset(state.stamp_recv, 0, state.stamp_bytes)) {
        throw std::runtime_error(
            "Ascend IPC all-reduce: cannot zero the stamp array");
    }
    if (!device_malloc_into(state.stamps, kStampTable * sizeof(uint16_t))) {
        throw std::runtime_error("Ascend IPC all-reduce: cannot allocate stamp table");
    }
    std::vector<uint16_t> table(kStampTable);
    for (long long i = 0; i < kStampTable; ++i) {
        table[static_cast<size_t>(i)] = stamp_value(i, rank, world);
    }
    if (!memcpy_h2d(state.stamps, table.data(), table.size() * sizeof(uint16_t))) {
        throw std::runtime_error("Ascend IPC all-reduce: cannot fill stamp table");
    }

    for (int s = 0; s < kSets * world; ++s) {
        export_region(state.recv[static_cast<size_t>(s)], state.slot_bytes, id_path,
                      rank, s);
    }
    export_region(state.stamp_recv, state.stamp_bytes, id_path, rank,
                  stamp_slot(world));

    // Wait only for the regions this rank imports: rank j's payload slot
    // `set * world + rank` in each set, and rank j's stamp array. Waiting for all
    // of them would also work and would say nothing extra.
    const double deadline = now_ms() + rendezvous_deadline_ms();
    for (int j = 0; j < world; ++j) {
        if (j == rank) continue;
        for (int set = 0; set < kSets; ++set) {
            const std::string path =
                key_path(id_path, j, set * world + rank);
            if (!wait_for_key(path, deadline)) {
                throw std::runtime_error(
                    "Ascend IPC all-reduce: rank " + std::to_string(j) +
                    " did not publish slot " + std::to_string(set * world + rank) +
                    " within the rendezvous window");
            }
        }
        if (!wait_for_key(key_path(id_path, j, stamp_slot(world)), deadline)) {
            throw std::runtime_error(
                "Ascend IPC all-reduce: rank " + std::to_string(j) +
                " did not publish its stamp array within the rendezvous window");
        }
    }
    for (int j = 0; j < world; ++j) {
        if (j == rank) continue;
        for (int set = 0; set < kSets; ++set) {
            state.remote[static_cast<size_t>(set * world + j)] =
                import_region(id_path, j, set * world + rank);
        }
        state.stamp_remote[static_cast<size_t>(j)] =
            import_region(id_path, j, stamp_slot(world));
    }
    return std::make_unique<IpcState>(std::move(state));
}

IpcState& state_for(const std::string& id_path, int world, int rank, int device,
                    size_t capacity) {
    const std::string key = id_path + ":" + std::to_string(world) + ":" +
                            std::to_string(rank) + ":" + std::to_string(device);
    std::unordered_map<std::string, std::unique_ptr<IpcState>>& states = ipc_states();
    auto it = states.find(key);
    if (it != states.end()) return *it->second;
    std::unique_ptr<IpcState> state = initialize(id_path, world, rank, device, capacity);
    IpcState& ref = *state;
    states.emplace(key, std::move(state));
    return ref;
}

}  // namespace

bool ascend_ipc_allreduce_f16_applies(int world, int count) {
    if (!env_flag("POCKET_ASCEND_IPC_ALLREDUCE")) return false;
    if (world <= 1 || world > kMaxWorld) return false;
    if (count <= 0) return false;
    if (count > static_cast<int>(kDefaultMaxElements)) return false;
    return static_cast<size_t>(count) <= max_elements();
}

bool ascend_ipc_allreduce_f16_inplace(int world, int rank, int device,
                                      const char* id_path, uint16_t* values,
                                      int count, void* stream) {
    const bool timing = stats_enabled();
    const double t_entry = timing ? now_ms() : 0.0;
    if (!ascend_ipc_allreduce_f16_applies(world, count)) return false;
    warn_diagnostics();
    if (id_path == nullptr || id_path[0] == '\0') {
        throw std::runtime_error(
            "Ascend IPC all-reduce needs the rendezvous id path the HCCL "
            "communicator uses");
    }
    if (values == nullptr || rank < 0 || rank >= world) {
        throw std::runtime_error("Ascend IPC all-reduce: invalid buffer or rank");
    }

    IpcState& state = state_for(id_path, world, rank, device, max_elements());
    if (static_cast<size_t>(count) > state.capacity) {
        throw std::runtime_error(
            "Ascend IPC all-reduce: plane is larger than the buffers allocated "
            "for it");
    }

    // The caller's stream, default stream included. See the header: substituting a
    // private stream would make these copies unordered against the kernels that
    // produce `values` and force a full drain before every one of the 129 calls.
    // `rstream_enabled` is the diagnostic that pays for both of those on purpose,
    // to separate "the null stream is handled differently" from every other
    // explanation; it is off by default and the two drains it adds are the price
    // of the answer, not a configuration to ship.
    aclrtStream acl_stream = static_cast<aclrtStream>(stream);
    const bool substitute_stream = acl_stream == nullptr && rstream_enabled();
    if (substitute_stream) {
        check_acl(aclrtSynchronizeStream(nullptr),
                  "drain the default stream before the IPC collective");
        acl_stream = private_stream(device);
    }

    const int set = static_cast<int>(state.round % kSets);
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(uint16_t);
    const double t0 = timing ? now_ms() : 0.0;

    if (skip_enabled()) {
        if (timing) {
            Stats& s = stats();
            s.prologue_ms += t0 - t_entry;
            ++s.calls;
            if (s.calls % kReportEvery == 0) report_stats();
        }
        ++state.round;
        return true;
    }

    // Push this rank's plane into every peer's slot for this rank, and the round's
    // stamp into every peer's stamp array at (set, this rank). Both copies are on
    // one stream, so the stamp lands after the plane it announces and a peer that
    // sees the stamp knows the plane before it is complete.
    for (int j = 0; j < world; ++j) {
        if (j == rank) continue;
        check_acl(aclrtMemcpyAsync(state.remote[static_cast<size_t>(set * world + j)],
                                   plane_bytes, values, plane_bytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream),
                  "push a plane to a peer");
        check_acl(aclrtMemcpyAsync(
                      state.stamp_remote[static_cast<size_t>(j)] +
                          (static_cast<size_t>(set * world + rank)) * kStampStride,
                      sizeof(uint16_t), state.stamps + (state.round % kStampTable),
                      sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream),
                  "push a stamp to a peer");
    }
    const double t1 = timing ? now_ms() : 0.0;

    // The barrier. Every rank's stamp for this round sits in one contiguous block of
    // `world * kStampStride` words, so a poll iteration costs one D2H of 256 bytes
    // rather than one per peer. The stamps inside it are `kStampStride` words apart
    // and only every stride-th word is a stamp; the rest is padding that exists so no
    // two writers share a block.
    //
    // The read is what the poll costs, not the comparison that follows it, and the
    // read is `pocket::memcpy_d2h`: it drains the default stream, then blocks in
    // `aclrtMemcpy`. The drain is what makes the read safe against this rank's own
    // outstanding pushes, and it is also what makes the loop short. The copy itself
    // waits on a write that belongs to another process, so no local stream can order
    // the answer -- the loop is a poll, not a fence -- and a read that skipped the
    // drain would return before the peers arrive and have to be repeated. That was
    // built as `POCKET_ASCEND_IPC_ALLREDUCE_ASYNCPOLL` and measured as a 10.7% loss;
    // the comment where that switch used to live has the numbers. It is one default
    // stream drain per collective, and since the collective is issued *on* the default
    // stream it is a drain of the layer queued behind it -- the same class of host
    // round trip the bracket below is removed to avoid -- but it is buying the wait,
    // not wasting it. See docs/performance/ascend_single_request_tps.md 5.5.3.
    const double deadline = now_ms() + poll_deadline_ms();
    long long poll_iters = 0;
    const int sleep_us = poll_sleep_us();
    while (!nopoll_enabled()) {
        uint16_t* const landing = state.seen;
        const uint16_t* const stamps_at =
            state.stamp_recv + static_cast<size_t>(set * world) * kStampStride;
        const size_t stamp_read_bytes =
            static_cast<size_t>(world) * kStampStride * sizeof(uint16_t);
        if (!memcpy_d2h(landing, stamps_at, stamp_read_bytes)) {
            throw std::runtime_error(
                "Ascend IPC all-reduce: cannot read the receive stamps");
        }
        ++poll_iters;
        int pending = 0;
        uint16_t first_want = 0;
        uint16_t first_got = 0;
        for (int k = 0; k < world; ++k) {
            if (k == rank) continue;
            const uint16_t want = stamp_value(state.round, k, world);
            const uint16_t got = state.seen[static_cast<size_t>(k) * kStampStride];
            if (got != want) {
                if (pending == 0) {
                    first_want = want;
                    first_got = got;
                }
                ++pending;
            }
        }
        if (pending == 0) break;
        if (now_ms() > deadline) {
            throw std::runtime_error(
                "Ascend IPC all-reduce: " + std::to_string(pending) +
                " of " + std::to_string(world - 1) +
                " peers sent nothing for round " + std::to_string(state.round) +
                " (expected stamp " + std::to_string(first_want) + ", saw " +
                std::to_string(first_got) +
                "); the group is dead or desynchronised");
        }
        if (sleep_us > 0) usleep(static_cast<unsigned int>(sleep_us));
    }
    // Diagnostic only, default off. See settle_us(): this is the delay that makes
    // the window between a peer's stamp landing and its plane landing empty, so an
    // engine run that stops being non-reproducible with it on was reading planes
    // that had not finished arriving. It is charged to wait_ms because that is
    // what it is -- waiting -- and a run with it set is flagged at startup.
    if (const int settle = settle_us(); settle > 0) {
        usleep(static_cast<unsigned int>(settle));
    }
    const double t2 = timing ? now_ms() : 0.0;

    // Only now, with every peer's plane known to have landed, is the reduction
    // correct. In place: `values` already holds this rank's own contribution.
    //
    // The sum is taken on FP16's own grid, which rounds after every add. That was
    // suspected of causing the step-0 scatter in
    // docs/performance/ascend_single_request_tps.md 5.5.4, and an FP32-accumulator
    // form was built and measured: it made the scatter wider, not narrower, and
    // cost 6.4 ms of a 77 ms step. It was removed. What the scatter actually
    // tracks is the size of the plane, and 5.5.4 has that measurement.
    for (int k = 0; k < world; ++k) {
        if (k == rank) continue;
        if (!qwen_add_inplace_f16_ascend(
                values, state.recv[static_cast<size_t>(set * world + k)], count,
                acl_stream)) {
            throw std::runtime_error("Ascend IPC all-reduce: the reduce failed");
        }
    }
    // Close the substitution opened above: the caller's next kernel runs on the
    // default stream and reads `values`, so it has to wait for a reduce that is no
    // longer on that stream. This is the second of the two drains and is the whole
    // cost of the diagnostic. Charged to reduce_ms, because it is the reduce's
    // completion being waited for.
    if (substitute_stream) {
        check_acl(aclrtSynchronizeStream(acl_stream),
                  "drain the IPC collective stream");
    }
    const double t3 = timing ? now_ms() : 0.0;

    if (timing) {
        Stats& s = stats();
        s.prologue_ms += t0 - t_entry;
        s.push_ms += t1 - t0;
        s.wait_ms += t2 - t1;
        s.reduce_ms += t3 - t2;
        s.poll_iters += poll_iters;
        ++s.calls;
        if (s.calls % kReportEvery == 0) report_stats();
    }

    ++state.round;
    return true;
}

}  // namespace pocket

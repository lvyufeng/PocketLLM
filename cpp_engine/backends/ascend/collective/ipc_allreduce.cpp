// Hand-written cross-process all-reduce. See ipc_allreduce.hpp for what it is and
// why each piece is shaped the way it is; the measurements behind it are in
// docs/performance/ascend_single_request_tps.md §5.

#include "ipc_allreduce.hpp"

#include "device_runtime.hpp"
#include "qwen_ascend_ops.hpp"

#include <acl/acl.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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

// Worlds above this would make the per-rank stamp ranges collide after the modulo
// above. The 910B host this backend targets has eight cards.
constexpr int kMaxWorld = 8;

// The largest plane the hand-written path accepts, in FP16 elements. It covers
// batched decode comfortably -- 16 rows of a 5120-wide hidden is 81920 -- and
// leaves prefill, where the payload is large enough for HCCL's bandwidth to matter
// and the extra copies to cost real time, on HCCL.
constexpr size_t kDefaultMaxElements = 1u << 20;  // 1 Mi elements, 2 MiB

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
    size_t stamp_bytes = 0; // kSets * world stamps
    std::vector<uint16_t*> recv;   // kSets * world planes this rank receives into
    std::vector<uint16_t*> remote; // kSets * world peers' planes for this rank
    uint16_t* stamp_recv = nullptr;      // kSets * world, this rank polls it
    std::vector<uint16_t*> stamp_remote; // world, peers' stamp arrays
    uint16_t* stamps = nullptr;          // kStampTable entries
    std::vector<uint16_t> seen;          // world, the poll's landing buffer
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
    state.stamp_bytes = static_cast<size_t>(kSets * world) * sizeof(uint16_t);

    state.recv.assign(static_cast<size_t>(kSets * world), nullptr);
    state.remote.assign(static_cast<size_t>(kSets * world), nullptr);
    state.stamp_remote.assign(static_cast<size_t>(world), nullptr);
    state.seen.assign(static_cast<size_t>(world), 0);

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
    if (!ascend_ipc_allreduce_f16_applies(world, count)) return false;
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
    aclrtStream acl_stream = static_cast<aclrtStream>(stream);

    const int set = static_cast<int>(state.round % kSets);
    const size_t plane_bytes = static_cast<size_t>(count) * sizeof(uint16_t);

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
                      state.stamp_remote[static_cast<size_t>(j)] + set * world + rank,
                      sizeof(uint16_t), state.stamps + (state.round % kStampTable),
                      sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream),
                  "push a stamp to a peer");
    }

    // The barrier. Every rank's stamps for this round are one contiguous read, and
    // the read is a blocking D2H that is deliberately not stream-ordered: the write
    // it waits for belongs to another process, so no local stream's completion
    // would order it.
    const double deadline = now_ms() + poll_deadline_ms();
    for (;;) {
        if (!memcpy_d2h(state.seen.data(), state.stamp_recv + set * world,
                        static_cast<size_t>(world) * sizeof(uint16_t))) {
            throw std::runtime_error(
                "Ascend IPC all-reduce: cannot read the receive stamps");
        }
        int pending = 0;
        uint16_t first_want = 0;
        uint16_t first_got = 0;
        for (int k = 0; k < world; ++k) {
            if (k == rank) continue;
            const uint16_t want = stamp_value(state.round, k, world);
            const uint16_t got = state.seen[static_cast<size_t>(k)];
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
    }

    // Only now, with every peer's plane known to have landed, is the reduction
    // correct. In place: `values` already holds this rank's own contribution.
    for (int k = 0; k < world; ++k) {
        if (k == rank) continue;
        if (!qwen_add_inplace_f16_ascend(
                values, state.recv[static_cast<size_t>(set * world + k)], count,
                acl_stream)) {
            throw std::runtime_error("Ascend IPC all-reduce: the reduce failed");
        }
    }

    ++state.round;
    return true;
}

}  // namespace pocket

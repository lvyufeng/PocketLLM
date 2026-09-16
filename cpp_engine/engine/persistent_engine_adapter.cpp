#include "persistent_engine_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace pocket {
namespace {

// Matches the server's rule and PersistentEngine's own contract: a temperature
// at or below this is greedy argmax, which is what keeps existing greedy runs
// reproducible when a request leaves temperature unset.
constexpr float kGreedyTemperatureEpsilon = 1.0e-5f;

SamplingParams to_persistent_sampling(const BatchSamplingParams& sampling) {
    SamplingParams sp;
    sp.temperature = sampling.temperature;
    sp.top_p = sampling.top_p;
    sp.greedy = sampling.temperature <= kGreedyTemperatureEpsilon;
    sp.seed = sampling.seed;
    // PersistentEngine's sampler has no top-k stage, so BatchSamplingParams::top_k
    // is dropped here rather than silently reinterpreted as something else.
    return sp;
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    const std::chrono::duration<double> delta =
        std::chrono::steady_clock::now() - start;
    return delta.count();
}

}  // namespace

PersistentEngineAdapter::PersistentEngineAdapter(const std::string& ckpt_dir,
                                                 const ForwardSmokeOptions& opts,
                                                 int layer_count,
                                                 int max_context,
                                                 int max_slots)
    : owned_(std::make_unique<PersistentEngine>(ckpt_dir, opts, layer_count, max_context, max_slots)),
      engine_(owned_.get()),
      max_slots_(max_slots),
      allocated_slots_(max_slots),
      slot_taken_(static_cast<size_t>(max_slots), false),
      slot_request_ids_(static_cast<size_t>(max_slots), 0),
      positions_(static_cast<size_t>(max_slots), 0) {}

PersistentEngineAdapter::PersistentEngineAdapter(PersistentEngine& engine)
    : engine_(&engine),
      max_slots_(engine.max_slots()),
      allocated_slots_(engine.max_slots()),
      slot_taken_(static_cast<size_t>(engine.max_slots()), false),
      slot_request_ids_(static_cast<size_t>(engine.max_slots()), 0),
      positions_(static_cast<size_t>(engine.max_slots()), 0) {}

PersistentEngineAdapter::~PersistentEngineAdapter() = default;

Capabilities PersistentEngineAdapter::caps() const {
    Capabilities c;
    c.paged_kv = false;
    // The engine owns slots and can run rows at different positions in their own
    // caches at once, but only when the batched forward is switched on: with it
    // off, batch_decode_step() runs its per-request reference loop however wide
    // the batch is, so handing it a second row would only make both requests
    // slower. Reporting that honestly is what the scheduler clamps on, and it
    // matters in both directions -- a hardcoded false silently discarded
    // --max-batch-size, and a hardcoded true would advertise concurrency the
    // engine does not deliver.
    c.continuous_batching = max_slots_ > 1 && engine_->batched_decode_enabled();
    // Chunked prefill is a separate question from decode batching and stays off:
    // it decides whether a prefill token budget means anything, and this engine
    // reports no paged KV (kv_paged() is false, every kv_*_blocks() is 0), so
    // there is no block allocator to resume an unfinished prompt against.
    c.chunked_prefill = false;
    c.max_slots = max_slots_;
    // Rows are sampled independently -- batch_decode_step() runs one selection
    // per row against that row's own sampling params and slot RNG.
    c.per_request_sampling = true;
    c.per_request_top_k = false;
    c.fixed_top_k = 0;
    return c;
}

int PersistentEngineAdapter::max_context() const { return engine_->max_context(); }

int PersistentEngineAdapter::device() const { return engine_->options().device; }

void PersistentEngineAdapter::allocate_batch_slots(int max_batch_size) {
    if (max_batch_size > max_slots_) {
        throw std::invalid_argument(
            "PersistentEngineAdapter: requested " + std::to_string(max_batch_size) +
            " slots but caps().max_slots is " + std::to_string(max_slots_));
    }
    allocated_slots_ = max_batch_size;
}

int PersistentEngineAdapter::allocate_slot(uint64_t request_id) {
    for (int slot_id = 0; slot_id < allocated_slots_; ++slot_id) {
        if (!slot_taken_[static_cast<size_t>(slot_id)]) {
            slot_taken_[static_cast<size_t>(slot_id)] = true;
            slot_request_ids_[static_cast<size_t>(slot_id)] = request_id;
            positions_[static_cast<size_t>(slot_id)] = 0;
            // Both halves are needed and neither implies the other: the local
            // call clears this rank's caches, the command clears every worker's.
            // A slot reused without the command keeps the finished request's KV
            // and compressor accumulators on the worker ranks, and the next
            // request admitted to that slot reads them through the all-reduce.
            engine_->worker_command_reset_slot(slot_id);
            engine_->reset_slot(slot_id);
            engine_->claim_slot(slot_id, request_id);
            return slot_id;
        }
    }
    return -1;  // No free slot
}

void PersistentEngineAdapter::free_slot(uint64_t request_id) {
    const int slot_id = find_slot(request_id);
    if (slot_id < 0) return;
    slot_taken_[static_cast<size_t>(slot_id)] = false;
    slot_request_ids_[static_cast<size_t>(slot_id)] = 0;
    positions_[static_cast<size_t>(slot_id)] = 0;
}

int PersistentEngineAdapter::find_slot(uint64_t request_id) const {
    for (int slot_id = 0; slot_id < allocated_slots_; ++slot_id) {
        if (slot_taken_[static_cast<size_t>(slot_id)] &&
            slot_request_ids_[static_cast<size_t>(slot_id)] == request_id) {
            return slot_id;
        }
    }
    return -1;
}

bool PersistentEngineAdapter::kv_paged() const { return false; }
int PersistentEngineAdapter::kv_free_blocks() const { return 0; }
int PersistentEngineAdapter::kv_total_blocks() const { return 0; }
int PersistentEngineAdapter::kv_blocks_for_tokens(int) const { return 0; }

bool PersistentEngineAdapter::is_stop_token(const BatchSamplingParams& sampling,
                                            int token) const {
    if (sampling.ignore_eos) return false;
    if (!sampling.stop_token_ids.empty()) {
        return std::find(sampling.stop_token_ids.begin(),
                         sampling.stop_token_ids.end(),
                         token) != sampling.stop_token_ids.end();
    }
    return token == engine_->eos_id();
}

BatchPrefillResult PersistentEngineAdapter::batch_prefill(
    const std::vector<BatchedRequest*>& requests, int /*token_budget*/) {
    BatchPrefillResult out;
    if (requests.empty()) return out;
    if (requests.size() > static_cast<size_t>(allocated_slots_)) {
        throw std::invalid_argument(
            "PersistentEngineAdapter::batch_prefill: requested " +
            std::to_string(requests.size()) + " requests but only " +
            std::to_string(allocated_slots_) + " slots allocated");
    }

    const auto started = std::chrono::steady_clock::now();

    for (BatchedRequest* req : requests) {
        if (req == nullptr) {
            throw std::invalid_argument(
                "PersistentEngineAdapter::batch_prefill: null request");
        }
        if (req->seq_len != 0) {
            // caps().chunked_prefill is false, so a prompt always completes in one
            // call and the scheduler never resumes one. Arriving here means it did
            // anyway, and replaying the prompt from token 0 into a cache that already
            // holds part of it would duplicate positions.
            throw std::invalid_argument(
                "PersistentEngineAdapter::batch_prefill: cannot resume a partially "
                "prefilled prompt; this engine does not declare chunked_prefill");
        }

        const int slot_id = find_slot(req->request_id);
        if (slot_id < 0) {
            throw std::runtime_error(
                "PersistentEngineAdapter::batch_prefill: request not allocated to a slot");
        }

        const SamplingParams sp = to_persistent_sampling(req->sampling);
        engine_->worker_command_prefill(req->prompt_tokens, slot_id);
        const int token = engine_->prefill(req->prompt_tokens, sp, slot_id);

        const int prompt_tokens = static_cast<int>(req->prompt_tokens.size());
        positions_[static_cast<size_t>(slot_id)] = prompt_tokens;

        ForwardResult result;
        result.token = token;
        result.top_token = token;
        result.position = positions_[static_cast<size_t>(slot_id)];

        req->seq_len = prompt_tokens;
        req->last_token = token;
        req->last_result = result;
        req->finished = is_stop_token(req->sampling, token);

        out.results.push_back(result);
        out.incomplete.push_back(false);
        out.total_tokens += prompt_tokens;
    }

    out.seconds = elapsed_seconds(started);
    return out;
}

BatchDecodeResult PersistentEngineAdapter::batch_decode_step(
    const std::vector<BatchedRequest*>& requests) {
    BatchDecodeResult out;
    if (requests.empty()) return out;
    if (requests.size() > static_cast<size_t>(allocated_slots_)) {
        throw std::invalid_argument(
            "PersistentEngineAdapter::batch_decode_step: requested " +
            std::to_string(requests.size()) + " requests but only " +
            std::to_string(allocated_slots_) + " slots allocated");
    }

    const auto started = std::chrono::steady_clock::now();

    // Build batch request array
    std::vector<PersistentBatchRequest> batch_requests;
    batch_requests.reserve(requests.size());

    for (BatchedRequest* req : requests) {
        if (req == nullptr) {
            throw std::invalid_argument(
                "PersistentEngineAdapter::batch_decode_step: null request");
        }

        const int slot_id = find_slot(req->request_id);
        if (slot_id < 0) {
            throw std::runtime_error(
                "PersistentEngineAdapter::batch_decode_step: request not allocated to a slot");
        }

        PersistentBatchRequest batch_req;
        batch_req.last_token = req->last_token;
        batch_req.position = positions_[static_cast<size_t>(slot_id)];
        batch_req.slot_id = slot_id;
        batch_req.sampling = to_persistent_sampling(req->sampling);
        batch_requests.push_back(batch_req);
    }

    // Single batched forward (currently sequential, will be optimized)
    std::vector<int> tokens = engine_->batch_decode_step(batch_requests);

    // Update per-request state
    for (size_t i = 0; i < requests.size(); ++i) {
        BatchedRequest* req = requests[i];
        const int token = tokens[i];
        const int slot_id = batch_requests[i].slot_id;

        positions_[static_cast<size_t>(slot_id)]++;

        const bool stopped = is_stop_token(req->sampling, token);
        req->last_token = token;
        req->seq_len += 1;
        req->finished = stopped;

        out.next_tokens.push_back(token);
        out.finished.push_back(stopped);
        out.hit_stop_token.push_back(stopped);
    }

    out.seconds = elapsed_seconds(started);
    return out;
}

}  // namespace pocket

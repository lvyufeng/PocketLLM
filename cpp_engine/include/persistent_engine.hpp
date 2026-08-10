#pragma once

#include "dsv4_engine.hpp"
#include "tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dsv4 {

struct SamplingParams {
    float temperature = 1.0f;
    float top_p = 1.0f;
    bool greedy = true;
    uint64_t seed = 0;
};

// Persistent inference engine that owns SafeForwardContext (weights, resident
// device caches, NCCL handle, FP4 host pinned buffers) for the full lifetime
// of a server process. Lets multiple requests reuse all heavy resident state.
class PersistentEngine {
public:
    // ctor allocates SafeForwardContext, prepares resident caches sized for
    // `max_context` tokens (prompt + generation combined). Throws on failure.
    PersistentEngine(const std::string& ckpt_dir,
                     const ForwardSmokeOptions& opts,
                     int layer_count,
                     int max_context);
    ~PersistentEngine();

    PersistentEngine(const PersistentEngine&) = delete;
    PersistentEngine& operator=(const PersistentEngine&) = delete;

    // Clear KV / indexer caches and reset internal position counter. Cheap.
    void reset_session();

    // Run prefill on token_ids (which must include the prompt's last token).
    // Returns the sampled token id for the next position (rank 0 valid; on
    // worker ranks the return value is the rank-local argmax, which the
    // server should discard).
    int prefill(const std::vector<int>& token_ids, const SamplingParams& sp);

    // Run one decode step using `last_token` as the input embedding at
    // position `position`. Returns the sampled next token (rank 0 valid).
    int decode_step(int last_token, int position, const SamplingParams& sp);

    // Speculative-decode verify: forward `draft_tokens` starting at
    // `start_position` and return what the target model samples at each one, so
    // the caller can find the accepted prefix. Only rank 0's values are
    // meaningful; workers return a rank-local argmax that must be discarded.
    //
    // Forwards the drafts one at a time, not as a batch -- see the comment on
    // the definition for why the batched form is not numerically interchangeable
    // with plain decode.
    std::vector<int> verify_step(const std::vector<int>& draft_tokens,
                                 int start_position,
                                 const SamplingParams& sp);

    int eos_id() const;
    int max_context() const;
    int layer_count() const;
    const Tokenizer& tokenizer() const;
    const ForwardSmokeOptions& options() const;

    // TP rank > 0 entry point. Blocks on a small NCCL int32 broadcast channel
    // driven by rank 0; runs the requested op until SHUTDOWN.
    void run_worker_loop();

    // Trigger NCCL communicator init on all ranks. Must be called by every
    // rank before any of the worker_command_* / run_worker_loop functions.
    // For TP=1 this is a no-op.
    void warmup_tp();

    // Rank 0 utilities to drive the worker loop. No-op for tp_world == 1.
    enum class WorkerCommand : int32_t {
        Prefill = 0,
        DecodeStep = 1,
        Reset = 2,
        Shutdown = 3,
    };
    void worker_command_prefill(const std::vector<int>& token_ids);
    void worker_command_decode(int32_t last_token, int32_t position);
    void worker_command_reset();
    void worker_command_shutdown();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace dsv4

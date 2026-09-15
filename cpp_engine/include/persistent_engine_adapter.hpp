#pragma once

#include "inference_engine.hpp"
#include "persistent_engine.hpp"

#include <memory>
#include <string>
#include <vector>

namespace pocket {

// Presents the DeepSeek-V4 PersistentEngine through the InferenceEngine surface.
//
// PersistentEngine owns a fixed set of contiguous request slots. KV and
// recurrent state are allocated once, and every scheduler row carries its slot
// id into the model forward. The cache is not paged, but independent rows can
// decode together through PersistentEngine's batched continuation path. Prefill
// remains a safe per-request operation; the scheduler may admit several prompts
// and the adapter fills their slots without resetting active requests.
//
// TP is handled the way QwenEngine's batched entry points handle it: each forward
// announces itself on the worker command channel first, so a scheduler driving
// this adapter is TP-safe without knowing the protocol exists. Rank 0 drives;
// worker ranks stay in run_worker_loop(), which the interface exposes.
class PersistentEngineAdapter : public InferenceEngine {
public:
    // Owning form: constructs the PersistentEngine. This is what the model
    // registry's factory uses.
    PersistentEngineAdapter(const std::string& ckpt_dir,
                            const ForwardSmokeOptions& opts,
                            int layer_count,
                            int max_context,
                            int max_slots = 1);

    // Borrowing form, for a caller that already owns a PersistentEngine and
    // still needs its non-batched entry points (speculative decoding, the TP
    // worker loop). The engine must outlive the adapter.
    explicit PersistentEngineAdapter(PersistentEngine& engine);

    ~PersistentEngineAdapter() override;

    PersistentEngineAdapter(const PersistentEngineAdapter&) = delete;
    PersistentEngineAdapter& operator=(const PersistentEngineAdapter&) = delete;

    // The wrapped engine, for the model-specific surface the interface does not
    // carry: speculative decoding, the tokenizer, the per-forward worker
    // commands. TP bring-up and teardown are on the interface itself.
    PersistentEngine& engine() { return *engine_; }
    const PersistentEngine& engine() const { return *engine_; }

    Capabilities caps() const override;
    int max_context() const override;
    int device() const override;

    // Allocates the fixed slot count selected at construction. The scheduler's
    // requested width is clamped by the capability negotiation before this call.
    void allocate_batch_slots(int max_batch_size) override;
    int allocate_slot(uint64_t request_id) override;
    void free_slot(uint64_t request_id) override;

    // Not paged; all three report 0, which caps().paged_kv == false is what makes
    // meaningful.
    bool kv_paged() const override;
    int kv_free_blocks() const override;
    int kv_total_blocks() const override;
    int kv_blocks_for_tokens(int tokens) const override;

    // Prefill each newly admitted prompt into its own slot. The full prompt
    // kernel remains the tuned single-sequence path; no active slot is reset
    // while another request is being filled.
    BatchPrefillResult batch_prefill(const std::vector<BatchedRequest*>& requests,
                                     int token_budget) override;

    // Decode all active rows in one continuation batch. Rows may have different
    // absolute positions but must have distinct slots.
    BatchDecodeResult batch_decode_step(
        const std::vector<BatchedRequest*>& requests) override;

    // Straight forwards. PersistentEngine spells the last one
    // worker_command_shutdown(); the interface asks for the intent, not the
    // wire op.
    void warmup_tp() override { engine_->warmup_tp(); }
    void run_worker_loop() override { engine_->run_worker_loop(); }
    void shutdown_tp_workers() override { engine_->worker_command_shutdown(); }

private:
    bool is_stop_token(const BatchSamplingParams& sampling, int token) const;
    int find_slot(uint64_t request_id) const;

    std::unique_ptr<PersistentEngine> owned_;
    PersistentEngine* engine_ = nullptr;
    int max_slots_ = 1;
    int allocated_slots_ = 1;
    std::vector<bool> slot_taken_;
    std::vector<uint64_t> slot_request_ids_;
    std::vector<int> positions_;
};

}  // namespace pocket

/* Qwen3, dense, on whatever device the session was opened on.
 *
 * The architecture is small enough to state in full: an embedding, twenty-eight
 * identical blocks, and a final norm feeding a (tied) output projection. Each
 * block is a pre-norm residual around grouped-query attention and a SwiGLU
 * feed-forward network:
 *
 *   x = x + wo( attn( qk_norm( qkv( rms_norm(x) ) ) ) )
 *   x = x + w_down( silu(w_gate(n)) * w_up(n) ),  n = rms_norm(x)
 *
 * Three details in that shape are Qwen3's and not the generic transformer's,
 * and each is a place a faithful-looking implementation goes quietly wrong:
 *
 *   - **The QK norm is per head, over `head_dim`.** It normalizes each of the
 *     sixteen (respectively eight) query and key heads separately, so the
 *     reduction is over 128 values and not over the 2048-wide projection row.
 *     Normalizing the row would be a different, plausible, wrong model.
 *   - **It runs before RoPE, not after.** A rotation of a normalized vector is
 *     still normalized, so the other order is not a crash -- it is a different
 *     set of angles applied to the same magnitudes.
 *   - **RoPE is split-half (NEOX).** `x[i]` rotates against `x[i + head_dim/2]`,
 *     not against its neighbour.
 *
 * The layout follows the file rather than the papers: a weight is stored
 * `(rows, cols)` with the output axis first, which is how GGUF writes it and
 * which makes `attn_q.weight` `(n_embd, n_head * head_dim)` = `(1024, 2048)` --
 * a *wider* output than the residual stream. `n_embd` is 1024 and the attention
 * head space is 2048, so those two numbers are not interchangeable anywhere in
 * this file.
 *
 * Everything here is written against `kernel::Backend` and holds only opaque
 * device handles. The one place a host value is read is `forward`, which
 * returns a *device* buffer; `Session` copies it back with `to_host`. That is
 * what lets the same source run on the CPU and on a card.
 */

#ifndef POCKETLLM_MODEL_QWEN3_H
#define POCKETLLM_MODEL_QWEN3_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"

namespace pocketllm {

/* How a stored weight is used by `matmul`.
 *
 * Exactly one of the two handles is live. A tensor this build can widen is read
 * once at load, converted on the host, and `data` is the device copy of the
 * result; a k-quant tensor has no widened form at all, and `blocks` holds the
 * checkpoint's own bytes so the kernel can decode each weight as it consumes
 * it. `quantized` says which, and it is a stored flag rather than a test of
 * `blocks.handle != 0` so that a reader finds the decision in one place.
 *
 * The second form is not an optimization of the first. A `q4_k_m` weight is
 * 4.5 bits per weight and its f32 expansion is 32 -- so widening one to run a
 * dense GEMM would allocate seven times the checkpoint on the device, which on
 * the card this project targets is the difference between fitting and not. */
struct Weight {
  kernel::DeviceBuffer data;       /* widened to f32, resident on the backend */
  kernel::DeviceBuffer blocks;     /* the raw bytes, for a quantized tensor */
  int64_t rows = 0;                /* output features (GGUF ne1) */
  int64_t cols = 0;                /* input features (GGUF ne0) */
  int type_id = 0;
  int64_t nbytes = 0;
  bool quantized = false;
  /* The bytes in `blocks` are eight-column panels (`Q4Kx8`), not the file's
   * per-row `q4_K` blocks.  The two are the same byte count, so this flag is the
   * only thing that says which the buffer holds -- and it is set in exactly one
   * place, so a reader of `matmul` does not have to re-derive the decision from
   * the shape.  False for everything else, including the embedding table, whose
   * gather reads the file layout. */
  bool panel = false;
  /* The bytes in `blocks` are `quant::Q6KRepacked` blocks (one `int8` per
   * weight, byte-expanded), on a 16-byte stride, not the file's 210-byte `q6_K`
   * blocks.  Unlike the q4_K panels this is a *different size*, so nothing about
   * the shape says which layout a buffer holds -- this flag is the only
   * authority.  CUDA-only; false everywhere else, including the embedding
   * table. */
  bool q6k_repacked = false;
  /* The bytes in `blocks` were released at load, after the backend built its own
   * decoded form of them (`Backend::materializes_packed_weights`).  The handle
   * is *kept*: it is the key that form is filed under and the forward path still
   * passes it -- but it is no longer an allocation, so only this flag stands
   * between the destructor and a double free.  False on every backend that reads
   * the packed bytes where they lie, which is every backend but Ascend. */
  bool packed_released = false;
};

class Qwen3Model {
 public:
  /* Read the hyperparameters and bind every tensor. Throws `Error` naming the
   * key or the tensor that was missing, because a checkpoint missing one of the
   * 199 weights produces nonsense rather than a failure, and the nonsense
   * arrives as a wrong token several hundred milliseconds later.
   *
   * The reader must outlive the load; the backend must outlive the model. */
  static std::unique_ptr<Qwen3Model> load(const GgufReader &checkpoint, kernel::Backend &backend);

  ~Qwen3Model();
  Qwen3Model(const Qwen3Model &) = delete;
  Qwen3Model &operator=(const Qwen3Model &) = delete;

  int64_t n_embd() const { return n_embd_; }
  int64_t n_layer() const { return n_layer_; }
  int64_t n_head() const { return n_head_; }
  int64_t n_head_kv() const { return n_head_kv_; }
  int64_t head_dim() const { return head_dim_; }
  int64_t n_vocab() const { return n_vocab_; }

  /* The longest sequence this model will accept, from `context_length`. */
  int64_t capacity() const { return capacity_; }

  /* How many positions the cache holds.  This is sequence 0's length; the
   * batch path keeps one length per sequence in `seq_len_`, and the
   * single-sequence path is the member at index 0. */
  int64_t cache_length() const { return seq_len_[0]; }

  /* Run `n` tokens occupying positions `start_pos .. start_pos + n - 1` and
   * return the logits of the *last* one, as a device buffer of `n_vocab`
   * floats. `start_pos` must equal `cache_length()` for a decode and 0 for a
   * fresh prompt -- the graph does not special-case a restart, so a caller that
   * wants one calls `reset()`.
   *
   * The returned handle is owned by the model and stays valid until the next
   * call. It is *not* host memory: read it with the backend's `copy_to_host`. */
  kernel::DeviceBuffer forward(const int32_t *tokens, int64_t n, int64_t start_pos);

  /* The Stage-1 batch decode.  `n_seq` sequences run in one pass, each
   * contributing `n_per_seq[s]` tokens (the decode path contributes one per
   * sequence), and the weight-reading GEMMs run once at `m = sum(n_per_seq)`
   * instead of once per sequence.  Sequence `s` continues from its own length
   * `seq_len_[s]` and writes to its own KV slab, so the attention windows are
   * per-sequence -- ragged -- and the op is called once per sequence per layer.
   *
   * Returns a device buffer of `m` logit rows (one per sequence, its last
   * token).  Single-sequence callers keep using `forward`; this is the batched
   * entry and is what `pocketllm_forward_batch` drives. */
  kernel::DeviceBuffer forward_batch(const int32_t *tokens, const int64_t *n_per_seq,
                                     int64_t n_seq);

  /* Drop the KV cache. The buffers are kept -- a decode after a reset is the
   * next thing that happens, and reallocating 300 MB to fill none of it is
   * pure loss. */
  void reset();

  /* Pre-size the KV slabs to `slots` sequence slots of `positions` rows each,
   * before any sequence is admitted.
   *
   * This exists because `ensure_kv` -- the growth the forward path still does
   * -- releases and reallocates the slabs *without copying* when either the
   * row count or the slot count must grow.  That is fine for one sequence,
   * whose history is being grown under its own live window, and catastrophic
   * for a batch: the first sequence to cross a capacity boundary would wipe
   * every other slot's KV and return fluent-wrong text with no error.  A
   * scheduler therefore fixes the slab before the group starts and chooses a
   * window it never lets a sequence exceed, so `ensure_kv`'s grow branch is
   * never reached while more than one slot is live.
   *
   * Idempotent and non-destructive: calling it again with a size the slabs
   * already hold is a no-op, so it does not throw away a live batch.  Throws
   * `ContextLengthError` if `positions` is past the checkpoint's context. */
  void reserve(int64_t positions, int64_t slots);

 private:
  explicit Qwen3Model(kernel::Backend &backend) : backend_(&backend) {}

  /* Everything a layer needs, resolved to handles once. An index-and-lookup
   * per weight would be a string comparison in the inner loop's preamble and a
   * null check in every layer of every forward. */
  struct Layer {
    kernel::DeviceBuffer attn_norm, ffn_norm, q_norm, k_norm;
    Weight wq, wk, wv, wo;
    Weight w_gate, w_up, w_down;
  };

  /* Read a tensor from the checkpoint, widen it to f32 on the host, and upload
   * it. The widening happens here and not in a kernel because it is a load-time
   * cost paid once. */
  kernel::DeviceBuffer bind_dense(const GgufReader &checkpoint, const std::string &name);

  /* A weight, through whichever path its stored type calls for: widened if it
   * is f32/f16, copied as packed blocks if it is a k-quant. The choice is made
   * once, here, and every other function reads `Weight::quantized`. */
  Weight bind_matrix(const GgufReader &checkpoint, const std::string &name);

  /* Repack a bound q4_K weight's blocks from the file's per-row 4-bit blocks into
   * eight-column panels, in place -- the two are the same byte count, so this
   * reinterprets the one buffer rather than allocating a second.  Called only
   * when the session selected the panel path and the shape divides; throws if it
   * does not, rather than leaving a matrix half in each layout. */
  void repack_weight(Weight &weight) const;
  void repack_weight_q6k(const GgufReader &checkpoint, const std::string &name,
                         Weight &weight) const;

  /* Copy a tensor's stored bytes to the device untouched and describe them as
   * a `Weight`. Nothing is decoded at load: the type id travels with the
   * handle and the kernel is what reads the layout. */
  Weight bind_packed(const GgufReader &checkpoint, const std::string &name);

  /* The embedding table, which is either a float matrix or a packed one. Kept
   * as a `Weight` rather than a handle because the output projection is tied to
   * it and needs the same shape and type. */
  Weight bind_table(const GgufReader &checkpoint, const std::string &name);

  void matmul(const Weight &w, kernel::DeviceBuffer x, kernel::DeviceBuffer out, int64_t m,
              bool accumulate) const;
  void embed(kernel::DeviceBuffer tokens, int64_t n, kernel::DeviceBuffer out) const;

  /* Build, in a backend that keeps a decoded copy of a packed weight, the copy
   * of every packed weight the forward path will read, and then release the
   * checkpoint's bytes.  A no-op on a backend that does not
   * (`Backend::materializes_packed_weights`) -- which is every backend but
   * Ascend, so their loads and their numerics are untouched.
   *
   * Called once, at the end of `load`, after every weight is bound.  It is
   * idempotent per weight, not per model: a weight already released is skipped,
   * which is what lets a retry after a partial failure be written the same way
   * as the first attempt. */
  void materialize_packed_weights();

  /* Drop whichever of a `Weight`'s two buffers is the live one. A member
   * function because the destructor and the tie-breaking in it both need to
   * ask, and the answer must be the same in both places. */
  void release_weight(const Weight &weight);

  /* Whether two `Weight`s denote the same bound tensor: the same form, and the
   * same live handle.  This is how a tied head is told from one the checkpoint
   * carries separately, and the test has to be identity rather than shape -- a
   * Qwen3 head is `(n_embd, n_vocab)` whether it is tied or not, so shapes
   * cannot tell the two apart. */
  static bool same_buffer(const Weight &a, const Weight &b);

  /* Size the scratch buffers for a batch of `n` tokens whose last position is
   `end_pos - 1`. The second argument is not derivable from the first because the
   attention score row is indexed by absolute key position: a one-token decode at
   position 300 needs a row 300 long, and `n` alone would say one.  `head_rows` is
   the number of logit rows the head must hold: 1 for the single-sequence path
   (only the last token is projected), `n` for the batch, whose every row is a
   sequence's token. */
  void ensure_capacity(int64_t n, int64_t end_pos, int64_t head_rows = 1);

  /* Size the KV slabs to `positions` per sequence and `n_seq` sequence slots.
   * The layout is `[layer][sequence][position][kv_head][head_dim]`, so the slab
   * is `n_layer_ * n_seq * positions * n_head_kv * head_dim` elements wide.  At
   * `n_seq == 1` this is the single-sequence slab the graph has always used, and
   * the offsets below add zero for sequence 0. */
  void ensure_kv(int64_t positions, int64_t n_seq);

  /* The shared body of `forward` (n_seq == 1) and `forward_batch`.  One
   * implementation of the graph, so the single-sequence path is not a second
   * copy that can drift from the batched one: at n_seq == 1 every loop below
   * runs exactly once with sequence 0's own start position, and the op stream
   * is the one the graph has always issued. */
  kernel::DeviceBuffer forward_impl(const int32_t *tokens, const int64_t *n_per_seq,
                                    const int64_t *start_pos, int64_t n_seq);
  void build_rope_table();
  void grow_rope_table(int64_t positions);

  kernel::Backend *backend_ = nullptr;

  /* Scratch, sized to the batch as it is needed. Held on the model because the
   * shape is the model's (n_embd, n_head, ...) and reallocating them per
   * forward would touch the allocator several times per token. */
  kernel::DeviceBuffer x_, x_norm_, q_, k_, v_, attn_, gate_, up_, ffn_, last_, logits_;
  kernel::DeviceBuffer scores_;
  kernel::DeviceBuffer tokens_;
  kernel::DeviceBuffer rope_cos_, rope_sin_;
  /* The packed activations the panel kernel reads -- one `quant::Q8KBlock` row
   * per activation row for `gemv_q4k_8x8`, the kernel every panel weight is read
   * by.  Sized (in `ensure_capacity`) to the wider `Q8Kx4` groups the retained
   * but not dispatched panel GEMM would pack, so the bound does not depend on
   * which of the two kernels a session runs.  Filled and consumed inside one
   * `matmul` call and may be released immediately; a member rather than a local
   * because reallocating it per weight per layer would be 200 allocations a
   * token, and because the only size that can change with the batch is the
   * batch. */
  kernel::DeviceBuffer act_packed_;
  /* Where an *accumulating* panel call builds its tile.  The two accumulating
   * calls in the graph (`wo`, `w_down`) write into a buffer the residual is
   * already in -- `out` and `x` are the same handle -- so the tile cannot be
   * written there, and `x` is still being read row by row.  A third buffer is
   * the honest answer; the panel kernel takes a destination pointer, so this is
   * one more argument rather than a second kernel.  Only allocated when the
   * panel path is on. */
  kernel::DeviceBuffer panel_tile_;
  /* Is the repacked panel layout in use for this run? Decided once at load --
   * from `repack_enabled()` and the backend's name -- rather than per `matmul`,
   * because it is a property of the session and a per-call string comparison
   * would be a per-op decision that cannot change.  The name is historical: the
   * layout no longer selects a GEMM (see `model/qwen3.cpp`'s `matmul`). */
  bool panel_gemm_ = false;
  /* Is the byte-expanded q6_K layout in use this run? Unlike the q4_K panels it
   * is sent by the *backend*, not the model: only CUDA reads the repacked block,
   * so a CPU session must keep the file layout or `gemm_quant` would walk 210-
   * byte blocks as 274-byte ones. Decided once at load, beside `panel_gemm_`. */
  bool q6k_repack_enabled_ = false;

  /* The KV cache, `[layer][position][kv_head][head_dim]`.
   *
   * The layer dimension is not optional and is the one thing about this layout
   * worth stating: every layer has its own keys and values, so a cache shared
   * across layers holds the last layer's data for every position the current
   * batch did not rewrite. A batched prefill never notices -- each layer writes
   * its own positions immediately before reading them -- and a second call
   * reads 27 layers of stale keys and produces a fluent, completely wrong
   * token. That is exactly the bug this dimension was added to fix. */
  kernel::DeviceBuffer k_cache_, v_cache_;
  int64_t cache_capacity_ = 0;
  /* The cache's element width, taken from the backend once at load rather than
   * from a constant here: a graph that named f16 would be a graph that decides
   * for the card too, and the card keeps an f32 cache deliberately -- see
   * `Backend::preferred_kv_dtype`.  Read once and stored because it also
   * decides every offset in the slabs, and asking the backend per layer would
   * be thirty virtual calls a token for a value that cannot change. */
  kernel::KVDtype kv_dtype_ = kernel::KVDtype::kF32;

  Weight tok_embd_;
  Weight output_;
  kernel::DeviceBuffer output_norm_;

  int64_t n_embd_ = 0, n_layer_ = 0, n_head_ = 0, n_head_kv_ = 0, head_dim_ = 0, n_ff_ = 0;
  int64_t n_vocab_ = 0, capacity_ = 0;
  float rms_eps_ = 1e-6F;
  float rope_theta_ = 10000.0F;

  /* How many positions the cache and the rotary table have room for. Grown
   * together, because a token that fits the cache but not the table would be a
   * read past the end of the table. */
  int64_t position_capacity_ = 0;
  /* Every sequence's cache length, one entry per live sequence.  Index 0 is the
   * single-sequence path's `cache_length_`; the batch path uses one entry per
   * sequence, and they drift apart exactly by how far each sequence has been
   * decoded -- which is the ragged window the batch has to honor. */
  std::vector<int64_t> seq_len_{0};
  /* How many sequence slots the KV slabs were allocated for.  Kept beside
   * `cache_capacity_` because the two together size the slab and either can grow
   * independently. */
  int64_t kv_slots_ = 0;

  std::vector<Layer> layers_;
};

}  // namespace pocketllm

#endif /* POCKETLLM_MODEL_QWEN3_H */
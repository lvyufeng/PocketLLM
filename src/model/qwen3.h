/* Qwen3, dense, running on the CPU.
 *
 * The architecture is small enough to state in full: an embedding, twenty-eight
 * identical blocks, and a final norm feeding a tied output projection. Each
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
 */

#ifndef POCKETLLM_MODEL_QWEN3_H
#define POCKETLLM_MODEL_QWEN3_H

#include <cstdint>
#include <memory>
#include <vector>

#include "gguf/reader.h"

namespace pocketllm {

/* How a stored weight is used by `matmul`.
 *
 * Today both spellings describe an f16/f32 tensor: the bytes are read once at
 * load and `data` points at the widened copy. The `blocks` form is where the
 * quantized path lands -- a step-5 kernel reads those bytes directly and never
 * expands them -- and it is declared now so that adding it is a change inside
 * `matmul` rather than a change to every call site. */
struct Weight {
  const float *data = nullptr;     /* dequantized, for an unquantized tensor */
  const uint8_t *blocks = nullptr; /* the raw bytes, for a quantized one */
  int64_t rows = 0;                /* output features (GGUF ne1) */
  int64_t cols = 0;                /* input features (GGUF ne0) */
  int type_id = 0;
  int64_t nbytes = 0;
};

class Qwen3Model {
 public:
  /* Read the hyperparameters and bind every tensor. Throws `Error` naming the
   * key or the tensor that was missing, because a checkpoint missing one of the
   * 199 weights produces nonsense rather than a failure, and the nonsense
   * arrives as a wrong token several hundred milliseconds later.
   *
   * The reader must outlive the model: unquantized weights are copied out of
   * it, but a quantized one would be a pointer into its mapping. */
  static std::unique_ptr<Qwen3Model> load(const GgufReader &checkpoint);

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

  /* How many positions the cache holds. */
  int64_t cache_length() const { return cache_length_; }

  /* Run `n` tokens occupying positions `start_pos .. start_pos + n - 1` and
   * return the logits of the *last* one. `logits` must have room for `n_vocab`
   * floats. `start_pos` must equal `cache_length()` for a decode and 0 for a
   * fresh prompt -- the graph does not special-case a restart, so a caller that
   * wants one calls `reset()`.
   *
   * The returned pointer is owned by the model and stays valid until the next
   * call. */
  const float *forward(const int32_t *tokens, int64_t n, int64_t start_pos);

  /* Drop the KV cache. The buffers are kept -- a decode after a reset is the
   * next thing that happens, and reallocating 300 MB to fill none of it is
   * pure loss. */
  void reset();

 private:
  Qwen3Model() = default;

  /* Everything a layer needs, resolved to pointers once. An index-and-lookup
   * per weight would be a string comparison in the inner loop's preamble and a
   * null check in every layer of every forward. */
  struct Layer {
    const float *attn_norm = nullptr;
    const float *ffn_norm = nullptr;
    const float *q_norm = nullptr;
    const float *k_norm = nullptr;
    Weight wq, wk, wv, wo;
    Weight w_gate, w_up, w_down;
  };

  /* The widened copy of a tensor, kept alive for the model's lifetime. Returned
   * by pointer into a stable allocation: `std::vector<Weight>` would move its
   * elements on growth and invalidate every `data` a layer holds. */
  const float *bind_dense(const GgufReader &checkpoint, const std::string &name);
  Weight bind_matrix(const GgufReader &checkpoint, const std::string &name);

  void matmul(const Weight &w, const float *x, float *out, int64_t m, bool accumulate) const;
  void ensure_capacity(int64_t n);
  void build_rope_table();

  /* Scratch, resized to the batch as it is needed. Held on the model because
   * the shape is the model's (n_embd, n_head, ...) and reallocating them per
   * forward would touch the allocator several times per token. */
  std::vector<float> x_, x_norm_, q_, k_, v_, attn_, gate_, up_, ffn_, last_, logits_;
  std::vector<float> scores_;
  std::vector<float> rope_cos_, rope_sin_;

  /* The KV cache, `[layer][position][kv_head][head_dim]`.
   *
   * The layer dimension is not optional and is the one thing about this layout
   * worth stating: every layer has its own keys and values, so a cache shared
   * across layers holds the last layer's data for every position the current
   * batch did not rewrite. A batched prefill never notices -- each layer writes
   * its own positions immediately before reading them -- and a second call
   * reads 27 layers of stale keys and produces a fluent, completely wrong
   * token. That is exactly the bug this dimension was added to fix. */
  std::vector<float> k_cache_, v_cache_;

  std::vector<std::unique_ptr<float[]>> blobs_;
  std::vector<Layer> layers_;

  Weight tok_embd_, output_;
  const float *output_norm_ = nullptr;

  int64_t n_embd_ = 0, n_layer_ = 0, n_head_ = 0, n_head_kv_ = 0, head_dim_ = 0, n_ff_ = 0;
  int64_t n_vocab_ = 0, capacity_ = 0;
  float rms_eps_ = 1e-6F;
  float rope_theta_ = 10000.0F;

  int64_t cache_capacity_ = 0;
  int64_t cache_length_ = 0;
};

}  // namespace pocketllm

#endif /* POCKETLLM_MODEL_QWEN3_H */
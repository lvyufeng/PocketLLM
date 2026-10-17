#include "runtime/session.h"

#include <utility>
#include <vector>

#include "kernel/backend.h"
#include "runtime/status.h"

namespace pocketllm {

Session::Session(std::string gguf_path, std::string backend_name,
                 std::unique_ptr<kernel::Backend> backend, std::unique_ptr<GgufReader> checkpoint,
                 std::unique_ptr<Tokenizer> tokenizer, std::unique_ptr<Qwen3Model> model)
    : gguf_path_(std::move(gguf_path)),
      backend_name_(std::move(backend_name)),
      backend_(std::move(backend)),
      checkpoint_(std::move(checkpoint)),
      tokenizer_(std::move(tokenizer)),
      model_(std::move(model)) {}

Session::~Session() = default;

std::string Session::no_graph_message() const {
  const std::string arch = checkpoint_->get_string("general.architecture", "?");
  const char *why = arch_refusal(arch);
  return "this checkpoint is architecture '" + arch + "', which this build cannot run" +
         (why == nullptr ? std::string() : std::string(": ") + why);
}

std::unique_ptr<Session> Session::open(const std::string &gguf_path, const std::string &backend) {
  /* An empty name means the caller took the default, which the header documents
   * as "cpu".  Anything else is resolved against what this *build* provides.
   *
   * The check happens here, before the file is mapped, and it is not merely an
   * optimization. The device is the argument the *caller* passed, so it is the
   * one they can always act on; the checkpoint is a file, and a caller told
   * "that is not a GGUF file" has learned nothing about the request they got
   * wrong. Asking about the device first is what makes the error name the thing
   * the caller controls -- and it means a two-gigabyte mapping is not paid for
   * on a request that was never going to work. */
  std::string device = backend.empty() ? "cpu" : backend;
  kernel::require_backend(device);

  /* The stat check is gone: `GgufReader` opens the file itself and reports a
   * missing one with the same errno message, so checking first would only be a
   * second place that can disagree about what "opens" means. */
  auto checkpoint = std::make_unique<GgufReader>(gguf_path);
  auto tokenizer = std::make_unique<Tokenizer>(*checkpoint);

  /* The model is optional in a way the reader and the tokenizer are not, and
   * the architecture, not the parse, is what decides: `GgufReader` reads any
   * well-formed GGUF, so a checkpoint of an architecture this build has no
   * graph for is a *successful* open that cannot run.  That is the state every
   * other architecture is in, and turning it into a failed open would make
   * `encode` -- which works fine -- unreachable for them.
   *
   * The backend is *created* only here and not before, so a checkpoint this
   * build has no graph for does not pay for a CUDA context. Creating it is a
   * heavier thing than asking whether it can be created, which is what the
   * check above already did. */
  std::unique_ptr<kernel::Backend> device_backend;
  std::unique_ptr<Qwen3Model> model;
  if (arch_refusal(checkpoint->get_string("general.architecture", "")) == nullptr) {
    device_backend = kernel::make_backend(device);
    model = Qwen3Model::load(*checkpoint, *device_backend);
    /* The model has copied every tensor it needs into the backend's memory, and
     * the tokenizer its vocabulary and merges, so nothing reads the file again.
     * Release the mapping here rather than at session teardown: `MemAvailable`
     * is what the caller watches, and it is measured while the session is open,
     * not after it closes.  A checkpoint of an architecture this build cannot
     * run stays mapped, because binding never happened and `tensor_data` may
     * still be the only thing keeping the loaded bytes alive. */
    checkpoint->release_mapping();
  }

  return std::unique_ptr<Session>(new Session(gguf_path, device, std::move(device_backend),
                                              std::move(checkpoint), std::move(tokenizer),
                                              std::move(model)));
}

std::vector<float> Session::forward(const int32_t *tokens, int64_t n, int64_t *argmax_out) {
  if (model_ == nullptr) {
    throw Error(no_graph_message());
  }
  const kernel::DeviceBuffer logits = model_->forward(tokens, n, position_);
  position_ = model_->cache_length();

  if (argmax_out != nullptr) {
    /* The caller wants the token and not the distribution, so the vocabulary
     * stays on the device.  This is the greedy fast path the header's
     * `pocketllm_forward_argmax` exists for: `Backend::argmax` writes an int64
     * on the device and eight bytes cross the bus, instead of the
     * `n_vocab * 4` bytes the logits transfer below would cost.  The id is the
     * same one `pocketllm_argmax` would pick over the host logits -- both are
     * `Backend::argmax`, ties to the lowest index -- so the two paths cannot
     * disagree, and it is returned as the empty vector's companion: the id is
     * in `*argmax_out` and there are no logits to hand back. */
    kernel::DeviceBuffer index = backend_->allocate(8);
    backend_->argmax(logits, model_->n_vocab(), index);
    backend_->synchronize();
    backend_->copy_to_host(argmax_out, index, 8);
    backend_->release(index);
    return {};
  }

  /* The logits come back in one transfer.  Not because they are wanted on the
   * host -- a decode wants one integer -- but because the ABI hands them to the
   * caller, and a caller that does not want them asked for the argmax above. */
  std::vector<float> host(static_cast<std::size_t>(model_->n_vocab()));
  backend_->copy_to_host(host.data(), logits, model_->n_vocab() * 4);
  return host;
}

std::vector<float> Session::forward_embeddings(const float *planes, const int64_t *positions,
                                                int64_t n, int64_t start_pos) {
  if (model_ == nullptr) {
    throw Error(no_graph_message());
  }
  /* The image span is decoded causally, which is a measured fact about this
   * architecture rather than a simplification: llama.cpp's rule is
   * `mtmd_decode_use_non_causal`, and it answers true only for the Gemma
   * projectors.  A `qwen3vl_merger` span sees the tokens before it and not the
   * ones after, so the causal arm is the correct one -- and it is also the arm
   * that does not need an f32 cache, which is why this call carries no
   * environment requirement the way the non-causal one would. */
  const kernel::DeviceBuffer logits =
      model_->forward_embeddings(planes, positions, n, start_pos, /*non_causal=*/false);
  position_ = model_->cache_length();

  std::vector<float> host(static_cast<std::size_t>(model_->n_vocab()));
  backend_->copy_to_host(host.data(), logits, model_->n_vocab() * 4);
  return host;
}

std::vector<float> Session::forward_positions(const int32_t *tokens, const int64_t *positions,
                                              int64_t n, int64_t start_pos) {
  if (model_ == nullptr) {
    throw Error(no_graph_message());
  }
  const kernel::DeviceBuffer logits = model_->forward_positions(tokens, positions, n, start_pos);
  position_ = model_->cache_length();

  std::vector<float> host(static_cast<std::size_t>(model_->n_vocab()));
  backend_->copy_to_host(host.data(), logits, model_->n_vocab() * 4);
  return host;
}

int64_t Session::cache_length() const {
  return model_ == nullptr ? 0 : model_->cache_length();
}

void Session::forward_batch(const int32_t *tokens, const int64_t *n_per_seq, int64_t n_seq,
                            int32_t *out_tokens) {
  if (model_ == nullptr) {
    throw Error(no_graph_message());
  }
  if (out_tokens == nullptr || n_seq <= 0) {
    throw Error("forward_batch needs an output and at least one sequence");
  }
  /* One pass over all the sequences' tokens; the model keeps one cached length
   * per sequence and writes one logit row per sequence.  The greedy id is read
   * from each row's device argmax, the same four-byte path `forward` uses when
   * asked for a token -- the vocabulary never comes to the host. */
  const kernel::DeviceBuffer logits = model_->forward_batch(tokens, n_per_seq, n_seq);
  const int64_t vocab = model_->n_vocab();
  kernel::DeviceBuffer index = backend_->allocate(8 * n_seq);
  for (int64_t s = 0; s < n_seq; ++s) {
    const kernel::DeviceBuffer row{logits.handle + static_cast<uintptr_t>(s * vocab * 4),
                                   vocab * 4};
    const kernel::DeviceBuffer slot{index.handle + static_cast<uintptr_t>(s * 8), 8};
    backend_->argmax(row, vocab, slot);
  }
  backend_->synchronize();
  std::vector<int64_t> ids(static_cast<std::size_t>(n_seq));
  backend_->copy_to_host(ids.data(), index, n_seq * 8);
  backend_->release(index);
  for (int64_t s = 0; s < n_seq; ++s) {
    out_tokens[s] = static_cast<int32_t>(ids[static_cast<std::size_t>(s)]);
  }
  /* `position_` tracks sequence 0, which is the single-sequence reader's
   * notion of "where the cache is"; the model holds every sequence's length. */
  position_ = model_->cache_length();
}

void Session::reserve(int64_t positions, int64_t slots) {
  if (model_ == nullptr) {
    throw Error(no_graph_message());
  }
  model_->reserve(positions, slots);
  /* The cache now holds no sequence's history; the reservation is the size,
   * not a decode position. */
  position_ = 0;
}

void Session::reset() {
  position_ = 0;
  if (model_ != nullptr) {
    model_->reset();
  }
}

}  // namespace pocketllm

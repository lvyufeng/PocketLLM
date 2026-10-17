/* The PocketLLM engine's C ABI.
 *
 * This header is the *only* contract between the Python host shell and the
 * native engine, and it is normative from the first commit.  The engine owns
 * the whole chain -- GGUF read, tokenize, graph walk, sample -- and Python
 * reaches all of it through the flat `extern "C"` surface below.  Nothing here
 * mentions a C++ type, an STL container or an exception, because every caller
 * of it is either a host language that has none of those or a test harness
 * that wants a stable symbol to dlsym.
 *
 * Conventions, in one place:
 *
 *   - Every function that can fail takes an `err` buffer and returns a negative
 *     value; a non-negative return is the success result.  0 is a valid
 *     success result (an empty token list), which is exactly why failure is
 *     negative rather than 0.
 *   - Errors do not cross the boundary as exceptions.  A C++ exception that
 *     escaped into a Python `ctypes` frame would be undefined behaviour.
 *   - A `pocketllm_session*` is opaque and owned by its opener; it is not
 *     thread-safe, and one session drives one device, which is the invariant
 *     the whole tree is built around.
 *   - Buffers are caller-allocated.  `_cap` is the caller's capacity in
 *     elements; a call that needs more than `_cap` fails rather than
 *     overflowing.
 *   - Negative returns are not all alike.  `-1` is a generic failure and `-2`
 *     an unbound model; `POCKETLLM_ERR_CONTEXT_LENGTH` (`-3`) is a
 *     specifically *recoverable* one -- the request ran past the checkpoint's
 *     context window -- and no other failure uses that value, so a host can
 *     tell a bad request from a broken engine without reading the message.
 */

#ifndef POCKETLLM_H
#define POCKETLLM_H

#include <stddef.h>
#include <stdint.h>

/* The return `pocketllm_forward` gives when the tokens would run past the
 * checkpoint's context window.  It is the one negative value a caller may
 * usefully act on: the request is too long, not malformed, and a shorter one
 * will succeed.  `pocketllm_forward` takes no error buffer, so this code is how
 * a caller tells that case apart; the limit itself is the checkpoint's context
 * length, a property of the model rather than of a given call. */
#define POCKETLLM_ERR_CONTEXT_LENGTH (-3)

/* The library is built with hidden visibility, so a function that is not
 * marked here is not in the symbol table at all -- which is the point: the ABI
 * is this list and nothing else, and a stray exported symbol is a commitment
 * nobody agreed to.  The empty case is for the Windows and static builds this
 * header is written to survive. */
#if defined(_WIN32) || defined(__CYGWIN__)
#define POCKETLLM_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define POCKETLLM_API __attribute__((visibility("default")))
#else
#define POCKETLLM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The major version is a handshake: the Python loader refuses a library whose
 * major differs from the one it was written against, because a change to it is
 * a change to a signature in this file.  The minor version moves for an
 * additive change and a host may ignore it. */
#define POCKETLLM_ABI_VERSION_MAJOR 1
#define POCKETLLM_ABI_VERSION_MINOR 2

/* The largest error message the engine writes, including the terminator.  A
 * caller that passes a smaller buffer gets a truncated message, never an
 * overflow. */
#define POCKETLLM_ERR_CAP 512

typedef struct pocketllm_session pocketllm_session;

/* The ABI this library implements, as "MAJOR.MINOR".  The returned string is
 * static and must not be freed.  This is the one call that is always safe --
 * it needs no session and cannot fail. */
POCKETLLM_API const char *pocketllm_abi_version(void);

/* Open a checkpoint and prepare a session.
 *
 * `backend` is a device name -- "cpu" today.  `err` may be NULL, in which case
 * `err_cap` is ignored; otherwise a message is written to it on failure.
 * Returns NULL on failure. */
POCKETLLM_API pocketllm_session *pocketllm_open(const char *gguf_path, const char *backend, char *err, size_t err_cap);

/* Release a session.  A NULL argument is a no-op, which makes the open/fail
 * path in a caller that always calls close safe. */
POCKETLLM_API void pocketllm_close(pocketllm_session *session);

/* Tokenize `text` into `out`.
 *
 * `add_special` asks for the model's BOS/EOS convention rather than raw text.
 *
 * `parse_special` decides whether a control token spelled inside `text` -- a
 * chat template's `<|im_start|>`, say -- becomes that one token or the
 * characters it is spelled with.  The two are not the same request: a caller
 * driving a chat template wants 1, and one feeding user-typed text wants 0,
 * because that text merely mentions the spelling.  User-defined tokens are
 * recognized either way.  The distinction is llama.cpp's, and the test uses
 * it to compare against `llama_tokenize` in both modes.
 *
 * Returns the number of token ids written, or a negative value on failure.
 * A return greater than `cap` means the result did not fit and `out` was not
 * written -- callers should treat that as an error and size the buffer from
 * `pocketllm_encode`'s first call with a NULL `out` and a zero `cap`. */
POCKETLLM_API int pocketllm_encode(const pocketllm_session *session, const char *text, int add_special,
                                   int parse_special, int32_t *out, int cap);

/* Detokenize `n` token ids into `out`, NUL-terminated.
 *
 * Returns the number of bytes written excluding the terminator, or a negative
 * value on failure. */
POCKETLLM_API int pocketllm_decode(const pocketllm_session *session, const int32_t *ids, int n, char *out, int cap);

/* The normative graph call: run `tokens` through the model and write the
 * logits for the last position into `logits`.
 *
 * This is the call the whole engine exists to serve.  It allocates its own
 * activations, keeps the KV cache in the session, and returns the vocabulary
 * size -- which is also the number of floats written -- or a negative value on
 * failure. */
POCKETLLM_API int pocketllm_forward(pocketllm_session *session, const int32_t *tokens, int n, float *logits,
                                    int logits_cap);

/* The greedy fast path: run `tokens` and write only the index of the largest
 * logit into `out_token`.
 *
 * This is `pocketllm_forward` with the vocabulary left on the device.  The
 * device computes the argmax (`Backend::argmax`) and four bytes cross the bus,
 * so a caller that wants one token never materializes 151936 floats -- which is
 * the whole cost of a decode step on a host whose logits surface is a Python
 * list rather than a ctypes buffer.  The token it returns is the *same* one
 * `pocketllm_argmax` would pick over the logits `pocketllm_forward` would have
 * written: both are `Backend::argmax`, and ties resolve to the lowest index
 * there, so the two paths cannot disagree.
 *
 * `out_token` is written only on success.  Returns 0 on success, a negative
 * value on failure -- `kErrContextLength` for a prompt past the checkpoint's
 * window, the same recoverable code `pocketllm_forward` returns.
 *
 * This is an addition, not a replacement: `pocketllm_forward` is unchanged and
 * stays the call a sampler needs, because temperature/top-k/top-p want the
 * distribution and not just its argmax. */
POCKETLLM_API int pocketllm_forward_argmax(pocketllm_session *session, const int32_t *tokens, int n,
                                           int32_t *out_token);

/* The Stage-1 batch decode: run `n_seq` sequences in one forward and write each
 * sequence's greedy token id into `out_tokens`.
 *
 * `tokens` is the concatenation of every sequence's tokens, `n_per_seq[s]` how
 * many of them belong to sequence s (decode passes one per sequence), and the
 * sequences run from their own cached lengths, so their attention windows are
 * ragged -- sequence 1 may be at position 40 while sequence 2 is at 300.  The
 * model writes one token per sequence, in order.
 *
 * The point is the weights: every stack reads its weights once for all
 * sequences, so the GEMM runs at `m = sum(n_per_seq)` instead of once per
 * sequence.  This is what `docs/architecture/batch_serving.md` calls Stage 1:
 * the sequence axis on the KV cache, with the single-sequence path unchanged
 * (call `pocketllm_forward` / `pocketllm_forward_argmax` for one sequence).
 *
 * Returns 0 on success, a negative value on failure -- `kErrContextLength` when
 * any sequence would run past the checkpoint's window.  `out_tokens` is written
 * only on success and must hold `n_seq` int32s.  It is additive:
 * `pocketllm_forward` and `pocketllm_forward_argmax` are unchanged. */
POCKETLLM_API int pocketllm_forward_batch(pocketllm_session *session, const int32_t *tokens,
                                          const int64_t *n_per_seq, int64_t n_seq,
                                          int32_t *out_tokens);

/* Pre-size the KV cache to `slots` sequence slots of `positions` rows each.
 *
 * The batch path exists so a scheduler can run `pocketllm_forward_batch` every
 * step.  It cannot, however, let the cache *grow* while more than one sequence
 * is live: the grow path releases and reallocates the slabs without copying,
 * so a sequence crossing a capacity boundary would wipe every other slot's KV
 * and return fluent-wrong text with no error.  A scheduler calls this once,
 * before admitting a group, and then keeps every sequence's position inside the
 * `positions` window; the forward path never reaches its grow branch.
 *
 * Idempotent, and it does not discard a cache it already covers.  `positions`
 * past the checkpoint's context length is `kErrContextLength`.  Returns 0 on
 * success, a negative value on failure.  Additive -- nothing else changes. */
POCKETLLM_API int pocketllm_reserve(pocketllm_session *session, int64_t positions,
                                    int64_t slots);

/* Drop the KV cache and the position counter.  The next `pocketllm_forward`
 * starts a fresh sequence.  Returns 0 on success, negative on failure. */
POCKETLLM_API int pocketllm_reset(pocketllm_session *session);

/* --- Vision-language -------------------------------------------------------
 *
 * A vision prompt is a text model, a projector, and a *composition* of three
 * calls: the text before the image, the image span as embeddings, and the text
 * after it.  That order is the engine's contract rather than a caller's
 * convenience -- the span's attention is a property of a batch and not of a range
 * inside one -- so it is one call here, not three.  A caller that assembled the
 * three itself would be a second implementation of the one thing this ABI exists
 * to fix.
 *
 * The image is raw RGB, `width * height * 3` bytes, three interleaved channels,
 * row-major, top to bottom, at any size: the engine resizes it the way the
 * reference does, choosing a target with the checkpoint's own pixel limits.  That
 * choice is not cosmetic -- the span's token count is `(nx / 32) * (ny / 32)` --
 * so `pocketllm_vlm_image_tokens` reports what a shape becomes before a caller
 * commits to it. */

typedef struct pocketllm_vlm pocketllm_vlm;

/* Open a tower and a text model on one device.  Both paths are required: the
 * language model without its projector cannot take an image, and the projector
 * without its language model produces features nothing reads.  Returns NULL on
 * failure, with `err` written the same way `pocketllm_open` writes it. */
POCKETLLM_API pocketllm_vlm *pocketllm_vlm_open(const char *mmproj_path, const char *gguf_path,
                                                const char *backend, char *err, size_t err_cap);

/* Release a vision session.  A NULL argument is a no-op. */
POCKETLLM_API void pocketllm_vlm_close(pocketllm_vlm *session);

/* How many image tokens an input of this shape becomes, or negative on failure.
 *
 * Worth its own call because the answer is not a function of the pixel count
 * alone: the preprocessing rounds the shape onto the patch grid and then clamps it
 * by the checkpoint's own limits, so a caller that guesses from area sizes its
 * prompt wrong.  An image already at the checkpoint's own size is unchanged. */
POCKETLLM_API int64_t pocketllm_vlm_image_tokens(const pocketllm_vlm *session, int64_t width,
                                                 int64_t height);

/* The marker the prompt splits on -- `"<__media__>"`, which is `libmtmd`'s
 * default.  A prompt holds exactly one of them; two is refused rather than
 * resolved to the first, because the composition above places one image per call.
 * The returned string is static and must not be freed. */
POCKETLLM_API const char *pocketllm_vlm_marker(void);

/* Run the whole path and write the logits of the last position into `logits`.
 *
 * `prompt` holds one marker; `rgb` is `width * height * 3` bytes as described
 * above.  Returns the vocabulary size -- also the number of floats written -- or a
 * negative value on failure, the same convention `pocketllm_forward` uses.
 *
 * This is a *prefill*: the three calls are one prompt, so the session's cache is
 * reset first and a caller that wants a second turn starts one here rather than
 * continuing from the last.  Leaving a previous composition's rows in place would
 * have the span attend to them, which is a different prompt and not an error
 * anything downstream would catch. */
POCKETLLM_API int pocketllm_vlm_forward(pocketllm_vlm *session, const char *prompt,
                                        const uint8_t *rgb, int64_t width, int64_t height,
                                        float *logits, int logits_cap);

/* The greedy fast path: run the composition and write only the index of the
 * largest logit into `out_token`.
 *
 * This is `pocketllm_vlm_forward` with the vocabulary left on the device, for the
 * reason `pocketllm_forward_argmax` gives -- a caller that wants one token should
 * not materialize 151936 floats -- and the id is the same one
 * `pocketllm_argmax` would pick over those logits.  Returns 0 on success, a
 * negative value on failure. */
POCKETLLM_API int pocketllm_vlm_forward_argmax(pocketllm_vlm *session, const char *prompt,
                                               const uint8_t *rgb, int64_t width, int64_t height,
                                               int32_t *out_token);

/* The index of the largest of `n` logits, which is greedy sampling.  Returns a
 * non-negative index, or negative if `n` is not positive.  It takes no session
 * because it is a pure function of its input, and a caller may sample outside
 * the engine. */
POCKETLLM_API int pocketllm_argmax(const float *logits, int n);

/* ``out[i] = logits[i] / temperature``, in place if `out == logits`.
 *
 * Refuses a `temperature <= 0` -- a temperature of zero is greedy, and greedy
 * is `pocketllm_argmax`, not a division by zero that fills the vector with
 * infinities.  Returns 0 on success and a negative value on failure, so a
 * caller can tell a refusal from a buffer it should have sized itself.
 *
 * Like `pocketllm_argmax` this takes no session: it is a pure function of its
 * input, and the engine holds no state for it. */
POCKETLLM_API int pocketllm_temperature(const float *logits, float *out, int n, float temperature);

/* Draw one token from the top-k/top-p/min-p truncated softmax of `logits`,
 * given a uniform variate in `[0, 1)`.
 *
 * Returns the token id, or a negative value if `n` is not positive.  `top_k`
 * of 0 means no top-k limit; `top_p` of 1.0 and `min_p` of 0.0 mean no
 * truncation, so the three trivial arguments sample the untruncated softmax.
 *
 * The variate is an argument and not something the engine draws: the library
 * holds no RNG, so a caller that wants a seed controls it, and the same
 * `(logits, uniform)` always produces the same token.  That is also what makes
 * this testable against the reference without an oracle for the RNG. */
POCKETLLM_API int pocketllm_sample(const float *logits, int n, float uniform, int top_k, float top_p,
                                   float min_p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POCKETLLM_H */
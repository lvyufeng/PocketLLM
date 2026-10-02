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
 */

#ifndef POCKETLLM_H
#define POCKETLLM_H

#include <stddef.h>
#include <stdint.h>

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
#define POCKETLLM_ABI_VERSION_MINOR 1

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

/* Drop the KV cache and the position counter.  The next `pocketllm_forward`
 * starts a fresh sequence.  Returns 0 on success, negative on failure. */
POCKETLLM_API int pocketllm_reset(pocketllm_session *session);

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
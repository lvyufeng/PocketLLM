/* The ABI vocabulary, mirrored from Python.
 *
 * `python/pocketllm/kernels/dtypes.py` is the spec and this file is the
 * transcription.  They answer different questions and are deliberately not
 * shared: the Python table is what a *kernel* needs, keyed by ABI name, and
 * these are what the *file* says, keyed by GGML type id.  ``file_type_id`` is
 * the GGML id; the runtime id a raw-block kernel switches on is a different
 * numbering and lives in dtypes.py.
 *
 * `python/pocketllm/loader/gguf/reader.py`'s GGML_TYPES is the authority for
 * this table.  A divergence is caught by the reader test, which compares every
 * tensor's type name and byte size against the Python reader on a real
 * checkpoint.
 */

#ifndef POCKETLLM_ABI_SPEC_H
#define POCKETLLM_ABI_SPEC_H

#include <cstdint>

namespace pocketllm {

/* The geometry of one GGML storage type: how many weights share a block, and
 * how many bytes that block occupies.  An unpacked type has a block of one
 * element, which makes the "blocked" and "plain" cases the same arithmetic in
 * the byte-size calculation. */
struct GgmlType {
  const char *name;
  int block_elems;
  int block_bytes;
};

/* The geometry for a GGML type id.
 *
 * An id the table does not carry comes back as ``{"unknown_<id>", 0, 0}`` --
 * the same shape the Python reader reports for the fork-private ternary packs.
 * Knowing a type is *not* the same as being able to run it: this function
 * answers "how many bytes", and dispatch is refused elsewhere, on purpose.
 *
 * The returned name points at static storage; the caller does not own it. */
GgmlType ggml_type_of(int type_id);

}  // namespace pocketllm

#endif /* POCKETLLM_ABI_SPEC_H */
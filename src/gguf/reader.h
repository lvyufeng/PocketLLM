/* The GGUF reader.
 *
 * A transcription of `python/pocketllm/loader/gguf/reader.py`, and it has to
 * stay one: the Python reader is the oracle the C one is checked against, so
 * where the two could differ the Python is right and the C moves.  The test
 * opens the same checkpoint with both and compares the metadata map, the tensor
 * count and every tensor's ``(dims, type, offset, nbytes)``.
 *
 * The file is memory-mapped rather than read into a buffer.  A 1.5 GB f16
 * checkpoint is 1.5 GB of page cache the kernel already has, and copying it
 * into a heap buffer per session would double the resident set for no gain --
 * the tensors are read once, in place, by the kernels.
 *
 * "Read once" is also what lets `release_mapping()` hand the mapping back when
 * the caller is done with it, which the library's own entry points do after
 * `Qwen3Model::load` has bound every tensor: on a board where the accelerator
 * and the host share one LPDDR pool, keeping the file mapped costs the pool
 * the whole checkpoint.  See `release_mapping()`.
 *
 * The layout, in order, is what makes the reader a straight-line program:
 *
 *   "GGUF" | version u32 | tensor_count u64 | metadata_count u64
 *   | metadata KV, repeated | tensor records, repeated
 *   | pad to ``alignment`` | tensor data
 *
 * Note that tensor_count precedes metadata_count.  That is the format, not a
 * typo, and it is the one ordering mistake that produces a plausible-looking
 * failure rather than a crash.
 */

#ifndef POCKETLLM_GGUF_READER_H
#define POCKETLLM_GGUF_READER_H

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace pocketllm {

/* A metadata value.
 *
 * GGUF has thirteen scalar types plus string and array.  They collapse into
 * four C++ shapes: an integer, a float, a string, and a typed list; the exact
 * GGUF type is kept alongside so a value can be written back, and so the reader
 * test can compare types and not merely values. */
struct GgufArray {
  int value_type = 0;
  std::vector<int64_t> ints;      /* every integer type, sign-extended */
  std::vector<double> floats;     /* f32 and f64 */
  std::vector<std::string> strings;
  std::vector<uint8_t> bools;
  uint64_t length = 0;
};

using GgufValue = std::variant<int64_t, double, bool, std::string, GgufArray>;

struct GgufMetadataEntry {
  int type = 0;
  GgufValue value;
};

struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> dimensions; /* as stored: fastest axis first */
  int type_id = 0;
  uint64_t offset = 0;          /* relative to data_start */
  uint64_t absolute_offset = 0; /* data_start + offset */
  uint64_t nbytes = 0;          /* 0 when the type geometry is unknown */
  bool size_known = true;
};

class GgufReader {
 public:
  /* Map ``path`` and parse its header, metadata and tensor directory.  Throws
   * `Error` -- with a message naming what was wrong -- for a missing file, a
   * bad magic, a truncated header, or a metadata type this build cannot read.
   * The tensor *data* is not touched here; only the directory is. */
  explicit GgufReader(const std::string &path);
  ~GgufReader();

  GgufReader(const GgufReader &) = delete;
  GgufReader &operator=(const GgufReader &) = delete;

  uint32_t version() const { return version_; }
  uint64_t tensor_count() const { return tensor_count_; }
  uint64_t metadata_count() const { return metadata_count_; }
  uint64_t data_start() const { return data_start_; }
  uint64_t alignment() const { return alignment_; }
  uint64_t size() const { return size_; }
  const std::string &path() const { return path_; }

  const std::map<std::string, GgufMetadataEntry> &metadata() const { return metadata_; }
  const std::vector<GgufTensorInfo> &tensors() const { return tensors_; }

  /* A metadata value by key, or nullptr.  The typed accessors below are what
   * callers use; this is for the reader test's comparison. */
  const GgufValue *find(const std::string &key) const;

  /* Typed lookups, each with a default rather than a throw: a missing
   * hyperparameter is worth a named failure at the point it is used, not at
   * the point it is read. */
  int64_t get_int(const std::string &key, int64_t fallback) const;
  double get_float(const std::string &key, double fallback) const;
  std::string get_string(const std::string &key, const std::string &fallback) const;

  /* A tensor's directory entry, or nullptr.  The pointer is stable for the
   * reader's lifetime. */
  const GgufTensorInfo *tensor(const std::string &name) const;

  /* A pointer to a tensor's bytes in the mapping, and its length.  Throws if
   * the tensor is unknown or its bytes run past the end of the file -- which is
   * the check that turns a truncated download into a message instead of a
   * segfault during the first matmul. */
  const uint8_t *tensor_data(const std::string &name, uint64_t *nbytes) const;

  /* Drop the file mapping once nothing will read the file again.
   *
   * The directory -- names, shapes, offsets and the metadata -- is parsed into
   * this object's own storage at construction, so `size()`, `metadata()`,
   * `tensors()` and `tensor()` keep working after the bytes are gone; only
   * `tensor_data()` needs the mapping, and it throws a named error rather than
   * returning a dangling pointer if it is called afterwards.
   *
   * This is what makes a checkpoint larger than the board's free host memory
   * runnable: a caller that has bound every tensor into device memory (or its
   * own buffers) no longer holds 4.7 GiB of reclaimable page-cache-backed
   * mapping against `MemAvailable`, so the pages the single shared LPDDR pool
   * reports as free are real.  A checkpoint the loader re-reads stays mapped --
   * this is a decision the caller makes, not a policy this class applies. */
  void release_mapping();

  /* False once `release_mapping()` has run. */
  bool mapping_held() const { return mapping_ != nullptr; }

 private:
  void parse();

  std::string path_;
  int fd_ = -1;
  const uint8_t *mapping_ = nullptr;
  uint64_t size_ = 0;

  uint32_t version_ = 0;
  uint64_t tensor_count_ = 0;
  uint64_t metadata_count_ = 0;
  uint64_t alignment_ = 32;
  uint64_t data_start_ = 0;

  std::map<std::string, GgufMetadataEntry> metadata_;
  std::vector<GgufTensorInfo> tensors_;
  std::map<std::string, std::size_t> tensor_index_;
};

/* The byte size of a tensor with ``elements`` values of ``type_id``, or 0 and
 * ``*known = false`` when the type's geometry is not in the table.  Rounded up
 * to a whole block, because a blocked type stores a partial block in full. */
uint64_t tensor_nbytes(int type_id, const std::vector<uint64_t> &dimensions, bool *known);

}  // namespace pocketllm

#endif /* POCKETLLM_GGUF_READER_H */
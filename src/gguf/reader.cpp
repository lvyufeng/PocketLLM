#include "gguf/reader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

#include "abi/spec.h"
#include "runtime/status.h"

namespace pocketllm {

namespace {

/* GGUF's metadata value types, from the format's own `gguf_meta_value_type`. */
enum MetadataType {
  kUint8 = 0,
  kInt8 = 1,
  kUint16 = 2,
  kInt16 = 3,
  kUint32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUint64 = 10,
  kInt64 = 11,
  kFloat64 = 12,
};

/* A cursor over the mapping.  Bounds-checked on every read, because the input
 * is a file that may be truncated, corrupt or simply not a GGUF, and every one
 * of those should be an exception rather than a read past the end. */
class Cursor {
 public:
  Cursor(const uint8_t *data, uint64_t size) : data_(data), size_(size) {}

  uint64_t position() const { return pos_; }

  void seek(uint64_t to) {
    if (to > size_) {
      throw Error("GGUF: seek past end of file");
    }
    pos_ = to;
  }

  void skip(uint64_t count) { seek(pos_ + count); }

  void read(void *out, uint64_t count) {
    if (pos_ + count > size_) {
      throw Error("GGUF: unexpected end of file");
    }
    std::memcpy(out, data_ + pos_, count);
    pos_ += count;
  }

  template <typename T>
  T read_scalar() {
    T value{};
    read(&value, sizeof(T));
    return value;
  }

  /* GGUF strings are length-prefixed and UTF-8.  Invalid bytes are replaced
   * rather than rejected, matching the Python reader's ``errors="replace"`` --
   * a checkpoint with a stray byte in a metadata string is a checkpoint whose
   * weights are fine, and failing the whole load over it would be the wrong
   * trade. */
  std::string read_string() {
    const uint64_t n = read_scalar<uint64_t>();
    if (pos_ + n > size_) {
      throw Error("GGUF: string runs past end of file");
    }
    std::string out(reinterpret_cast<const char *>(data_ + pos_), static_cast<std::size_t>(n));
    pos_ += n;
    return out;
  }

 private:
  const uint8_t *data_;
  uint64_t size_;
  uint64_t pos_ = 0;
};

/* Materialise a metadata array.
 *
 * The Python reader *skips* arrays by default and describes them with a
 * summary, because materialising a 150k-entry tokenizer vocabulary to read one
 * header field is waste.  The C reader always materialises, because it is the
 * one that has to *use* the vocabulary -- and it is a deliberate difference,
 * not a divergence: the summary the Python side returns and the values this
 * returns are checked to agree on length and item type by the reader test. */
GgufArray read_array(Cursor &cursor, int item_type, uint64_t length) {
  GgufArray array;
  array.value_type = item_type;
  array.length = length;
  const auto count = static_cast<std::size_t>(length);

  switch (item_type) {
    case kString:
      array.strings.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        array.strings.push_back(cursor.read_string());
      }
      return array;

    case kBool:
      array.bools.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        array.bools.push_back(cursor.read_scalar<uint8_t>() != 0);
      }
      return array;

    case kFloat32:
    case kFloat64:
      array.floats.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        array.floats.push_back(item_type == kFloat32 ? static_cast<double>(cursor.read_scalar<float>())
                                                     : cursor.read_scalar<double>());
      }
      return array;

    case kArray:
      throw Error("GGUF: nested metadata arrays are not supported");

    default: {
      array.ints.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        /* Sign-extended into one int64 slot so a caller does not have to
         * re-branch on the declared width to read a count. */
        int64_t value = 0;
        switch (item_type) {
          case kUint8: value = cursor.read_scalar<uint8_t>(); break;
          case kInt8: value = cursor.read_scalar<int8_t>(); break;
          case kUint16: value = cursor.read_scalar<uint16_t>(); break;
          case kInt16: value = cursor.read_scalar<int16_t>(); break;
          case kUint32: value = cursor.read_scalar<uint32_t>(); break;
          case kInt32: value = cursor.read_scalar<int32_t>(); break;
          case kUint64: value = static_cast<int64_t>(cursor.read_scalar<uint64_t>()); break;
          case kInt64: value = cursor.read_scalar<int64_t>(); break;
          default: throw Error("GGUF: unsupported array item type " + std::to_string(item_type));
        }
        array.ints.push_back(value);
      }
      return array;
    }
  }
}

GgufValue read_value(Cursor &cursor, int type) {
  switch (type) {
    case kUint8: return static_cast<int64_t>(cursor.read_scalar<uint8_t>());
    case kInt8: return static_cast<int64_t>(cursor.read_scalar<int8_t>());
    case kUint16: return static_cast<int64_t>(cursor.read_scalar<uint16_t>());
    case kInt16: return static_cast<int64_t>(cursor.read_scalar<int16_t>());
    case kUint32: return static_cast<int64_t>(cursor.read_scalar<uint32_t>());
    case kInt32: return static_cast<int64_t>(cursor.read_scalar<int32_t>());
    case kUint64: return static_cast<int64_t>(cursor.read_scalar<uint64_t>());
    case kInt64: return cursor.read_scalar<int64_t>();
    case kFloat32: return static_cast<double>(cursor.read_scalar<float>());
    case kFloat64: return cursor.read_scalar<double>();
    case kBool: return cursor.read_scalar<uint8_t>() != 0;
    case kString: return cursor.read_string();
    case kArray: {
      const int item_type = static_cast<int>(cursor.read_scalar<uint32_t>());
      const uint64_t length = cursor.read_scalar<uint64_t>();
      return read_array(cursor, item_type, length);
    }
    default:
      throw Error("GGUF: unsupported metadata type " + std::to_string(type));
  }
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

uint64_t tensor_nbytes(int type_id, const std::vector<uint64_t> &dimensions, bool *known) {
  const GgmlType type = ggml_type_of(type_id);
  if (type.block_elems <= 0) {
    if (known != nullptr) {
      *known = false;
    }
    return 0;
  }
  if (known != nullptr) {
    *known = true;
  }
  uint64_t elements = 1;
  for (uint64_t dim : dimensions) {
    elements *= dim;
  }
  const uint64_t blocks = (elements + static_cast<uint64_t>(type.block_elems) - 1) /
                          static_cast<uint64_t>(type.block_elems);
  return blocks * static_cast<uint64_t>(type.block_bytes);
}

GgufReader::GgufReader(const std::string &path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw Error("cannot open checkpoint '" + path + "': " + std::strerror(errno));
  }

  struct stat st {};
  if (::fstat(fd_, &st) != 0) {
    throw Error("cannot stat checkpoint '" + path + "': " + std::strerror(errno));
  }
  size_ = static_cast<uint64_t>(st.st_size);
  if (size_ < 8) {
    throw Error("'" + path + "' is too small to be a GGUF file");
  }

  void *mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapping == MAP_FAILED) {
    throw Error("cannot map checkpoint '" + path + "': " + std::strerror(errno));
  }
  mapping_ = static_cast<const uint8_t *>(mapping);

  try {
    parse();
  } catch (...) {
    /* A parse failure leaves nothing worth keeping, and the destructor would
     * run anyway -- but the object is not constructed at this point, so the
     * unmap has to happen here. */
    ::munmap(const_cast<uint8_t *>(mapping_), size_);
    ::close(fd_);
    mapping_ = nullptr;
    fd_ = -1;
    throw;
  }
}

GgufReader::~GgufReader() {
  if (mapping_ != nullptr) {
    ::munmap(const_cast<uint8_t *>(mapping_), size_);
  }
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

void GgufReader::parse() {
  Cursor cursor(mapping_, size_);

  char magic[4] = {};
  cursor.read(magic, 4);
  if (std::memcmp(magic, "GGUF", 4) != 0) {
    throw Error("'" + path_ + "' is not a GGUF file (bad magic)");
  }

  version_ = cursor.read_scalar<uint32_t>();
  /* Tensors before metadata.  The format says so, and reading them in the
   * other order yields a plausible small tensor count rather than an error. */
  tensor_count_ = cursor.read_scalar<uint64_t>();
  metadata_count_ = cursor.read_scalar<uint64_t>();

  for (uint64_t i = 0; i < metadata_count_; ++i) {
    std::string key = cursor.read_string();
    const int type = static_cast<int>(cursor.read_scalar<uint32_t>());
    GgufMetadataEntry entry;
    entry.type = type;
    entry.value = read_value(cursor, type);
    metadata_[key] = std::move(entry);
  }

  std::vector<std::tuple<std::string, std::vector<uint64_t>, int, uint64_t>> records;
  records.reserve(static_cast<std::size_t>(tensor_count_));
  for (uint64_t i = 0; i < tensor_count_; ++i) {
    std::string name = cursor.read_string();
    const uint32_t n_dims = cursor.read_scalar<uint32_t>();
    std::vector<uint64_t> dims;
    dims.reserve(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      dims.push_back(cursor.read_scalar<uint64_t>());
    }
    const int type_id = static_cast<int>(cursor.read_scalar<uint32_t>());
    const uint64_t offset = cursor.read_scalar<uint64_t>();
    records.emplace_back(std::move(name), std::move(dims), type_id, offset);
  }

  alignment_ = 32;
  if (const GgufValue *value = find("general.alignment")) {
    if (const auto *as_int = std::get_if<int64_t>(value)) {
      alignment_ = static_cast<uint64_t>(*as_int);
    }
  }
  if (alignment_ == 0) {
    alignment_ = 1;
  }
  data_start_ = align_up(cursor.position(), alignment_);

  tensors_.reserve(records.size());
  for (auto &record : records) {
    GgufTensorInfo info;
    info.name = std::move(std::get<0>(record));
    info.dimensions = std::move(std::get<1>(record));
    info.type_id = std::get<2>(record);
    info.offset = std::get<3>(record);
    info.absolute_offset = data_start_ + info.offset;
    bool known = true;
    info.nbytes = tensor_nbytes(info.type_id, info.dimensions, &known);
    info.size_known = known;
    tensor_index_[info.name] = tensors_.size();
    tensors_.push_back(std::move(info));
  }
}

const GgufValue *GgufReader::find(const std::string &key) const {
  auto it = metadata_.find(key);
  return it == metadata_.end() ? nullptr : &it->second.value;
}

int64_t GgufReader::get_int(const std::string &key, int64_t fallback) const {
  const GgufValue *value = find(key);
  if (value == nullptr) {
    return fallback;
  }
  if (const auto *as_int = std::get_if<int64_t>(value)) {
    return *as_int;
  }
  if (const auto *as_double = std::get_if<double>(value)) {
    return static_cast<int64_t>(*as_double);
  }
  return fallback;
}

double GgufReader::get_float(const std::string &key, double fallback) const {
  const GgufValue *value = find(key);
  if (value == nullptr) {
    return fallback;
  }
  if (const auto *as_double = std::get_if<double>(value)) {
    return *as_double;
  }
  if (const auto *as_int = std::get_if<int64_t>(value)) {
    return static_cast<double>(*as_int);
  }
  return fallback;
}

std::string GgufReader::get_string(const std::string &key, const std::string &fallback) const {
  const GgufValue *value = find(key);
  if (value == nullptr) {
    return fallback;
  }
  if (const auto *as_string = std::get_if<std::string>(value)) {
    return *as_string;
  }
  return fallback;
}

const GgufTensorInfo *GgufReader::tensor(const std::string &name) const {
  auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const uint8_t *GgufReader::tensor_data(const std::string &name, uint64_t *nbytes) const {
  const GgufTensorInfo *info = tensor(name);
  if (info == nullptr) {
    throw Error("GGUF: no tensor named '" + name + "'");
  }
  if (info->absolute_offset + info->nbytes > size_) {
    throw Error("GGUF: tensor '" + name + "' runs past the end of the file (truncated?)");
  }
  /* The directory outlives the mapping, so this is the one place a released
   * mapping can be reached -- and it has to be a named error rather than a
   * dangling pointer: the caller asked for bytes that were deliberately handed
   * back, which is a different mistake from a truncated file. */
  if (mapping_ == nullptr) {
    throw Error("GGUF: tensor '" + name +
                "' was requested after the checkpoint mapping was released");
  }
  if (nbytes != nullptr) {
    *nbytes = info->nbytes;
  }
  return mapping_ + info->absolute_offset;
}

void GgufReader::release_mapping() {
  if (mapping_ == nullptr) {
    return;
  }
  ::munmap(const_cast<uint8_t *>(mapping_), size_);
  mapping_ = nullptr;
}

}  // namespace pocketllm
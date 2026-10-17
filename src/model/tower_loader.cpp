#include "model/tower_loader.h"

#include <cstring>

#include "abi/spec.h"
#include "quant/half.h"
#include "quant/q8_0.h"
#include "runtime/status.h"

namespace pocketllm {
namespace tower {
namespace {

/* The type ids this loader reads.  They are GGML's storage ids, which is what the
 * file states; the runtime dispatch numbering is a different one and lives in
 * `abi/spec.h`. */
constexpr int kTypeF32 = 0;
constexpr int kTypeF16 = 1;
constexpr int kTypeQ8_0 = 8;

int64_t element_count(const GgufTensorInfo &info) {
  int64_t elements = 1;
  for (uint64_t dim : info.dimensions) {
    elements *= static_cast<int64_t>(dim);
  }
  return elements;
}

}  // namespace

int64_t require_int(const GgufReader &checkpoint, const std::string &key) {
  const int64_t value = checkpoint.get_int(key, 0);
  if (value <= 0) {
    throw Error("the mmproj has no positive metadata key '" + key +
                "'; this build reads a tower's geometry from the file rather than assuming it");
  }
  return value;
}

kernel::DeviceBuffer bind_tensor(const GgufReader &checkpoint, kernel::Backend &backend,
                                std::vector<kernel::DeviceBuffer> &owned, const std::string &name) {
  uint64_t nbytes = 0;
  const uint8_t *bytes = checkpoint.tensor_data(name, &nbytes);
  const GgufTensorInfo *info = checkpoint.tensor(name);
  if (info == nullptr) {
    throw Error("the tower is missing tensor '" + name +
                "'; a tower missing one weight answers with a plausible embedding rather than an "
                "error");
  }
  const int64_t elements = element_count(*info);

  std::vector<float> widened(static_cast<std::size_t>(elements));
  if (info->type_id == kTypeF32) {
    if (static_cast<int64_t>(nbytes) != elements * 4) {
      throw Error("tensor '" + name + "' is declared f32 and " + std::to_string(nbytes) +
                  " bytes for " + std::to_string(elements) + " elements");
    }
    std::memcpy(widened.data(), bytes, static_cast<std::size_t>(nbytes));
  } else if (info->type_id == kTypeF16) {
    if (static_cast<int64_t>(nbytes) != elements * 2) {
      throw Error("tensor '" + name + "' is declared f16 and " + std::to_string(nbytes) +
                  " bytes for " + std::to_string(elements) + " elements");
    }
    for (int64_t i = 0; i < elements; ++i) {
      widened[static_cast<std::size_t>(i)] = half_to_float(load_u16(bytes + 2 * i));
    }
  } else if (info->type_id == kTypeQ8_0) {
    if (static_cast<int64_t>(nbytes) != quant::q8_0_packed_bytes(elements)) {
      throw Error("tensor '" + name + "' is declared q8_0 and " + std::to_string(nbytes) +
                  " bytes for " + std::to_string(elements) + " elements");
    }
    quant::q8_0_dequantize(bytes, elements, widened.data());
  } else {
    throw Error("tensor '" + name + "' is " + ggml_type_of(info->type_id).name +
                ", and this loader reads f32, f16 and q8_0");
  }

  kernel::DeviceBuffer device = backend.allocate(elements * 4);
  backend.copy_to_device(device, widened.data(), elements * 4);
  owned.push_back(device);
  return device;
}

kernel::DeviceBuffer bind_optional_tensor(const GgufReader &checkpoint, kernel::Backend &backend,
                                          std::vector<kernel::DeviceBuffer> &owned,
                                          const std::string &name) {
  if (checkpoint.tensor(name) == nullptr) {
    /* A null handle, not a zero buffer: the graph skips the add.  For a quantized
     * tower the two are not the same thing, which is why the distinction is carried
     * rather than smoothed over. */
    return kernel::DeviceBuffer{};
  }
  return bind_tensor(checkpoint, backend, owned, name);
}

Linear bind_linear(const GgufReader &checkpoint, kernel::Backend &backend,
                   std::vector<kernel::DeviceBuffer> &owned, const std::string &base,
                   int64_t rows, int64_t cols) {
  const std::string name = base + ".weight";
  const GgufTensorInfo *info = checkpoint.tensor(name);
  if (info == nullptr) {
    throw Error("the tower is missing tensor '" + name +
                "'; a tower missing one weight answers with a plausible embedding rather than an "
                "error");
  }
  /* GGUF writes the fastest-varying axis first, so a matrix arrives as
   * (contraction, output) -- the transpose of how the product is written.  The bytes
   * are already row-major in that order, so nothing is transposed here and these two
   * lines are the only place the conventions meet. */
  if (info->dimensions.size() != 2 || static_cast<int64_t>(info->dimensions[0]) != cols ||
      static_cast<int64_t>(info->dimensions[1]) != rows) {
    throw Error("tensor '" + name + "' is not (" + std::to_string(cols) + ", " +
                std::to_string(rows) + "), which is (contraction, output)");
  }
  Linear linear;
  linear.weight = bind_tensor(checkpoint, backend, owned, name);
  linear.bias = bind_optional_tensor(checkpoint, backend, owned, base + ".bias");
  return linear;
}

}  // namespace tower
}  // namespace pocketllm
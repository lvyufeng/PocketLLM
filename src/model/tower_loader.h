/* Binding a projector's tensors, shared by the two towers this tree reads.
 *
 * A vision tower and an audio tower are different graphs over the same kind of file:
 * an `mmproj` that is dense in one case and eight-bit in the other, with biases that
 * are always f32.  What they have in common is the *loading*, and it is the part
 * where being nearly right is worst -- a tensor bound with the wrong width produces a
 * plausible embedding and a wrong answer several hundred milliseconds later, with
 * nothing in between to say which tensor it was.
 *
 * So this is one implementation rather than two.  The vision tower had its own
 * (`bind_dense`, originally in `qwen3vl_vision.cpp`) and the audio tower needed the
 * same thing plus `q8_0`; copying it would have been two places to fix the next time a
 * format arrives, and the `q8_0` branch is exactly the kind of change that gets made
 * in one of them.
 *
 * The surfaces are deliberately small: read a tensor, read a matrix, and fail by
 * *name*.  A loader that fails by shape rather than by name is a loader whose error
 * message sends the reader to the wrong file.
 */

#ifndef POCKETLLM_MODEL_TOWER_LOADER_H
#define POCKETLLM_MODEL_TOWER_LOADER_H

#include <cstdint>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"

namespace pocketllm {
namespace tower {

/* A matrix projection with a bias, which every tower's block is made of.  `rows` and
 * `cols` are checked against what the file declares rather than trusted, because a
 * transpose is a shape error the arithmetic below would happily accept. */
struct Linear {
  kernel::DeviceBuffer weight;
  kernel::DeviceBuffer bias;
};

/* A whole tensor, widened to f32 on the device.
 *
 * f32 is copied, f16 is widened, `q8_0` is decoded (`quant/q8_0.h`), and anything
 * else is refused by name.  The decoded form is what the graph reads: a GEMM that
 * keeps eight-bit weights has to quantize the activation to match, which is a
 * different change with its own measurements, and this is the shortest road to a
 * tower whose numbers can be compared against the reference.
 *
 * `owned` receives the buffer so the caller can free it; the towers keep one such
 * vector apiece and release it in their destructors. */
kernel::DeviceBuffer bind_tensor(const GgufReader &checkpoint, kernel::Backend &backend,
                                 std::vector<kernel::DeviceBuffer> &owned, const std::string &name);

/* `name + ".weight"` and `name + ".bias"`, as a pair, with the declared shape of the
 * weight checked against `rows x cols`.
 *
 * A tensor with no bias in the file comes back with a null buffer handle, which is
 * how every caller below distinguishes "no bias" from "a zero bias" -- the graph
 * skips the add rather than adding nothing, and for a quantized path the difference
 * is a real one. */
Linear bind_linear(const GgufReader &checkpoint, kernel::Backend &backend,
                   std::vector<kernel::DeviceBuffer> &owned, const std::string &base,
                   int64_t rows, int64_t cols);

/* A bias-only tensor, or a null buffer if the file has none. */
kernel::DeviceBuffer bind_optional_tensor(const GgufReader &checkpoint, kernel::Backend &backend,
                                          std::vector<kernel::DeviceBuffer> &owned,
                                          const std::string &name);

/* The integer a metadata key holds, or a thrown `Error` naming the key.  A tower
 * whose geometry is guessed rather than read is a tower that runs and answers
 * wrongly. */
int64_t require_int(const GgufReader &checkpoint, const std::string &key);

}  // namespace tower
}  // namespace pocketllm

#endif /* POCKETLLM_MODEL_TOWER_LOADER_H */
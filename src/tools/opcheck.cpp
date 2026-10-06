/* Run one kernel over inputs a caller supplies, for checking against an oracle.
 *
 * Internal, like the other tools: it links the engine's objects directly so that
 * it can call a single kernel and print exactly what came out.
 *
 * The inputs are not generated here.  The caller writes a **request** file, this
 * tool runs the op on it, and it writes a **response**.  That direction matters
 * and is worth stating plainly: the oracle for a quantized kernel is the Python
 * reference decoder, which means the Python side has to author the call anyway --
 * so if this tool also generated inputs, there would be two descriptions of what
 * "the canonical call to `gemm`" is, in two languages, and they would drift.
 * Making Python the author and this tool a pure evaluator removes that
 * possibility structurally rather than by discipline.
 *
 * The format is line-oriented rather than JSON, and deliberately: it is a few
 * dozen lines of parsing instead of a dependency, and a failing case is a text
 * file a person can read side by side.
 *
 *     op rms_norm
 *     device cpu
 *     param eps 1e-06
 *     tensor x f32 2 256
 *     <512 floats, space separated>
 *     tensor weight f32 256
 *     <256 floats>
 *
 * The response is the same shape, with the output tensor:
 *
 *     op rms_norm
 *     device cpu
 *     status ok
 *     tensor out f32 2 256
 *     <512 floats>
 *
 * A failure writes `status error` and a `message` line rather than exiting with
 * a code, so the caller can report what the engine said.
 *
 * `--poison` fills the output with a sentinel before the call and reports how
 * many elements are still the sentinel afterwards.  That is the check a kernel
 * that only wrote part of its output fails: the untouched elements hold the
 * previous token's values, which are finite, plausible, and wrong.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kernel/backend.h"
#include "quant/blocks.h"
#include "runtime/status.h"

namespace {

using pocketllm::kernel::Backend;
using pocketllm::kernel::DeviceBuffer;
using pocketllm::kernel::KVDtype;

/* Deliberately not a value any of these kernels can produce: an odd exponent far
 * outside the range the graph occupies.  A true positive is still possible in
 * principle -- nothing constrains an output to avoid it -- which is why this is
 * reported as a count for the caller to judge rather than asserted here. */
constexpr float kPoison = -1.2345678e33F;

/* A tensor as the request file describes it: the dtype is one of `f32` or `i32`,
 * because those are the two the kernels read. */
struct Tensor {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<float> floats;
  std::vector<int32_t> ints;
  /* A packed weight is neither: it is the checkpoint's own bytes, and the whole
   * point of these kernels is that the bytes are never reinterpreted as
   * numbers on the way to the device. */
  std::vector<uint8_t> bytes;

  int64_t elements() const {
    int64_t n = 1;
    for (int64_t d : shape) {
      n *= d;
    }
    /* An empty shape is a scalar -- one element -- which is what every tensor
     * library means by rank 0 and what the schema uses for `topk_sample`'s
     * `uniform`. Returning 0 here made a 0-dim tensor unreachable: the parser
     * would demand zero values for it. The product over an empty range is
     * already 1, so the special case is the `0` and not the `1`. */
    return n;
  }
};

struct Request {
  std::string op;
  std::string device = "cpu";
  bool poison = false;
  std::vector<std::pair<std::string, std::string>> params;
  std::vector<std::pair<std::string, Tensor>> tensors;

  const Tensor *tensor(const std::string &name) const {
    for (const auto &entry : tensors) {
      if (entry.first == name) {
        return &entry.second;
      }
    }
    return nullptr;
  }

  const Tensor &require(const std::string &name) const {
    const Tensor *found = tensor(name);
    if (found == nullptr) {
      throw pocketllm::Error("the request has no tensor '" + name + "'");
    }
    return *found;
  }

  bool has(const std::string &name) const { return tensor(name) != nullptr; }

  bool flag(const std::string &name) const {
    const std::string value = param(name, "0");
    return value == "1" || value == "true";
  }

  std::string param(const std::string &name, const std::string &fallback) const {
    for (const auto &entry : params) {
      if (entry.first == name) {
        return entry.second;
      }
    }
    return fallback;
  }

  int64_t int_param(const std::string &name, int64_t fallback) const {
    for (const auto &entry : params) {
      if (entry.first == name) {
        return std::stoll(entry.second);
      }
    }
    return fallback;
  }

  float float_param(const std::string &name, float fallback) const {
    for (const auto &entry : params) {
      if (entry.first == name) {
        return std::stof(entry.second);
      }
    }
    return fallback;
  }
};

std::vector<std::string> words(const std::string &line) {
  std::istringstream stream(line);
  std::vector<std::string> out;
  std::string word;
  while (stream >> word) {
    out.push_back(word);
  }
  return out;
}

/* Parse a request file.  The one structural rule is that a `tensor` line is
 * *always* followed by exactly one line of values, whose count must match the
 * shape -- a mismatch is a request that would silently read a neighbour's data,
 * so it is refused rather than padded. */
Request read_request(std::istream &in) {
  Request request;
  std::string line;
  while (std::getline(in, line)) {
    const std::vector<std::string> parts = words(line);
    if (parts.empty() || parts[0][0] == '#') {
      continue;
    }
    if (parts[0] == "op" && parts.size() == 2) {
      request.op = parts[1];
    } else if (parts[0] == "device" && parts.size() == 2) {
      request.device = parts[1];
    } else if (parts[0] == "poison") {
      request.poison = true;
    } else if (parts[0] == "param" && parts.size() == 3) {
      request.params.emplace_back(parts[1], parts[2]);
    } else if (parts[0] == "tensor" && parts.size() >= 3) {
      Tensor tensor;
      tensor.dtype = parts[2];
      for (std::size_t i = 3; i < parts.size(); ++i) {
        tensor.shape.push_back(std::stoll(parts[i]));
      }
      std::string values;
      if (!std::getline(in, values)) {
        throw pocketllm::Error("tensor '" + parts[1] + "' has no values line");
      }
      const std::vector<std::string> fields = words(values);
      if (static_cast<int64_t>(fields.size()) != tensor.elements()) {
        throw pocketllm::Error("tensor '" + parts[1] + "' has " + std::to_string(fields.size()) +
                               " values but its shape needs " + std::to_string(tensor.elements()));
      }
      if (tensor.dtype == "f32") {
        for (const std::string &field : fields) {
          tensor.floats.push_back(std::stof(field));
        }
      } else if (tensor.dtype == "i32") {
        for (const std::string &field : fields) {
          tensor.ints.push_back(static_cast<int32_t>(std::stol(field)));
        }
      } else if (tensor.dtype == "u8") {
        /* One value per byte, each in 0..255. The count rule is the same as
         * for a float tensor -- the shape's product is the number of *bytes* --
         * which is what lets the caller say `w_blocks u8 4 1 144` and have the
         * tool check that four 144-byte blocks arrived rather than four
         * numbers. */
        for (const std::string &field : fields) {
          const long value = std::stol(field);
          if (value < 0 || value > 255) {
            throw pocketllm::Error("tensor '" + parts[1] + "' has byte value " + field +
                                   ", which is outside 0..255");
          }
          tensor.bytes.push_back(static_cast<uint8_t>(value));
        }
      } else {
        throw pocketllm::Error("tensor '" + parts[1] + "' has dtype '" + tensor.dtype +
                               "', which is neither f32, i32 nor u8");
      }
      request.tensors.emplace_back(parts[1], std::move(tensor));
    } else {
      throw pocketllm::Error("unknown directive: " + line);
    }
  }
  if (request.op.empty()) {
    throw pocketllm::Error("the request names no op");
  }
  return request;
}

/* `%.9g` round-trips a float32 exactly, which is what lets the caller compare
 * bit patterns rather than a printed approximation of them. */
void write_floats(std::ostream &out, const float *data, int64_t n) {
  char buffer[32];
  for (int64_t i = 0; i < n; ++i) {
    std::snprintf(buffer, sizeof(buffer), "%.9g", static_cast<double>(data[i]));
    out << (i == 0 ? "" : " ") << buffer;
  }
  out << "\n";
}

void write_floats(std::ostream &out, const std::vector<float> &values) {
  write_floats(out, values.data(), static_cast<int64_t>(values.size()));
}

/* Upload a host vector and hand back the device handle.  The caller owns it and
 * must release it -- `DeviceBuffer` has no destructor, which is the same bargain
 * the graph makes. */
DeviceBuffer upload(Backend &backend, const std::vector<float> &values) {
  DeviceBuffer buffer = backend.allocate(static_cast<int64_t>(values.size()) * 4);
  backend.copy_to_device(buffer, values.data(), static_cast<int64_t>(values.size()) * 4);
  return buffer;
}

DeviceBuffer upload(Backend &backend, const std::vector<uint8_t> &values) {
  DeviceBuffer buffer = backend.allocate(static_cast<int64_t>(values.size()));
  backend.copy_to_device(buffer, values.data(), static_cast<int64_t>(values.size()));
  return buffer;
}

DeviceBuffer upload(Backend &backend, const std::vector<int32_t> &values) {
  DeviceBuffer buffer = backend.allocate(static_cast<int64_t>(values.size()) * 4);
  backend.copy_to_device(buffer, values.data(), static_cast<int64_t>(values.size()) * 4);
  return buffer;
}

/* Everything below is one op: read its tensors out of the request, run it, and
 * print the result.  The dispatch is a plain if-chain rather than a table
 * because each branch has its own argument shape, and a table would only move
 * that shape into a struct nobody reads. */

int64_t count_poison(const std::vector<float> &values) {
  int64_t untouched = 0;
  for (float value : values) {
    if (value == kPoison) {
      ++untouched;
    }
  }
  return untouched;
}

/* Allocate the output, poison it if asked, run `call`, and print.  `call`
 * receives the output handle; the shape printed is the caller's. */
template <typename Call>
void run_and_report(std::ostream &out, Backend &backend, const Request &request,
                    const std::vector<int64_t> &shape, Call call) {
  int64_t elements = 1;
  for (int64_t d : shape) {
    elements *= d;
  }
  DeviceBuffer result = backend.allocate(elements * 4);
  if (request.poison) {
    backend.fill(result, kPoison);
  }
  call(result);
  backend.synchronize();

  std::vector<float> values(static_cast<std::size_t>(elements));
  backend.copy_to_host(values.data(), result, elements * 4);
  backend.release(result);

  out << "status ok\n";
  out << "tensor out f32";
  for (int64_t d : shape) {
    out << " " << d;
  }
  out << "\n";
  write_floats(out, values);
  if (request.poison) {
    out << "int unwritten " << count_poison(values) << "\n";
  }
}

void run_rms_norm(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &x = request.require("x");
  const Tensor &weight = request.require("weight");
  const float eps = request.float_param("eps", 1e-6F);
  const int64_t n_tokens = x.shape[0];
  const int64_t d = x.shape[1];
  if (weight.shape.size() != 1 || weight.shape[0] != d) {
    throw pocketllm::Error("rms_norm: the weight must have the row length as its only dimension");
  }
  DeviceBuffer dx = upload(backend, x.floats);
  DeviceBuffer dw = upload(backend, weight.floats);
  run_and_report(out, backend, request, {n_tokens, d}, [&](DeviceBuffer result) {
    backend.rms_norm(dx, dw, result, n_tokens, d, eps);
  });
  backend.release(dx);
  backend.release(dw);
}

void run_gemm(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &x = request.require("x");
  const Tensor &w = request.require("w");
  const int64_t m = x.shape[0];
  const int64_t k = x.shape[1];
  const int64_t n = w.shape[0];
  if (w.shape[1] != k) {
    throw pocketllm::Error("gemm: x is (m, k) and w must be (n, k) with the same k");
  }
  DeviceBuffer dx = upload(backend, x.floats);
  DeviceBuffer dw = upload(backend, w.floats);
  DeviceBuffer dbias;
  std::vector<float> bias;
  if (request.has("bias")) {
    bias = request.require("bias").floats;
    dbias = upload(backend, bias);
  }
  const bool accumulate = request.flag("accumulate");
  if (accumulate && request.poison) {
    /* The residual form adds into whatever is already there, so the sentinel
     * would become part of the sum and every element would differ from the
     * oracle by a constant large enough to swamp the answer.  The two flags ask
     * for incompatible things; refusing says so rather than printing a poisoned
     * comparison the caller has to interpret. */
    throw pocketllm::Error(
        "gemm: --poison and accumulate are incompatible; the sentinel would be summed into the "
        "residual");
  }
  run_and_report(out, backend, request, {m, n}, [&](DeviceBuffer result) {
    if (accumulate) {
      /* The residual form adds into whatever is already there, so the caller
       * seeds `out` through the request and the seed is what gets added to. */
      const Tensor &seed = request.require("out_seed");
      backend.copy_to_device(result, seed.floats.data(),
                             static_cast<int64_t>(seed.floats.size()) * 4);
    }
    backend.gemm(dx, dw, dbias, result, m, n, k, accumulate);
  });
  backend.release(dx);
  backend.release(dw);
  if (dbias.handle != 0) {
    backend.release(dbias);
  }
}

/* The packed product. Shape and the weight's *storage* are stated separately
 * and checked against each other, because this is the one op where the two can
 * disagree: `w_blocks n k/256 144` says how the row is walked, the block count
 * in the last dimension says how many bytes arrived, and a mismatch would
 * otherwise decode a weight from the neighbouring row and report a small
 * numerical difference rather than a bad request. */
void run_gemm_quant(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &x = request.require("x");
  const Tensor &w = request.require("w_blocks");
  const int64_t m = x.shape[0];
  const int64_t k = x.shape[1];
  const int64_t n = w.shape[0];
  const int type_id = static_cast<int>(request.int_param("type_id", 0));

  const int block_bytes = pocketllm::quant::block_bytes_of(type_id);
  if (block_bytes == 0) {
    throw pocketllm::Error("gemm_quant: no packed kernel for GGML type id " +
                           std::to_string(type_id));
  }
  if (k % pocketllm::quant::kBlockWeights != 0) {
    throw pocketllm::Error("gemm_quant: k must be a whole number of 256-weight blocks, got " +
                           std::to_string(k));
  }
  if (w.shape.size() != 3 || w.shape[1] != k / pocketllm::quant::kBlockWeights ||
      w.shape[2] != block_bytes) {
    throw pocketllm::Error("gemm_quant: w_blocks must be (n, k/256, " + std::to_string(block_bytes) +
                           ") for this type id");
  }
  int64_t expected_bytes = 1;
  for (int64_t dim : w.shape) {
    expected_bytes *= dim;
  }
  if (static_cast<int64_t>(w.bytes.size()) != expected_bytes) {
    throw pocketllm::Error("gemm_quant: w_blocks shape and byte count disagree");
  }

  DeviceBuffer dx = upload(backend, x.floats);
  DeviceBuffer dw = upload(backend, w.bytes);
  const bool accumulate = request.flag("accumulate");
  if (accumulate && request.poison) {
    throw pocketllm::Error(
        "gemm_quant: --poison and accumulate are incompatible; the sentinel would be summed into "
        "the residual");
  }
  run_and_report(out, backend, request, {m, n}, [&](DeviceBuffer result) {
    if (accumulate) {
      const Tensor &seed = request.require("out_seed");
      backend.copy_to_device(result, seed.floats.data(),
                             static_cast<int64_t>(seed.floats.size()) * 4);
    }
    backend.gemm_quant(dx, dw, DeviceBuffer{}, result, m, n, k, type_id, accumulate);
  });
  backend.release(dx);
  backend.release(dw);
}

void run_embedding_quant(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &tokens = request.require("tokens");
  const Tensor &table = request.require("table_blocks");
  const int64_t n_tokens = tokens.elements();
  const int64_t vocab = table.shape[0];
  const int64_t d = static_cast<int64_t>(table.shape[1]) * pocketllm::quant::kBlockWeights;
  const int type_id = static_cast<int>(request.int_param("type_id", 0));
  const int block_bytes = pocketllm::quant::block_bytes_of(type_id);
  if (block_bytes == 0) {
    throw pocketllm::Error("embedding_quant: no packed kernel for GGML type id " +
                           std::to_string(type_id));
  }
  if (table.shape.size() != 3 || table.shape[2] != block_bytes) {
    throw pocketllm::Error("embedding_quant: table_blocks must be (vocab, d/256, " +
                           std::to_string(block_bytes) + ") for this type id");
  }

  DeviceBuffer dtokens = upload(backend, tokens.ints);
  DeviceBuffer dtable = upload(backend, table.bytes);
  run_and_report(out, backend, request, {n_tokens, d}, [&](DeviceBuffer result) {
    backend.embedding_quant(dtokens, n_tokens, dtable, vocab, d, type_id, result);
  });
  backend.release(dtokens);
  backend.release(dtable);
}

void run_embedding(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &tokens = request.require("tokens");
  const Tensor &table = request.require("table");
  const int64_t n_tokens = tokens.elements();
  const int64_t vocab = table.shape[0];
  const int64_t d = table.shape[1];
  DeviceBuffer dtokens = upload(backend, tokens.ints);
  DeviceBuffer dtable = upload(backend, table.floats);
  run_and_report(out, backend, request, {n_tokens, d}, [&](DeviceBuffer result) {
    backend.embedding(dtokens, n_tokens, dtable, vocab, d, result);
  });
  backend.release(dtokens);
  backend.release(dtable);
}

void run_silu_mul(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &gate = request.require("gate");
  const Tensor &up = request.require("up");
  if (gate.shape != up.shape) {
    throw pocketllm::Error("silu_mul: gate and up must have the same shape");
  }
  DeviceBuffer dgate = upload(backend, gate.floats);
  DeviceBuffer dup = upload(backend, up.floats);
  run_and_report(out, backend, request, gate.shape, [&](DeviceBuffer result) {
    backend.silu_mul(dgate, dup, result, gate.elements());
  });
  backend.release(dgate);
  backend.release(dup);
}

void run_rope(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &x = request.require("x");
  const Tensor &cos = request.require("cos");
  const Tensor &sin = request.require("sin");
  const int64_t n_tokens = x.shape[0];
  const int64_t n_heads = x.shape[1];
  const int64_t d = x.shape[2];
  const int64_t start_pos = request.int_param("start_pos", 0);
  DeviceBuffer dx = upload(backend, x.floats);
  DeviceBuffer dcos = upload(backend, cos.floats);
  DeviceBuffer dsin = upload(backend, sin.floats);
  run_and_report(out, backend, request, {n_tokens, n_heads, d}, [&](DeviceBuffer result) {
    /* The op is in-place, so the result buffer is a copy of the input and the
     * rotation happens to it.  Copying rather than rotating `dx` directly keeps
     * the input and the output as separate things the caller can diff, which is
     * the only reason this reads better than mutating. */
    backend.copy_device_to_device(result, dx, n_tokens * n_heads * d * 4);
    backend.rope_neox(result, n_tokens, n_heads, d, start_pos, dcos, dsin);
  });
  backend.release(dx);
  backend.release(dcos);
  backend.release(dsin);
}

void run_attention(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &q = request.require("q");
  const Tensor &k_cache = request.require("k_cache");
  const Tensor &v_cache = request.require("v_cache");
  const int64_t q_len = q.shape[0];
  const int64_t n_heads = q.shape[1];
  const int64_t d = q.shape[2];
  const int64_t n_head_kv = k_cache.shape[1];
  const int64_t first_key = request.int_param("first_key", 0);
  const int64_t q_offset = request.int_param("q_offset", 0);
  const float scale = request.float_param("scale", 1.0F / std::sqrt(static_cast<float>(d)));

  DeviceBuffer dq = upload(backend, q.floats);
  /* The cache's width is chosen here rather than read from the backend.
   *
   * f32 is the default because this tool's contract is to check an *op* against
   * a reference, and the reference is computed in f32: defaulting to the
   * backend's preference would make every existing attention comparison a
   * comparison of a rounded cache, and the tolerances those tests use would
   * have to be widened to hide it.  The `kv_dtype` field is how a test asks for
   * the lossy width explicitly, and doing so is the only way both widths come
   * out of one backend -- which is what makes the f16 path's reference the f32
   * result beside it.
   *
   * The *graph* does read the preference (see `Qwen3Model::load`), so the f16
   * cache a shipped run uses is covered end to end by the token tests rather
   * than by this tool. */
  const KVDtype kv_dtype =
      request.int_param("kv_dtype", 0) == 1 ? KVDtype::kF16 : KVDtype::kF32;
  const int64_t kv_elem = pocketllm::kernel::kv_dtype_size(kv_dtype);
  const int64_t kv_width = n_head_kv * d;
  const int64_t cache_bytes = k_cache.shape[0] * kv_width * kv_elem;
  DeviceBuffer dk = backend.allocate(cache_bytes);
  DeviceBuffer dv = backend.allocate(cache_bytes);
  {
    /* Built through the same `kv_append` the graph uses, one cache row at a
     * time: an f16 cache whose rows were written by a flat elementwise cast
     * would be the head-interleaving bug `kv_append` exists to prevent, and a
     * test that built its own fixture that way would be testing the bug. */
    DeviceBuffer host_k = upload(backend, k_cache.floats);
    DeviceBuffer host_v = upload(backend, v_cache.floats);
    backend.kv_append(DeviceBuffer{dk.handle, cache_bytes}, host_k, k_cache.shape[0],
                      n_head_kv, d, kv_elem);
    backend.kv_append(DeviceBuffer{dv.handle, cache_bytes}, host_v, v_cache.shape[0],
                      n_head_kv, d, kv_elem);
    backend.release(host_k);
    backend.release(host_v);
  }
  /* The scratch is the backend's to size -- see `attention_scratch` -- and this
   * is the one place a caller outside the graph has to ask for it. */
  const int64_t span = q_offset + q_len - first_key;
  DeviceBuffer scores = backend.allocate(backend.attention_scratch(q_len, n_heads, span));
  run_and_report(out, backend, request, {q_len, n_heads, d}, [&](DeviceBuffer result) {
    backend.attention(dq, q_len, n_heads, dk, dv, n_head_kv, d, first_key, q_offset, scale, result,
                      scores, kv_dtype);
  });
  backend.release(dq);
  backend.release(dk);
  backend.release(dv);
  backend.release(scores);
}

void run_argmax(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &values = request.require("values");
  DeviceBuffer dvalues = upload(backend, values.floats);
  DeviceBuffer result = backend.allocate(8);
  backend.argmax(dvalues, values.elements(), result);
  backend.synchronize();
  int64_t index = -1;
  backend.copy_to_host(&index, result, 8);
  backend.release(dvalues);
  backend.release(result);
  out << "status ok\n";
  /* An index is an index: printed as an integer and compared exactly, because a
   * value one off is a different answer rather than a rounding. */
  out << "int out " << index << "\n";
}

/* ``out = softmax(x)`` over the last axis, matching the schema's `axis=-1`.
 * A float op, so it goes through `run_and_report` and is poisonable. */
void run_softmax(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &x = request.require("x");
  if (x.shape.size() != 2) {
    throw pocketllm::Error("softmax: x must be (rows, cols)");
  }
  const int64_t rows = x.shape[0];
  const int64_t cols = x.shape[1];
  DeviceBuffer dx = upload(backend, x.floats);
  run_and_report(out, backend, request, {rows, cols}, [&](DeviceBuffer result) {
    backend.softmax(dx, result, rows, cols);
  });
  backend.release(dx);
}

void run_logits_temperature(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &logits = request.require("logits");
  const float temperature = request.float_param("temperature", 1.0F);
  DeviceBuffer dlogits = upload(backend, logits.floats);
  run_and_report(out, backend, request, logits.shape, [&](DeviceBuffer result) {
    backend.logits_temperature(dlogits, result, logits.elements(), temperature);
  });
  backend.release(dlogits);
}

/* The sampler. Like `run_argmax`, its answer is a token id and not a value, so
 * it prints `int out` and is compared exactly rather than at a tolerance -- a
 * token one away is a different word, not a rounding. */
void run_topk_sample(std::ostream &out, Backend &backend, const Request &request) {
  const Tensor &logits = request.require("logits");
  const Tensor &uniform = request.require("uniform");
  if (uniform.floats.empty()) {
    throw pocketllm::Error("topk_sample: uniform must carry one value");
  }
  const int64_t vocab = logits.elements();
  const int64_t top_k = request.int_param("top_k", 0);
  const float top_p = request.float_param("top_p", 1.0F);
  const float min_p = request.float_param("min_p", 0.0F);
  DeviceBuffer dlogits = upload(backend, logits.floats);
  /* The scratch the ranked index list needs; the backend does not read it back
   * and the caller does not either, but it has to exist and be `vocab` wide. */
  DeviceBuffer order = backend.allocate(vocab * 8);
  DeviceBuffer result = backend.allocate(8);
  backend.topk_sample(dlogits, vocab, uniform.floats[0], top_k, top_p, min_p, order, result);
  backend.synchronize();
  int64_t token = -1;
  backend.copy_to_host(&token, result, 8);
  backend.release(dlogits);
  backend.release(order);
  backend.release(result);
  out << "status ok\n";
  out << "int out " << token << "\n";
}

using Runner = void (*)(std::ostream &, Backend &, const Request &);

struct Entry {
  const char *name;
  Runner runner;
};

const Entry kOps[] = {
    {"rms_norm", run_rms_norm},   {"gemm", run_gemm},
    {"gemm_quant", run_gemm_quant}, {"embedding", run_embedding},
    {"embedding_quant", run_embedding_quant}, {"silu_mul", run_silu_mul},
    {"rope", run_rope},           {"attention", run_attention},
    {"argmax", run_argmax},       {"softmax", run_softmax},
    {"logits_temperature", run_logits_temperature}, {"topk_sample", run_topk_sample},
};

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s --request FILE [--out FILE] [--device cpu|cuda]\n"
               "       the request names the op; --device here overrides the one in it\n",
               argv0);
  std::fprintf(stderr, "ops:");
  for (const Entry &entry : kOps) {
    std::fprintf(stderr, " %s", entry.name);
  }
  std::fprintf(stderr, "\n");
  return 2;
}

}  // namespace

int main(int argc, char **argv) {
  std::string request_path;
  std::string out_path;
  std::string device_override;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--request" && i + 1 < argc) {
      request_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device_override = argv[++i];
    } else {
      return usage(argv[0]);
    }
  }
  if (request_path.empty()) {
    return usage(argv[0]);
  }

  std::ofstream file;
  std::ostream *out = &std::cout;
  if (!out_path.empty()) {
    file.open(out_path);
    if (!file) {
      std::fprintf(stderr, "%s: cannot write %s\n", argv[0], out_path.c_str());
      return 1;
    }
    out = &file;
  }

  try {
    std::ifstream in(request_path);
    if (!in) {
      throw pocketllm::Error("cannot read the request at " + request_path);
    }
    Request request = read_request(in);
    if (!device_override.empty()) {
      request.device = device_override;
    }

    const Runner runner = [&] {
      for (const Entry &entry : kOps) {
        if (request.op == entry.name) {
          return entry.runner;
        }
      }
      throw pocketllm::Error("this tool has no op '" + request.op + "'");
    }();

    auto backend = pocketllm::kernel::make_backend(request.device);
    *out << "op " << request.op << "\n";
    *out << "device " << request.device << "\n";
    runner(*out, *backend, request);
    return 0;
  } catch (const std::exception &e) {
    /* The failure goes into the response rather than only to stderr, so the
     * caller reports *what the engine said* instead of "the tool exited 1". */
    *out << "status error\n";
    *out << "message " << e.what() << "\n";
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}
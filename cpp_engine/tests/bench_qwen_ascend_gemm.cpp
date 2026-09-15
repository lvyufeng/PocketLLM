// GEMM microbenchmark at the shapes the Qwen3.8 TP4 prefill actually issues.
//
// The full-model loop hides the linear layers behind a 44 s weight load, so a
// per-shape number is the only way to tell which of them is slow and whether the
// cost is the Cube unit or the layout marshalling around it. The two layouts
// measured are the ones the engine passes: a dense [cols, rows] weight and a
// strided view into a wider buffer (`weight_stride > rows`), which is what the
// packed QKV/MLP weight tensors look like.
//
//   ./tests/bench_qwen_ascend_gemm [--device N] [--iters N]

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count == 0) return;
        if (!pocket::device_malloc_into(ptr_, count * sizeof(uint16_t))) {
            throw std::runtime_error("device_malloc failed");
        }
        if (!pocket::device_memset(ptr_, 0, count * sizeof(uint16_t))) {
            throw std::runtime_error("device_memset failed");
        }
    }
    explicit DeviceBuffer(const std::vector<uint16_t>& host) : DeviceBuffer(host.size()) {
        if (!pocket::memcpy_h2d(ptr_, host.data(), count_ * sizeof(uint16_t))) {
            throw std::runtime_error("memcpy_h2d failed");
        }
    }
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    uint16_t* get() const { return ptr_; }

private:
    uint16_t* ptr_ = nullptr;
    size_t count_ = 0;
};

struct Shape {
    const char* name;
    int batch;  // token rows
    int cols;   // contraction dimension (input features)
    int rows;   // output features
};

// One timed launch of the same operator the engine calls, including the device
// synchronize that makes the number meaningful on a queued device.
//
// The weight is a [rows, weight_stride] row-major buffer read as a [cols, rows]
// tensor with a column stride, which is the contract `matmul_rows` enforces
// (`weight_stride >= cols`): the projection weights are stored output-major, so
// one output row's weights are contiguous. A dense weight is simply
// weight_stride == cols, and a larger stride models the padding the packed
// QKV/MLP tensors carry.
double time_matmul(uint16_t* x, uint16_t* w, uint16_t* y, const Shape& s,
                   int weight_stride, int iters) {
    // Warm-up: the first aclnn call on a device pays the kernel-binary load, and
    // that one-time cost is several times a 4096-token projection. Timing it
    // would make whichever shape runs first look broken.
    if (!pocket::qwen_fp16_matmul_rows_f16(x, w, y, s.batch, s.rows, s.cols,
                                           s.cols, s.rows, weight_stride) ||
        !pocket::device_synchronize()) {
        throw std::runtime_error("matmul warm-up failed");
    }
    if (!pocket::device_synchronize()) throw std::runtime_error("pre-sync failed");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        if (!pocket::qwen_fp16_matmul_rows_f16(x, w, y, s.batch, s.rows, s.cols,
                                               s.cols, s.rows, weight_stride)) {
            throw std::runtime_error("matmul launch failed");
        }
    }
    if (!pocket::device_synchronize()) throw std::runtime_error("post-sync failed");
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int iters = 5;
    int batch = 4096;
    bool scan = false;
    bool mem = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch = std::stoi(argv[++i]);
        } else if (arg == "--scan") {
            scan = true;
        } else if (arg == "--mem") {
            mem = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (!pocket::device_runtime_available()) {
        std::printf("[SKIP] no device runtime available\n");
        return 0;
    }
    if (!pocket::device_set(device)) {
        std::printf("[SKIP] device_set failed for device %d\n", device);
        return 0;
    }

    // TP4 shapes for one token row count. The 4096-token chunk is what the
    // long-context prefill benchmark runs; batch 1 is the decode step, where the
    // Cube is idle and the cost is whatever the weight read actually achieves.
    const Shape prefill_shapes[] = {
        {"q_proj", 4096, 5120, 1536},
        {"o_proj", 4096, 1536, 5120},
        {"mlp_up", 4096, 5120, 4352},
        {"mlp_dn", 4096, 4352, 5120},
        {"square", 4096, 4096, 4096},
    };
    const Shape decode_shapes[] = {
        {"q_proj", 1, 5120, 1536},
        {"o_proj", 1, 1536, 5120},
        {"mlp_up", 1, 5120, 4352},
        {"mlp_dn", 1, 4352, 5120},
    };
    const std::vector<Shape> shapes(batch == 1
                                        ? std::begin(decode_shapes)
                                        : std::begin(prefill_shapes),
                                    batch == 1 ? std::end(decode_shapes)
                                               : std::end(prefill_shapes));

    // Bandwidth scan: one token row against growing weights. Without it the
    // batch=1 gbs figure above cannot be told apart from a per-call launch cost
    // that a larger weight would amortize, and the two imply opposite fixes.
    if (scan) {
        std::printf("device=%d iters=%d scan=1\n", device, iters);
        for (int rows : {1536, 6144, 24576, 98304, 393216}) {
            const Shape s{"scan", 1, 5120, rows};
            std::mt19937 rng(99u);
            std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
            std::vector<uint16_t> host_x(static_cast<size_t>(s.cols));
            for (uint16_t& v : host_x) v = float_to_half(dist(rng));
            DeviceBuffer x(host_x);
            DeviceBuffer w(static_cast<size_t>(rows) * s.cols);
            DeviceBuffer y(static_cast<size_t>(rows));
            const double seconds = time_matmul(x.get(), w.get(), y.get(), s, s.cols, iters);
            const double weight_bytes =
                static_cast<double>(rows) * s.cols * sizeof(uint16_t);
            std::printf("scan rows=%-7d mib=%8.1f seconds=%.6f gbs=%.1f\n", rows,
                        weight_bytes / (1024.0 * 1024.0), seconds,
                        weight_bytes / seconds / 1e9);
            std::fflush(stdout);
        }
        return 0;
    }

    // Raw HBM probe. The GEMM numbers above are only interpretable against the
    // memory system's actual ceiling: if this reports ~1 TB/s and the GEMV
    // reports 320 GB/s, the kernel is the problem, and if it reports 320 GB/s
    // then the matmul is already at the hardware limit and the fix is elsewhere.
    if (mem) {
        const size_t bytes = 512ull * 1024 * 1024;
        DeviceBuffer a(bytes / sizeof(uint16_t));
        DeviceBuffer b(bytes / sizeof(uint16_t));
        if (!pocket::memcpy_d2d(b.get(), a.get(), bytes) ||
            !pocket::device_synchronize()) {
            throw std::runtime_error("d2d warm-up failed");
        }
        if (!pocket::device_synchronize()) throw std::runtime_error("pre-sync failed");
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!pocket::memcpy_d2d(b.get(), a.get(), bytes)) {
                throw std::runtime_error("d2d failed");
            }
        }
        if (!pocket::device_synchronize()) throw std::runtime_error("post-sync failed");
        const auto stop = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(stop - start).count() / iters;
        // A copy moves the buffer twice through HBM, once out and once in.
        std::printf("device=%d iters=%d mem=1 mib=%.1f seconds=%.6f gbs=%.1f\n",
                    device, iters, bytes / (1024.0 * 1024.0), seconds,
                    2.0 * bytes / seconds / 1e9);
        return 0;
    }

    std::printf("device=%d iters=%d\n", device, iters);
    for (const Shape& s : shapes) {
        std::mt19937 rng(1234u + static_cast<unsigned>(s.rows));
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        std::vector<uint16_t> host_x(static_cast<size_t>(s.batch) * s.cols);
        for (uint16_t& v : host_x) v = float_to_half(dist(rng));
        std::vector<uint16_t> host_w(static_cast<size_t>(s.rows) * s.cols);
        for (uint16_t& v : host_w) v = float_to_half(dist(rng));
        DeviceBuffer x(host_x), w(host_w);
        DeviceBuffer y(static_cast<size_t>(s.batch) * s.rows);

        const double seconds = time_matmul(x.get(), w.get(), y.get(), s, s.cols, iters);
        // 2 flops per multiply-accumulate.
        const double flops =
            2.0 * static_cast<double>(s.batch) * s.cols * s.rows;
        // Every weight element is read once per call, so the decode case is
        // bandwidth-bound: this is the number that caps decode tokens/s.
        const double weight_bytes =
            static_cast<double>(s.rows) * s.cols * sizeof(uint16_t);
        std::printf("%-8s batch=%d cols=%d rows=%d seconds=%.6f tflops=%.3f gbs=%.1f\n",
                    s.name, s.batch, s.cols, s.rows, seconds, flops / seconds / 1e12,
                    weight_bytes / seconds / 1e9);
        std::fflush(stdout);
    }
    return 0;
}

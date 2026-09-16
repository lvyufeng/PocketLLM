// Where decode time goes, split into host and device.
//
// Decode on this backend is memory bound in principle and not in practice. A TP4
// shard re-reads 13.4 GB of resident weights per token; the HBM read probe streams
// that at 1152 GB/s, so the floor is 11.7 ms, and the model runs at 108 ms. The
// same weight through the decode matmul reads at 265 GB/s. Everything between
// those two numbers is what this file is for: the weight operand is the one that
// has to move, and the sweeps below are the ones that say why it does not.
//
// `void qwen_fp16_matmul_rows_f16_ascend` passes the weight as a transposed view
// of the [rows, columns] runtime layout. `qwen_fp16_gemv_f16_ascend` puts it on
// the other Cube operand. Both read the same bytes and neither moves the rate, so
// the operand *position* is ruled out; what is left is the operand's *layout*, and
// the B-dense block below stores the weight transposed so the fractal loader can
// read it in bursts.
//
// Per-op cost has two halves -- the kernel and the host work that precedes it --
// so the same op is timed twice:
//
//   enqueue  N launches back to back, one synchronize at the end. If the host is
//            the bottleneck this is the host cost per op; otherwise it is the
//            kernel cost.
//   sync     one launch and one synchronize per op. Always kernel + host.
//
// The two numbers together say which half to attack. They are reported separately
// rather than as a ratio because a host-bound op and a device-bound op can have
// the same `sync` number and need opposite fixes.
//
//   ./tests/bench_qwen_ascend_decode_ops [--device N] [--iters 200]

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The TP4 shard of Qwen3.8-27B, read off the checkpoint's config: hidden 5120,
// intermediate 17408 split four ways, and this rank's slice of each projection.
constexpr int kHidden = 5120;
constexpr int kMlp = 17408 / 4;
constexpr int kQHeads = 6;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 256;
constexpr int kLinearKeyHeads = 16 / 4;
constexpr int kLinearValueHeads = 48 / 4;
constexpr int kLinearHeadDim = 128;

// The engine's resident cache for the benchmark prompt. The attention kernels take
// it as a pitch, and the neutral entry rejects contexts that round past it.
constexpr int kMaxContext = 8192;

// Round-to-nearest-even FP16, written out rather than taken from a header so the
// benchmark has no dependency beyond the device runtime. `half_to_float` is exact
// for every input, including the subnormals this generator can produce.
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
    uint32_t half = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1fu;
    const uint32_t mantissa = h & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // Subnormal: renormalise into a normal FP32 exponent.
            int shift = 0;
            uint32_t value = mantissa;
            while ((value & 0x400u) == 0) {
                value <<= 1;
                ++shift;
            }
            const uint32_t normal = 127u - 15u - static_cast<uint32_t>(shift);
            bits = sign | (normal << 23) | ((value & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
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

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<uint16_t> out(count);
    for (uint16_t& value : out) value = float_to_half(dist(rng));
    return out;
}

struct Timing {
    double enqueue_seconds = 0.0;
    double sync_seconds = 0.0;
};

// The op is a lambda so the two loops below are written once. `enqueue` runs the
// body `iters` times with no synchronize between them.
template <typename Op>
Timing time_op(Op op, int iters) {
    Timing timing;
    if (!op()) throw std::runtime_error("warmup launch failed");
    if (!pocket::device_synchronize()) throw std::runtime_error("warmup sync failed");

    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!op()) throw std::runtime_error("launch failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        timing.enqueue_seconds =
            std::chrono::duration<double>(stop - start).count() / iters;
        if (!pocket::device_synchronize()) throw std::runtime_error("sync failed");
    }
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!op()) throw std::runtime_error("launch failed");
            if (!pocket::device_synchronize()) throw std::runtime_error("sync failed");
        }
        const auto stop = std::chrono::steady_clock::now();
        timing.sync_seconds =
            std::chrono::duration<double>(stop - start).count() / iters;
    }
    return timing;
}

void report(const char* name, const Timing& timing) {
    std::printf("%-28s enqueue=%8.1f us  sync=%8.1f us\n", name,
                timing.enqueue_seconds * 1.0e6, timing.sync_seconds * 1.0e6);
    std::fflush(stdout);
}

// Same figure as `report`, plus the weight bytes the op had to read and the rate
// that implies. `sync` is host plus device, so the device-only time is the part
// of it the host could not have been issuing already -- `sync - enqueue`. Both
// rates are printed: the device one is the operand-streaming rate, and the `sync`
// one is what the layer budget actually pays, because a layer of 448 ops pays the
// host half 448 times.
void report_gbs(const char* name, double weight_bytes, const Timing& timing) {
    const double device_seconds =
        std::max(0.0, timing.sync_seconds - timing.enqueue_seconds);
    const double device_gbs = device_seconds > 0.0
                                  ? weight_bytes / device_seconds / 1e9
                                  : 0.0;
    const double sync_gbs = timing.sync_seconds > 0.0
                                ? weight_bytes / timing.sync_seconds / 1e9
                                : 0.0;
    std::printf(
        "%-28s enqueue=%8.1f us  sync=%8.1f us  device=%8.1f us  "
        "gbs_device=%6.1f  gbs_sync=%6.1f\n",
        name, timing.enqueue_seconds * 1.0e6, timing.sync_seconds * 1.0e6,
        device_seconds * 1.0e6, device_gbs, sync_gbs);
    std::fflush(stdout);
}

// `report_gbs` with the arithmetic throughput alongside it, for the points whose
// shape is chosen to expose the Cube rather than the memory path.
void report_flops(const char* name, double flops, const Timing& timing) {
    const double device_seconds =
        std::max(0.0, timing.sync_seconds - timing.enqueue_seconds);
    const double tflops = device_seconds > 0.0 ? flops / device_seconds / 1e12 : 0.0;
    std::printf("%-28s enqueue=%8.1f us  sync=%8.1f us  device=%8.1f us  "
                "tflops=%7.2f\n",
                name, timing.enqueue_seconds * 1.0e6,
                timing.sync_seconds * 1.0e6, device_seconds * 1.0e6, tflops);
    std::fflush(stdout);
}

// Largest absolute difference between two device FP16 buffers, read back through
// the host. Used only by the parity check below, which runs once per shape
// rather than per timed iteration.
float max_abs_diff(const uint16_t* a, const uint16_t* b, size_t count) {
    std::vector<uint16_t> host_a(count);
    std::vector<uint16_t> host_b(count);
    if (!pocket::memcpy_d2h(host_a.data(), a, count * sizeof(uint16_t)) ||
        !pocket::memcpy_d2h(host_b.data(), b, count * sizeof(uint16_t))) {
        throw std::runtime_error("memcpy_d2h failed");
    }
    float worst = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(half_to_float(host_a[i]) - half_to_float(host_b[i]));
        if (diff > worst) worst = diff;
    }
    return worst;
}

// The operands here are values in [-0.5, 0.5) and a row dot product is 5120 terms
// of them, so an individual output is order-one. A reordering of the same
// accumulation differs in the last bits of FP16; a wrong weight layout differs by
// an order-one amount. The threshold separates those two by a wide margin.
const char* parity_label(const uint16_t* a, const uint16_t* b, size_t count) {
    return max_abs_diff(a, b, count) < 0.5f ? "ok" : "FAIL";
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
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

    std::mt19937 rng(20260914u);
    std::printf("iters=%d device=%d\n", iters, device);

    // MLP gate and up: [1, 5120] x [5120, 8704]. The two together are the largest
    // pair of weight reads in the layer.
    {
        DeviceBuffer x(random_halves(kHidden, rng));
        DeviceBuffer w(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(kMlp));
        report("matmul 1x5120x8704",
               time_op([&] {
                   return pocket::qwen_fp16_matmul_rows_f16(x.get(), w.get(), y.get(),
                                                            1, kMlp, kHidden, kHidden,
                                                            kMlp, kHidden);
               }, iters));
    }
    // MLP down: [1, 8704] x [8704, 5120].
    {
        DeviceBuffer x(random_halves(kMlp, rng));
        DeviceBuffer w(random_halves(static_cast<size_t>(kHidden) * kMlp, rng));
        DeviceBuffer y(static_cast<size_t>(kHidden));
        report("matmul 1x8704x5120",
               time_op([&] {
                   return pocket::qwen_fp16_matmul_rows_f16(x.get(), w.get(), y.get(),
                                                            1, kHidden, kMlp, kMlp,
                                                            kHidden, kMlp);
               }, iters));
    }
    // The MLP shard at its real width, swept over the query-row count.
    //
    // A decode GEMV reads its weight once and multiplies it by one row of
    // activations, so the device time is a pure operand stream: 44.6 MB at the
    // 1152 GB/s the HBM probe measures is 38.7 us, and anything above that is
    // overhead the kernel is paying for something other than memory. Sweeping M
    // separates the two candidates, because a Cube tile that is too small at M=1
    // fills up at M=16 and the operand read does not change at all.
    for (int m : {1, 4, 16, 64}) {
        DeviceBuffer x(random_halves(static_cast<size_t>(m) * kHidden, rng));
        DeviceBuffer w(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(m) * kMlp);
        char label[64];
        std::snprintf(label, sizeof(label), "matmul %dx5120x4352", m);
        report_gbs(label, static_cast<double>(kMlp) * kHidden * 2,
                   time_op([&] {
                       return pocket::qwen_fp16_matmul_rows_f16(
                           x.get(), w.get(), y.get(), m, kMlp, kHidden, kHidden,
                           kMlp, kHidden);
                   }, iters));
    }
    // The same sweep with the weight on the Cube's A operand instead of its B.
    //
    // Operand order is the whole difference: A is indexed by the uncontracted
    // axis, so a row-major weight is read as one sequential run, while B is
    // indexed by the contracted one and is described by a transposed view over
    // the same memory. Same arithmetic, same bytes, different access pattern --
    // so the pair of tables says whether the 265 GB/s ceiling belongs to the
    // memory path or to the operand the weight landed on.
    for (int m : {1, 4, 16}) {
        DeviceBuffer x(random_halves(static_cast<size_t>(m) * kHidden, rng));
        DeviceBuffer w(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(m) * kMlp);
        char label[64];
        std::snprintf(label, sizeof(label), "gemv %dx5120x4352", m);
        report_gbs(label, static_cast<double>(kMlp) * kHidden * 2,
                   time_op([&] {
                       return pocket::qwen_fp16_gemv_f16_ascend(
                           x.get(), w.get(), y.get(), m, kMlp, kHidden, kHidden,
                           kMlp, nullptr);
                   }, iters));
    }
    // The same product with the weight physically transposed to [columns, rows].
    //
    // Every block above passes the weight as a transposed *view* of a
    // [rows, columns] buffer -- shape [5120, 4352] with stride {1, 5120}. That is
    // legal and aclnn accepts it, but the B operand is read fractal by fractal and
    // a fractal's contiguous run is along N, which in that view is the axis with
    // stride 5120. Each 32-byte burst the loader wants is a sixteen-way gather.
    //
    // Storing the weight the other way round makes B dense in the layout ND2NZ is
    // shaped for. A weight is read-only, so the transpose is paid once at upload
    // and never again at decode; if this measures faster, that is a one-off cost.
    //
    // The two blocks also have to agree numerically, so the first iteration of
    // each is checked against the other before any timing is reported.
    for (int m : {1, 16}) {
        const size_t count = static_cast<size_t>(kMlp) * kHidden;
        const std::vector<uint16_t> w_host = random_halves(count, rng);
        // w_host is [rows, columns]; w_kn holds the same values as [columns, rows].
        std::vector<uint16_t> w_kn(count);
        for (int r = 0; r < kMlp; ++r) {
            for (int c = 0; c < kHidden; ++c) {
                w_kn[static_cast<size_t>(c) * kMlp + r] =
                    w_host[static_cast<size_t>(r) * kHidden + c];
            }
        }
        DeviceBuffer x(random_halves(static_cast<size_t>(m) * kHidden, rng));
        DeviceBuffer w(w_kn);
        DeviceBuffer w_row_major(w_host);
        DeviceBuffer y(static_cast<size_t>(m) * kMlp);
        DeviceBuffer y_ref(static_cast<size_t>(m) * kMlp);
        char label[64];
        std::snprintf(label, sizeof(label), "matmul %dx5120x4352 B-dense", m);
        report_gbs(label, static_cast<double>(count) * 2,
                   time_op([&] {
                       return pocket::qwen_fp16_matmul_weight_transposed_f16_ascend(
                           x.get(), w.get(), y.get(), m, kMlp, kHidden, kHidden,
                           kMlp, nullptr);
                   }, iters));
        if (!pocket::qwen_fp16_matmul_rows_f16(x.get(), w_row_major.get(),
                                               y_ref.get(), m, kMlp, kHidden,
                                               kHidden, kMlp, kHidden)) {
            std::printf("    parity check: reference launch failed\n");
        } else {
            std::printf("    parity vs the strided-B view: %s (max |diff| %.4f)\n",
                        parity_label(y.get(), y_ref.get(),
                                     static_cast<size_t>(m) * kMlp),
                        max_abs_diff(y.get(), y_ref.get(),
                                     static_cast<size_t>(m) * kMlp));
        }
    }
    // Where the Cube tops out once the shapes stop being degenerate.
    //
    // M=1 is the one shape a decode step runs, and it is the one shape a Cube
    // cannot fill: the unit is 16 rows tall, so a batch of one leaves fifteen
    // sixteenths of it idle. That is harmless here -- a GEMV does two flops per
    // byte read, which at 1152 GB/s asks for 2.3 TFLOP/s and nothing more -- but
    // it does not explain a *slower* result than the same weight read at M=16.
    // These four points say whether the ceiling is the Cube itself or the shape:
    // if a square GEMM also lands near ten TFLOP/s then the unit is the limit and
    // the GEMV has to be moved off it, and if the square GEMM is a hundred times
    // faster then M=1 is a tiling problem that a kernel of our own can fix.
    for (int m : {128, 512, 1024}) {
        DeviceBuffer x(random_halves(static_cast<size_t>(m) * kHidden, rng));
        DeviceBuffer w(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(m) * kMlp);
        char label[64];
        std::snprintf(label, sizeof(label), "matmul %dx5120x4352", m);
        report_flops(label, static_cast<double>(m) * kMlp * kHidden * 2,
                     time_op([&] {
                         return pocket::qwen_fp16_matmul_rows_f16(
                             x.get(), w.get(), y.get(), m, kMlp, kHidden, kHidden,
                             kMlp, kHidden);
                     }, iters));
    }
    {
        const int n = 4096;
        DeviceBuffer x(random_halves(static_cast<size_t>(n) * n, rng));
        DeviceBuffer w(random_halves(static_cast<size_t>(n) * n, rng));
        DeviceBuffer y(static_cast<size_t>(n) * n);
        report_flops("matmul 4096x4096x4096",
                     2.0 * n * n * n,
                     time_op([&] {
                         return pocket::qwen_fp16_matmul_rows_f16(
                             x.get(), w.get(), y.get(), n, n, n, n, n, n);
                     }, std::max(3, iters / 10)));
    }
    // SwiGLU: two matmuls plus a SiLU and a multiply, all issued from here.
    {
        DeviceBuffer x(random_halves(kHidden, rng));
        DeviceBuffer gate(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer up(random_halves(static_cast<size_t>(kMlp) * kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(kMlp));
        report("swiglu 1x5120x8704",
               time_op([&] {
                   return pocket::qwen_fp16_swiglu_matmul_rows_f16(
                       x.get(), gate.get(), up.get(), y.get(), 1, kMlp, kHidden,
                       kHidden, kMlp, kHidden);
               }, iters));
    }
    // RMSNorm is small but runs twice per layer, and its host cost does not scale
    // with the row count.
    {
        DeviceBuffer x(random_halves(kHidden, rng));
        DeviceBuffer gamma(random_halves(kHidden, rng));
        DeviceBuffer y(static_cast<size_t>(kHidden));
        report("rmsnorm 1x5120",
               time_op([&] {
                   return pocket::qwen_rmsnorm_fp16_gamma_rows_f16(
                       x.get(), gamma.get(), y.get(), 1, kHidden, 1e-6f);
               }, iters));
    }
    // The linear-attention projections and the gated-delta step, which the profile
    // shows is the second largest single item in the linear layers.
    {
        const int heads = kLinearValueHeads;
        const int key_heads = kLinearKeyHeads;
        const int dim = kLinearHeadDim;
        const int value_dim = heads * dim;
        const int key_dim = key_heads * dim;
        DeviceBuffer state(static_cast<size_t>(heads) * dim * dim);
        DeviceBuffer q(random_halves(static_cast<size_t>(key_dim), rng));
        DeviceBuffer k(random_halves(static_cast<size_t>(key_dim), rng));
        DeviceBuffer v(random_halves(static_cast<size_t>(value_dim), rng));
        DeviceBuffer g(random_halves(static_cast<size_t>(heads), rng));
        DeviceBuffer beta(random_halves(static_cast<size_t>(heads), rng));
        DeviceBuffer out(static_cast<size_t>(value_dim));
        report("gated_delta_step",
               time_op([&] {
                   return pocket::qwen_gated_delta_step_f16(
                       reinterpret_cast<float*>(const_cast<uint16_t*>(state.get())),
                       q.get(), k.get(), v.get(), g.get(), beta.get(), out.get(),
                       heads, key_heads, dim, dim, 1.0f);
               }, iters));
    }
    // Decode attention at the context the engine actually runs it at. Two
    // candidate kernels are timed side by side because they parallelize on
    // different axes: the neutral entry uses the Cube unit but, at one query row,
    // has only kv_heads items to spread over cores, while FlashDecoding splits the
    // context and gives every core a piece. Which one wins is a question about
    // this part's core count, not about arithmetic.
    for (int context : {2048, 4097, 8192}) {
        DeviceBuffer q(random_halves(static_cast<size_t>(kQHeads) * kHeadDim, rng));
        DeviceBuffer k(random_halves(static_cast<size_t>(context) * kKvHeads * kHeadDim, rng));
        DeviceBuffer v(random_halves(static_cast<size_t>(context) * kKvHeads * kHeadDim, rng));
        DeviceBuffer out(static_cast<size_t>(kQHeads) * kHeadDim);
        float* scores = nullptr;
        if (!pocket::device_malloc_into(
                scores, static_cast<size_t>(kQHeads) * context * sizeof(float))) {
            throw std::runtime_error("scratch malloc failed");
        }
        char label[64];
        std::snprintf(label, sizeof(label), "decode attn %d cube", context);
        report(label,
               time_op([&] {
                   return pocket::qwen_gqa_decode_attention_f16(
                       q.get(), k.get(), v.get(), out.get(), scores, kQHeads,
                       kKvHeads, kHeadDim, context, kMaxContext);
               }, iters));

        // Same split geometry the engine's long-context branch uses: one
        // partition per 256 tokens, capped at the core count.
        const int partitions = std::min(30, (context + 255) / 256);
        float* partials = nullptr;
        if (!pocket::device_malloc_into(
                partials, static_cast<size_t>(kQHeads) * partitions *
                              (kHeadDim + 2) * sizeof(float))) {
            throw std::runtime_error("partials malloc failed");
        }
        std::snprintf(label, sizeof(label), "decode attn %d flashdec(%d)", context,
                      partitions);
        report(label,
               time_op([&] {
                   return pocket::qwen_gqa_decode_attention_flashdec_f16(
                       q.get(), k.get(), v.get(), out.get(), partials, kQHeads,
                       kKvHeads, kHeadDim, context, kMaxContext, partitions);
               }, iters));

        // The Cube decode kernel with the context split across cores. `partitions
        // = 0` asks for the widest split the context allows, which is what the
        // engine would use in production; the third timing pins the count so the
        // scaling against core count is visible rather than inferred.
        std::snprintf(label, sizeof(label), "decode attn %d cube-split(auto)", context);
        report(label,
               time_op([&] {
                   return pocket::qwen_gqa_decode_attention_cube_split_f16(
                       q.get(), k.get(), v.get(), out.get(), kQHeads, kKvHeads,
                       kHeadDim, context, kMaxContext, 0);
               }, iters));
        if (partitions > 1) {
            std::snprintf(label, sizeof(label), "decode attn %d cube-split(%d)",
                          context, partitions);
            report(label,
                   time_op([&] {
                       return pocket::qwen_gqa_decode_attention_cube_split_f16(
                           q.get(), k.get(), v.get(), out.get(), kQHeads, kKvHeads,
                           kHeadDim, context, kMaxContext, partitions);
                   }, iters));
        }
        pocket::device_free(partials);
        pocket::device_free(scores);
    }
    // The causal convolution of the linear-attention layers, which the device
    // profile puts at 33% of a batched decode step -- the largest single op, ahead
    // of every matmul.
    //
    // What makes that suspicious is not the share but the shape of it: the profiled
    // instances sit in a plateau, min 289 us and max 328 us, across calls that do
    // very different amounts of work. Four taps over 384 channels is a handful of
    // microseconds of arithmetic, so almost all of that is a per-call constant.
    // Two candidates are timed apart here. The length sweep holds the operands
    // fixed and grows the sequence, so the slope is the per-token cost and the
    // intercept is everything a call pays regardless; the kernel sweep holds the
    // channel count fixed and grows the taps, which is the axis the kernel's own
    // weight gather scales on, so a slope here that matches the gather's size says
    // the gather is the constant.
    {
        const int channels = 384;
        const int tail_rows = 7;
        DeviceBuffer tail(random_halves(static_cast<size_t>(tail_rows) * channels,
                                        rng));
        for (int seq : {1, 4, 16, 64}) {
            DeviceBuffer x(random_halves(static_cast<size_t>(seq) * channels, rng));
            DeviceBuffer y(static_cast<size_t>(seq) * channels);
            DeviceBuffer w(random_halves(static_cast<size_t>(channels) * 4, rng));
            char label[64];
            std::snprintf(label, sizeof(label), "conv %2dx%d k4", seq, channels);
            report(label, time_op([&] {
                       return pocket::qwen_causal_depthwise_conv_silu_f16(
                           x.get(), w.get(), tail.get(), y.get(), seq, channels, 4,
                           true, nullptr);
                   }, iters));
        }
        for (int kernel : {2, 4, 8}) {
            DeviceBuffer x(random_halves(channels, rng));
            DeviceBuffer y(channels);
            DeviceBuffer w(random_halves(static_cast<size_t>(channels) * kernel,
                                         rng));
            char label[64];
            std::snprintf(label, sizeof(label), "conv 1x%d k%d", channels, kernel);
            report(label, time_op([&] {
                       return pocket::qwen_causal_depthwise_conv_silu_f16(
                           x.get(), w.get(), tail.get(), y.get(), 1, channels,
                           kernel, true, nullptr);
                   }, iters));
        }
    }

    // The greedy top-1 over this rank's vocabulary shard, sampled at the end of
    // every step. The device profile puts it third in the decode window at 2.3 ms
    // per call, which is out of proportion to the work: it reads one 248 KiB row
    // of logits per token and the lm_head GEMM that wrote those is itself only
    // ~1 ms. Two axes separate the cost. Growing `rows` at a fixed width says how
    // much of the call is per row and how much is per launch; growing `count` says
    // whether the per-row part scales with the bytes, which would be bandwidth, or
    // with the lanes, which is the scan itself -- the kernel folds each tile on the
    // vector unit and walks only the folded lanes, so the two move together now.
    //
    // The engine's own widths are 62,080 (a TP4 slice of 248,320) and rows 1 or the
    // batch size; the other widths are there to give the fit something to stand on.
    {
        const int channels = 62080;
        const int max_rows = 16;
        // One row feeds the width sweep and a full batch feeds the row sweep, so
        // the buffer is sized for the larger of the two.
        const size_t max_count = static_cast<size_t>(max_rows) * channels;
        std::vector<float> logits_host(max_count);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& value : logits_host) value = dist(rng);

        float* d_logits = nullptr;
        int* d_tokens = nullptr;
        float* d_values = nullptr;
        if (!pocket::device_malloc_into(d_logits, max_count * sizeof(float)) ||
            !pocket::device_malloc_into(d_tokens, max_rows * sizeof(int)) ||
            !pocket::device_malloc_into(d_values, max_rows * sizeof(float))) {
            throw std::runtime_error("device_malloc failed");
        }
        if (!pocket::memcpy_h2d(d_logits, logits_host.data(),
                                max_count * sizeof(float))) {
            throw std::runtime_error("memcpy_h2d failed");
        }
        for (int rows : {1, 2, 4, 8, 16}) {
            char label[64];
            std::snprintf(label, sizeof(label), "argmax %2dx%d", rows, channels);
            report(label, time_op([&] {
                       return pocket::qwen_argmax_fp32_rows(
                           d_logits, d_tokens, d_values, rows, channels, 0);
                   }, iters));
        }
        for (int width : {2048, 8192, 32768, 62080}) {
            char label[64];
            std::snprintf(label, sizeof(label), "argmax  1x%d", width);
            report(label, time_op([&] {
                       return pocket::qwen_argmax_fp32_rows(
                           d_logits, d_tokens, d_values, 1, width, 0);
                   }, iters));
        }
        pocket::device_free(d_logits);
        pocket::device_free(d_tokens);
        pocket::device_free(d_values);
    }

    return 0;
}

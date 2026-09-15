// Cube-path validation: C[m, n] = A[m, k] * B[n, k]^T computed with AscendC::Gemm
// on one AI core, checked against a double-precision host reference.
//
// The reason this test exists rather than a read of the CANN headers: the L1
// operand layout Gemm's v1 LoadData path consumes is only described indirectly,
// and a layout mismatch does not fault -- it returns a plausible-looking matrix.
// The only way to know the staging convention is to compute a known product.
//
//   ./tests/test_qwen_ascend_cube_gemm [--device N]

#include "device_runtime.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Not declared in qwen_ops.hpp: this probe is a Cube bring-up tool, not part of
// the neutral operator set, so it is declared here next to its only caller.
namespace pocket {
bool qwen_cube_gemm_probe(const uint16_t* d_a, const uint16_t* d_b, uint16_t* d_c,
                          int m, int n, int k, uint32_t mode, uint32_t readout_len,
                          void* stream);
bool qwen_cube_transpose_probe(const uint16_t* d_a, const uint16_t* d_b, uint16_t* d_c,
                               int m, int n, int k, uint32_t transpose, void* stream);
bool qwen_transpose_f16_ascend(const uint16_t* d_src, uint16_t* d_dst, int rows, int cols,
                               void* stream);
}  // namespace pocket

namespace {

// Same rounding as the bench harness, kept local so the test has no link-time
// dependency on the benchmark translation unit.
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

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3ffu;
            const uint32_t exp32 = static_cast<uint32_t>(127 - 15 - shift);
            bits = sign | (exp32 << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        const uint32_t exp32 = exponent - 15 + 127;
        bits = sign | (exp32 << 23) | (mantissa << 13);
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
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    uint16_t* get() const { return ptr_; }

private:
    uint16_t* ptr_ = nullptr;
    size_t count_ = 0;
};

// Largest relative error over the tile, plus the worst absolute error, which is
// what distinguishes "one element is wrong" from "the whole layout is wrong".
struct Error {
    double max_rel = 0.0;
    double max_abs = 0.0;
    double ref_scale = 0.0;
    size_t bad = 0;
    size_t total = 0;
};

Error check(int m, int n, int k, unsigned seed, bool report) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<float> host_a(static_cast<size_t>(m) * k);
    std::vector<float> host_b(static_cast<size_t>(n) * k);
    for (float& v : host_a) v = dist(rng);
    for (float& v : host_b) v = dist(rng);

    std::vector<uint16_t> a_h(host_a.size()), b_h(host_b.size());
    for (size_t i = 0; i < host_a.size(); ++i) a_h[i] = float_to_half(host_a[i]);
    for (size_t i = 0; i < host_b.size(); ++i) b_h[i] = float_to_half(host_b[i]);

    DeviceBuffer a(a_h.size()), b(b_h.size());
    DeviceBuffer c(static_cast<size_t>(m) * n);
    if (!pocket::memcpy_h2d(a.get(), a_h.data(), a_h.size() * sizeof(uint16_t)) ||
        !pocket::memcpy_h2d(b.get(), b_h.data(), b_h.size() * sizeof(uint16_t)) ||
        !pocket::qwen_cube_gemm_probe(a.get(), b.get(), c.get(), m, n, k, 0, 0, nullptr) ||
        !pocket::device_synchronize()) {
        std::printf("[FAIL] launch failed for m=%d n=%d k=%d\n", m, n, k);
        return Error{};
    }
    std::vector<uint16_t> c_h(static_cast<size_t>(m) * n);
    if (!pocket::memcpy_d2h(c_h.data(), c.get(), c_h.size() * sizeof(uint16_t))) {
        std::printf("[FAIL] readback failed for m=%d n=%d k=%d\n", m, n, k);
        return Error{};
    }

    // Reference in double, using the half-rounded inputs, so the only difference
    // left is the accumulation order inside the Cube.
    Error err;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double acc = 0.0;
            for (int d = 0; d < k; ++d) {
                acc += static_cast<double>(half_to_float(a_h[static_cast<size_t>(row) * k + d])) *
                       static_cast<double>(half_to_float(b_h[static_cast<size_t>(col) * k + d]));
            }
            const double got = half_to_float(c_h[static_cast<size_t>(row) * n + col]);
            err.ref_scale = std::max(err.ref_scale, std::fabs(acc));
            const double abs_err = std::fabs(got - acc);
            err.max_abs = std::max(err.max_abs, abs_err);
            if (std::fabs(acc) > 1e-3) {
                err.max_rel = std::max(err.max_rel, abs_err / std::fabs(acc));
            }
            ++err.total;
            // A layout error is never uniformly small; it is a small number of
            // elements landing in the wrong place. Counting the outliers and
            // printing where they are separates the two far better than the
            // worst-case magnitude does.
            if (abs_err > 1e-2) {
                ++err.bad;
                if (report && err.bad <= 6) {
                    std::printf("    BAD c[%d,%d] got=%.6f ref=%.6f\n", row, col, got, acc);
                }
            }
            if (report && (row < 2 && col < 2)) {
                std::printf("    c[%d,%d] got=%.6f ref=%.6f\n", row, col, got, acc);
            }
        }
    }
    return err;
}

}  // namespace

// Does LoadData2DParams::ifTranspose transpose the B operand, and if so what L1
// layout does it expect? The host cannot answer that from the headers -- the
// dav_c100 impl just forwards the flag to load_cbuf_to_cb -- so the question is
// settled by computing a known product with `d_b` handed over in its natural
// [k, n] form, which is what the V cache actually looks like.
//
// transpose == 0 stages [n, k] and should reproduce A * B^T, i.e. the same
// product AscendC::Gemm already produces; it exists to prove the hand-rolled
// L0A/L0B/Mmad sequence below is sound before the flag is believed.
int transpose_probe(int device, bool report) {
    (void)device;
    struct Case {
        const char* name;
        int m, n, k;
    };
    const Case cases[] = {
        {"sq-16", 16, 16, 16},
        {"sq-32", 32, 32, 32},
        {"sq-64", 64, 64, 64},
        {"m64-k256", 64, 32, 256},
        {"n64-k256", 32, 64, 256},
    };
    int total_fail = 0;
    for (const Case& c : cases) {
        for (uint32_t mode = 0; mode < 3; ++mode) {
            std::mt19937 rng(4242u + static_cast<unsigned>(c.m * 7 + c.n * 3 + c.k));
            std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
            std::vector<float> host_a(static_cast<size_t>(c.m) * c.k);
            // Always generated in [k, n] order: that is the operand's natural
            // shape and the only layout the V cache ever has.
            std::vector<float> host_b(static_cast<size_t>(c.k) * c.n);
            for (float& v : host_a) v = dist(rng);
            for (float& v : host_b) v = dist(rng);
            std::vector<uint16_t> a_h(host_a.size()), b_h(host_b.size());
            for (size_t i = 0; i < host_a.size(); ++i) a_h[i] = float_to_half(host_a[i]);
            for (size_t i = 0; i < host_b.size(); ++i) b_h[i] = float_to_half(host_b[i]);

            DeviceBuffer da(a_h.size());
            // mode 0 and 1 feed the natural [k, n] operand and differ only in the
            // transpose flag. mode 2 feeds the pre-transposed [n, k] operand with
            // no flag, so it must reproduce the plain Gemm product.
            std::vector<uint16_t> b_used = b_h;
            if (mode == 2) {
                for (int t = 0; t < c.k; ++t) {
                    for (int j = 0; j < c.n; ++j) {
                        b_used[static_cast<size_t>(j) * c.k + t] = b_h[static_cast<size_t>(t) * c.n + j];
                    }
                }
            }
            DeviceBuffer db(b_used.size());
            DeviceBuffer dc(static_cast<size_t>(c.m) * c.n);
            if (!pocket::memcpy_h2d(da.get(), a_h.data(), a_h.size() * sizeof(uint16_t)) ||
                !pocket::memcpy_h2d(db.get(), b_used.data(), b_used.size() * sizeof(uint16_t)) ||
                !pocket::qwen_cube_transpose_probe(da.get(), db.get(), dc.get(), c.m, c.n, c.k,
                                                   mode == 1 ? 1u : 0u, nullptr) ||
                !pocket::device_synchronize()) {
                std::printf("%-9s m=%d n=%d k=%d mode=%u [FAIL] launch\n", c.name, c.m, c.n, c.k, mode);
                ++total_fail;
                continue;
            }
            std::vector<uint16_t> got(static_cast<size_t>(c.m) * c.n);
            if (!pocket::memcpy_d2h(got.data(), dc.get(), got.size() * sizeof(uint16_t))) {
                std::printf("%-9s m=%d n=%d k=%d mode=%u [FAIL] readback\n", c.name, c.m, c.n, c.k, mode);
                ++total_fail;
                continue;
            }
            double max_abs = 0.0, scale = 0.0;
            size_t bad = 0;
            for (int i = 0; i < c.m; ++i) {
                for (int j = 0; j < c.n; ++j) {
                    double acc = 0.0;
                    for (int t = 0; t < c.k; ++t) {
                        acc += static_cast<double>(half_to_float(a_h[static_cast<size_t>(i) * c.k + t])) *
                               static_cast<double>(half_to_float(b_h[static_cast<size_t>(t) * c.n + j]));
                    }
                    const double v = half_to_float(got[static_cast<size_t>(i) * c.n + j]);
                    scale = std::max(scale, std::fabs(acc));
                    const double e = std::fabs(v - acc);
                    max_abs = std::max(max_abs, e);
                    if (e > 1e-2) {
                        ++bad;
                        if (report && bad <= 3) {
                            std::printf("      BAD c[%d,%d] got=%.6f ref=%.6f\n", i, j, v, acc);
                        }
                    }
                }
            }
            const bool ok = max_abs < 1e-2 * std::max(1.0, scale);
            if (!ok) ++total_fail;
            std::printf("%-9s m=%4d n=%4d k=%4d mode=%u max_abs=%.3e ref=%.3f bad=%zu/%d %s\n",
                        c.name, c.m, c.n, c.k, mode, max_abs, scale, bad, c.m * c.n,
                        ok ? "ok" : "MISMATCH");
            std::fflush(stdout);
        }
    }
    return total_fail == 0 ? 0 : 1;
}

// Does the vector transpose unit actually produce a transpose on this part?
//
// The kernel uses AscendC::Transpose<half>, which the dav_c100 impl lowers to
// vtranspose -- a 16x16 uint16 transpose with no fractal precondition. That is a
// read of the header, and header reads have already been wrong twice on this SoC
// (ifTranspose, and the UB -> L1 ND2NZ overload), so it gets executed here.
//
// A transpose is a pure permutation: every output element is a copy of exactly
// one input element and no arithmetic runs at all. So the check is exact bit
// equality, not a tolerance, and a single misplaced 16x16 block shows up as a
// run of mismatches rather than a plausible-looking near miss.
int transpose2d_probe(int device, bool report) {
    (void)device;
    struct Case {
        const char* name;
        int rows, cols;
    };
    const Case cases[] = {
        {"single", 16, 16},
        {"wide", 32, 64},
        {"tall", 64, 32},
        {"sq-256", 256, 256},
        {"kv-4096", 4096, 256},
        {"3x6", 48, 96},
    };
    int failures = 0;
    for (const Case& c : cases) {
        const size_t count = static_cast<size_t>(c.rows) * c.cols;
        std::vector<uint16_t> src(count);
        for (size_t i = 0; i < count; ++i) {
            src[i] = float_to_half(static_cast<float>((i * 2654435761u) % 4096) - 2048.0f);
        }
        DeviceBuffer ds(count), dd(count);
        std::vector<uint16_t> got(count, 0);
        if (!pocket::memcpy_h2d(ds.get(), src.data(), count * sizeof(uint16_t)) ||
            !pocket::qwen_transpose_f16_ascend(ds.get(), dd.get(), c.rows, c.cols, nullptr) ||
            !pocket::device_synchronize() ||
            !pocket::memcpy_d2h(got.data(), dd.get(), count * sizeof(uint16_t))) {
            std::printf("%-8s rows=%5d cols=%5d [FAIL] launch or readback\n", c.name, c.rows, c.cols);
            ++failures;
            continue;
        }
        size_t bad = 0;
        size_t first_bad = count;
        for (int i = 0; i < c.rows; ++i) {
            for (int j = 0; j < c.cols; ++j) {
                const uint16_t want = src[static_cast<size_t>(i) * c.cols + j];
                const uint16_t have = got[static_cast<size_t>(j) * c.rows + i];
                if (want != have) {
                    if (first_bad == count) first_bad = static_cast<size_t>(j) * c.rows + i;
                    ++bad;
                    if (report && bad <= 3) {
                        std::printf("      first mismatch dst[%d,%d] want=%u have=%u\n", j, i,
                                    static_cast<unsigned>(want), static_cast<unsigned>(have));
                    }
                }
            }
        }
        if (bad != 0) ++failures;
        if (bad == 0) {
            std::printf("%-8s rows=%5d cols=%5d bad=0/%zu first=none ok\n", c.name, c.rows, c.cols,
                        count);
        } else {
            std::printf("%-8s rows=%5d cols=%5d bad=%zu/%zu first=%zu MISMATCH\n", c.name, c.rows,
                        c.cols, bad, count, first_bad);
        }
        std::fflush(stdout);
    }
    return failures == 0 ? 0 : 1;
}

// Reads back a product whose structure makes the buffer layout self-evident.
// With A = B = identity the product is the identity too, so wherever a one lands
// in the returned buffer names the (row, column) the Cube put there. Two modes:
// the logical store should put them on the main diagonal, while the raw store
// exposes the fractal arrangement the L0C readout left behind.
int layout_probe(int device, bool raw) {
    (void)device;
    const int m = 32, n = 32, k = 32;
    std::vector<uint16_t> a(static_cast<size_t>(m) * k, float_to_half(0.0f));
    std::vector<uint16_t> b(static_cast<size_t>(n) * k, float_to_half(0.0f));
    for (int i = 0; i < m; ++i) a[static_cast<size_t>(i) * k + i] = float_to_half(1.0f);
    for (int i = 0; i < n; ++i) b[static_cast<size_t>(i) * k + i] = float_to_half(1.0f);

    // The raw image covers the tile rounded up to GemmTiling::blockSize = 16,
    // so 32 x 32 stays 32 x 32 -- but read a wider window than that anyway so a
    // runaway copy shows up instead of being clipped.
    const size_t count = raw ? static_cast<size_t>(64) * 64
                             : static_cast<size_t>(m) * n;
    DeviceBuffer da(a.size()), db(b.size()), dc(count);
    if (!pocket::memcpy_h2d(da.get(), a.data(), a.size() * sizeof(uint16_t)) ||
        !pocket::memcpy_h2d(db.get(), b.data(), b.size() * sizeof(uint16_t)) ||
        !pocket::qwen_cube_gemm_probe(da.get(), db.get(), dc.get(), m, n, k, raw ? 1u : 0u, 0u,
                                      nullptr) ||
        !pocket::device_synchronize()) {
        std::printf("[FAIL] layout probe launch failed\n");
        return 1;
    }
    std::vector<uint16_t> c(count);
    if (!pocket::memcpy_d2h(c.data(), dc.get(), c.size() * sizeof(uint16_t))) {
        std::printf("[FAIL] layout probe readback failed\n");
        return 1;
    }
    std::printf("%s layout probe: A = B = identity, %dx%d, ones at 0-based index\n",
                raw ? "raw L0C" : "logical", m, n);
    int ones = 0;
    for (size_t idx = 0; idx < c.size(); ++idx) {
        if (std::fabs(half_to_float(c[idx])) > 0.5f) {
            // For the logical store the diagonal position is the claim under
            // test; for the raw image the index itself is the datum, so print
            // it alongside the diagonal slot it corresponds to.
            if (ones < 96) {
                std::printf("  [%6zu]%s\n", idx,
                            raw ? "" : (idx / n == idx % n ? " diag" : " OFF-DIAGONAL"));
            }
            ++ones;
        }
    }
    std::printf("  %d ones out of %d expected\n", ones, m);
    return 0;
}

// Sweeps the `blockLen` Gemm's c100 readout uses for its L0C -> UB copy.
//
// That value is hardcoded inside Gemm as `roundM * roundN * sizeof(T) / 1024`,
// and the resulting buffer is not what its size implies -- so rather than trust
// any single reading of the unit, stage the same 32x32 identity and try every
// plausible unit until the UB image is the layout the arithmetic says it should
// be. The expected image is the zN fractal form of the identity: a one wherever
// row == column, at fractal offsets (col / 16) * roundM * 16 + row, i.e. 17 * i
// for the first 16 and 768 + 17 * (i - 16) for the second.
int readout_probe(int device) {
    (void)device;
    const int m = 32, n = 32, k = 32;
    const size_t count = 65536;  // one whole UB, so a runaway copy is visible too
    std::vector<uint16_t> a(static_cast<size_t>(m) * k, float_to_half(0.0f));
    std::vector<uint16_t> b(static_cast<size_t>(n) * k, float_to_half(0.0f));
    for (int i = 0; i < m; ++i) a[static_cast<size_t>(i) * k + i] = float_to_half(1.0f);
    for (int i = 0; i < n; ++i) b[static_cast<size_t>(i) * k + i] = float_to_half(1.0f);

    std::vector<size_t> expected;
    // zN fractal image of a 32x32 identity: row == column lands at
    // (col / 16) * roundM * 16 + row * 16 + col % 16 with roundM = 32, i.e. 17 * i
    // inside the first column fractal and 512 + 17 * (i - 16) inside the second.
    for (int i = 0; i < 16; ++i) expected.push_back(static_cast<size_t>(i) * 17);
    for (int i = 16; i < 32; ++i) expected.push_back(512 + static_cast<size_t>(i - 16) * 17);

    const uint32_t candidates[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    std::printf("readout sweep: 32x32 identity, %zu ones expected in the zN image\n",
                expected.size());
    for (uint32_t len : candidates) {
        DeviceBuffer da(a.size()), db(b.size()), dc(count);
        if (!pocket::memcpy_h2d(da.get(), a.data(), a.size() * sizeof(uint16_t)) ||
            !pocket::memcpy_h2d(db.get(), b.data(), b.size() * sizeof(uint16_t)) ||
            !pocket::qwen_cube_gemm_probe(da.get(), db.get(), dc.get(), m, n, k, 2u, len,
                                          nullptr) ||
            !pocket::device_synchronize()) {
            std::printf("  blockLen=%-6u [FAIL] launch\n", len);
            continue;
        }
        std::vector<uint16_t> c(count);
        if (!pocket::memcpy_d2h(c.data(), dc.get(), c.size() * sizeof(uint16_t))) {
            std::printf("  blockLen=%-6u [FAIL] readback\n", len);
            continue;
        }
        size_t hits = 0;
        for (size_t e : expected) {
            if (std::fabs(half_to_float(c[e])) > 0.5f) ++hits;
        }
        size_t ones = 0;
        size_t first_bad = c.size();
        for (size_t i = 0; i < 2048; ++i) {
            if (std::fabs(half_to_float(c[i])) > 0.5f) {
                ++ones;
                if (first_bad == c.size()) first_bad = i;
            }
        }
        std::printf("  blockLen=%-6u hits=%2zu/%zu ones=%3zu first=%zu%s\n", len, hits,
                    expected.size(), ones, first_bad, hits == expected.size() ? "  <-- MATCH" : "");
        std::fflush(stdout);
    }
    return 0;
}

int main(int argc, char** argv) {
    int device = 0;
    bool layout = false;
    bool raw = false;
    bool readout = false;
    bool transpose = false;
    bool transpose2d = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--layout") {
            layout = true;
        } else if (arg == "--raw") {
            raw = true;
            layout = true;
        } else if (arg == "--readout") {
            readout = true;
        } else if (arg == "--transpose") {
            transpose = true;
        } else if (arg == "--transpose2d") {
            transpose2d = true;
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
    if (readout) return readout_probe(device);
    if (layout) return layout_probe(device, raw);
    if (transpose) return transpose_probe(device, true);
    if (transpose2d) return transpose2d_probe(device, true);

    // 256x256x256 is the smallest tile whose every operand is a whole number of
    // 16-element fractals in both directions, so a transposed or fractal-strided
    // read shows up as a large error rather than a subtle one.
    struct Case {
        const char* name;
        int m, n, k;
    };
    const Case cases[] = {
        {"square-256", 256, 256, 256},
        {"skinny-m", 64, 256, 256},
        {"skinny-n", 256, 64, 256},
        {"wide-k", 128, 128, 512},
    };

    int failures = 0;
    for (const Case& c : cases) {
        const Error err = check(c.m, c.n, c.k, 900u + static_cast<unsigned>(c.m + c.n + c.k),
                                failures > 0);
        // fp16 accumulation over k terms puts the floor at roughly 1e-3 of the
        // result scale, so the pass condition is "no outliers", not "tiny
        // worst-case relative error" -- near-zero reference values make the
        // latter meaningless.
        const bool ok = err.max_abs < 1e-2 * std::max(1.0, err.ref_scale);
        if (!ok) ++failures;
        std::printf("%-10s m=%d n=%d k=%d max_abs=%.3e ref_scale=%.3f bad=%zu/%zu %s\n",
                    c.name, c.m, c.n, c.k, err.max_abs, err.ref_scale, err.bad, err.total,
                    ok ? "ok" : "MISMATCH");
        std::fflush(stdout);
    }
    return failures == 0 ? 0 : 1;
}

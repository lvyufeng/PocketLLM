// Does a UB vector instruction honour a source operand that is offset by less
// than a 32-byte block?
//
// `AscendC::Add(dst, src0, src1, count)` hands its operands to the vector unit as
// raw `__ubuf__` pointers and does no alignment arithmetic of its own, so the C++
// layer is silent on the question. Every kernel in this tree works around it --
// `fold_sum` stops its halving fold at kAlignFloat and finishes the last eight
// lanes on the scalar unit, and the attention row fold does the same -- which is
// evidence that the answer is "no", but not the answer.
//
// The answer decides the shape of the incoherence rotation the ternary checkpoint
// needs: a Walsh-Hadamard transform over 1024-point runs is ten butterfly passes
// whose pairing stride runs 1, 2, 4, ... 512 *elements*, and the first three are
// 4, 8 and 16 bytes in FP32. If the unit honours them, the kernel is ten passes of
// paired Add/Sub over one UB tile. If it rounds them down, a straight-line port
// computes `2 * src[i]` at three of the ten levels -- a different transform,
// silently -- and the kernel has to stage the run in a layout where every level's
// offset is a whole number of blocks, paying an element-level transpose that no
// vector or copy instruction on this part can express.
//
// Measured on the 910B (2026-09-26): they are **not** honoured, and not by
// rounding either. Every offset from 1 to 7 elements raises an aicore exception --
// `error code = 0x10`, `errorStr: Illegal instruction, which is usually caused by
// unaligned UUB addresses` -- and the launch is lost; the eight-element offset, one
// whole block, returns the sum at every element. So a source operand below a block
// is not addressable at all, and the three sub-block levels of the butterfly are
// unreachable from the vector unit by any single offset. They have to be computed
// on the scalar unit, which is the only place in UB that addresses an element.
//
// The values are small exact integers, so the reference sum is exact in FP32 and
// the comparison is equality rather than a tolerance. This is a question about
// which bytes were read, and the check says so.
//
//   ./tests/test_qwen_ascend_vec_offset [--device N]

#include "device_runtime.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// Not declared in qwen_ops.hpp: this probe is an addressing oracle, not part of
// the neutral operator set, so it is declared here next to its only caller.
namespace pocket {
bool qwen_vec_offset_probe(const float* d_source, float* d_dest, int count,
                           int shift, void* stream);
}  // namespace pocket

namespace {

// Three chunks: long enough that the kernel's loop runs more than once, so a
// per-iteration hand-off bug shows up as a wrong second chunk rather than as a
// clean single pass.
constexpr int kCount = 12288;

int failures = 0;
int checks = 0;

// The input is a fixed pseudo-random sequence of small integers: exact in FP32,
// no common factor, and no run of equal neighbours that could hide a
// one-lane shift.
std::vector<float> make_input(int count) {
    std::vector<float> values(static_cast<size_t>(count));
    uint32_t state = 0x9e3779b9u;
    for (int i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        values[static_cast<size_t>(i)] =
            static_cast<float>(static_cast<int>(state >> 24) - 128);
    }
    // The shifted operand of the last lanes reads past the payload, so the
    // buffer the launcher sees is longer than the one compared.
    values.resize(static_cast<size_t>(count) + 16, 0.0f);
    return values;
}

// 0 = the sum the kernel is asked for, 1 = `2 * src[i]` (what a source offset
// rounded down to the enclosing block produces), 2 = anything else.
int classify(float got, float low, float high) {
    if (got == low + high) return 0;
    if (got == 2.0f * low) return 1;
    return 2;
}

int run(int device) {
    if (!pocket::device_runtime_available()) {
        std::printf("[SKIP] no device runtime available\n");
        return 0;
    }
    if (!pocket::device_set(device)) {
        std::printf("[SKIP] device_set failed for device %d\n", device);
        return 0;
    }

    const std::vector<float> input = make_input(kCount);
    float* d_source = nullptr;
    float* d_dest = nullptr;
    if (!pocket::device_malloc_into(d_source, input.size() * sizeof(float)) ||
        !pocket::device_malloc_into(d_dest, static_cast<size_t>(kCount) * sizeof(float))) {
        std::printf("[FAIL] device_malloc failed\n");
        return 1;
    }
    struct Free {
        float* a;
        float* b;
        ~Free() {
            pocket::device_free(a);
            pocket::device_free(b);
        }
    } free_guard{d_source, d_dest};
    if (!pocket::memcpy_h2d(d_source, input.data(), input.size() * sizeof(float)) ||
        !pocket::device_memset(d_dest, 0, static_cast<size_t>(kCount) * sizeof(float))) {
        std::printf("[FAIL] upload failed\n");
        return 1;
    }

    std::printf("vec offset probe: count=%d\n", kCount);

    // Aligned first, and on its own: it is the control the sub-block arms are
    // read against, and if the harness itself is broken it is better to learn
    // that from the arm that is supposed to work.
    //
    // Four outcomes per arm, and a fault is one of them rather than an error to
    // be scored separately: an arm that raises a device error has answered the
    // question, and the answer is that the offset is not addressable. It has to
    // be counted as an arm result or a run where every sub-block arm faults
    // reports the same verdict as a run where none of them were tried.
    enum Arm { kExact, kDoubled, kOther, kFaulted };

    const int shifts[] = {8, 1, 2, 3, 4, 5, 6, 7};
    Arm control = kOther;
    int sub_exact = 0;
    int sub_doubled = 0;
    int sub_other = 0;
    int sub_faulted = 0;
    for (const int shift : shifts) {
        ++checks;
        const bool is_control = shift == 8;
        Arm arm = kOther;
        if (!pocket::qwen_vec_offset_probe(d_source, d_dest, kCount, shift, nullptr)) {
            std::printf("  shift %d: launch refused\n", shift);
            arm = kFaulted;
        } else {
            // The synchronise is unconditional and comes first: a launch that
            // faults reports through the error slot, not through this call's
            // return value.
            pocket::device_synchronize();
            const std::string sync_error = pocket::device_last_error();
            if (!sync_error.empty()) {
                std::printf("  shift %d: device error: %s\n", shift, sync_error.c_str());
                arm = kFaulted;
            } else {
                std::vector<float> got(static_cast<size_t>(kCount));
                if (!pocket::memcpy_d2h(got.data(), d_dest,
                                        static_cast<size_t>(kCount) * sizeof(float))) {
                    std::printf("  shift %d: download failed\n", shift);
                    arm = kFaulted;
                } else {
                    int exact = 0;
                    int doubled = 0;
                    int other = 0;
                    for (int i = 0; i < kCount; ++i) {
                        const int verdict = classify(got[static_cast<size_t>(i)],
                                                     input[static_cast<size_t>(i)],
                                                     input[static_cast<size_t>(i) + shift]);
                        if (verdict == 0) {
                            ++exact;
                        } else if (verdict == 1) {
                            ++doubled;
                        } else {
                            ++other;
                        }
                    }
                    std::printf("  shift %d: exact %d  doubled %d  other %d  %s\n", shift,
                                exact, doubled, other,
                                exact == kCount ? "sum"
                                                : (doubled == kCount ? "2*src" : "mixed"));
                    // Every element has to land on one of the two models. A
                    // result that is neither is not an answer to this question:
                    // it means the unit did something per-lane and per-block,
                    // which no reading of the result supports.
                    if (exact == kCount) {
                        arm = kExact;
                    } else if (doubled == kCount) {
                        arm = kDoubled;
                    } else {
                        ++failures;
                        std::printf(
                            "  FAIL shift %d is neither the sum nor 2*src at every "
                            "element\n",
                            shift);
                        arm = kOther;
                    }
                }
            }
        }
        if (is_control) {
            control = arm;
        } else {
            switch (arm) {
                case kExact: ++sub_exact; break;
                case kDoubled: ++sub_doubled; break;
                case kOther: ++sub_other; break;
                case kFaulted: ++sub_faulted; break;
            }
        }
    }

    const int sub_total = sub_exact + sub_doubled + sub_other + sub_faulted;
    std::printf("sub-block arms: %d sum, %d 2*src, %d other, %d faulted (of %d)\n",
                sub_exact, sub_doubled, sub_other, sub_faulted, sub_total);

    if (control != kExact) {
        std::printf("verdict: the aligned control did not return the sum, so nothing here "
                    "is readable\n");
        return 1;
    }
    if (sub_faulted > 0) {
        std::printf("verdict: sub-block source offsets are not addressable -- the core "
                    "takes an aicore exception and the launch is lost. The butterfly "
                    "cannot run in place; levels whose pairing stride is under a block "
                    "have to be reached another way.\n");
    } else if (sub_exact == sub_total) {
        std::printf("verdict: sub-block source offsets are honoured -- the butterfly can "
                    "run in place\n");
    } else if (sub_doubled == sub_total) {
        std::printf("verdict: sub-block source offsets are rounded down to the enclosing "
                    "block -- the run needs a staged layout\n");
    } else {
        std::printf("verdict: sub-block source offsets produce something else again\n");
    }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    return run(device);
}

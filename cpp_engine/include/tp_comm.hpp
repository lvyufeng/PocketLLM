#pragma once

#include <cstdint>

namespace dsv4 {

struct TpTopResult {
    int token = 0;
    float logit = 0.0f;
};

bool nccl_available();

#ifdef DSV4_HAVE_NCCL
void run_nccl_float_sum_smoke(int world, int rank, int device, const char* id_path, float value);
TpTopResult nccl_global_top1(int world, int rank, int device, const char* id_path, int local_token, float local_logit);
void nccl_global_top1_rows(int world, int rank, int device, const char* id_path,
                           const int* d_local_tokens,
                           const float* d_local_logits, int rows,
                           int* global_tokens, float* global_logits);
void nccl_all_reduce_sum_float_inplace(int world, int rank, int device, const char* id_path, float* d_values, int count);
void nccl_all_reduce_sum_f16_inplace(int world, int rank, int device, const char* id_path, uint16_t* d_values, int count);
void nccl_all_reduce_sum_bf16_inplace(int world, int rank, int device, const char* id_path, uint16_t* d_values, int count);
// Broadcast a small int32 buffer from rank `root` to all ranks. Synchronous.
void nccl_broadcast_int32(int world, int rank, int device, const char* id_path, int32_t* buf, int count, int root);
// Gather equal-sized float32 chunks from every rank to `root`. `h_local` is read on
// each rank; on rank == root, `h_root_out` (size world * local_count) receives the
// concatenated result. May pass nullptr for h_root_out on non-root ranks.
void nccl_gather_floats_to_root(int world, int rank, int device, const char* id_path, const float* h_local, int local_count, float* h_root_out, int root);
#endif

}  // namespace dsv4

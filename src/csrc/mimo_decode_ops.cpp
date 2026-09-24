// MiMo-V2.6's decode step, reached without the Python layer.
//
// A decode step is 48 layers of small operations -- one row against a short span -- and the host
// is the whole of its cost: `tests/probe_mimo_v2_decode_ops.py` counts about 8500 device-API calls
// and 2000 Python-level dispatches for one token, and `tests/probe_mimo_v2_host_phases.py` puts
// 23.5 ms of the token in the router alone. The arithmetic is a few hundred microseconds of card
// time; the rest is the cost of *asking* for it.
//
// So this is the same arithmetic, in the same order, with the same ATen calls, reached from C++.
// That is the whole design and it is why the answer is bit-identical rather than close: the
// kernels are the ones PyTorch would have run, so an op removed here cannot have changed the
// result, only what it cost. The one op that *is* removed -- the group mask, which the released
// configuration cannot see -- is removed only where that argument holds exactly; the comment on
// the function states it. `tests/test_models_mimo_v2_device_router.py` holds the two paths to
// `torch.equal` over a randomised batch and both groupings, and the reference is
// `layers.gate_and_route`, which stays where it is for the host path and for a chunk.

#include <torch/extension.h>

#include <c10/util/Optional.h>

#include <limits>
#include <vector>

// The reference's router, on the card: `(topk_idx, topk_weight)`.
//
// `layers.gate_and_route` is the definition and this is a transcription of it, line for line, so
// the two can be diffed by eye. `noaux_tc` scores every expert with a sigmoid, adds a learned
// correction bias to decide *which* experts a token takes, and then weights the chosen ones by the
// uncorrected score -- renormalised, because the reference discards the top-k ordering and divides
// by the actual sum.
//
// One branch is not in the reference: at `n_group == 1` and `topk_group == 1` -- which is the
// released configuration -- the group block is skipped, because with one group the mask it builds
// is all ones and `masked_fill` with an all-false mask leaves every element *unchanged*, not merely
// equal. That is twelve of the router's dispatches and the whole of its middle, and it is skipped
// only where the skip is provable rather than measured.
std::vector<torch::Tensor> mimo_noaux_tc_route(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    const torch::Tensor& correction_bias,
    int64_t top_k,
    int64_t n_group,
    int64_t topk_group,
    bool norm_topk_prob,
    double routed_scaling_factor) {
  // `[batch, seq, hidden]` and a flat `[rows, hidden]` are the same arithmetic on the same rows;
  // the reference's first act is this reshape.
  torch::Tensor flat = hidden_states.reshape({-1, hidden_states.size(-1)});
  const int64_t rows = flat.size(0);

  // Both operands are upcast explicitly; the released config stores the router weight in bf16 in
  // the checkpoint, so this upcast is part of the arithmetic and not an optimisation.
  torch::Tensor logits = at::linear(flat.to(at::kFloat), weight.to(at::kFloat));
  torch::Tensor scores = logits.sigmoid();

  torch::Tensor scores_for_choice = scores.view({rows, -1}) + correction_bias.unsqueeze(0);

  torch::Tensor masked;
  if (n_group == 1 && topk_group == 1) {
    // One group holds every expert and the group that is kept is that group, so `score_mask` is
    // all ones and `masked` is `scores_for_choice` to the bit.
    masked = scores_for_choice;
  } else {
    // Group selection: each group is scored by its best two experts.
    torch::Tensor group_scores =
        std::get<0>(scores_for_choice.view({rows, n_group, -1}).topk(2, -1)).sum(-1);
    torch::Tensor group_idx = std::get<1>(group_scores.topk(topk_group, -1, true, false));
    torch::Tensor group_mask = at::zeros_like(group_scores);
    group_mask.scatter_(1, group_idx, 1);
    torch::Tensor score_mask = group_mask.unsqueeze(-1)
                                   .expand({rows, n_group, scores_for_choice.size(-1) / n_group})
                                   .reshape({rows, -1});
    masked = scores_for_choice.masked_fill(
        score_mask.logical_not(), -std::numeric_limits<float>::infinity());
  }
  torch::Tensor topk_idx = std::get<1>(masked.topk(top_k, -1, true, false));
  torch::Tensor topk_weight = scores.gather(1, topk_idx);

  if (top_k > 1 && norm_topk_prob) {
    torch::Tensor denominator = topk_weight.sum(-1, /*keepdim=*/true) + 1e-20;
    topk_weight = topk_weight / denominator;
  }
  topk_weight = topk_weight * routed_scaling_factor;
  return {topk_idx, topk_weight};
}

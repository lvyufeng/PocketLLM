"""GLM-DSA / GLM-5.2 GGUF source-name mapping tables."""

from __future__ import annotations

from src.components.moe.spec import TensorMapping

GLOBAL_TENSORS = {
    "token_embd.weight": ("embed_tokens.weight", "embedding", "transpose"),
    "output.weight": ("lm_head.weight", "lm_head", "transpose"),
    "output_norm.weight": ("final_norm.weight", "final_norm", "direct"),
}

DENSE_LAYER_TENSORS = {
    "attn_k_b.weight": ("self_attn.k_b_proj.weight", "attn_k_b", "direct"),
    "attn_kv_a_mqa.weight": ("self_attn.kv_a_mqa_proj.weight", "attn_kv_a", "transpose"),
    "attn_kv_a_norm.weight": ("self_attn.kv_a_norm.weight", "attn_norm", "direct"),
    "attn_norm.weight": ("input_layernorm.weight", "attn_norm", "direct"),
    "attn_output.weight": ("self_attn.o_proj.weight", "attn_o", "transpose"),
    "attn_q_a.weight": ("self_attn.q_a_proj.weight", "attn_q_a", "transpose"),
    "attn_q_a_norm.weight": ("self_attn.q_a_norm.weight", "attn_norm", "direct"),
    "attn_q_b.weight": ("self_attn.q_b_proj.weight", "attn_q_b", "transpose"),
    "attn_v_b.weight": ("self_attn.v_b_proj.weight", "attn_v_b", "direct"),
    "ffn_down.weight": ("mlp.down_proj.weight", "dense_w2", "transpose"),
    "ffn_gate.weight": ("mlp.gate_proj.weight", "dense_w1", "transpose"),
    "ffn_norm.weight": ("post_attention_layernorm.weight", "ffn_norm", "direct"),
    "ffn_up.weight": ("mlp.up_proj.weight", "dense_w3", "transpose"),
    "indexer.attn_k.weight": ("self_attn.indexer.k_proj.weight", "indexer_attn_k", "transpose"),
    "indexer.attn_q_b.weight": ("self_attn.indexer.q_b_proj.weight", "indexer_attn_q_b", "transpose"),
    "indexer.k_norm.bias": ("self_attn.indexer.k_norm.bias", "indexer_norm", "direct"),
    "indexer.k_norm.weight": ("self_attn.indexer.k_norm.weight", "indexer_norm", "direct"),
    "indexer.proj.weight": ("self_attn.indexer.proj.weight", "indexer_proj", "transpose"),
}

MOE_LAYER_TENSORS = {
    "attn_k_b.weight": DENSE_LAYER_TENSORS["attn_k_b.weight"],
    "attn_kv_a_mqa.weight": DENSE_LAYER_TENSORS["attn_kv_a_mqa.weight"],
    "attn_kv_a_norm.weight": DENSE_LAYER_TENSORS["attn_kv_a_norm.weight"],
    "attn_norm.weight": DENSE_LAYER_TENSORS["attn_norm.weight"],
    "attn_output.weight": DENSE_LAYER_TENSORS["attn_output.weight"],
    "attn_q_a.weight": DENSE_LAYER_TENSORS["attn_q_a.weight"],
    "attn_q_a_norm.weight": DENSE_LAYER_TENSORS["attn_q_a_norm.weight"],
    "attn_q_b.weight": DENSE_LAYER_TENSORS["attn_q_b.weight"],
    "attn_v_b.weight": DENSE_LAYER_TENSORS["attn_v_b.weight"],
    "exp_probs_b.bias": ("mlp.router.bias", "gate_bias", "direct"),
    "ffn_down_exps.weight": ("mlp.experts.routed.w2", "routed_w2", "routed_expert_transpose"),
    "ffn_down_shexp.weight": ("mlp.shared_experts.w2", "shared_w2", "transpose"),
    "ffn_gate_exps.weight": ("mlp.experts.routed.w1", "routed_w1", "routed_expert_transpose"),
    "ffn_gate_inp.weight": ("mlp.router.weight", "gate", "transpose"),
    "ffn_gate_shexp.weight": ("mlp.shared_experts.w1", "shared_w1", "transpose"),
    "ffn_norm.weight": DENSE_LAYER_TENSORS["ffn_norm.weight"],
    "ffn_up_exps.weight": ("mlp.experts.routed.w3", "routed_w3", "routed_expert_transpose"),
    "ffn_up_shexp.weight": ("mlp.shared_experts.w3", "shared_w3", "transpose"),
    "indexer.attn_k.weight": DENSE_LAYER_TENSORS["indexer.attn_k.weight"],
    "indexer.attn_q_b.weight": DENSE_LAYER_TENSORS["indexer.attn_q_b.weight"],
    "indexer.k_norm.bias": DENSE_LAYER_TENSORS["indexer.k_norm.bias"],
    "indexer.k_norm.weight": DENSE_LAYER_TENSORS["indexer.k_norm.weight"],
    "indexer.proj.weight": DENSE_LAYER_TENSORS["indexer.proj.weight"],
}

NEXTN_TENSORS = {
    "nextn.eh_proj.weight": ("nextn.eh_proj.weight", "nextn", "transpose"),
    "nextn.enorm.weight": ("nextn.enorm.weight", "nextn_norm", "direct"),
    "nextn.hnorm.weight": ("nextn.hnorm.weight", "nextn_norm", "direct"),
    "nextn.shared_head_norm.weight": ("nextn.shared_head_norm.weight", "nextn_norm", "direct"),
}


def layer_tensor_table(layer: int, leading_dense_layers: int) -> dict[str, tuple[str, str, str]]:
    return DENSE_LAYER_TENSORS if int(layer) < int(leading_dense_layers) else MOE_LAYER_TENSORS


def classify_tensor_name(name: str, *, leading_dense_layers: int = 3) -> str:
    if name in GLOBAL_TENSORS:
        return GLOBAL_TENSORS[name][1]
    if name in NEXTN_TENSORS:
        return NEXTN_TENSORS[name][1]
    if not name.startswith("blk."):
        return "other"
    parts = name.split(".", 2)
    if len(parts) != 3:
        return "other"
    try:
        layer = int(parts[1])
    except ValueError:
        return "other"
    suffix = parts[2]
    if suffix in NEXTN_TENSORS:
        return NEXTN_TENSORS[suffix][1]
    return layer_tensor_table(layer, leading_dense_layers).get(suffix, ("", "other", ""))[1]


def build_tensor_mappings(n_layers: int, *, leading_dense_layers: int = 3, nextn_layers: int = 0) -> list[TensorMapping]:
    mappings: list[TensorMapping] = []
    for source_name, (logical, role, transform) in GLOBAL_TENSORS.items():
        mappings.append(TensorMapping(source_name, logical, role, transform))
    for layer in range(int(n_layers)):
        for suffix, (logical_suffix, role, transform) in layer_tensor_table(layer, leading_dense_layers).items():
            source = f"blk.{layer}.{suffix}"
            logical = f"layers.{layer}.{logical_suffix}"
            mappings.append(TensorMapping(source, logical, role, transform, layer=layer))
    if int(nextn_layers) > 0 and int(n_layers) > 0:
        nextn_layer = int(n_layers) - 1
        for suffix, (logical_suffix, role, transform) in NEXTN_TENSORS.items():
            source = f"blk.{nextn_layer}.{suffix}"
            logical = f"layers.{nextn_layer}.{logical_suffix}"
            mappings.append(TensorMapping(source, logical, role, transform, layer=nextn_layer))
    return mappings

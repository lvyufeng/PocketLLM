from __future__ import annotations

import time
from collections import Counter
from typing import TYPE_CHECKING

import torch

from src.components.moe.capability import capability_status_for_role
from src.components.moe.placement import HardwareProfile, heterogeneous_expert_decision, lowbit_device_resident_decision
from src.components.moe.spec import (
    CapabilityItem,
    CapabilityReport,
    MoEArchitectureParams,
    PlacementDecision,
    SpecValidation,
    TensorMapping,
    bytes_by,
    counts_by,
    metadata_float,
    metadata_int,
)
from src.loader.gguf.bundle import GGUFBundle
from src.loader.mappings.glm_dsa import build_tensor_mappings, classify_tensor_name

if TYPE_CHECKING:
    from src.runtime.generation import GGUFTokenRuntime


class GLMDSASpec:
    architecture = "glm-dsa"
    display_name = "GLM-DSA / GLM-5.2"

    def parse_params(self, bundle: GGUFBundle) -> MoEArchitectureParams:
        md = bundle.metadata
        return MoEArchitectureParams(
            architecture=self.architecture,
            n_layers=metadata_int(md, "glm-dsa.block_count", 0),
            hidden_size=metadata_int(md, "glm-dsa.embedding_length", 0),
            vocab_size=metadata_int(md, "tokenizer.ggml.tokens", metadata_int(md, "glm-dsa.vocab_size", 0)),
            context_length=metadata_int(md, "glm-dsa.context_length", 0),
            n_heads=metadata_int(md, "glm-dsa.attention.head_count", 0),
            n_kv_heads=metadata_int(md, "glm-dsa.attention.head_count_kv", 0),
            head_dim=metadata_int(md, "glm-dsa.attention.key_length", 0),
            rope_dim=metadata_int(md, "glm-dsa.rope.dimension_count", 0) or None,
            rope_base=metadata_float(md, "glm-dsa.rope.freq_base", 0.0) or None,
            n_routed_experts=metadata_int(md, "glm-dsa.expert_count", 0),
            top_k=metadata_int(md, "glm-dsa.expert_used_count", 0),
            expert_intermediate_size=metadata_int(md, "glm-dsa.expert_feed_forward_length", 0),
            n_shared_experts=metadata_int(md, "glm-dsa.expert_shared_count", 0),
            gate_function=f"gguf_enum:{metadata_int(md, 'glm-dsa.expert_gating_func', 0)}",
            attention_kind="glm_dsa_mla_indexed",
            routed_expert_layout="packed_3d_after_dense_prefix",
            norm_eps=metadata_float(md, "glm-dsa.attention.layer_norm_rms_epsilon", 0.0) or None,
        )

    def leading_dense_layers(self, bundle: GGUFBundle) -> int:
        return metadata_int(bundle.metadata, "glm-dsa.leading_dense_block_count", 0)

    def nextn_layers(self, bundle: GGUFBundle) -> int:
        return metadata_int(bundle.metadata, "glm-dsa.nextn_predict_layers", 0)

    def classify_tensor(self, name: str) -> str:
        # The real classification for a bundle depends on leading_dense_block_count.
        # GLM-5.2 uses 3 dense layers; pass that default here for registry summaries.
        return classify_tensor_name(name)

    def build_tensor_mappings(self, bundle: GGUFBundle) -> list[TensorMapping]:
        return build_tensor_mappings(
            self.parse_params(bundle).n_layers,
            leading_dense_layers=self.leading_dense_layers(bundle),
            nextn_layers=self.nextn_layers(bundle),
        )

    def _expected_shape_ok(
        self,
        source_name: str,
        dims: tuple[int, ...],
        params: MoEArchitectureParams,
        leading_dense_layers: int,
        metadata: dict[str, object],
    ) -> bool:
        h = params.hidden_size
        v = params.vocab_size
        e = params.n_routed_experts
        i = params.expert_intermediate_size
        rope_dim = metadata_int(metadata, "glm-dsa.rope.dimension_count", int(params.rope_dim or 0))
        q_lora = metadata_int(metadata, "glm-dsa.attention.q_lora_rank", max(1, h // 3))
        kv_lora = metadata_int(metadata, "glm-dsa.attention.kv_lora_rank", max(1, params.head_dim - rope_dim))
        key_mla = metadata_int(metadata, "glm-dsa.attention.key_length_mla", max(1, params.head_dim - rope_dim))
        value_len = metadata_int(metadata, "glm-dsa.attention.value_length", key_mla)
        value_mla = metadata_int(metadata, "glm-dsa.attention.value_length_mla", key_mla)
        ff = metadata_int(metadata, "glm-dsa.feed_forward_length", h * 2)
        indexer_heads = metadata_int(metadata, "glm-dsa.attention.indexer.head_count", max(1, params.n_heads))
        indexer_key = metadata_int(metadata, "glm-dsa.attention.indexer.key_length", max(1, key_mla))
        k_nope = max(1, key_mla - rope_dim)

        if source_name in {"token_embd.weight", "output.weight"}:
            return dims == (h, v)
        if source_name == "output_norm.weight":
            return dims == (h,)
        parts = source_name.split(".", 2)
        if len(parts) != 3:
            return True
        try:
            layer = int(parts[1])
        except ValueError:
            return True
        suffix = parts[2]
        dense = layer < leading_dense_layers
        expected_common = {
            "attn_k_b.weight": (k_nope, kv_lora, params.n_heads),
            "attn_kv_a_mqa.weight": (h, kv_lora + rope_dim),
            "attn_kv_a_norm.weight": (kv_lora,),
            "attn_norm.weight": (h,),
            "attn_output.weight": (params.n_heads * value_mla, h),
            "attn_q_a.weight": (h, q_lora),
            "attn_q_a_norm.weight": (q_lora,),
            "attn_q_b.weight": (q_lora, params.n_heads * key_mla),
            "attn_v_b.weight": (value_len, value_mla, params.n_heads),
            "ffn_norm.weight": (h,),
            "indexer.attn_k.weight": (h, indexer_key),
            "indexer.attn_q_b.weight": (q_lora, indexer_heads * indexer_key),
            "indexer.k_norm.bias": (indexer_key,),
            "indexer.k_norm.weight": (indexer_key,),
            "indexer.proj.weight": (h, indexer_heads),
        }
        expected_dense = {
            "ffn_down.weight": (ff, h),
            "ffn_gate.weight": (h, ff),
            "ffn_up.weight": (h, ff),
        }
        expected_moe = {
            "exp_probs_b.bias": (e,),
            "ffn_down_exps.weight": (i, h, e),
            "ffn_down_shexp.weight": (i, h),
            "ffn_gate_exps.weight": (h, i, e),
            "ffn_gate_inp.weight": (h, e),
            "ffn_gate_shexp.weight": (h, i),
            "ffn_up_exps.weight": (h, i, e),
            "ffn_up_shexp.weight": (h, i),
        }
        expected_nextn = {
            "nextn.eh_proj.weight": (ff, h),
            "nextn.enorm.weight": (h,),
            "nextn.hnorm.weight": (h,),
            "nextn.shared_head_norm.weight": (h,),
        }
        expected = expected_common.get(suffix)
        if expected is None:
            expected = expected_nextn.get(suffix)
        if expected is None:
            expected = (expected_dense if dense else expected_moe).get(suffix)
        return expected is None or dims == expected

    def validate_bundle(self, bundle: GGUFBundle) -> SpecValidation:
        params = self.parse_params(bundle)
        leading_dense_layers = self.leading_dense_layers(bundle)
        errors: list[str] = []
        warnings: list[str] = []
        names = bundle.tensors_by_name
        mapped_sources: set[str] = set()
        role_counts: Counter[str] = Counter()

        if bundle.metadata.get("general.architecture") != self.architecture:
            errors.append(f"general.architecture expected {self.architecture!r}, got {bundle.metadata.get('general.architecture')!r}")
        for key in (
            "glm-dsa.block_count",
            "glm-dsa.embedding_length",
            "glm-dsa.context_length",
            "glm-dsa.attention.head_count",
            "glm-dsa.attention.key_length",
            "glm-dsa.attention.key_length_mla",
            "glm-dsa.attention.value_length",
            "glm-dsa.attention.value_length_mla",
            "glm-dsa.attention.q_lora_rank",
            "glm-dsa.attention.kv_lora_rank",
            "glm-dsa.attention.indexer.head_count",
            "glm-dsa.attention.indexer.key_length",
            "glm-dsa.feed_forward_length",
            "glm-dsa.expert_count",
            "glm-dsa.expert_used_count",
            "glm-dsa.expert_feed_forward_length",
            "glm-dsa.leading_dense_block_count",
        ):
            if key not in bundle.metadata:
                errors.append(f"missing metadata {key}")
        if params.n_layers <= 0:
            errors.append("glm-dsa.block_count must be positive")
        if leading_dense_layers < 0 or leading_dense_layers > params.n_layers:
            errors.append(f"invalid leading dense layer count {leading_dense_layers}")

        for mapping in self.build_tensor_mappings(bundle):
            tensor = names.get(mapping.source_name)
            if tensor is None:
                errors.append(f"missing tensor {mapping.source_name}")
                continue
            mapped_sources.add(mapping.source_name)
            role_counts[mapping.role] += 1
            if not self._expected_shape_ok(mapping.source_name, tuple(tensor.dimensions), params, leading_dense_layers, bundle.metadata):
                errors.append(f"{mapping.source_name} unexpected shape {tuple(tensor.dimensions)}")

        unmapped = sorted(set(names) - mapped_sources)
        if unmapped:
            warnings.append(f"{len(unmapped)} unmapped tensors, e.g. {unmapped[:10]}")
        return SpecValidation(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            mapped_sources=len(mapped_sources),
            unmapped_sources=unmapped,
            role_counts=dict(role_counts),
        )

    def capability_report(self, bundle: GGUFBundle, *, gpu_count: int = 4, gpu_memory_gib: float = 22.0) -> CapabilityReport:
        params = self.parse_params(bundle)
        leading_dense_layers = self.leading_dense_layers(bundle)
        role_by_name = {t.name: classify_tensor_name(t.name, leading_dense_layers=leading_dense_layers) for t in bundle.tensors}
        tensor_role_counts = counts_by(bundle.tensors, lambda t: role_by_name[t.name])
        tensor_type_counts = counts_by(bundle.tensors, lambda t: t.type_name)
        bytes_by_type = bytes_by(bundle.tensors, lambda t: t.type_name)
        bytes_by_role = bytes_by(bundle.tensors, lambda t: role_by_name[t.name])

        caps: list[CapabilityItem] = []
        seen: set[tuple[str, str]] = set()
        for tensor in bundle.tensors:
            role = role_by_name[tensor.name]
            key = (role, tensor.type_name)
            if key in seen:
                continue
            seen.add(key)
            status, reason = capability_status_for_role(role, tensor.type_name, architecture=self.architecture)
            caps.append(CapabilityItem(f"{role}:{tensor.type_name}", status, reason))
        caps.append(CapabilityItem("generation", "candidate", "GLM-DSA dense-prefix reference GGUF generation is implemented; MoE layers remain deferred"))

        tensor_bytes = sum(int(t.nbytes or 0) for t in bundle.tensors)
        routed_bytes = sum(int(t.nbytes or 0) for t in bundle.tensors if role_by_name[t.name] in {"routed_w1", "routed_w2", "routed_w3"})
        hardware = HardwareProfile(gpu_count=gpu_count, gpu_memory_gib=gpu_memory_gib)
        placements: list[PlacementDecision] = [
            lowbit_device_resident_decision(tensor_bytes, hardware),
            heterogeneous_expert_decision(routed_bytes),
        ]
        return CapabilityReport(
            architecture=self.architecture,
            params=params,
            tensor_type_counts=tensor_type_counts,
            tensor_role_counts=tensor_role_counts,
            bytes_by_type=bytes_by_type,
            bytes_by_role=bytes_by_role,
            capabilities=caps,
            placements=placements,
        )

    def build_token_runtime(
        self,
        bundle: GGUFBundle,
        *,
        world: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
        n_layers: int | None,
        gpu_memory_gib: float,
    ) -> "GGUFTokenRuntime":
        """Build the first GLM-DSA GGUF token runtime.

        The current implementation is a correctness bring-up path: it dequantizes
        tensors into reference PyTorch modules and only supports the dense-prefix
        layers.  If ``n_layers`` is omitted, default to ``leading_dense_layers`` so
        the generic generation CLI can smoke-test GLM attention/dense FFN without
        accidentally entering the not-yet-implemented 76 MoE layers.
        """
        from src.runtime.generation import GGUFTokenRuntime
        from src.models.glm_dsa.gguf_model import load_glm_dsa_gguf_model

        _ = (world, rank, gpu_memory_gib)
        leading_dense = self.leading_dense_layers(bundle)
        effective_layers = int(leading_dense if n_layers is None else n_layers)
        if effective_layers > leading_dense:
            raise NotImplementedError(
                f"GLM-DSA reference generation currently supports dense-prefix only: "
                f"n_layers={effective_layers}, leading_dense_layers={leading_dense}"
            )

        t_load = time.perf_counter()
        model, _info = load_glm_dsa_gguf_model(
            bundle,
            device=device,
            dtype=dtype,
            n_layers=effective_layers,
            allow_moe_layers=False,
        )
        load_seconds = time.perf_counter() - t_load
        eos = bundle.metadata.get("tokenizer.ggml.eos_token_id")
        eos_id = int(eos) if isinstance(eos, int) else None
        return GGUFTokenRuntime(
            model=model,
            expert_start=0,
            expert_count=0,
            eos_token_id=eos_id,
            load_seconds=float(load_seconds),
        )

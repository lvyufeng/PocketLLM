from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from src.loader.gguf.bundle import GGUFBundle, GGUFTensorRef, read_gguf_bundle
from src.loader.gguf.tensor_reader import GGUFTensorDataReader
from src.models.glm_dsa.architecture import (
    GLMDSAArgs,
    GLMDSAAttention,
    GLMDSABlock,
    GLMDSADenseMLP,
    GLMDSAMoE,
    GLMDSAMoEPlaceholder,
    GLMDSARawBlockMoE,
    GLMDSATransformer,
    ReferenceEmbedding,
    ReferenceLinear,
)


class GLMDSAGGUFModelLoader:
    """Assemble a GLM-DSA GGUF runtime from a GGUF bundle.

    Routed MoE experts run through the raw-block CUDA grouped kernel
    (``GLMDSARawBlockMoE``) when the routed dtypes and block layout are eligible;
    otherwise the loader falls back to the dequantized fp32 reference MoE
    (``GLMDSAMoE``).  Dense/attention projections use raw quantized GGUF linears.
    """

    def __init__(
        self,
        bundle_or_path: GGUFBundle | str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        n_layers: int | None = None,
        allow_moe_layers: bool = False,
        expert_start: int = 0,
        expert_count: int | None = None,
        use_raw_block_moe: bool = True,
        world: int = 1,
        rank: int = 0,
    ):
        self.bundle = read_gguf_bundle(bundle_or_path) if not isinstance(bundle_or_path, GGUFBundle) else bundle_or_path
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"GLM-DSA reference runtime currently requires CUDA device, got {resolved}")
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved
        self.dtype = dtype
        self.args = GLMDSAArgs.from_bundle(self.bundle, n_layers=n_layers)
        self.allow_moe_layers = bool(allow_moe_layers)
        self.use_raw_block_moe = bool(use_raw_block_moe)
        self.world = max(1, int(world))
        self.rank = max(0, int(rank))
        self.expert_start = int(expert_start)
        available = max(0, int(self.args.n_routed_experts) - self.expert_start)
        self.expert_count = available if expert_count is None else int(expert_count)
        if self.expert_start < 0 or self.expert_count < 0 or self.expert_start + self.expert_count > int(self.args.n_routed_experts):
            raise ValueError(
                f"invalid GLM-DSA expert range [{self.expert_start}, {self.expert_start + self.expert_count}) "
                f"for expert_count={self.args.n_routed_experts}"
            )
        self._readers: dict[str, GGUFTensorDataReader] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "GLMDSAGGUFModelLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _tensor_ref(self, name: str) -> GGUFTensorRef:
        try:
            return self.bundle.tensors_by_name[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor not found: {name}") from exc

    def _reader_for(self, tensor: GGUFTensorRef) -> GGUFTensorDataReader:
        reader = self._readers.get(tensor.shard_path)
        if reader is None:
            reader = GGUFTensorDataReader(tensor.shard_path)
            self._readers[tensor.shard_path] = reader
        return reader

    def _read_tensor_cpu(self, name: str) -> torch.Tensor:
        tensor = self._tensor_ref(name)
        return self._reader_for(tensor).read_tensor(tensor.name)

    def _read_dense(self, name: str) -> torch.Tensor:
        return self._read_tensor_cpu(name).to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous()

    def _linear(self, name: str) -> ReferenceLinear:
        return ReferenceLinear(self._read_dense(name), out_dtype=self.dtype)

    def _embedding(self, name: str) -> ReferenceEmbedding:
        return ReferenceEmbedding(self._read_dense(name), out_dtype=self.dtype)

    def _read_q8_0_3d(self, name: str, *, expected_shape: tuple[int, int, int]) -> torch.Tensor:
        tensor = self._read_dense(name)
        # GGUFTensorDataReader returns storage shape reversed for non-2D tensors.
        # Convert from [H, M, D] to the GGUF logical [D, M, H] used by shape docs.
        if tuple(tensor.shape) == (expected_shape[2], expected_shape[1], expected_shape[0]):
            tensor = tensor.permute(2, 1, 0).contiguous()
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} expected shape {expected_shape}, got {tuple(tensor.shape)}")
        return tensor

    def _read_routed_expert_cpu(self, name: str, expert: int) -> torch.Tensor:
        tensor = self._tensor_ref(name)
        return self._reader_for(tensor).read_routed_expert(tensor.name, expert=int(expert)).float().contiguous()

    def _glm_moe_expert_loader(self, prefix: str):
        def load_expert(expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                self._read_routed_expert_cpu(f"{prefix}.ffn_gate_exps.weight", expert),
                self._read_routed_expert_cpu(f"{prefix}.ffn_up_exps.weight", expert),
                self._read_routed_expert_cpu(f"{prefix}.ffn_down_exps.weight", expert),
            )

        return load_expert

    def _quant_tensor(self, name: str, *, row_start: int = 0, row_count: int | None = None):
        """Read a 2D quantized GGUF tensor into device-resident raw blocks.

        Avoids fp32 weight expansion for dense/attention/shared weights.  When
        ``row_count`` is given, only that output-row slice is read (TP sharding
        of vocab/lm_head).
        """
        from src.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS
        from src.loader.gguf.quantized_tensor import QuantizedGGUFTensor

        tensor = self._tensor_ref(name)
        reader = self._reader_for(tensor)
        if row_count is None:
            blocks, type_name, row_elems = reader.read_quantized_matrix_blocks(tensor.name)
        else:
            blocks, type_name, row_elems = reader.read_quantized_matrix_block_rows(
                tensor.name, int(row_start), int(row_count)
            )
        try:
            type_id = GGUF_DENSE_TYPE_IDS[type_name]
        except KeyError as exc:
            raise NotImplementedError(
                f"GLM weight {name} type {type_name!r} is not supported by the raw-block runtime"
            ) from exc
        cuda_blocks = blocks.to(device=self.device, non_blocking=False).contiguous()
        return QuantizedGGUFTensor(
            source_name=name,
            blocks=cuda_blocks,
            type_name=type_name,
            type_id=int(type_id),
            row_elems=int(row_elems),
            out_dim=int(cuda_blocks.shape[0]),
            row_start=int(row_start),
        )

    def _quant_linear(self, name: str, *, row_start: int = 0, row_count: int | None = None):
        """Build a raw-block CUDA linear over a 2D quantized GGUF tensor."""
        from src.components.gguf.quantized_ops import QuantizedGGUFLinear

        quant = self._quant_tensor(name, row_start=row_start, row_count=row_count)
        return QuantizedGGUFLinear(quant, out_dtype=self.dtype)

    def _quant_embedding(self, name: str):
        """Build a raw-block CUDA embedding over a 2D quantized GGUF tensor."""
        from src.components.gguf.quantized_ops import QuantizedGGUFEmbedding

        quant = self._quant_tensor(name)
        return QuantizedGGUFEmbedding(quant, out_dtype=self.dtype)

    def _quant_lm_head(self, name: str):
        """Build a raw-block CUDA lm_head, vocab-row-sharded across TP ranks.

        When ``world > 1`` each rank loads only its ``[row_start, row_start +
        row_count)`` slice of the vocab rows.  The transformer forward combines
        the per-rank logit slices via ``distributed_argmax_local_logits`` (decode)
        or ``gather_sharded_logits`` (return-logits), using ``lm_head.row_start``.
        Embedding stays replicated.
        """
        if self.world <= 1:
            return self._quant_linear(name)
        from src.components.gguf.tp_logits import tp_vocab_row_range

        total_rows = int(self._tensor_ref(name).dimensions[1])
        row_start, row_count = tp_vocab_row_range(total_rows, self.world, self.rank)
        return self._quant_linear(name, row_start=row_start, row_count=row_count)

    def _quant_linear_q8_0(self, name: str):
        """Build a raw-block CUDA linear over a 2D q8_0 GGUF tensor.

        q8_0 uses its own ``q8_0_gemm_forward`` kernel, not the grid/type-id
        block-dot path, so it reads raw 34-byte blocks and skips the dense
        type-id map.
        """
        from src.components.gguf.quantized_ops import Q8_0GGUFLinear

        tensor = self._tensor_ref(name)
        reader = self._reader_for(tensor)
        blocks = reader.read_q8_0_blocks(tensor.name)
        row_elems = int(tensor.dimensions[0])
        cuda_blocks = blocks.to(device=self.device, non_blocking=False).contiguous()
        return Q8_0GGUFLinear(
            cuda_blocks,
            row_elems,
            source_name=name,
            out_dtype=self.dtype,
        )

    def _quant_linear_auto(self, name: str):
        """Build a raw-block CUDA linear, dispatching by GGUF dtype.

        ``q8_0`` uses its dedicated ``q8_0_gemm_forward`` kernel; every other
        supported dtype goes through the grid/type-id block-dot path.  GLM shared
        experts are mostly q5_k/q6_k, but ``blk.8.ffn_down_shexp`` is q8_0, so the
        raw-block MoE path must handle both.
        """
        if self._tensor_ref(name).type_name == "q8_0":
            return self._quant_linear_q8_0(name)
        return self._quant_linear(name)

    def _glm_routed_type_names(self, prefix: str) -> tuple[str, str, str]:
        w1 = self._tensor_ref(f"{prefix}.ffn_gate_exps.weight").type_name
        w3 = self._tensor_ref(f"{prefix}.ffn_up_exps.weight").type_name
        w2 = self._tensor_ref(f"{prefix}.ffn_down_exps.weight").type_name
        return w1, w3, w2

    def _routed_grid_for(self, type_names: tuple[str, str, str]) -> torch.Tensor:
        """Pick the signed grid for the routed dtypes, or an empty tensor.

        GLM routed experts use iq2_xs (w1/w3) + iq3_xxs (w2), which share a
        single packed codebook.  Non-grid dtypes get an empty grid.
        """
        from src.loader.gguf.tensor_reader import get_iq2xs_iq3xxs_signed_grid_tensor

        if any(tn in ("iq2_xs", "iq3_xxs") for tn in type_names):
            return get_iq2xs_iq3xxs_signed_grid_tensor()
        return torch.empty(0, dtype=torch.int8)

    def _routed_dtypes_raw_supported(self, type_names: tuple[str, str, str]) -> bool:
        from src.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS

        return all(tn in GGUF_DENSE_TYPE_IDS for tn in type_names)

    def _routed_blocks_256_aligned(self, prefix: str) -> bool:
        """Raw-block/CUDA grouped kernels require 256-element block alignment.

        Tiny synthetic test bundles use non-aligned dims (e.g. in_dim=12); those
        must fall back to the fp32 reference MoE instead of the raw-block path.
        """
        for suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            tensor = self._tensor_ref(f"{prefix}.{suffix}")
            in_dim = int(tensor.dimensions[0])
            if in_dim % 256 != 0:
                return False
        return True

    def _build_moe_layer(self, prefix: str, layer_id: int):
        routed_types = self._glm_routed_type_names(prefix)
        use_raw = (
            self.use_raw_block_moe
            and self._routed_dtypes_raw_supported(routed_types)
            and self._routed_blocks_256_aligned(prefix)
        )
        if use_raw:
            gate = self._tensor_ref(f"{prefix}.ffn_gate_exps.weight")
            return GLMDSARawBlockMoE(
                self.args,
                layer_id,
                self._read_dense(f"{prefix}.ffn_gate_inp.weight"),
                self._read_dense(f"{prefix}.exp_probs_b.bias"),
                self._quant_linear_auto(f"{prefix}.ffn_gate_shexp.weight"),
                self._quant_linear_auto(f"{prefix}.ffn_up_shexp.weight"),
                self._quant_linear_auto(f"{prefix}.ffn_down_shexp.weight"),
                gguf_path=gate.shard_path,
                w1_name=f"{prefix}.ffn_gate_exps.weight",
                w3_name=f"{prefix}.ffn_up_exps.weight",
                w2_name=f"{prefix}.ffn_down_exps.weight",
                signed_grid=self._routed_grid_for(routed_types),
                device=self.device,
                dtype=self.dtype,
                expert_start=self.expert_start,
                expert_count=self.expert_count,
            )
        return GLMDSAMoE(
            self.args,
            layer_id,
            self._read_dense(f"{prefix}.ffn_gate_inp.weight"),
            self._read_dense(f"{prefix}.exp_probs_b.bias"),
            self._linear(f"{prefix}.ffn_gate_shexp.weight"),
            self._linear(f"{prefix}.ffn_up_shexp.weight"),
            self._linear(f"{prefix}.ffn_down_shexp.weight"),
            self._glm_moe_expert_loader(prefix),
            device=self.device,
            dtype=self.dtype,
            expert_start=self.expert_start,
            expert_count=self.expert_count,
        )

    def load(self) -> GLMDSATransformer:
        if self.args.n_layers > self.args.leading_dense_layers and not self.allow_moe_layers:
            raise NotImplementedError(
                f"GLM-DSA MoE layers are implemented but disabled for this load "
                f"(allow_moe_layers=False); requested n_layers={self.args.n_layers} exceeds "
                f"leading_dense_layers={self.args.leading_dense_layers}. Enable MoE to load these layers."
            )

        embedding = self._quant_embedding("token_embd.weight")
        lm_head = self._quant_lm_head("output.weight")
        final_norm = self._read_dense("output_norm.weight")

        layers: list[GLMDSABlock] = []
        k_nope = self.args.key_mla_dim - self.args.rope_dim
        for layer_id in range(self.args.n_layers):
            prefix = f"blk.{layer_id}"
            attention = GLMDSAAttention(
                self.args,
                layer_id,
                self._quant_linear(f"{prefix}.attn_q_a.weight"),
                self._read_dense(f"{prefix}.attn_q_a_norm.weight"),
                self._quant_linear_q8_0(f"{prefix}.attn_q_b.weight"),
                self._quant_linear_q8_0(f"{prefix}.attn_kv_a_mqa.weight"),
                self._read_dense(f"{prefix}.attn_kv_a_norm.weight"),
                self._read_q8_0_3d(
                    f"{prefix}.attn_k_b.weight",
                    expected_shape=(k_nope, self.args.kv_lora_rank, self.args.n_heads),
                ),
                self._read_q8_0_3d(
                    f"{prefix}.attn_v_b.weight",
                    expected_shape=(self.args.value_dim, self.args.value_mla_dim, self.args.n_heads),
                ),
                self._quant_linear(f"{prefix}.attn_output.weight"),
                device=self.device,
                dtype=self.dtype,
            )
            if layer_id < self.args.leading_dense_layers:
                mlp = GLMDSADenseMLP(
                    self._quant_linear(f"{prefix}.ffn_gate.weight"),
                    self._quant_linear(f"{prefix}.ffn_up.weight"),
                    self._quant_linear(f"{prefix}.ffn_down.weight"),
                    dtype=self.dtype,
                )
            else:
                if self.allow_moe_layers:
                    mlp = self._build_moe_layer(prefix, layer_id)
                else:
                    mlp = GLMDSAMoEPlaceholder(layer_id)
            layers.append(
                GLMDSABlock(
                    self.args,
                    layer_id,
                    self._read_dense(f"{prefix}.attn_norm.weight"),
                    self._read_dense(f"{prefix}.ffn_norm.weight"),
                    attention,
                    mlp,
                    dtype=self.dtype,
                )
            )

        return GLMDSATransformer(
            self.args,
            embedding,
            layers,
            final_norm,
            lm_head,
            device=self.device,
            dtype=self.dtype,
        )


# Backward-compatible/simple alias.
GLMDSAGGUFLoader = GLMDSAGGUFModelLoader


def load_glm_dsa_gguf_model(
    gguf_path: str | Path | GGUFBundle,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    n_layers: int | None = None,
    allow_moe_layers: bool = False,
    expert_start: int = 0,
    expert_count: int | None = None,
    world: int = 1,
    rank: int = 0,
) -> tuple[GLMDSATransformer, dict[str, Any]]:
    start = time.perf_counter()
    loader = GLMDSAGGUFModelLoader(
        gguf_path,
        device=device,
        dtype=dtype,
        n_layers=n_layers,
        allow_moe_layers=allow_moe_layers,
        expert_start=expert_start,
        expert_count=expert_count,
        world=world,
        rank=rank,
    )
    try:
        model = loader.load()
    finally:
        loader.close()
    elapsed = time.perf_counter() - start
    info = {
        "load_seconds": elapsed,
        "layers": model.args.n_layers,
        "dim": model.args.dim,
        "vocab_size": model.args.vocab_size,
        "context_length": model.args.context_length,
        "rope_dim": model.args.rope_dim,
        "rope_base": model.args.rope_base,
        "device": str(model.device),
        "dtype": str(dtype),
        "allow_moe_layers": bool(allow_moe_layers),
        "expert_start": int(loader.expert_start),
        "expert_count": int(loader.expert_count),
        "world": int(loader.world),
        "rank": int(loader.rank),
        "lm_head_out_dim": int(model.lm_head.out_dim),
        "lm_head_row_start": int(model.lm_head.row_start),
    }
    return model, info

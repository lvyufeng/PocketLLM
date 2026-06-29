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
    GLMDSAMoEPlaceholder,
    GLMDSATransformer,
    ReferenceEmbedding,
    ReferenceLinear,
)


class GLMDSAGGUFModelLoader:
    """Assemble a first-pass GLM-DSA reference runtime from a GGUF bundle.

    This loader intentionally uses dequantized reference tensors for correctness
    bring-up. It is only meant for small-layer smoke tests at first; optimized
    raw-block kernels and PocketMoE active-expert staging should replace the
    fallback paths once the GLM-DSA math is validated.
    """

    def __init__(
        self,
        bundle_or_path: GGUFBundle | str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        n_layers: int | None = None,
        allow_moe_layers: bool = False,
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

    def load(self) -> GLMDSATransformer:
        if self.args.n_layers > self.args.leading_dense_layers and not self.allow_moe_layers:
            raise NotImplementedError(
                f"GLM-DSA MoE layers are not implemented in the first reference runtime; "
                f"requested n_layers={self.args.n_layers}, leading_dense_layers={self.args.leading_dense_layers}"
            )

        embedding = self._embedding("token_embd.weight")
        lm_head = self._linear("output.weight")
        final_norm = self._read_dense("output_norm.weight")

        layers: list[GLMDSABlock] = []
        k_nope = self.args.key_mla_dim - self.args.rope_dim
        for layer_id in range(self.args.n_layers):
            prefix = f"blk.{layer_id}"
            attention = GLMDSAAttention(
                self.args,
                layer_id,
                self._linear(f"{prefix}.attn_q_a.weight"),
                self._read_dense(f"{prefix}.attn_q_a_norm.weight"),
                self._linear(f"{prefix}.attn_q_b.weight"),
                self._linear(f"{prefix}.attn_kv_a_mqa.weight"),
                self._read_dense(f"{prefix}.attn_kv_a_norm.weight"),
                self._read_q8_0_3d(
                    f"{prefix}.attn_k_b.weight",
                    expected_shape=(k_nope, self.args.kv_lora_rank, self.args.n_heads),
                ),
                self._read_q8_0_3d(
                    f"{prefix}.attn_v_b.weight",
                    expected_shape=(self.args.value_dim, self.args.value_mla_dim, self.args.n_heads),
                ),
                self._linear(f"{prefix}.attn_output.weight"),
                device=self.device,
                dtype=self.dtype,
            )
            if layer_id < self.args.leading_dense_layers:
                mlp = GLMDSADenseMLP(
                    self._linear(f"{prefix}.ffn_gate.weight"),
                    self._linear(f"{prefix}.ffn_up.weight"),
                    self._linear(f"{prefix}.ffn_down.weight"),
                    dtype=self.dtype,
                )
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
) -> tuple[GLMDSATransformer, dict[str, Any]]:
    start = time.perf_counter()
    loader = GLMDSAGGUFModelLoader(
        gguf_path,
        device=device,
        dtype=dtype,
        n_layers=n_layers,
        allow_moe_layers=allow_moe_layers,
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
    }
    return model, info

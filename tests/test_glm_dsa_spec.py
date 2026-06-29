from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.components.moe.registry import detect_spec, known_architectures
from src.loader.gguf.bundle import read_gguf_bundle
from src.models.glm_dsa.spec import GLMDSASpec
from tests.gguf_test_utils import GGML_F32, glm_dsa_metadata, write_gguf, write_glm_dsa_bundle


REAL_GLM_DSA_Q2_PATH = Path("/mnt/data3/GLM-5.2-GGUF/UD-Q2_K_XL")


def test_glm_dsa_spec_parses_tiny_bundle_and_validates_schema(tmp_path: Path) -> None:
    root = write_glm_dsa_bundle(tmp_path / "bundle", n_layers=4, leading_dense=1)
    bundle = read_gguf_bundle(root)
    spec = GLMDSASpec()

    params = spec.parse_params(bundle)
    validation = spec.validate_bundle(bundle)

    assert params.architecture == "glm-dsa"
    assert params.n_layers == 4
    assert params.hidden_size == 12
    assert params.vocab_size == 32
    assert params.context_length == 1024
    assert params.n_heads == 3
    assert params.n_kv_heads == 1
    assert params.head_dim == 5
    assert params.rope_dim == 1
    assert params.n_routed_experts == 4
    assert params.top_k == 2
    assert params.expert_intermediate_size == 8
    assert params.n_shared_experts == 1
    assert params.attention_kind == "glm_dsa_mla_indexed"
    assert params.routed_expert_layout == "packed_3d_after_dense_prefix"
    assert validation.ok, validation.errors
    assert validation.mapped_sources == 94
    assert validation.role_counts["dense_w1"] == 1
    assert validation.role_counts["dense_w2"] == 1
    assert validation.role_counts["dense_w3"] == 1
    assert validation.role_counts["routed_w1"] == 3
    assert validation.role_counts["routed_w2"] == 3
    assert validation.role_counts["routed_w3"] == 3
    assert validation.role_counts["nextn"] == 1
    assert validation.role_counts["nextn_norm"] == 3


def test_glm_dsa_registry_detects_architecture(tmp_path: Path) -> None:
    root = write_glm_dsa_bundle(tmp_path / "bundle")
    bundle = read_gguf_bundle(root)

    assert "glm-dsa" in known_architectures()
    assert detect_spec(bundle).architecture == "glm-dsa"


def test_glm_dsa_capability_is_inventory_supported_but_generation_deferred(tmp_path: Path) -> None:
    root = write_glm_dsa_bundle(tmp_path / "bundle", n_layers=4, leading_dense=1)
    bundle = read_gguf_bundle(root)
    report = GLMDSASpec().capability_report(bundle, gpu_count=4, gpu_memory_gib=22.0)

    caps = {item.name: item for item in report.capabilities}
    placements = {item.name: item for item in report.placements}

    assert report.tensor_type_counts["iq2_xs"] == 6
    assert report.tensor_type_counts["iq3_xxs"] == 3
    assert report.tensor_type_counts["q6_k"] == 4
    assert caps["routed_w1:iq2_xs"].status == "deferred"
    assert caps["routed_w2:iq3_xxs"].status == "deferred"
    assert caps["generation"].status == "candidate"
    assert placements["heterogeneous_routed_experts"].status == "candidate"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GLM-DSA reference runtime smoke")
def test_glm_dsa_dense_prefix_reference_runtime_forward(tmp_path: Path) -> None:
    root = write_glm_dsa_bundle(tmp_path / "bundle", n_layers=4, leading_dense=1)
    bundle = read_gguf_bundle(root)
    spec = GLMDSASpec()
    runtime = spec.build_token_runtime(
        bundle,
        world=1,
        rank=0,
        device=torch.device("cuda", 0),
        dtype=torch.float16,
        n_layers=1,
        gpu_memory_gib=22.0,
    )

    model = runtime.model
    tokens = torch.tensor([[1, 2, 3]], device=model.device, dtype=torch.long)
    model.reset_cache(batch_size=1, max_seq_len=4)
    logits = model.forward(tokens, 0)
    next_token = model.forward(tokens, 0, return_next_token=True)

    assert logits.shape == (1, 1, 32)
    assert next_token.shape == (1,)
    assert torch.isfinite(logits).all()


def test_glm_dsa_shape_validation_catches_wrong_tensor_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.gguf"
    metadata = glm_dsa_metadata(n_layers=1, leading_dense=1)
    write_gguf(path, metadata=metadata, tensors=[("token_embd.weight", (11, 32), GGML_F32)])
    bundle = read_gguf_bundle(path)

    validation = GLMDSASpec().validate_bundle(bundle)

    assert not validation.ok
    assert any("token_embd.weight unexpected shape" in error for error in validation.errors)


@pytest.mark.skipif(not REAL_GLM_DSA_Q2_PATH.exists(), reason="local GLM-5.2 Q2 GGUF bundle not present")
def test_real_glm_dsa_q2_bundle_validates_when_available() -> None:
    bundle = read_gguf_bundle(REAL_GLM_DSA_Q2_PATH)
    spec = GLMDSASpec()
    params = spec.parse_params(bundle)
    validation = spec.validate_bundle(bundle)
    report = spec.capability_report(bundle)

    assert params.n_layers == 79
    assert params.hidden_size == 6144
    assert params.vocab_size == 154880
    assert params.context_length == 1048576
    assert params.n_heads == 64
    assert params.n_kv_heads == 1
    assert params.head_dim == 576
    assert params.rope_dim == 64
    assert params.n_routed_experts == 256
    assert params.top_k == 8
    assert params.expert_intermediate_size == 2048
    assert params.n_shared_experts == 1
    assert bundle.tensor_count == 1809
    assert validation.ok, validation.errors[:20]
    assert validation.mapped_sources == 1809
    assert report.tensor_type_counts["iq2_xs"] == 148
    assert report.tensor_type_counts["iq3_xxs"] == 73
    assert any(item.name == "generation" and item.status == "candidate" for item in report.capabilities)

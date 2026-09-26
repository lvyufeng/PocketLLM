"""Xing4.0-29B-A4B, end to end, out of the published IQ4_NL GGUF.

The whole checkpoint is 18.7 GiB, so this is the first model in the repository
that fits on one 2080 Ti with room left over -- and the way the weights are held
follows from that rather than from the architecture.  Three kinds of tensor and
three treatments:

- **Raw IQ4_NL blocks, run in place.**  The routed experts (14.7 GiB) and the two
  leading dense blocks' MLPs.  Nothing is dequantized: the blocks go to the card
  and the kernels consume them there, which is what keeps the checkpoint's size
  and its resident footprint the same number.
- **Resident bf16, cast to fp16.**  The attention path is 2.117 GiB of bf16 in
  the file and the card has no bf16 tensor cores, so it is cast once at load.
  This is also the byte table's other half: the attention, not the MoE, is what a
  decode step reads most of.
- **fp32, kept as fp32.**  Every norm, the router, and the hyper-connections'
  gates, which the quantizer excludes by name as precision-sensitive.

What is deliberately not here: the NextN block at `blk.40`.  It is a
speculative head, the repository has its own history with MTP, and serving
without it changes no number this stage records.

Stage 5 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.xing4_0.attention import MLAAttentionWeights
from src.models.xing4_0.block import DecoderLayer, DecoderLayerWeights, rms_norm
from src.models.xing4_0.config import Xing4_0Params
from src.models.xing4_0.hyper_connection import HyperConnectionWeights
from src.models.xing4_0.mlp import GroupedExpertStack, MoEWeights, RoutedMoE, SwiGLUMLP

__all__ = ["Xing4_0GGUFModel"]


class Xing4_0GGUFModel:
    """The trunk, loaded from the released GGUF and resident on one device.

    `forward` takes real token ids and returns fp32 logits for every position it
    was given, which is what a sampling loop and a parity check both want.  The
    KV cache is the absorbed one -- `KVLatentCache`, 576 floats a token a layer
    against the expanded form's 10240 -- because that is the form the file is
    shaped for and the form decode has to use.
    """

    def __init__(
        self,
        gguf_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        block_count: int | None = None,
        config_path: str | Path | None = None,
        use_kernel: bool = True,
    ):
        from src.loader.gguf.quantized_loader import GGUFQuantizedTensorLoader

        self.path = str(gguf_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.loader = GGUFQuantizedTensorLoader(self.path, device=self.device)
        self.params = self._read_params(config_path)
        self.block_count = int(self.params.n_layers if block_count is None else block_count)
        self.use_kernel = bool(use_kernel)

        self.embedding = self.loader.read_dense("token_embd.weight", dtype=dtype)
        self.output_norm = self.loader.read_dense("output_norm.weight", dtype=torch.float32)
        from src.components.gguf.quantized_ops import QuantizedGGUFLinear

        self.lm_head = QuantizedGGUFLinear(
            self.loader.read_quant("output.weight", self._type_of("output.weight")),
            out_dtype=torch.float32,
        )
        cuda_mod = self._cuda()
        self.blocks: list[DecoderLayer] = []
        for index in range(self.block_count):
            prefix = f"blk.{index}."
            weights = DecoderLayerWeights(
                attn_hc=HyperConnectionWeights.from_gguf(self.loader, f"{prefix}hc_attn", self.params),
                ffn_hc=HyperConnectionWeights.from_gguf(self.loader, f"{prefix}hc_ffn", self.params),
                attention=MLAAttentionWeights.from_gguf(self.loader, prefix, self.params, dtype=dtype),
                input_layernorm=self.loader.read_dense(f"{prefix}attn_norm.weight", dtype=torch.float32),
                post_attention_layernorm=self.loader.read_dense(f"{prefix}ffn_norm.weight", dtype=torch.float32),
            )
            self.blocks.append(
                DecoderLayer(
                    self.params,
                    weights,
                    self._mlp(prefix, index, cuda_mod),
                    dtype=dtype,
                    device=self.device,
                    use_kernel=self.use_kernel,
                )
            )

    # -- loading ------------------------------------------------------------- #

    def _read_params(self, config_path: str | Path | None = None) -> Xing4_0Params:
        """The config from `config.json`, cross-checked against the GGUF header.

        The header is not sufficient on its own and the missing keys are the
        interesting ones: `mhc_h_res_clamp_min` / `_max` are the "+-30" the
        Sinkhorn logits are clamped to, and no GGUF key carries them, so a
        params object built from the file alone would default them to something
        and change the model.  So the side car is the source.

        What the header *is* good for is agreeing, and every key the two share is
        compared here.  A disagreement means the GGUF and the safetensors are
        different checkpoints, which is exactly the mistake that would otherwise
        show up as slightly-wrong text rather than an error.
        """
        if config_path is None:
            config_path = Path(self.path).parent.parent / "Xing4.0-29B-A4B" / "config.json"
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Xing4.0 needs the checkpoint's config.json next to the GGUF; looked for {config_path}"
            )
        params = Xing4_0Params.from_json(config_path)
        self._check_agreement(params)
        return params

    def _check_agreement(self, params: Xing4_0Params) -> None:
        """Every key the GGUF header and `config.json` both carry.

        Floats are compared with a tolerance, because the header stores them as
        fp32 and `config.json` writes `1e-6`: the same epsilon, two spellings.
        The gating function is compared as a name, since llama.cpp writes it as
        an enum -- 1 softmax, 2 sigmoid -- and `config.json` writes the word.
        """
        header = {
            key[len("xing4_0.") :]: value
            for key, value in self.loader.bundle.metadata.items()
            if key.startswith("xing4_0.")
        }
        gating = {1: "softmax", 2: "sigmoid"}.get(int(header["expert_gating_func"]))
        pairs = [
            ("embedding_length", params.hidden_size),
            ("block_count", params.n_layers + params.nextn_layers),
            ("attention.head_count", params.n_heads),
            ("attention.q_lora_rank", params.q_lora_rank),
            ("attention.kv_lora_rank", params.kv_lora_rank),
            ("attention.key_length_mla", params.qk_head_dim),
            ("attention.value_length_mla", params.v_head_dim),
            ("attention.layer_norm_rms_epsilon", params.rms_norm_eps),
            ("expert_count", params.n_routed_experts),
            ("expert_used_count", params.n_experts_per_tok),
            ("expert_shared_count", params.n_shared_experts),
            ("expert_feed_forward_length", params.moe_intermediate_size),
            ("feed_forward_length", params.intermediate_size),
            ("leading_dense_block_count", params.first_k_dense_replace),
            ("expert_weights_scale", params.routed_scaling_factor),
            ("expert_weights_norm", params.norm_topk_prob),
            ("expert_group_count", params.n_group),
            ("expert_group_used_count", params.topk_group),
            ("hyper_connection.count", params.hc_mult),
            ("hyper_connection.sinkhorn_iterations", params.hc_sinkhorn_iters),
            ("hyper_connection.epsilon", params.hc_eps),
            ("rope.freq_base", params.rope_theta),
            ("rope.dimension_count", params.qk_rope_head_dim),
            ("rope.scaling.factor", params.yarn.factor),
            ("rope.scaling.original_context_length", params.yarn.original_max_position_embeddings),
            ("rope.scaling.yarn_beta_fast", params.yarn.beta_fast),
            ("rope.scaling.yarn_beta_slow", params.yarn.beta_slow),
            ("context_length", params.context_length),
            ("vocab_size", params.vocab_size),
            ("nextn_predict_layers", params.nextn_layers),
        ]
        mismatched = []
        for key, value in pairs:
            if key not in header:
                mismatched.append(f"{key}: not in the GGUF header, config says {value!r}")
                continue
            found = header[key]
            if isinstance(value, float) and isinstance(found, float):
                agree = math.isclose(value, found, rel_tol=1e-6)
            else:
                agree = found == value
            if not agree:
                mismatched.append(f"{key}: header {found!r} vs config {value!r}")
        if gating != params.scoring_func:
            mismatched.append(f"expert_gating_func: header {gating!r} vs config {params.scoring_func!r}")
        if mismatched:
            raise ValueError(
                f"{self.path} and the config it was read with describe different checkpoints: "
                + "; ".join(mismatched)
            )

    def _type_of(self, name: str) -> str:
        return self.loader.tensor_ref(name).type_name

    def _cuda(self):
        from src.kernels.cuda_loader import load_cuda_kernel

        module = load_cuda_kernel()
        if module is None:
            raise RuntimeError("the cuda_kernel extension is required to run the MoE")
        return module

    def _mlp(self, prefix: str, index: int, cuda_mod):
        """A dense MLP for the two leading blocks, a routed MoE after them.

        `first_k_dense_replace` is 2, so blocks 0 and 1 have a 9216-wide ordinary
        FFN and every block after them has the 64-expert one.  The dense blocks
        are still IQ4_NL -- the quantizer applies to the whole model -- so they
        are raw blocks behind `QuantizedGGUFLinear` and not fp16 weights.
        """
        from src.components.gguf.quantized_ops import QuantizedGGUFLinear

        if index < int(self.params.first_k_dense_replace):
            return SwiGLUMLP(
                QuantizedGGUFLinear(self.loader.read_quant(f"{prefix}ffn_gate.weight", "iq4_nl")),
                QuantizedGGUFLinear(self.loader.read_quant(f"{prefix}ffn_up.weight", "iq4_nl")),
                QuantizedGGUFLinear(self.loader.read_quant(f"{prefix}ffn_down.weight", "iq4_nl")),
                out_dtype=self.dtype,
            )
        weights = MoEWeights.from_gguf(self.loader, prefix, device=str(self.device))
        stack = GroupedExpertStack(
            self.params,
            weights,
            cuda_mod,
            in_dim=int(self.params.hidden_size),
            inter_dim=int(self.params.moe_intermediate_size),
        )
        return RoutedMoE(self.params, weights, stack, dtype=self.dtype)

    # -- running ------------------------------------------------------------- #

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`[tokens]` -> `[1, tokens, hc, hidden]`, every stream equal.

        The reference starts the four streams identical
        (`inputs_embeds.unsqueeze(2).expand(...)`) and lets the hyper-connections
        pull them apart, so a port that began with four streams of different
        values would diverge before the first block.
        """
        ids = input_ids.reshape(-1).to(self.device)
        hidden = F.embedding(ids, self.embedding)
        return hidden.unsqueeze(0).unsqueeze(2).expand(1, -1, int(self.params.hc_mult), -1).contiguous()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache=None,
        start_pos: int = 0,
        absorbed: bool = True,
    ) -> torch.Tensor:
        """`[tokens]` -> `[tokens, vocab]` fp32 logits."""
        hidden = self.embed(input_ids)
        positions = torch.arange(
            int(start_pos), int(start_pos) + hidden.shape[1], device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        for index, block in enumerate(self.blocks):
            hidden = block.forward(
                hidden,
                positions,
                # One cache per layer.  The absorbed cache is 576 floats a token
                # per layer -- a 512 latent and its 64-wide rope key -- so a list
                # is 40 x 576 a token and not a single tensor of that width.
                cache=None if cache is None else cache[index],
                start_pos=start_pos,
                absorbed=absorbed,
            )
        collapsed = block_collapse(hidden)
        collapsed = rms_norm(collapsed, self.output_norm, self.params.rms_norm_eps)
        return self.lm_head(collapsed).reshape(-1, self.lm_head.out_dim).float()

    def make_cache(self, capacity: int, *, batch: int = 1):
        from src.models.xing4_0.attention import KVLatentCache

        return [
            KVLatentCache(batch, capacity, self.params, device=self.device, dtype=self.dtype)
            for _ in self.blocks
        ]

    @property
    def nbytes(self) -> int:
        """Resident weight bytes, so the card's unused room is a number.

        Raw blocks count at their stored size; a `QuantizedGGUFLinear` holds one
        tensor, an fp16/bf16/fp32 weight holds its own, and this walks the blocks
        rather than measuring the device so that it answers "how much did the
        checkpoint cost" and not "what else is on the card".
        """
        seen: set[int] = set()
        total = 0

        def visit(obj) -> None:
            nonlocal total
            if isinstance(obj, torch.Tensor):
                total += obj.numel() * obj.element_size()
                return
            if isinstance(obj, (int, float, str, bool, bytes, type(None))):
                return
            if id(obj) in seen:
                return
            seen.add(id(obj))
            # A raw-block tensor's payload is its `blocks`, and the `blocks`
            # tensor itself is reachable from there -- but visiting an
            # `mlp.weights` first and the `blocks` second would count the same
            # 15 GiB twice without this arm.
            blocks = getattr(getattr(obj, "tensor", None), "blocks", None)
            if isinstance(blocks, torch.Tensor):
                total += blocks.numel() * blocks.element_size()
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    visit(item)
                return
            if isinstance(obj, dict):
                for item in obj.values():
                    visit(item)
                return
            for value in getattr(obj, "__dict__", {}).values():
                visit(value)

        visit([self.embedding, self.output_norm, self.lm_head, self.blocks])
        return total


def block_collapse(hidden: torch.Tensor) -> torch.Tensor:
    """`hidden_states.mean(dim=2)` -- the four streams, averaged, not summed."""
    return hidden.mean(dim=-2)

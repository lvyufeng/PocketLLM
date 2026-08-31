"""Qwen4-Exp decoder stack.

Weights arrive as plain tensors from a provider (see `weights.py`), so the same
module code serves the single-GPU reference path and the TP4 heterogeneous path
— the latter simply hands over sharded tensors and a host-resident MoE backend.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.models.qwen4_exp.attention import (
    GatedDeltaNet,
    GatedDeltaNetCache,
    QSAAttention,
    QSAAttentionCache,
)
from src.models.qwen4_exp.config import Qwen4ExpTextConfig
from src.models.qwen4_exp.layers import (
    DenseMLP,
    GatedResidual,
    MRoPE,
    NGramHasher,
    RMSNorm,
    TopKRouter,
    inject_into_streams,
    prefill_linear,
)
from src.models.qwen4_exp.moe import MoEBackend


@contextmanager
def _noop_context():
    """No-op context manager when profiler is disabled."""
    yield


@dataclass
class Qwen4ExpCache:
    """Per-layer cache state plus the PLE token-history window.

    `ple_token_history` holds the last `ngram_size - 1` token ids so decode steps
    can hash n-grams that reach back before the current token.
    """

    layers: list[object] = field(default_factory=list)
    ple_token_history: torch.Tensor | None = None
    length: int = 0


class PLELayer:
    """Injects hashed n-gram features into every hyper-connection stream.

    The n-gram embedding is projected to one value and `hc_count` keys; the
    normalized stream activations gate the value per stream, then a dilated
    depthwise conv adds local lexical context.  The dilation is `ngram_size`, so
    the conv state spans `(kernel - 1) * ngram_size` positions.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        weights: dict[str, torch.Tensor],
        ple_layer_index: int,
        device: torch.device,
        *,
        ngram_table: object,
    ) -> None:
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.hasher = NGramHasher(config, ple_layer_index, device)
        self.ngram_table = ngram_table
        self.key_proj = weights["key_proj"]
        self.value_proj = weights["value_proj"]
        self.norm_key = RMSNorm(weights["norm_key"], config.rms_norm_eps, group_size=config.hidden_size)
        self.norm_query = RMSNorm(weights["norm_query"], config.rms_norm_eps, group_size=config.hidden_size)
        self.norm_conv = RMSNorm(weights["norm_conv"], config.rms_norm_eps, group_size=config.hidden_size)
        self.conv1d_weight = weights["conv1d"]  # (hc_hidden, 1, kernel)
        self.conv_dilation = config.ngram_size
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        self.conv_state: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.conv_state = None

    def _short_conv(self, hidden_states: torch.Tensor, *, use_cache: bool) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        x = hidden_states.transpose(1, 2)  # (b, hc_hidden, seq)
        if use_cache and self.conv_state is not None:
            x = torch.cat([self.conv_state.to(x.dtype), x], dim=-1)
        if use_cache:
            self.conv_state = x[:, :, -self.short_conv_state_len :].clone()
        x = F.pad(x, (self.short_conv_state_len, 0))
        x = x[..., -(self.short_conv_state_len + seq_len) :]
        channels = self.conv1d_weight.shape[0]
        x = F.conv1d(
            x,
            self.conv1d_weight,
            None,
            padding=0,
            dilation=self.conv_dilation,
            groups=channels,
        )
        return F.silu(x).transpose(1, 2)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        token_history: torch.Tensor,
        out_len: int,
        *,
        use_cache: bool,
    ) -> torch.Tensor:
        row_ids = self.hasher.row_ids(token_history, out_len)
        embeddings = self.ngram_table(row_ids).flatten(-2).to(hidden_states.dtype)
        key_normed = self.norm_key(prefill_linear(embeddings, self.key_proj)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        value = prefill_linear(embeddings, self.value_proj)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        # Signed sqrt keeps the gate's sign while compressing its magnitude.
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        return gated_value + self._short_conv(gated_value_normed, use_cache=use_cache)


class Qwen4ExpDecoderLayer:
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_idx: int,
        weights: dict[str, torch.Tensor],
        moe: MoEBackend,
        *,
        device: torch.device,
        ple: PLELayer | None = None,
        tp_shard: "TPShard | None" = None,
        all_reduce=None,
    ) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.is_linear = config.is_linear_layer(layer_idx)
        self.ple = ple
        self.moe = moe
        # Row-parallel outputs (attention o_proj/out_proj and the MLP sum) are
        # partial sums on each rank; the hyper-connection streams themselves are
        # replicated, so the reduce must land on the block output *before* it is
        # injected back into the streams.
        self.all_reduce = all_reduce
        shard = tp_shard or TPShard.single(config)

        if self.is_linear:
            self.attn = GatedDeltaNet(
                config,
                {k[len("linear_attn.") :]: v for k, v in weights.items() if k.startswith("linear_attn.")},
                num_v_heads=shard.linear_v_heads,
                num_k_heads=shard.linear_k_heads,
            )
        else:
            self.attn = QSAAttention(
                config,
                {k[len("self_attn.") :]: v for k, v in weights.items() if k.startswith("self_attn.")},
                num_heads=shard.attn_heads,
                num_kv_heads=shard.attn_kv_heads,
                indexer_heads=shard.indexer_heads,
            )

        self.attn_hc = GatedResidual(
            config,
            weights["attn_hyper_connection.hc_norm"],
            weights["attn_hyper_connection.input_mix_weight_down"],
            weights["attn_hyper_connection.input_mix_weight_up"],
            weights["attn_hyper_connection.block_inject_weight"],
        )
        self.mlp_hc = GatedResidual(
            config,
            weights["mlp_hyper_connection.hc_norm"],
            weights["mlp_hyper_connection.input_mix_weight_down"],
            weights["mlp_hyper_connection.input_mix_weight_up"],
            weights["mlp_hyper_connection.block_inject_weight"],
        )

        self.router = TopKRouter(weights["mlp.gate"], config.num_experts_per_tok, norm_topk_prob=config.norm_topk_prob)
        self.shared_expert = DenseMLP(
            weights["mlp.shared_expert.gate_proj"],
            weights["mlp.shared_expert.up_proj"],
            weights["mlp.shared_expert.down_proj"],
        )
        self.shared_expert_gate = weights["mlp.shared_expert_gate"]

    def make_cache(self, batch_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype):
        if self.is_linear:
            return GatedDeltaNetCache(
                self.config,
                batch_size,
                device,
                dtype,
                num_v_heads=self.attn.num_v_heads,
                num_k_heads=self.attn.num_k_heads,
            )
        return QSAAttentionCache(
            self.config,
            batch_size,
            max_seq_len,
            device,
            dtype,
            num_kv_heads=self.attn.num_kv_heads,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache,
        past_len: int,
        token_history: torch.Tensor | None,
        use_cache: bool,
        profiler=None,
    ) -> torch.Tensor:
        prof_scope = profiler.scope if profiler else lambda x: _noop_context()

        with prof_scope(f"layer_{self.layer_idx}"):
            if self.ple is not None:
                assert token_history is not None, "PLE layers need the token id history"
                with prof_scope("ple"):
                    ple_out = self.ple(
                        hidden_states, token_history, hidden_states.shape[1], use_cache=use_cache
                    )
                hidden_states = hidden_states + ple_out

            with prof_scope("attn_hc"):
                mixed, hyper_input, inject = self.attn_hc(hidden_states)

            with prof_scope("attention"):
                if self.is_linear:
                    attn_out = self.attn(mixed, cache)
                else:
                    attn_out = self.attn(mixed, cos, sin, cache, past_len=past_len)

            if profiler is not None and hasattr(self.attn, "last_profile"):
                for name, elapsed in self.attn.last_profile.items():
                    profiler.add_external(f"attention_{name}", elapsed)
                self.attn.last_profile.clear()

            if self.all_reduce is not None:
                with prof_scope("attn_reduce"):
                    attn_out = self.all_reduce(attn_out)

            with prof_scope("attn_inject"):
                hidden_states = inject_into_streams(attn_out, hyper_input, inject)

            with prof_scope("mlp_hc"):
                mixed, hyper_input, inject = self.mlp_hc(hidden_states)

            flat = mixed.reshape(-1, mixed.shape[-1])

            with prof_scope("router"):
                weights, indices = self.router(flat)

            with prof_scope("moe"):
                routed = self.moe(self.layer_idx, flat, indices, weights)

            with prof_scope("shared_expert"):
                shared = self.shared_expert(flat)
                shared = torch.sigmoid(
                    prefill_linear(flat, self.shared_expert_gate)
                ) * shared

            mlp_out = (routed + shared).reshape(mixed.shape)

            if self.all_reduce is not None:
                with prof_scope("mlp_reduce"):
                    mlp_out = self.all_reduce(mlp_out)

            with prof_scope("mlp_inject"):
                return inject_into_streams(mlp_out, hyper_input, inject)


@dataclass
class TPShard:
    """Head counts owned by this rank after tensor-parallel splitting."""

    attn_heads: int
    attn_kv_heads: int
    indexer_heads: int
    linear_v_heads: int
    linear_k_heads: int

    @classmethod
    def single(cls, config: Qwen4ExpTextConfig) -> "TPShard":
        return cls(
            attn_heads=config.num_attention_heads,
            attn_kv_heads=config.num_key_value_heads,
            indexer_heads=config.indexer_n_heads or 0,
            linear_v_heads=config.linear_num_value_heads,
            linear_k_heads=config.linear_num_key_heads,
        )


class Qwen4ExpModel:
    """The 48-layer decoder stack with 4-stream hyper-connections."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layers: list[Qwen4ExpDecoderLayer],
        *,
        embed_tokens,
        lm_head: torch.Tensor,
        final_mixer: GatedResidual,
        device: torch.device,
        dtype: torch.dtype,
        profiler=None,
    ) -> None:
        self.config = config
        self.layers = layers
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.final_mixer = final_mixer
        self.device = device
        self.dtype = dtype
        self.profiler = profiler
        self.rope = MRoPE(config, device)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def make_cache(self, batch_size: int, max_seq_len: int) -> Qwen4ExpCache:
        cache = Qwen4ExpCache(
            layers=[layer.make_cache(batch_size, max_seq_len, self.device, self.dtype) for layer in self.layers]
        )
        for layer in self.layers:
            if layer.ple is not None:
                layer.ple.reset_cache()
        cache.ple_token_history = torch.full(
            (batch_size, self.config.ngram_size - 1),
            self.config.primary_eos_token_id,
            dtype=torch.long,
        )
        return cache

    def _rope_cache(self, total_len: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for absolute positions [0, total_len). The QSA indexer needs
        every past position, not just the current ones, so we keep the full span
        and grow it in place."""
        if self._cos is None or self._cos.shape[1] < total_len or self._cos.shape[0] != batch_size:
            position_ids = torch.arange(total_len, device=self.device).view(1, 1, -1).expand(3, batch_size, -1)
            self._cos, self._sin = self.rope(position_ids, self.dtype)
        return self._cos[:, :total_len], self._sin[:, :total_len]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Qwen4ExpCache | None = None,
        past_len: int = 0,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        prof_scope = self.profiler.scope if self.profiler else lambda x: _noop_context()

        batch_size, seq_len = input_ids.shape
        total_len = past_len + seq_len

        with prof_scope("rope_cache"):
            cos, sin = self._rope_cache(total_len, batch_size)

        token_history = None
        if any(layer.ple is not None for layer in self.layers):
            with prof_scope("token_history"):
                # Token ids and the n-gram hash live on the CPU (the PLE table is
                # host-resident), so keep the history there too.
                ids_cpu = input_ids.to("cpu", dtype=torch.long)
                if cache is not None and cache.ple_token_history is not None:
                    token_history = torch.cat([cache.ple_token_history, ids_cpu], dim=1)
                else:
                    pad = torch.full(
                        (batch_size, self.config.ngram_size - 1),
                        self.config.primary_eos_token_id,
                        dtype=torch.long,
                    )
                    token_history = torch.cat([pad, ids_cpu], dim=1)

        with prof_scope("embed"):
            hidden = self.embed_tokens(input_ids).to(self.dtype)
            hidden = hidden.repeat(1, 1, self.config.hc_count)

        for idx, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                cos=cos,
                sin=sin,
                cache=cache.layers[idx] if cache is not None else None,
                past_len=past_len,
                token_history=token_history,
                use_cache=cache is not None,
                profiler=self.profiler,
            )

        if cache is not None:
            cache.length = total_len
            if token_history is not None:
                cache.ple_token_history = token_history[:, -(self.config.ngram_size - 1) :].clone()

        if last_token_only:
            hidden = hidden[:, -1:]
        with prof_scope("final_mixer"):
            hidden = self.final_mixer(hidden)

        with prof_scope("lm_head"):
            return F.linear(hidden, self.lm_head)

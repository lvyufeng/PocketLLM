"""DSpark multi-token draft stages for DeepSeek-V4-Flash checkpoints.

The 0731 checkpoint ships a 3-stage speculative-decoding module under the mtp.*
namespace. Its per-stage body (attn / ffn / hc_*) is named exactly like the
repo's MTPBlock, so `Block` is reused verbatim; only the stage-specific heads
are new:

    stage 0     main_proj + main_norm       project concat(h40,h41,h42) -> dim
    stage n-1   norm + hc_head_* +          markov bias per draft position,
                markov_head + confidence    plus a confidence score per token

Reference: the checkpoint's own inference/model.py (DSparkBlock). This module
mirrors that file's math rather than inventing its own.

One round drafts `dspark_block_size` tokens from a single committed token, then
the main model verifies all of them in one forward. Verify cost grows with the
draft length, so drafting tokens that will be rejected is a net loss -- see
`DSparkGate` for the truncation rule and `docs/dspark.md` for the measurements.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

import src.models.deepseek_v4.runtime as rt
from src.kernels.ops import act_quant, sparse_attn
from src.models.deepseek_v4.runtime import (
    Attention,
    Block,
    Linear,
    ParallelEmbedding,
    ParallelHead,
    RMSNorm,
    apply_rotary_emb,
    set_dtype,
)


def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int):
    """Key indices each draft position attends to: the whole KV window, then the
    draft block's own keys. Every draft position sees the same set, which is why
    one row is expanded rather than built per position."""
    assert start_pos > 0
    matrix = torch.cat([
        torch.arange(min(window_size, start_pos + 1), device="cuda"),
        window_size + torch.arange(block_size, device="cuda"),
    ])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(Attention):
    """Attention for a DSpark draft stage.

    Deliberately overrides the whole forward rather than reusing Attention's:
    the draft block's keys and values come from the *main* model's hidden state
    (main_x), not from its own input, and only the query comes from x. The base
    class has no way to express that, and reusing it silently produces a draft
    that attends to the wrong thing.
    """

    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor):
        assert self.compress_ratio == 0, "DSpark stages are not compressed layers"
        bsz, seqlen, _ = main_x.size()
        win = self.window_size
        rd = self.rope_head_dim

        # KV for the committed positions, derived from the main model's hidden.
        main_freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)
        act_quant(main_kv[..., :-rd], 64, rt.scale_fmt, rt.scale_dtype, True)

        if start_pos == 0:
            # Prefill only fills the window; there is nothing to draft yet.
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = \
                    main_kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            return x

        bsz, block_size, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos + seqlen:start_pos + seqlen + block_size]

        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant(kv[..., :-rd], 64, rt.scale_fmt, rt.scale_dtype, True)

        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos)
        self.kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)
        kv = torch.cat([self.kv_cache[:bsz], kv], dim=1)
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))

    @torch.inference_mode()
    def write_main_kv(self, main_x: torch.Tensor, start_pos: int) -> None:
        """Write the main model's KV for positions start_pos .. start_pos+seqlen-1
        into the ring cache, without running attention.

        forward() only writes the single position it drafts from, which is all the
        reference's one-token-at-a-time loop ever needs. A real spec loop commits
        several positions per round, and each one has to land in the window or the
        next draft attends to stale slots.
        """
        bsz, seqlen, _ = main_x.size()
        win = self.window_size
        rd = self.rope_head_dim
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], self.freqs_cis[start_pos:start_pos + seqlen])
        act_quant(main_kv[..., :-rd], 64, rt.scale_fmt, rt.scale_dtype, True)
        if seqlen > win:
            # Only the last `win` positions survive in the ring anyway.
            main_kv = main_kv[:, -win:]
            start_pos += seqlen - win
            seqlen = win
        slots = (torch.arange(seqlen, device=main_kv.device) + start_pos) % win
        self.kv_cache[:bsz].index_copy_(1, slots, main_kv)


def _gather_vocab(logits: torch.Tensor) -> torch.Tensor:
    """Concatenate the per-rank vocab shards into full-vocab logits.

    ParallelHead.get_logits returns only this rank's slice; the reference's
    `full_logits=True` path does this all_gather. Without it a TP>1 argmax picks
    the best token out of 1/tp_world_size of the vocabulary.
    """
    if rt.tp_world_size <= 1:
        return logits
    shards = [torch.empty_like(logits) for _ in range(rt.tp_world_size)]
    dist.all_gather(shards, logits.contiguous())
    return torch.cat(shards, dim=-1)


class DSparkMarkovHead(nn.Module):
    """Rank-256 bigram bias: embed the previous draft token, project to vocab.

    Cheap enough to run per draft position, which is the point -- it lets each
    draft token condition on the one before it without a full block forward.
    """

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = ParallelEmbedding(vocab_size, markov_rank)
        # markov_w2 maps rank -> vocab. ParallelHead owns the vocab sharding, so
        # the gathered bias lines up with the gathered main-head logits.
        self.markov_w2 = ParallelHead(vocab_size, markov_rank)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.markov_w1(token_ids)
        logits = self.markov_w2.get_logits(embed.unsqueeze(1), keep_all_positions=True).squeeze(1)
        return _gather_vocab(logits), embed


class DSparkConfidenceHead(nn.Module):
    """Scalar accept-confidence per draft token, computed in fp32.

    The checkpoint stores proj in bf16; the reference implementation keeps the
    parameter in fp32 so the score itself is fp32, and this follows suit.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = Linear(input_dim, 1, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)


class DSparkBlock(Block):
    """One DSpark draft stage. The MoE half is inherited from Block unchanged;
    only the attention differs, because it reads its KV from the main model."""

    def __init__(self, layer_id: int, args, stage_id: int, n_stages: int):
        super().__init__(layer_id, args)
        # Block.__init__ built a plain Attention; swap in the DSpark one, which
        # takes main_x and keeps its own ring cache.
        self.attn = DSparkAttention(layer_id, args)
        self.stage_id = stage_id
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        hc_dim = self.hc_mult * args.dim

        if stage_id == 0:
            n_target = len(args.dspark_target_layer_ids)
            assert n_target > 0, "DSpark needs target layers"
            self.main_proj = Linear(args.dim * n_target, args.dim)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)

        if stage_id == n_stages - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)
            with set_dtype(torch.float32):
                self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))
                self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))
                self.hc_head_scale = nn.Parameter(torch.empty(1))

        # Tied to the main model's tables by the builder, not owned here.
        self.embed: ParallelEmbedding = None
        self.head: ParallelHead = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor,
                main_x: torch.Tensor) -> torch.Tensor:
        """Same hc/attn/ffn structure as Block.forward, but the attention call
        carries main_x. Block.forward cannot be reused because its attn call
        signature has no room for it."""
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, main_x)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return self.hc_post(x, residual, post, comb)

    @torch.inference_mode()
    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        """Turn the main model's hidden states into the draft block's input.

        Only position 0 gets a real token; the rest are the noise token, so the
        draft block predicts block_size positions from one committed token.
        Returns the draft ids too -- the MoE gate hashes them, so every stage
        needs them, not just the embedding.
        """
        assert self.embed is not None, "stage 0 needs the main model's embedding"
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size],
                                            self.noise_token_id)
        draft_input_ids[:, 0] = input_ids
        x = self.embed(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x, draft_input_ids

    @torch.inference_mode()
    def forward_head(self, x: torch.Tensor, input_ids: torch.Tensor,
                     temperature: float = 0.0):
        """Emit block_size draft tokens, each biased by its predecessor."""
        assert self.head is not None, "last stage needs the main model's head"
        x = self.head.hc_head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        logits = self.head.get_logits(self.norm(x), keep_all_positions=True)
        logits = _gather_vocab(logits)

        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, i])
            logits[:, i].add_(logits_bias)
            markov_embeds.append(markov_embed)
            if temperature <= 0.0:
                output_ids[:, i + 1] = logits[:, i].argmax(dim=-1)
            else:
                probs = torch.softmax(logits[:, i] / temperature, dim=-1)
                output_ids[:, i + 1] = torch.multinomial(probs, 1).squeeze(-1)

        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(x, markov_embed)
        return output_ids, logits, confidence

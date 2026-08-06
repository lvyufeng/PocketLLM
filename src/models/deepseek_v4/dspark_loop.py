"""DSpark speculative decoding loop with adaptive draft-length gating.

Assembles the pieces: build the draft stages onto a trunk (`attach_dspark`),
then run rounds of draft -> gate -> verify -> commit (`DSparkLoop`).

Greedy verification only. A round drafts up to `dspark_block_size` tokens, the
main model verifies them in one forward, and the longest prefix whose greedy
argmax matches the draft is committed along with one bonus token (the main
model's own prediction at the first mismatch), so the output is identical to
plain greedy decoding regardless of how many tokens the draft got right -- or
how many the gate chose to submit.

Requires the small-batch FP4 MoE kernel (DEEPSEEK_GPU_MOE_MULTI_TOKEN_FP4=1) to
be worth running at all: without it a multi-token verify falls into the prefill
grouped MoE path and costs ~3.3s instead of ~850ms, which puts speculation at
0.37x plain decode. See docs/dspark.md.
"""
from __future__ import annotations

import torch

from src.models.deepseek_v4.dspark import DSparkBlock
from src.models.deepseek_v4.dspark_gate import DSparkGate


def attach_dspark(model, args) -> None:
    """Replace the trunk's MTP list with the checkpoint's DSpark stages.

    The trunk must be built with n_mtp_layers=0; DSpark stages live in the same
    mtp.* namespace and would otherwise collide with MTPBlocks.
    """
    n_stages = args.n_dspark_stages
    if n_stages <= 0:
        raise ValueError("attach_dspark needs args.n_dspark_stages > 0")
    if len(model.mtp) > 0:
        raise ValueError(
            "trunk already has MTP blocks; build it with n_mtp_layers=0 so the "
            "DSpark stages can own the mtp.* namespace")
    # Stages are uncompressed layers; the trunk's compress_ratios list is indexed
    # by layer id, so it has to cover theirs too.
    args.compress_ratios = list(args.compress_ratios) + [0] * n_stages
    model.mtp = torch.nn.ModuleList()
    for stage_id in range(n_stages):
        blk = DSparkBlock(args.n_layers + stage_id, args, stage_id, n_stages)
        # embed/head are tied to the main model's tables, not stored per stage.
        blk.embed = model.embed
        blk.head = model.head
        model.mtp.append(blk)
    model.dspark_target_layer_ids = tuple(args.dspark_target_layer_ids)


def dspark_state_dict_filter(model, args):
    """Wrap model.state_dict so the loader's completeness check ignores the
    tied embed/head of stages 1..n-1.

    The checkpoint stores those tables once. The loader pre-registers only
    mtp.0's pair as satisfied, so without this the later stages look unloaded.
    Returns a restore callable.
    """
    real = model.state_dict

    def filtered(*a, **kw):
        sd = real(*a, **kw)
        for s in range(1, args.n_dspark_stages):
            sd.pop(f"mtp.{s}.embed.weight", None)
            sd.pop(f"mtp.{s}.head.weight", None)
        return sd

    model.state_dict = filtered

    def restore():
        model.state_dict = real

    return restore


class DSparkLoop:
    """Runs gated DSpark speculative decoding against a loaded trunk.

    One instance per generation stream: it owns the gate's learned calibration
    and the running position, neither of which is shareable.
    """

    def __init__(self, model, args, gate: DSparkGate | None = None):
        if len(model.mtp) == 0:
            raise ValueError("call attach_dspark(model, args) first")
        self.model = model
        self.args = args
        self.block_size = args.dspark_block_size
        self.target_layer_ids = tuple(args.dspark_target_layer_ids)
        self.gate = gate if gate is not None else DSparkGate.from_env(self.block_size)
        self.rounds = 0
        self.accepted_tokens = 0
        self.drafted_tokens = 0

    @torch.inference_mode()
    def _main_forward(self, tokens: torch.Tensor, start_pos: int, want_hidden: bool):
        """Main model forward; optionally capture the DSpark target-layer hiddens.

        The hiddens are mean-pooled over the hc dimension, matching what
        DSparkBlock.main_proj was trained to consume.
        """
        captured: dict[int, torch.Tensor] = {}
        handles = []
        if want_hidden:
            def make_hook(idx):
                def hook(_mod, _inp, out):
                    captured[idx] = out.mean(dim=2)
                return hook

            for idx in self.target_layer_ids:
                handles.append(self.model.layers[idx].register_forward_hook(make_hook(idx)))
        try:
            logits = self.model(tokens, start_pos, keep_all_positions=True)
        finally:
            for h in handles:
                h.remove()

        greedy = logits.argmax(dim=-1)
        hidden = None
        if want_hidden and captured:
            hidden = torch.cat([captured[i] for i in self.target_layer_ids], dim=-1)
        return greedy, hidden

    @torch.inference_mode()
    def _write_draft_kv(self, main_hidden: torch.Tensor, start_pos: int) -> None:
        """Fill every stage's KV window from the main model's hiddens.

        Every committed position must land in the ring, not just the one drafted
        from, or the next round's draft attends to stale slots.
        """
        stage0 = self.model.mtp[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))
        for stage in self.model.mtp:
            stage.attn.write_main_kv(main_x, start_pos)

    @torch.inference_mode()
    def _draft(self, main_hidden: torch.Tensor, committed_id: torch.Tensor, start_pos: int):
        """Run all stages; returns (draft_ids [b, block_size], confidence)."""
        stage0 = self.model.mtp[0]
        x, main_x, draft_input_ids = stage0.forward_embed(main_hidden, committed_id)
        for stage in self.model.mtp:
            x = stage(x, start_pos, draft_input_ids, main_x)
        output_ids, _logits, confidence = self.model.mtp[-1].forward_head(x, committed_id)
        return output_ids[:, 1:], confidence

    @torch.inference_mode()
    def prefill(self, tokens: torch.Tensor):
        """Prefill the prompt and prime the draft stages.

        Returns (first_token, position) where first_token is the main model's
        greedy continuation, exactly as plain decoding would produce.
        """
        greedy, main_hidden = self._main_forward(tokens, 0, want_hidden=True)
        self._write_draft_kv(main_hidden, 0)
        self._pos = tokens.size(1)
        self._committed = greedy[:, -1:]
        self._hidden = main_hidden[:, -1:]
        return self._committed, self._pos

    @torch.inference_mode()
    def step(self):
        """One speculative round. Returns (new_tokens [b, n], n_drafted).

        n_drafted is 0 when the gate chose a plain single-token decode, which is
        the same output path with none of the draft submitted for verification.
        """
        n = self.gate.choose_draft_len(self.gate_confidence_source())
        if n == 0:
            greedy, hidden = self._main_forward(self._committed, self._pos, want_hidden=True)
            self._write_draft_kv(hidden, self._pos)
            out = self._committed.clone()
            self._pos += 1
            self._committed = greedy[:, :1]
            self._hidden = hidden[:, :1]
            self.gate.note_skipped_round()
            self.rounds += 1
            return out, 0

        draft_ids = self._pending_draft[:, :n]
        verify_in = torch.cat([self._committed, draft_ids], dim=1)
        v_greedy, v_hidden = self._main_forward(verify_in, self._pos, want_hidden=True)

        # Longest matching prefix. Position i of the verify output is the main
        # model's prediction for the token after draft position i-1, so draft
        # token i is accepted iff v_greedy[i] == draft_ids[i].
        n_ok = 0
        for i in range(n):
            if not bool((draft_ids[:, i] == v_greedy[:, i]).all()):
                break
            n_ok += 1

        # Commit the accepted prefix plus the main model's own next token. In
        # exact arithmetic the bonus token makes this equivalent to plain greedy
        # decode; measured on TP=4 FP4 it is not, because the multi-token verify
        # forward is not even run-to-run reproducible (docs/dspark.md,
        # "Output determinism"). That divergence is a property of the batched
        # attention path, not of this loop or of gating.
        out = torch.cat([self._committed, draft_ids[:, :n_ok]], dim=1)
        self._write_draft_kv(v_hidden[:, :n_ok + 1], self._pos)
        self._pos += n_ok + 1
        self._committed = v_greedy[:, n_ok:n_ok + 1]
        self._hidden = v_hidden[:, n_ok:n_ok + 1]

        self.gate.observe(self._pending_conf, n_ok, n)
        self.rounds += 1
        self.accepted_tokens += n_ok
        self.drafted_tokens += n
        return out, n

    @torch.inference_mode()
    def gate_confidence_source(self):
        """Draft now so the gate can score it, caching the result for `step`.

        The confidence score is produced *by* the draft, so the draft always runs
        before the gate decides -- gating can only avoid the verify cost, never
        the draft cost. This is why a skipped round still costs draft_ms and the
        gate cannot fully close the gap to plain decode on a hard workload.
        """
        self._pending_draft, conf = self._draft(self._hidden, self._committed[:, 0],
                                                self._pos - 1)
        self._pending_conf = [float(c) for c in conf[0]]
        return self._pending_conf

    def stats(self) -> dict:
        accept_rate = (self.accepted_tokens / self.drafted_tokens
                       if self.drafted_tokens else None)
        return {
            "rounds": self.rounds,
            "accepted_tokens": self.accepted_tokens,
            "drafted_tokens": self.drafted_tokens,
            "accept_rate": round(accept_rate, 4) if accept_rate is not None else None,
            "gate": self.gate.stats(),
        }

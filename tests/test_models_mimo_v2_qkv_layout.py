"""Row order of the released fused `qkv_proj`.

`attention_projection_layout` is `fused_qkv`, so a layer reads one projection
whose output holds query, key and value concatenated. The config says the
projection is fused; it does not say in which order the rows are stored, and
that is a silent failure: every candidate order yields tensors of the right
shape, and only the logits say which one is right.

Two orders are in play. The release's own `MiMoV2Attention` splits the fused
output contiguously as `[q | k | v]`. The serving stack's loader instead takes a
*contiguous quarter* of the stored tensor and hands it to a parameter laid out
as `[q_rank | k_rank | v_rank]`, and refuses to run at any attention
tensor-parallel size other than 4 -- which is only correct if the stored tensor
is four equal groups of `[q_i | k_i | v_i]`. The released weights are the second:
the tensor's row-magnitude profile turns over once every 3392 rows on a global
layer and every 3712 on a windowed one, which is exactly one quarter of it, and
the low stretch is the layer's value block.

These tests pin the arithmetic of every reading on synthetic tensors whose rows
are tagged by the section they came from. Which reading a given checkpoint uses
is a property of the file and is checked against the released weights in
`test_models_mimo_v2_loader.py`.
"""

from __future__ import annotations

import pytest
import torch

from src.models.mimo_v2.config import MimoV2TextConfig
from src.models.mimo_v2.layers import split_fused_qkv

#: The released geometry, for the two attention families. A global-attention
#: layer fuses 64 query heads of width 192 with 4 key/value heads of width
#: 192/128; a sliding-window layer fuses the same queries with 8 key/value heads.
RELEASED = {
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 192,
    "v_head_dim": 128,
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 8,
    "swa_head_dim": 192,
    "swa_v_head_dim": 128,
    "attention_projection_layout": "fused_qkv",
    "hybrid_layer_pattern": [0, 1] + [1] * 46,
}
SECTIONS = {"global": (12288, 768, 512), "sliding": (12288, 1536, 1024)}
LAYER_OF = {"global": 0, "sliding": 1}
TAG = {"q": 1.0, "k": 2.0, "v": 3.0}
LAYOUTS = ("tp4_interleaved", "tp4_interleaved_vk")


def released(family: str):
    """The released attention shape for a layer of this family."""
    return MimoV2TextConfig.from_dict(RELEASED).attention(LAYER_OF[family])


def tagged(q: int, k: int, v: int, groups: int, order: tuple[str, ...]) -> torch.Tensor:
    """A fused projection output in which every row is filled with its section's tag.

    A wrong cut then shows up as a section holding another section's value,
    rather than as a shape error. The projection's output is its last axis, the
    way `F.linear` leaves it.
    """
    widths = {"q": q // groups, "k": k // groups, "v": v // groups}
    return torch.cat(
        [torch.full((widths[name],), TAG[name]) for _ in range(groups) for name in order]
    )


@pytest.mark.parametrize("family", ["global", "sliding"])
def test_the_released_sections_are_what_the_families_need(family):
    q, k, v = SECTIONS[family]
    shape = released(family)
    assert (shape.q_size, shape.k_size, shape.v_size) == (q, k, v)
    assert shape.qkv_out == q + k + v


def test_a_fused_projection_defaults_to_the_row_order_the_serving_stack_needs():
    config = MimoV2TextConfig.from_dict(RELEASED)
    assert config.resolved_projection_layout == "fused_qkv"
    assert config.resolved_qkv_row_layout == "tp4_interleaved"
    assert config.attention(0).qkv_row_layout == "tp4_interleaved"


def test_a_split_projection_has_no_row_order_to_get_wrong():
    config = MimoV2TextConfig.from_dict({**RELEASED, "attention_projection_layout": "split"})
    assert config.resolved_qkv_row_layout == "contiguous"


def test_the_row_order_can_be_stated_outright():
    """The field exists so a fixture built in the other order can say so."""
    config = MimoV2TextConfig.from_dict(
        {**RELEASED, "attention_qkv_row_layout": "contiguous"}
    )
    assert config.resolved_qkv_row_layout == "contiguous"


def test_a_row_order_that_is_not_one_of_the_readings_is_a_problem():
    bad = MimoV2TextConfig.from_dict({**RELEASED, "attention_qkv_row_layout": "quarter"})
    assert any("row layout" in problem for problem in bad.verify())


@pytest.mark.parametrize("family", ["global", "sliding"])
def test_contiguous_reading_cuts_once(family):
    q, k, v = SECTIONS[family]
    fused = tagged(q, k, v, groups=1, order=("q", "k", "v"))
    query, key, value = split_fused_qkv(fused, released(family), "contiguous")
    assert (query.shape[0], key.shape[0], value.shape[0]) == (q, k, v)
    assert query.unique().tolist() == [TAG["q"]]
    assert key.unique().tolist() == [TAG["k"]]
    assert value.unique().tolist() == [TAG["v"]]


@pytest.mark.parametrize("family", ["global", "sliding"])
def test_interleaved_reading_regroups_the_four_quarters(family):
    q, k, v = SECTIONS[family]
    fused = tagged(q, k, v, groups=4, order=("q", "k", "v"))
    assert fused.shape[0] == q + k + v
    query, key, value = split_fused_qkv(fused, released(family), "tp4_interleaved")
    assert (query.shape[0], key.shape[0], value.shape[0]) == (q, k, v)
    assert query.unique().tolist() == [TAG["q"]]
    assert key.unique().tolist() == [TAG["k"]]
    assert value.unique().tolist() == [TAG["v"]]


@pytest.mark.parametrize("family", ["global", "sliding"])
def test_the_interleaved_reading_recovers_the_contiguous_one_rung_by_rung(family):
    """The two readings read the same rows; only the cut differs.

    Numbering every row of a contiguous tensor, regrouping it into four quarters
    the way the serving stack's loader implies, and then cutting it has to recover
    exactly the rows the contiguous cut calls `q`, `k` and `v`, in that order.
    """
    q, k, v = SECTIONS[family]
    shape = released(family)
    contiguous = torch.arange(float(q + k + v))
    cq, ck, cv = contiguous[:q], contiguous[q : q + k], contiguous[q + k :]

    q_rank, k_rank, v_rank = q // 4, k // 4, v // 4
    grouped = torch.cat(
        [
            torch.cat(
                [
                    cq[r * q_rank : (r + 1) * q_rank],
                    ck[r * k_rank : (r + 1) * k_rank],
                    cv[r * v_rank : (r + 1) * v_rank],
                ]
            )
            for r in range(4)
        ]
    )

    query, key, value = split_fused_qkv(grouped, shape, "tp4_interleaved")
    assert torch.equal(query, cq)
    assert torch.equal(key, ck)
    assert torch.equal(value, cv)


@pytest.mark.parametrize("family", ["global", "sliding"])
def test_the_swap_of_the_two_wide_sections_is_not_a_no_op(family):
    """`tp4_interleaved_vk` reads the same four quarters with the wide sections swapped.

    Both orders are shape-valid whenever the key and value sections differ in
    width, so the only thing worth pinning is that the second one really does
    exchange them rather than quietly falling back to the first.
    """
    q, k, v = SECTIONS[family]
    shape = released(family)
    fused = tagged(q, k, v, groups=4, order=("q", "k", "v"))
    _, key_kv, value_kv = split_fused_qkv(fused, shape, "tp4_interleaved")
    _, key_vk, value_vk = split_fused_qkv(fused, shape, "tp4_interleaved_vk")
    assert (key_kv.shape[0], value_kv.shape[0]) == (k, v)
    assert (key_vk.shape[0], value_vk.shape[0]) == (k, v)
    # Under the swapped reading the section the first order calls `k` is read as
    # the value, so the two disagree on both.
    assert not torch.equal(key_kv, key_vk)
    assert not torch.equal(value_kv, value_vk)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_width_the_layout_cannot_tile_is_refused(layout):
    shape = released("global")
    with pytest.raises(ValueError, match="not 4 groups"):
        split_fused_qkv(torch.zeros(shape.qkv_out - 1), shape, layout)


def test_an_unknown_layout_is_refused():
    shape = released("global")
    with pytest.raises(ValueError, match="unknown qkv row layout"):
        split_fused_qkv(torch.zeros(shape.qkv_out), shape, "quarter")

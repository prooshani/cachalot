"""Contracts of the DSpark draft head that do not need the checkpoint.

The draft's arithmetic is scored end to end by benchmarks/dspark_acceptance.py,
which needs the model. What is pinned here is the part that is easy to get
silently wrong and impossible to notice afterwards: which keys a draft position
attends to, and which checkpoint tensors belong to the draft at all. An
off-by-one in either produces a draft that still runs, still emits tokens, and
quietly accepts at a lower rate -- a false null rather than a crash.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from cachalot.model.dspark_draft import (
    BLOCK_SIZE,
    N_MTP_LAYERS,
    WINDOW_SIZE,
    dspark_topk_idxs,
    is_mtp_expert_tensor,
    is_mtp_tensor,
)
from cachalot.model.text_decode_runtime import MTP_TARGET_LAYERS, DecodeResult


def test_target_layers_match_the_checkpoint_config():
    # text_config.dspark_target_layer_ids, and one draft stage per target layer.
    assert MTP_TARGET_LAYERS == (37, 38, 39)
    assert len(MTP_TARGET_LAYERS) == N_MTP_LAYERS


def test_decode_result_carries_no_main_hidden_by_default():
    result = DecodeResult(
        logits=mx.zeros((4,)),
        hidden=mx.zeros((4,)),
        routes=(),
        position=0,
    )
    assert result.main_hidden is None


@pytest.mark.parametrize("start_pos", [1, 5, 127])
def test_topk_covers_exactly_the_filled_window_before_it_wraps(start_pos):
    idxs = dspark_topk_idxs(start_pos)
    assert idxs.shape == (BLOCK_SIZE, start_pos + 1 + BLOCK_SIZE)
    row = idxs[0].tolist()
    # Slots 0..start_pos hold positions 0..start_pos, and nothing else is filled.
    assert row[: start_pos + 1] == list(range(start_pos + 1))
    assert row[start_pos + 1 :] == list(range(WINDOW_SIZE, WINDOW_SIZE + BLOCK_SIZE))


@pytest.mark.parametrize("start_pos", [128, 129, 1000])
def test_topk_covers_the_whole_window_once_it_is_full(start_pos):
    idxs = dspark_topk_idxs(start_pos)
    assert idxs.shape == (BLOCK_SIZE, WINDOW_SIZE + BLOCK_SIZE)
    row = idxs[0].tolist()
    assert row == list(range(WINDOW_SIZE + BLOCK_SIZE))


def test_every_draft_position_attends_to_the_whole_block():
    """Bidirectional inside the block: no query may be given a causal prefix."""
    idxs = dspark_topk_idxs(64)
    rows = [idxs[i].tolist() for i in range(BLOCK_SIZE)]
    assert all(row == rows[0] for row in rows)
    block = list(range(WINDOW_SIZE, WINDOW_SIZE + BLOCK_SIZE))
    assert all(set(block).issubset(row) for row in rows)


def test_topk_rejects_a_negative_position():
    with pytest.raises(ValueError):
        dspark_topk_idxs(-1)


def test_draft_tensors_are_recognised_and_experts_separated():
    assert is_mtp_tensor("mtp.0.attn.wq_a.weight")
    assert is_mtp_tensor("mtp.2.markov_head.head.weight")
    assert not is_mtp_tensor("layers.37.attn.wq_a.weight")
    assert not is_mtp_tensor("head.weight")

    assert is_mtp_expert_tensor("mtp.1.ffn.experts.17.w2.scale")
    assert not is_mtp_expert_tensor("mtp.1.ffn.shared_experts.w2.scale")
    assert not is_mtp_expert_tensor("layers.3.ffn.experts.17.w2.scale")

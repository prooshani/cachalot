"""Repetition controls in generation.sample_logits.

These exist because of a measured failure: on the 2-bit expert bank, free-running
generation on code prompts falls into a repeating loop -- 828 tokens of period 2
in one measured reply -- which the teacher-forced quality gate cannot see.
What matters is that the penalties are off by default, that the frequency
penalty grows with the count (a classic once-per-token penalty does not break a
confident loop), and that a banned n-gram is actually unreachable.
"""
from __future__ import annotations

import mlx.core as mx

from cachalot.model.generation import SamplingParams, sample_logits


def _logits(values: list[float]) -> mx.array:
    return mx.array(values, dtype=mx.float32)


def test_penalties_are_off_by_default():
    params = SamplingParams()
    assert params.frequency_penalty == 0.0
    assert params.presence_penalty == 0.0
    assert params.no_repeat_ngram_size == 0


def test_no_penalty_leaves_the_official_path_untouched():
    logits = _logits([0.0, 10.0, 0.0])
    # Greedy is deterministic, so this pins the unpenalized choice.
    assert sample_logits(logits, temperature=0) == 1
    assert sample_logits(logits, temperature=0, recent=[1, 1, 1, 1]) == 1


def test_frequency_penalty_grows_with_the_count():
    """The property that breaks a loop: ten occurrences must cost ten times one."""
    logits = _logits([0.0, 5.0, 0.0])

    # One occurrence is not enough to overturn a 5.0 lead at penalty 1.0.
    assert sample_logits(logits, temperature=0, recent=[1], frequency_penalty=1.0) == 1
    # Ten are.
    assert sample_logits(logits, temperature=0, recent=[1] * 10, frequency_penalty=1.0) != 1


def test_presence_penalty_is_flat_however_often_the_token_appeared():
    logits = _logits([0.0, 5.0, 0.0])
    once = [1]
    many = [1] * 50
    # A flat 1.0 never overturns a 5.0 lead, no matter the count.
    assert sample_logits(logits, temperature=0, recent=once, presence_penalty=1.0) == 1
    assert sample_logits(logits, temperature=0, recent=many, presence_penalty=1.0) == 1
    # A flat 6.0 does, and it does so at one occurrence.
    assert sample_logits(logits, temperature=0, recent=once, presence_penalty=6.0) != 1


def test_no_repeat_ngram_bans_the_completion_of_a_seen_ngram():
    # History "0 1 2", and we are about to extend "1 2" -> banning bigram-completion
    # of the context must make token 2 unreachable after "1".
    logits = _logits([0.0, 0.0, 100.0])
    recent = [0, 1, 2, 1]
    # size 2 bans any token that followed the last token (1) before: that is 2.
    assert sample_logits(logits, temperature=0, recent=recent, no_repeat_ngram_size=2) != 2


def test_a_banned_token_stays_unreachable_under_sampling():
    logits = _logits([0.0, 0.0, 100.0])
    recent = [0, 1, 2, 1]
    picks = {
        sample_logits(
            logits, temperature=1.0, recent=recent, no_repeat_ngram_size=2
        )
        for _ in range(40)
    }
    assert 2 not in picks


def test_penalties_compose_without_removing_every_option():
    logits = _logits([1.0, 2.0, 3.0, 4.0])
    token = sample_logits(
        logits,
        temperature=0,
        recent=[3] * 20,
        frequency_penalty=0.5,
        presence_penalty=0.5,
    )
    assert token in (0, 1, 2)

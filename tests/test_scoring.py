from __future__ import annotations

from memory_engine.scoring import score_importance


def test_low_signal_message_scores_low() -> None:
    assert score_importance("Hi") < 0.55


def test_high_signal_message_scores_high() -> None:
    assert score_importance("I study AP Physics and my goal is to pass") >= 0.55

from __future__ import annotations


def score_importance(message: str, context: str = "") -> float:
    text = message.lower().strip()
    if len(text) <= 3:
        return 0.1

    low_signal = {"hi", "hello", "thanks", "ok", "cool", "lol", "bye"}
    if text in low_signal:
        return 0.1

    high_signal_keywords = {
        "i am",
        "i'm",
        "i prefer",
        "my",
        "goal",
        "deadline",
        "study",
        "project",
        "task",
        "remember",
    }

    score = 0.35
    if any(token in text for token in high_signal_keywords):
        score += 0.4
    if any(char.isdigit() for char in text):
        score += 0.1
    if context:
        score += 0.05
    return min(score, 1.0)

"""Shared helpers for the agentic banking clustering project."""

from __future__ import annotations

import math
import re
from collections import defaultdict


def normalize_text(value: str) -> str:
    """Normalize a short text for stable comparisons."""
    value = str(value).strip().lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я\s-]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_list(value: object) -> list[str]:
    """Return a clean list even if the model sends a scalar value."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def token_overlap(left: str, right: str) -> float:
    """Compute token overlap relative to the smaller set."""
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def choose_weighted_label(candidates: list[tuple[str, int]]) -> str:
    """Choose a stable label from weighted candidates."""
    scores: dict[str, int] = defaultdict(int)
    original_forms: dict[str, str] = {}

    for label, weight in candidates:
        normalized_label = normalize_text(label)
        if not normalized_label:
            continue
        scores[normalized_label] += weight
        original_forms.setdefault(normalized_label, str(label).strip())

    if not scores:
        return "Не определено"

    winner = min(
        scores,
        key=lambda normalized_label: (
            normalized_label == "не определено",
            -scores[normalized_label],
            len(normalized_label.split()),
            len(normalized_label),
        ),
    )
    return original_forms.get(winner, "Не определено") or "Не определено"

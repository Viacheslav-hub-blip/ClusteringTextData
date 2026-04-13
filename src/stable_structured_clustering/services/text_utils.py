"""Shared text and similarity helpers."""

from __future__ import annotations

import math
import re


def normalize_text(value: str) -> str:
    """Normalize a short text for stable comparisons."""
    value = value.strip().lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я\s-]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_list(value: object) -> list[str]:
    """Return a clean list even if the model returns a scalar value."""
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


def is_undefined(value: str) -> bool:
    """Check if a label is effectively undefined."""
    return normalize_text(value) in {"", "не определено"}

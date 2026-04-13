"""Utility helpers for text normalization and similarity."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Iterable


UNDEFINED_LABEL = "Не определено"


def normalize_text(value: str) -> str:
    """Normalize short text for stable comparisons."""
    value = value.strip().lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я\s-]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_list(value: object) -> list[str]:
    """Return a clean list from scalar or list input."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def is_undefined(value: str) -> bool:
    """Check whether the label is effectively undefined."""
    return normalize_text(value) in {"", normalize_text(UNDEFINED_LABEL)}


def token_overlap(left: str, right: str) -> float:
    """Compute overlap relative to the smaller token set."""
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


def best_label(candidates: Iterable[tuple[str, int]]) -> str:
    """Choose a stable human-readable label from weighted candidates."""
    scores: dict[str, int] = defaultdict(int)
    original_forms: dict[str, str] = {}

    for label, weight in candidates:
        normalized = normalize_text(label)
        if not normalized:
            continue
        scores[normalized] += int(weight)
        original_forms.setdefault(normalized, label.strip())

    if not scores:
        return UNDEFINED_LABEL

    winner = min(
        scores,
        key=lambda item: (
            is_undefined(item),
            -scores[item],
            len(item.split()),
            len(item),
        ),
    )
    return original_forms.get(winner, UNDEFINED_LABEL) or UNDEFINED_LABEL


def best_key(candidates: Iterable[tuple[str, int]]) -> str:
    """Choose a stable machine key from weighted key candidates."""
    scores: dict[str, int] = defaultdict(int)
    for label, weight in candidates:
        normalized = normalize_text(label)
        if normalized:
            scores[normalized] += int(weight)
    if not scores:
        return normalize_text(UNDEFINED_LABEL)
    return min(
        scores,
        key=lambda item: (
            item == normalize_text(UNDEFINED_LABEL),
            -scores[item],
            len(item.split()),
            len(item),
        ),
    )


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    """Deduplicate while preserving the first occurrence order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def dump_json(data: object) -> str:
    """Serialize a payload in readable UTF-8 JSON."""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)

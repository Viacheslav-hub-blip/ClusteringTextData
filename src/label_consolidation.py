"""Embedding-based consolidation for semantically duplicate labels."""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Callable, TypeVar

from langchain_core.embeddings import Embeddings

ItemT = TypeVar("ItemT")


class SemanticLabelConsolidator:
    """Collapse near-duplicate labels without using a fixed category list."""

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings

    @staticmethod
    def _normalize_text(value: str) -> str:
        value = value.strip().lower().replace("ё", "е")
        value = re.sub(r"[^0-9a-zа-я\s-]+", " ", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    @classmethod
    def _token_overlap(cls, left: str, right: str) -> float:
        left_tokens = set(cls._normalize_text(left).split())
        right_tokens = set(cls._normalize_text(right).split())
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        denominator = min(len(left_tokens), len(right_tokens))
        if denominator == 0:
            return 0.0
        return intersection / denominator

    def consolidate(
        self,
        items: list[ItemT],
        *,
        item_id_getter: Callable[[ItemT], str],
        label_getter: Callable[[ItemT], str],
        semantic_text_getter: Callable[[ItemT], str],
        family_key_getter: Callable[[ItemT], str],
        size_getter: Callable[[ItemT], int],
        min_similarity: float,
        min_token_overlap: float,
    ) -> dict[str, str]:
        """Return item_id -> canonical label mapping."""
        if len(items) < 2:
            return {}

        groups_by_family: dict[str, list[ItemT]] = defaultdict(list)
        for item in items:
            family_key = self._normalize_text(family_key_getter(item)) or "__global__"
            groups_by_family[family_key].append(item)

        reconciled: dict[str, str] = {}
        for family_items in groups_by_family.values():
            if len(family_items) < 2:
                continue

            item_ids = [item_id_getter(item) for item in family_items]
            labels = [str(label_getter(item)).strip() for item in family_items]
            normalized_labels = [self._normalize_text(label) for label in labels]
            texts = [semantic_text_getter(item) for item in family_items]
            embeddings = self._embeddings.embed_documents(texts)
            parent = {item_id: item_id for item_id in item_ids}

            def find(node_id: str) -> str:
                while parent[node_id] != node_id:
                    parent[node_id] = parent[parent[node_id]]
                    node_id = parent[node_id]
                return node_id

            def union(left_id: str, right_id: str) -> None:
                left_root = find(left_id)
                right_root = find(right_id)
                if left_root != right_root:
                    parent[right_root] = left_root

            for left_index, left_id in enumerate(item_ids):
                for right_index in range(left_index + 1, len(item_ids)):
                    right_id = item_ids[right_index]
                    same_normalized_label = (
                        normalized_labels[left_index]
                        and normalized_labels[left_index] == normalized_labels[right_index]
                    )
                    token_overlap = self._token_overlap(
                        labels[left_index],
                        labels[right_index],
                    )
                    similarity = self._cosine_similarity(
                        embeddings[left_index],
                        embeddings[right_index],
                    )
                    if same_normalized_label or (
                        similarity >= min_similarity and token_overlap >= min_token_overlap
                    ):
                        union(left_id, right_id)

            components: dict[str, list[ItemT]] = defaultdict(list)
            items_by_id = {item_id_getter(item): item for item in family_items}
            for item_id in item_ids:
                components[find(item_id)].append(items_by_id[item_id])

            for component in components.values():
                if len(component) < 2:
                    continue
                best_item = min(
                    component,
                    key=lambda item: (
                        self._normalize_text(label_getter(item)) == "не определено",
                        -size_getter(item),
                        len(label_getter(item).split()),
                        len(label_getter(item)),
                    ),
                )
                canonical_label = str(label_getter(best_item)).strip() or "Не определено"
                for item in component:
                    reconciled[item_id_getter(item)] = canonical_label

        return reconciled

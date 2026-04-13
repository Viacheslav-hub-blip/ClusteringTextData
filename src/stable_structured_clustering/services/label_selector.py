"""Deterministic label selection for stable clusters."""

from __future__ import annotations

from collections import defaultdict

from ..models import ParentCluster, SpecificCluster, SpecificPrototype
from .text_utils import is_undefined, normalize_text


class LabelSelector:
    """Choose stable labels from cluster members instead of generating new text."""

    @staticmethod
    def _best_label(candidates: list[tuple[str, int]]) -> str:
        """Choose a canonical label from weighted candidates."""
        scores: dict[str, int] = defaultdict(int)
        original_forms: dict[str, str] = {}

        for label, weight in candidates:
            normalized_label = normalize_text(label)
            if not normalized_label:
                continue
            scores[normalized_label] += weight
            original_forms.setdefault(normalized_label, label.strip())

        if not scores:
            return "Не определено"

        normalized_winner = min(
            scores,
            key=lambda normalized_label: (
                is_undefined(normalized_label),
                -scores[normalized_label],
                len(normalized_label.split()),
                len(normalized_label),
            ),
        )
        return original_forms.get(normalized_winner, "Не определено") or "Не определено"

    def assign_labels(
        self,
        specific_clusters: list[SpecificCluster],
        parent_clusters_by_id: dict[str, ParentCluster],
        prototypes_by_id: dict[str, SpecificPrototype],
    ) -> None:
        """Assign stable specific and parent labels in-place."""
        specific_by_id = {
            cluster.specific_cluster_id: cluster for cluster in specific_clusters
        }

        for cluster in specific_clusters:
            label_candidates: list[tuple[str, int]] = []
            for prototype_id in cluster.prototype_ids:
                prototype = prototypes_by_id[prototype_id]
                signal = prototype.representative_signal
                label_candidates.append(
                    (signal.specific_focus, len(prototype.member_comment_ids))
                )
            cluster.specific_group = self._best_label(label_candidates)

        for parent_cluster in parent_clusters_by_id.values():
            label_candidates = []
            for specific_cluster_id in parent_cluster.child_specific_cluster_ids:
                specific_cluster = specific_by_id[specific_cluster_id]
                prototype = prototypes_by_id[specific_cluster.representative_prototype_id]
                signal = prototype.representative_signal
                label_candidates.append(
                    (signal.parent_focus, len(specific_cluster.member_comment_ids))
                )
            parent_cluster.parent_group = self._best_label(label_candidates)

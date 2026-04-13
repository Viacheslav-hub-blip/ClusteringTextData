"""Deterministic specific clustering over structured prototypes."""

from __future__ import annotations

from collections import defaultdict
import logging

from langchain_core.embeddings import Embeddings

from ..models import SpecificCluster, SpecificPrototype
from .candidate_retriever import CandidateRetriever
from .text_utils import cosine_similarity, token_overlap

logger = logging.getLogger(__name__)


class SpecificClusterBuilder:
    """Build specific clusters from structured prototypes."""

    def __init__(self, embeddings: Embeddings, top_k: int = 8):
        self._retriever = CandidateRetriever(embeddings=embeddings, top_k=top_k)

    @staticmethod
    def _signature_text(prototype: SpecificPrototype) -> str:
        """Build a stable signature text for similarity search."""
        signal = prototype.representative_signal
        return " | ".join(
            [
                signal.polarity,
                signal.phenomenon,
                signal.subject,
                signal.parent_focus,
                signal.parent_key,
                signal.specific_focus,
                signal.specific_key,
                " ".join(signal.material_details),
                " ".join(signal.entities),
            ]
        )

    @staticmethod
    def _should_merge(
        left: SpecificPrototype,
        right: SpecificPrototype,
        similarity: float,
    ) -> bool:
        """Apply conservative merge rules for exact-case clusters."""
        left_signal = left.representative_signal
        right_signal = right.representative_signal

        if left_signal.is_meaningful != right_signal.is_meaningful:
            return False
        if not left_signal.is_meaningful:
            return True
        if left_signal.polarity != right_signal.polarity:
            return False
        if left_signal.specific_key == right_signal.specific_key:
            return True

        parent_overlap = token_overlap(left_signal.parent_key, right_signal.parent_key)
        phenomenon_overlap = token_overlap(left_signal.phenomenon, right_signal.phenomenon)
        focus_overlap = token_overlap(left_signal.specific_focus, right_signal.specific_focus)

        return (
            similarity >= 0.90
            and focus_overlap >= 0.50
            and (parent_overlap >= 0.50 or phenomenon_overlap >= 0.50)
        )

    def build(
        self,
        prototypes: list[SpecificPrototype],
    ) -> tuple[list[SpecificCluster], dict[str, str]]:
        """Build specific clusters and return prototype-to-cluster mapping."""
        if not prototypes:
            return [], {}

        parent = {prototype.prototype_id: prototype.prototype_id for prototype in prototypes}
        prototypes_by_id = {prototype.prototype_id: prototype for prototype in prototypes}
        texts_by_id = {
            prototype.prototype_id: self._signature_text(prototype)
            for prototype in prototypes
        }
        candidate_map, vectors_by_id = self._retriever.retrieve(texts_by_id)

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

        for left_id, candidate_ids in candidate_map.items():
            left_prototype = prototypes_by_id[left_id]
            left_vector = vectors_by_id.get(left_id)
            if left_vector is None:
                continue
            for right_id in candidate_ids:
                right_prototype = prototypes_by_id[right_id]
                right_vector = vectors_by_id.get(right_id)
                if right_vector is None:
                    continue
                similarity = cosine_similarity(left_vector, right_vector)
                if self._should_merge(left_prototype, right_prototype, similarity):
                    union(left_id, right_id)

        grouped_prototype_ids: dict[str, list[str]] = defaultdict(list)
        for prototype in prototypes:
            grouped_prototype_ids[find(prototype.prototype_id)].append(prototype.prototype_id)

        clusters: list[SpecificCluster] = []
        prototype_to_cluster: dict[str, str] = {}

        for index, prototype_ids in enumerate(grouped_prototype_ids.values(), start=1):
            member_comment_ids: list[str] = []
            for prototype_id in prototype_ids:
                member_comment_ids.extend(prototypes_by_id[prototype_id].member_comment_ids)

            representative_prototype_id = max(
                prototype_ids,
                key=lambda prototype_id: len(prototypes_by_id[prototype_id].member_comment_ids),
            )
            cluster_id = f"stable_specific_cluster_{index}"
            clusters.append(
                SpecificCluster(
                    specific_cluster_id=cluster_id,
                    prototype_ids=prototype_ids,
                    member_comment_ids=member_comment_ids,
                    representative_prototype_id=representative_prototype_id,
                )
            )
            for prototype_id in prototype_ids:
                prototype_to_cluster[prototype_id] = cluster_id

        logger.info(
            "Stable specific clustering: %d prototypes -> %d clusters",
            len(prototypes),
            len(clusters),
        )
        return clusters, prototype_to_cluster

"""Deterministic parent clustering over stable specific clusters."""

from __future__ import annotations

from collections import defaultdict
import logging

from langchain_core.embeddings import Embeddings

from ..models import ParentCluster, SpecificCluster, SpecificPrototype
from .candidate_retriever import CandidateRetriever
from .text_utils import cosine_similarity, token_overlap

logger = logging.getLogger(__name__)


class ParentClusterBuilder:
    """Build parent clusters from representative specific clusters."""

    def __init__(self, embeddings: Embeddings, top_k: int = 8):
        self._retriever = CandidateRetriever(embeddings=embeddings, top_k=top_k)

    @staticmethod
    def _signature_text(
        specific_cluster: SpecificCluster,
        prototypes_by_id: dict[str, SpecificPrototype],
    ) -> str:
        """Build a stable parent signature text."""
        signal = prototypes_by_id[specific_cluster.representative_prototype_id].representative_signal
        return " | ".join(
            [
                signal.polarity,
                signal.phenomenon,
                signal.parent_focus,
                signal.parent_key,
                signal.subject,
            ]
        )

    @staticmethod
    def _should_merge(
        left: SpecificCluster,
        right: SpecificCluster,
        prototypes_by_id: dict[str, SpecificPrototype],
        similarity: float,
    ) -> bool:
        """Apply conservative merge rules for parent clusters."""
        left_signal = prototypes_by_id[left.representative_prototype_id].representative_signal
        right_signal = prototypes_by_id[right.representative_prototype_id].representative_signal

        if left_signal.is_meaningful != right_signal.is_meaningful:
            return False
        if not left_signal.is_meaningful:
            return True
        if left_signal.polarity != right_signal.polarity:
            return False
        if left_signal.parent_key == right_signal.parent_key:
            return True

        return (
            similarity >= 0.92
            and token_overlap(left_signal.parent_focus, right_signal.parent_focus) >= 0.50
            and token_overlap(left_signal.phenomenon, right_signal.phenomenon) >= 0.50
        )

    def build(
        self,
        specific_clusters: list[SpecificCluster],
        prototypes_by_id: dict[str, SpecificPrototype],
    ) -> dict[str, ParentCluster]:
        """Build parent clusters keyed by parent_cluster_id."""
        if not specific_clusters:
            return {}

        parent = {
            cluster.specific_cluster_id: cluster.specific_cluster_id for cluster in specific_clusters
        }
        specific_by_id = {
            cluster.specific_cluster_id: cluster for cluster in specific_clusters
        }
        texts_by_id = {
            cluster.specific_cluster_id: self._signature_text(cluster, prototypes_by_id)
            for cluster in specific_clusters
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
            left_cluster = specific_by_id[left_id]
            left_vector = vectors_by_id.get(left_id)
            if left_vector is None:
                continue
            for right_id in candidate_ids:
                right_cluster = specific_by_id[right_id]
                right_vector = vectors_by_id.get(right_id)
                if right_vector is None:
                    continue
                similarity = cosine_similarity(left_vector, right_vector)
                if self._should_merge(left_cluster, right_cluster, prototypes_by_id, similarity):
                    union(left_id, right_id)

        grouped_specific_ids: dict[str, list[str]] = defaultdict(list)
        for cluster in specific_clusters:
            grouped_specific_ids[find(cluster.specific_cluster_id)].append(cluster.specific_cluster_id)

        parent_clusters: dict[str, ParentCluster] = {}
        for index, specific_cluster_ids in enumerate(grouped_specific_ids.values(), start=1):
            representative_specific_cluster_id = max(
                specific_cluster_ids,
                key=lambda specific_cluster_id: len(
                    specific_by_id[specific_cluster_id].member_comment_ids
                ),
            )
            member_comment_ids: list[str] = []
            for specific_cluster_id in specific_cluster_ids:
                specific_cluster = specific_by_id[specific_cluster_id]
                member_comment_ids.extend(specific_cluster.member_comment_ids)

            parent_cluster_id = f"stable_parent_cluster_{index}"
            parent_clusters[parent_cluster_id] = ParentCluster(
                parent_cluster_id=parent_cluster_id,
                child_specific_cluster_ids=specific_cluster_ids,
                member_comment_ids=member_comment_ids,
                representative_specific_cluster_id=representative_specific_cluster_id,
            )
            for specific_cluster_id in specific_cluster_ids:
                specific_by_id[specific_cluster_id].parent_cluster_id = parent_cluster_id

        logger.info(
            "Stable parent clustering: %d specific clusters -> %d parent clusters",
            len(specific_clusters),
            len(parent_clusters),
        )
        return parent_clusters

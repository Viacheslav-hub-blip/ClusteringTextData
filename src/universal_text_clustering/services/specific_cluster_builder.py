"""Specific cluster building service."""

from __future__ import annotations

from collections import defaultdict
import logging

from ..models import PairDecision, Prototype, RelationType, SpecificCluster

logger = logging.getLogger(__name__)


class SpecificClusterBuilder:
    """Build exact-case clusters from SAME relations."""

    def build(
        self,
        prototypes: list[Prototype],
        decisions: list[PairDecision],
    ) -> tuple[list[SpecificCluster], dict[str, str]]:
        """Build specific clusters and return prototype-to-cluster mapping."""
        parent = {prototype.prototype_id: prototype.prototype_id for prototype in prototypes}
        prototypes_by_id = {prototype.prototype_id: prototype for prototype in prototypes}

        def find(node_id: str) -> str:
            while parent[node_id] != node_id:
                parent[node_id] = parent[parent[node_id]]
                node_id = parent[node_id]
            return node_id

        def union(left_id: str, right_id: str) -> None:
            left_root, right_root = find(left_id), find(right_id)
            if left_root != right_root:
                parent[right_root] = left_root

        for decision in decisions:
            if decision.relation == RelationType.SAME:
                union(decision.left_prototype_id, decision.right_prototype_id)

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
            cluster_id = f"specific_cluster_{index}"
            cluster = SpecificCluster(
                specific_cluster_id=cluster_id,
                prototype_ids=prototype_ids,
                member_comment_ids=member_comment_ids,
                representative_prototype_id=representative_prototype_id,
            )
            clusters.append(cluster)

            for prototype_id in prototype_ids:
                prototype_to_cluster[prototype_id] = cluster_id

        logger.info(
            "Specific clustering: %d prototypes -> %d specific clusters",
            len(prototypes),
            len(clusters),
        )
        return clusters, prototype_to_cluster

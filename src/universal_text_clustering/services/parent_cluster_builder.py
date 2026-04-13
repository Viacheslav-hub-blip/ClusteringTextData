"""Parent cluster building service."""

from __future__ import annotations

from collections import Counter, defaultdict
import logging
import re

from ..models import PairDecision, ParentCluster, Prototype, RelationType, SpecificCluster

logger = logging.getLogger(__name__)


class ParentClusterBuilder:
    """Build parent links and parent clusters over specific clusters."""

    @staticmethod
    def _normalize_parent_key(value: str) -> str:
        """Normalize a model-produced parent key without mapping it to fixed categories."""
        value = value.strip().lower()
        value = re.sub(r"[^\w\s-]+", " ", value, flags=re.UNICODE)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def build_parent_links(
        self,
        decisions: list[PairDecision],
        prototype_to_specific_cluster: dict[str, str],
    ) -> dict[str, str]:
        """Build specific-cluster to parent-specific-cluster links."""
        candidate_links: dict[str, list[str]] = defaultdict(list)

        for decision in decisions:
            left_cluster = prototype_to_specific_cluster.get(decision.left_prototype_id)
            right_cluster = prototype_to_specific_cluster.get(decision.right_prototype_id)
            if not left_cluster or not right_cluster or left_cluster == right_cluster:
                continue

            if decision.relation == RelationType.A_SPECIFIC_OF_B:
                candidate_links[left_cluster].append(right_cluster)
            elif decision.relation == RelationType.B_SPECIFIC_OF_A:
                candidate_links[right_cluster].append(left_cluster)

        parent_links: dict[str, str] = {}
        for child_cluster_id, parent_candidates in candidate_links.items():
            parent_links[child_cluster_id] = Counter(parent_candidates).most_common(1)[0][0]

        logger.info(
            "Parent linking: built %d specific->parent links",
            len(parent_links),
        )
        return parent_links

    def build_parent_clusters(
        self,
        specific_clusters: list[SpecificCluster],
        parent_links: dict[str, str],
        prototypes_by_id: dict[str, Prototype],
    ) -> list[ParentCluster]:
        """Build parent clusters from specific links and data-driven parent keys."""
        specific_by_id = {
            cluster.specific_cluster_id: cluster for cluster in specific_clusters
        }
        cluster_parent = {
            cluster.specific_cluster_id: cluster.specific_cluster_id
            for cluster in specific_clusters
        }

        def find(cluster_id: str) -> str:
            while cluster_parent[cluster_id] != cluster_id:
                cluster_parent[cluster_id] = cluster_parent[cluster_parent[cluster_id]]
                cluster_id = cluster_parent[cluster_id]
            return cluster_id

        def union(left_cluster_id: str, right_cluster_id: str) -> None:
            left_root = find(left_cluster_id)
            right_root = find(right_cluster_id)
            if left_root != right_root:
                cluster_parent[right_root] = left_root

        for child_cluster_id, parent_cluster_id in parent_links.items():
            if child_cluster_id in specific_by_id and parent_cluster_id in specific_by_id:
                union(child_cluster_id, parent_cluster_id)

        clusters_by_parent_key: dict[str, list[str]] = defaultdict(list)
        for cluster in specific_clusters:
            frame = prototypes_by_id[cluster.representative_prototype_id].representative_frame
            parent_key = self._normalize_parent_key(frame.parent_key or frame.general_topic)
            if parent_key and parent_key != "не определено":
                clusters_by_parent_key[parent_key].append(cluster.specific_cluster_id)

        for cluster_ids in clusters_by_parent_key.values():
            first_cluster_id = cluster_ids[0]
            for cluster_id in cluster_ids[1:]:
                union(first_cluster_id, cluster_id)

        children_by_root: dict[str, list[str]] = defaultdict(list)
        for cluster in specific_clusters:
            children_by_root[find(cluster.specific_cluster_id)].append(
                cluster.specific_cluster_id
            )

        parent_clusters: list[ParentCluster] = []
        for index, child_cluster_ids in enumerate(children_by_root.values(), start=1):
            parent_cluster_id = f"parent_cluster_{index}"
            for child_cluster_id in child_cluster_ids:
                specific_by_id[child_cluster_id].parent_cluster_id = parent_cluster_id

            representative_cluster = max(
                (
                    specific_by_id[child_cluster_id]
                    for child_cluster_id in child_cluster_ids
                ),
                key=lambda cluster: len(cluster.member_comment_ids),
            )
            representative_prototype = prototypes_by_id[
                representative_cluster.representative_prototype_id
            ]
            parent_clusters.append(
                ParentCluster(
                    parent_cluster_id=parent_cluster_id,
                    child_specific_cluster_ids=child_cluster_ids,
                    representative_topic=(
                        representative_prototype.representative_frame.parent_key
                        or representative_prototype.representative_frame.general_topic
                    ),
                )
            )

        logger.info(
            "Parent clustering: %d specific clusters -> %d parent clusters",
            len(specific_clusters),
            len(parent_clusters),
        )
        return parent_clusters

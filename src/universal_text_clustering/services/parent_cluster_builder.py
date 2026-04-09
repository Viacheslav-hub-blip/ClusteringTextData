"""Parent cluster building service."""

from __future__ import annotations

from collections import Counter, defaultdict
import logging

from ..models import PairDecision, ParentCluster, Prototype, RelationType, SpecificCluster

logger = logging.getLogger(__name__)


class ParentClusterBuilder:
    """Build parent links and parent clusters over specific clusters."""

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
        """Build parent clusters from specific cluster links."""
        specific_by_id = {
            cluster.specific_cluster_id: cluster for cluster in specific_clusters
        }
        children_by_parent: dict[str, list[str]] = defaultdict(list)

        for cluster in specific_clusters:
            parent_id = parent_links.get(cluster.specific_cluster_id, cluster.specific_cluster_id)
            cluster.parent_cluster_id = parent_id
            children_by_parent[parent_id].append(cluster.specific_cluster_id)

        parent_clusters: list[ParentCluster] = []
        for parent_id, child_cluster_ids in children_by_parent.items():
            representative_cluster = specific_by_id[parent_id]
            representative_prototype = prototypes_by_id[
                representative_cluster.representative_prototype_id
            ]
            parent_clusters.append(
                ParentCluster(
                    parent_cluster_id=parent_id,
                    child_specific_cluster_ids=child_cluster_ids,
                    representative_topic=representative_prototype.representative_frame.general_topic,
                )
            )

        logger.info(
            "Parent clustering: %d specific clusters -> %d parent clusters",
            len(specific_clusters),
            len(parent_clusters),
        )
        return parent_clusters

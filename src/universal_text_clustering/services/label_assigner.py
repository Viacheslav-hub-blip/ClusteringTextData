"""Output label assignment service."""

from __future__ import annotations

from ..models import OutputRecord, ParentCluster, SpecificCluster


class LabelAssigner:
    """Assign final labels back to input comments."""

    def assign(
        self,
        comment_to_prototype: dict[str, str],
        prototype_to_specific_cluster: dict[str, str],
        specific_clusters_by_id: dict[str, SpecificCluster],
        parent_clusters_by_id: dict[str, ParentCluster],
    ) -> list[OutputRecord]:
        """Build final output records."""
        outputs: list[OutputRecord] = []

        for comment_id, prototype_id in comment_to_prototype.items():
            specific_cluster_id = prototype_to_specific_cluster[prototype_id]
            specific_cluster = specific_clusters_by_id[specific_cluster_id]
            parent_cluster = parent_clusters_by_id[specific_cluster.parent_cluster_id or specific_cluster_id]
            outputs.append(
                OutputRecord(
                    comment_id=comment_id,
                    specific_group=specific_cluster.specific_group,
                    parent_group=parent_cluster.parent_group,
                )
            )

        return outputs

"""Deterministic snapshot builder over structured signals."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.embeddings import Embeddings

from ..models import (
    CommentAssignment,
    InputComment,
    ParentClusterSummary,
    Snapshot,
    SpecificClusterSummary,
    StructuredSignal,
)
from .text_utils import (
    UNDEFINED_LABEL,
    best_key,
    best_label,
    cosine_similarity,
    dedupe_preserve_order,
    is_undefined,
    normalize_text,
    token_overlap,
)


class SnapshotBuilder:
    """Build a global clustering snapshot from precomputed signals."""

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings

    def build(
        self,
        comments: list[InputComment],
        signals_by_comment_id: dict[str, StructuredSignal],
    ) -> Snapshot:
        """Build the full clustering snapshot."""
        comments_by_id = {comment.comment_id: comment for comment in comments}
        specific_clusters = self._build_specific_clusters(signals_by_comment_id, comments_by_id)
        parent_clusters = self._build_parent_clusters(specific_clusters)
        assignments = self._build_assignments(signals_by_comment_id, specific_clusters, parent_clusters)
        return Snapshot(
            signals_by_comment_id=dict(signals_by_comment_id),
            assignments_by_comment_id=assignments,
            specific_clusters_by_id=specific_clusters,
            parent_clusters_by_id=parent_clusters,
        )

    def _build_specific_clusters(
        self,
        signals_by_comment_id: dict[str, StructuredSignal],
        comments_by_id: dict[str, InputComment],
    ) -> dict[str, SpecificClusterSummary]:
        grouped_signals: dict[tuple[str, str], list[StructuredSignal]] = defaultdict(list)
        for signal in signals_by_comment_id.values():
            key = (signal.polarity, signal.specific_key)
            grouped_signals[key].append(signal)

        if not grouped_signals:
            return {}

        proto_keys = list(grouped_signals)
        signature_texts = {
            proto_key: self._signal_group_signature(grouped_signals[proto_key])
            for proto_key in proto_keys
        }
        vectors = self._embed_map(signature_texts)
        parents = {proto_key: proto_key for proto_key in proto_keys}

        def find(proto_key: tuple[str, str]) -> tuple[str, str]:
            while parents[proto_key] != proto_key:
                parents[proto_key] = parents[parents[proto_key]]
                proto_key = parents[proto_key]
            return proto_key

        def union(left: tuple[str, str], right: tuple[str, str]) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for index, left_key in enumerate(proto_keys):
            for right_key in proto_keys[index + 1:]:
                similarity = cosine_similarity(vectors.get(left_key, []), vectors.get(right_key, []))
                if self._should_merge_specific(
                    grouped_signals[left_key],
                    grouped_signals[right_key],
                    similarity,
                ):
                    union(left_key, right_key)

        merged_groups: dict[tuple[str, str], list[StructuredSignal]] = defaultdict(list)
        for proto_key, group_signals in grouped_signals.items():
            merged_groups[find(proto_key)].extend(group_signals)

        specific_clusters: dict[str, SpecificClusterSummary] = {}
        for index, group_signals in enumerate(merged_groups.values(), start=1):
            cluster_id = f"specific_cluster_{index}"
            specific_group = best_label((signal.specific_focus, 1) for signal in group_signals)
            parent_group = best_label((signal.parent_focus, 1) for signal in group_signals)
            specific_key = best_key((signal.specific_key, 1) for signal in group_signals)
            parent_key = best_key((signal.parent_key, 1) for signal in group_signals)
            polarity = best_label((signal.polarity, 1) for signal in group_signals)
            member_comment_ids = [signal.comment_id for signal in group_signals]
            sample_texts = [
                comments_by_id[comment_id].text
                for comment_id in member_comment_ids[:3]
                if comment_id in comments_by_id
            ]
            signature_text = " | ".join(
                part
                for part in [
                    specific_group,
                    parent_group,
                    specific_key,
                    parent_key,
                    " ".join(sample_texts),
                ]
                if part
            )
            specific_clusters[cluster_id] = SpecificClusterSummary(
                cluster_id=cluster_id,
                specific_group=specific_group or UNDEFINED_LABEL,
                specific_key=specific_key or normalize_text(UNDEFINED_LABEL),
                parent_group=parent_group or UNDEFINED_LABEL,
                parent_key=parent_key or normalize_text(UNDEFINED_LABEL),
                polarity=polarity or "neutral",
                member_comment_ids=member_comment_ids,
                sample_texts=sample_texts,
                signature_text=signature_text,
            )

        return specific_clusters

    def _build_parent_clusters(
        self,
        specific_clusters: dict[str, SpecificClusterSummary],
    ) -> dict[str, ParentClusterSummary]:
        if not specific_clusters:
            return {}

        grouped_specific_ids: dict[str, list[str]] = defaultdict(list)
        for cluster_id, cluster in specific_clusters.items():
            grouped_specific_ids[cluster.parent_key].append(cluster_id)

        parent_keys = list(grouped_specific_ids)
        signature_texts = {
            parent_key: self._parent_group_signature(
                [specific_clusters[cluster_id] for cluster_id in grouped_specific_ids[parent_key]]
            )
            for parent_key in parent_keys
        }
        vectors = self._embed_map(signature_texts)
        parents = {parent_key: parent_key for parent_key in parent_keys}

        def find(parent_key: str) -> str:
            while parents[parent_key] != parent_key:
                parents[parent_key] = parents[parents[parent_key]]
                parent_key = parents[parent_key]
            return parent_key

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for index, left_key in enumerate(parent_keys):
            for right_key in parent_keys[index + 1:]:
                similarity = cosine_similarity(vectors.get(left_key, []), vectors.get(right_key, []))
                if self._should_merge_parent(
                    [specific_clusters[cluster_id] for cluster_id in grouped_specific_ids[left_key]],
                    [specific_clusters[cluster_id] for cluster_id in grouped_specific_ids[right_key]],
                    similarity,
                ):
                    union(left_key, right_key)

        merged_parent_groups: dict[str, list[str]] = defaultdict(list)
        for parent_key, cluster_ids in grouped_specific_ids.items():
            merged_parent_groups[find(parent_key)].extend(cluster_ids)

        parent_clusters: dict[str, ParentClusterSummary] = {}
        for index, child_cluster_ids in enumerate(merged_parent_groups.values(), start=1):
            parent_cluster_id = f"parent_cluster_{index}"
            child_clusters = [specific_clusters[cluster_id] for cluster_id in child_cluster_ids]
            parent_group = best_label((cluster.parent_group, len(cluster.member_comment_ids)) for cluster in child_clusters)
            parent_key = best_key((cluster.parent_key, len(cluster.member_comment_ids)) for cluster in child_clusters)
            member_comment_ids = dedupe_preserve_order(
                comment_id
                for cluster in child_clusters
                for comment_id in cluster.member_comment_ids
            )
            sample_texts = dedupe_preserve_order(
                text
                for cluster in child_clusters
                for text in cluster.sample_texts
            )[:4]
            signature_text = " | ".join(
                part
                for part in [
                    parent_group,
                    parent_key,
                    " ".join(cluster.specific_group for cluster in child_clusters),
                    " ".join(sample_texts),
                ]
                if part
            )
            parent_clusters[parent_cluster_id] = ParentClusterSummary(
                parent_cluster_id=parent_cluster_id,
                parent_group=parent_group or UNDEFINED_LABEL,
                parent_key=parent_key or normalize_text(UNDEFINED_LABEL),
                child_specific_cluster_ids=child_cluster_ids,
                member_comment_ids=member_comment_ids,
                sample_texts=sample_texts,
                signature_text=signature_text,
            )
            for cluster in child_clusters:
                cluster.parent_cluster_id = parent_cluster_id
                cluster.parent_group = parent_group or UNDEFINED_LABEL
                cluster.parent_key = parent_key or normalize_text(UNDEFINED_LABEL)

        return parent_clusters

    def _build_assignments(
        self,
        signals_by_comment_id: dict[str, StructuredSignal],
        specific_clusters: dict[str, SpecificClusterSummary],
        parent_clusters: dict[str, ParentClusterSummary],
    ) -> dict[str, CommentAssignment]:
        cluster_by_comment_id: dict[str, SpecificClusterSummary] = {}
        for cluster in specific_clusters.values():
            for comment_id in cluster.member_comment_ids:
                cluster_by_comment_id[comment_id] = cluster

        assignments: dict[str, CommentAssignment] = {}
        for comment_id, signal in signals_by_comment_id.items():
            specific_cluster = cluster_by_comment_id.get(comment_id)
            if specific_cluster is None:
                assignments[comment_id] = CommentAssignment(
                    comment_id=comment_id,
                    specific_cluster_id="",
                    specific_group=signal.specific_focus or UNDEFINED_LABEL,
                    specific_key=signal.specific_key or normalize_text(UNDEFINED_LABEL),
                    parent_cluster_id="",
                    parent_group=signal.parent_focus or UNDEFINED_LABEL,
                    parent_key=signal.parent_key or normalize_text(UNDEFINED_LABEL),
                )
                continue
            parent_cluster = parent_clusters.get(specific_cluster.parent_cluster_id)
            assignments[comment_id] = CommentAssignment(
                comment_id=comment_id,
                specific_cluster_id=specific_cluster.cluster_id,
                specific_group=specific_cluster.specific_group,
                specific_key=specific_cluster.specific_key,
                parent_cluster_id=specific_cluster.parent_cluster_id,
                parent_group=(parent_cluster.parent_group if parent_cluster else specific_cluster.parent_group),
                parent_key=(parent_cluster.parent_key if parent_cluster else specific_cluster.parent_key),
            )
        return assignments

    @staticmethod
    def _signal_group_signature(group_signals: list[StructuredSignal]) -> str:
        representative = group_signals[0]
        material = " ".join(
            dedupe_preserve_order(
                detail
                for signal in group_signals
                for detail in signal.material_details
            )
        )
        entities = " ".join(
            dedupe_preserve_order(
                entity
                for signal in group_signals
                for entity in signal.entities
            )
        )
        return " | ".join(
            [
                representative.polarity,
                representative.banking_area,
                representative.phenomenon,
                representative.object_name,
                representative.parent_focus,
                representative.parent_key,
                representative.specific_focus,
                representative.specific_key,
                material,
                entities,
            ]
        )

    @staticmethod
    def _parent_group_signature(clusters: list[SpecificClusterSummary]) -> str:
        return " | ".join(
            [
                " ".join(cluster.parent_group for cluster in clusters),
                " ".join(cluster.parent_key for cluster in clusters),
                " ".join(cluster.specific_group for cluster in clusters),
                " ".join(text for cluster in clusters for text in cluster.sample_texts[:2]),
            ]
        )

    @staticmethod
    def _should_merge_specific(
        left_signals: list[StructuredSignal],
        right_signals: list[StructuredSignal],
        similarity: float,
    ) -> bool:
        left = left_signals[0]
        right = right_signals[0]

        if left.polarity != right.polarity:
            return False
        if left.specific_key == right.specific_key:
            return True
        if is_undefined(left.specific_key) and is_undefined(right.specific_key):
            return True

        specific_overlap = max(
            token_overlap(left.specific_key, right.specific_key),
            token_overlap(left.specific_focus, right.specific_focus),
        )
        parent_overlap = max(
            token_overlap(left.parent_key, right.parent_key),
            token_overlap(left.parent_focus, right.parent_focus),
        )
        object_overlap = max(
            token_overlap(left.object_name, right.object_name),
            token_overlap(left.phenomenon, right.phenomenon),
        )
        return (
            similarity >= 0.90
            and specific_overlap >= 0.45
            and (parent_overlap >= 0.30 or object_overlap >= 0.30)
        )

    @staticmethod
    def _should_merge_parent(
        left_clusters: list[SpecificClusterSummary],
        right_clusters: list[SpecificClusterSummary],
        similarity: float,
    ) -> bool:
        left_parent_group = best_label((cluster.parent_group, len(cluster.member_comment_ids)) for cluster in left_clusters)
        right_parent_group = best_label((cluster.parent_group, len(cluster.member_comment_ids)) for cluster in right_clusters)
        left_parent_key = best_key((cluster.parent_key, len(cluster.member_comment_ids)) for cluster in left_clusters)
        right_parent_key = best_key((cluster.parent_key, len(cluster.member_comment_ids)) for cluster in right_clusters)

        if left_parent_key == right_parent_key:
            return True
        if is_undefined(left_parent_key) and is_undefined(right_parent_key):
            return True

        overlap = max(
            token_overlap(left_parent_key, right_parent_key),
            token_overlap(left_parent_group, right_parent_group),
        )
        specific_overlap = max(
            (
                max(
                    token_overlap(left_cluster.specific_group, right_cluster.specific_group),
                    token_overlap(left_cluster.specific_key, right_cluster.specific_key),
                )
                for left_cluster in left_clusters
                for right_cluster in right_clusters
            ),
            default=0.0,
        )
        return similarity >= 0.88 and overlap >= 0.40 and specific_overlap >= 0.20

    def _embed_map(self, texts_by_id: dict) -> dict:
        if len(texts_by_id) <= 1:
            return {item_id: [] for item_id in texts_by_id}
        ordered_ids = list(texts_by_id)
        ordered_texts = [texts_by_id[item_id] for item_id in ordered_ids]
        vectors = self._embeddings.embed_documents(ordered_texts)
        return {
            item_id: vector
            for item_id, vector in zip(ordered_ids, vectors, strict=True)
        }

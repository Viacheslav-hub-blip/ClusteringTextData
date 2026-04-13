"""Selection of suspicious local neighborhoods for repair."""

from __future__ import annotations

from itertools import combinations

from langchain_core.embeddings import Embeddings

from ..models import Neighborhood, Snapshot
from .text_utils import cosine_similarity, dedupe_preserve_order, is_undefined, token_overlap


class NeighborhoodSelector:
    """Find small suspicious neighborhoods instead of reclustering everything."""

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings

    def select(
        self,
        snapshot: Snapshot,
        *,
        limit: int = 12,
    ) -> list[Neighborhood]:
        """Return suspicious neighborhoods ordered by score."""
        specific_clusters = list(snapshot.specific_clusters_by_id.values())
        parent_clusters = list(snapshot.parent_clusters_by_id.values())
        neighborhoods: list[Neighborhood] = []

        specific_vectors = self._embed_map(
            {cluster.cluster_id: cluster.signature_text for cluster in specific_clusters}
        )
        parent_vectors = self._embed_map(
            {cluster.parent_cluster_id: cluster.signature_text for cluster in parent_clusters}
        )

        for left, right in combinations(specific_clusters, 2):
            similarity = cosine_similarity(
                specific_vectors.get(left.cluster_id, []),
                specific_vectors.get(right.cluster_id, []),
            )
            specific_overlap = token_overlap(left.specific_group, right.specific_group)
            parent_overlap = token_overlap(left.parent_group, right.parent_group)
            specific_key_overlap = token_overlap(left.specific_key, right.specific_key)
            parent_key_overlap = token_overlap(left.parent_key, right.parent_key)
            score = (
                (0.55 * similarity)
                + (0.25 * max(specific_overlap, parent_overlap))
                + (0.15 * specific_key_overlap)
                + (0.05 * parent_key_overlap)
            )
            if (
                similarity >= 0.84
                and max(specific_overlap, parent_overlap, specific_key_overlap, parent_key_overlap) >= 0.25
                and max(specific_overlap, specific_key_overlap) >= 0.15
            ):
                neighborhoods.append(
                    Neighborhood(
                        neighborhood_id="",
                        cluster_ids=[left.cluster_id, right.cluster_id],
                        parent_cluster_ids=dedupe_preserve_order([left.parent_cluster_id, right.parent_cluster_id]),
                        reason=(
                            f"Близкие clusters: sim={similarity:.2f}, "
                            f"specific_overlap={specific_overlap:.2f}, parent_overlap={parent_overlap:.2f}"
                        ),
                        score=score,
                        sample_comment_ids=dedupe_preserve_order(
                            left.member_comment_ids[:2] + right.member_comment_ids[:2]
                        ),
                    )
                )

        for parent_left, parent_right in combinations(parent_clusters, 2):
            similarity = cosine_similarity(
                parent_vectors.get(parent_left.parent_cluster_id, []),
                parent_vectors.get(parent_right.parent_cluster_id, []),
            )
            overlap = max(
                token_overlap(parent_left.parent_group, parent_right.parent_group),
                token_overlap(parent_left.parent_key, parent_right.parent_key),
            )
            specific_overlap = max(
                (
                    max(
                        token_overlap(snapshot.specific_clusters_by_id[left_id].specific_group, snapshot.specific_clusters_by_id[right_id].specific_group),
                        token_overlap(snapshot.specific_clusters_by_id[left_id].specific_key, snapshot.specific_clusters_by_id[right_id].specific_key),
                    )
                    for left_id in parent_left.child_specific_cluster_ids
                    for right_id in parent_right.child_specific_cluster_ids
                    if left_id in snapshot.specific_clusters_by_id and right_id in snapshot.specific_clusters_by_id
                ),
                default=0.0,
            )
            score = (0.70 * similarity) + (0.30 * overlap)
            if similarity >= 0.86 and overlap >= 0.30 and specific_overlap >= 0.15:
                cluster_ids = dedupe_preserve_order(
                    parent_left.child_specific_cluster_ids + parent_right.child_specific_cluster_ids
                )
                sample_comment_ids = dedupe_preserve_order(
                    parent_left.member_comment_ids[:2] + parent_right.member_comment_ids[:2]
                )
                neighborhoods.append(
                    Neighborhood(
                        neighborhood_id="",
                        cluster_ids=cluster_ids,
                        parent_cluster_ids=[parent_left.parent_cluster_id, parent_right.parent_cluster_id],
                        reason=f"Близкие parent clusters: sim={similarity:.2f}, overlap={overlap:.2f}",
                        score=score,
                        sample_comment_ids=sample_comment_ids,
                    )
                )

        for cluster in specific_clusters:
            if is_undefined(cluster.specific_group) or is_undefined(cluster.parent_group):
                neighborhoods.append(
                    Neighborhood(
                        neighborhood_id="",
                        cluster_ids=[cluster.cluster_id],
                        parent_cluster_ids=[cluster.parent_cluster_id] if cluster.parent_cluster_id else [],
                        reason="Есть неопределенный label, возможно нужен локальный пересмотр",
                        score=0.65,
                        sample_comment_ids=cluster.member_comment_ids[:3],
                    )
                )

        deduped = self._dedupe(neighborhoods)
        ranked = sorted(deduped, key=lambda item: item.score, reverse=True)[:limit]
        for index, neighborhood in enumerate(ranked, start=1):
            neighborhood.neighborhood_id = f"neighborhood_{index}"
        return ranked

    @staticmethod
    def _dedupe(neighborhoods: list[Neighborhood]) -> list[Neighborhood]:
        grouped: dict[frozenset[str], Neighborhood] = {}
        for neighborhood in neighborhoods:
            key = frozenset(neighborhood.cluster_ids)
            existing = grouped.get(key)
            if existing is None or neighborhood.score > existing.score:
                grouped[key] = neighborhood
        return list(grouped.values())

    def _embed_map(self, texts_by_id: dict[str, str]) -> dict[str, list[float]]:
        if len(texts_by_id) <= 1:
            return {item_id: [] for item_id in texts_by_id}
        ordered_ids = list(texts_by_id)
        ordered_texts = [texts_by_id[item_id] for item_id in ordered_ids]
        vectors = self._embeddings.embed_documents(ordered_texts)
        return {
            item_id: vector
            for item_id, vector in zip(ordered_ids, vectors, strict=True)
        }

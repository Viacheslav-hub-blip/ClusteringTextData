"""Stateful session that powers the LangChain repair agent."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ..models import CandidatePatch, InputComment, Neighborhood, OutputRecord, Snapshot, StructuredSignal
from .critic import NeighborhoodCritic
from .excel_loader import load_comments_from_excel
from .local_reclusterer import LocalReclusterer
from .neighborhood_selector import NeighborhoodSelector
from .patch_evaluator import PatchEvaluator
from .snapshot_builder import SnapshotBuilder
from .structure_extractor import StructureExtractor
from .text_utils import dump_json


class AgenticClusteringSession:
    """Own the current dataset, clustering snapshot, issues, and pending patches."""

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings):
        self._llm = llm
        self._embeddings = embeddings
        self._extractor = StructureExtractor(llm)
        self._snapshot_builder = SnapshotBuilder(embeddings)
        self._selector = NeighborhoodSelector(embeddings)
        self._critic = NeighborhoodCritic(llm)
        self._local_reclusterer = LocalReclusterer(llm)
        self._patch_evaluator = PatchEvaluator(llm)

        self.comments_by_id: dict[str, InputComment] = {}
        self.current_signals_by_comment_id: dict[str, StructuredSignal] = {}
        self.snapshot: Snapshot | None = None
        self.neighborhoods_by_id: dict[str, Neighborhood] = {}
        self.issues_by_neighborhood_id: dict[str, dict] = {}
        self.patches_by_id: dict[str, CandidatePatch] = {}
        self._patch_counter = 0

    def initialize_from_excel(
        self,
        path: str | Path,
        *,
        text_column: str,
        id_column: str,
        sheet_name: str | None,
        limit: int | None,
    ) -> None:
        """Load comments from Excel and build the initial snapshot."""
        raw_comments = load_comments_from_excel(
            path,
            text_column=text_column,
            id_column=id_column,
            sheet_name=sheet_name,
            limit=limit,
        )
        comments = [InputComment(comment_id=item["comment_id"], text=item["text"]) for item in raw_comments]
        self.initialize_from_comments(comments)

    def initialize_from_comments(self, comments: list[InputComment]) -> None:
        """Build initial signals and snapshot."""
        self.comments_by_id = {comment.comment_id: comment for comment in comments}
        signals = self._extractor.extract_batch(comments)
        self.current_signals_by_comment_id = {signal.comment_id: signal for signal in signals}
        self.snapshot = self._snapshot_builder.build(comments, self.current_signals_by_comment_id)
        self._refresh_neighborhoods()

    def summarize_state(self) -> dict:
        """Return a compact summary of the current global state."""
        snapshot = self._require_snapshot()
        return {
            "comment_count": len(snapshot.assignments_by_comment_id),
            "specific_cluster_count": len(snapshot.specific_clusters_by_id),
            "parent_cluster_count": len(snapshot.parent_clusters_by_id),
            "suspicious_neighborhood_count": len(self.neighborhoods_by_id),
            "top_neighborhoods": [
                {
                    "neighborhood_id": neighborhood.neighborhood_id,
                    "reason": neighborhood.reason,
                    "score": round(neighborhood.score, 3),
                    "cluster_ids": neighborhood.cluster_ids,
                }
                for neighborhood in list(self.neighborhoods_by_id.values())[:5]
            ],
        }

    def list_neighborhoods(self, limit: int = 5) -> dict:
        """List the top suspicious local neighborhoods."""
        items = list(self.neighborhoods_by_id.values())[: max(1, limit)]
        return {
            "neighborhoods": [
                {
                    "neighborhood_id": neighborhood.neighborhood_id,
                    "cluster_ids": neighborhood.cluster_ids,
                    "parent_cluster_ids": neighborhood.parent_cluster_ids,
                    "reason": neighborhood.reason,
                    "score": round(neighborhood.score, 3),
                    "sample_comment_ids": neighborhood.sample_comment_ids,
                }
                for neighborhood in items
            ]
        }

    def inspect_neighborhood(self, neighborhood_id: str) -> dict:
        """Inspect one neighborhood and ask the critic for a local diagnosis."""
        neighborhood = self._require_neighborhood(neighborhood_id)
        neighborhood_summary = self._build_neighborhood_summary(neighborhood, include_vectors=False)
        issue = self._critic.critique(neighborhood, dump_json(neighborhood_summary))
        payload = {
            "neighborhood": neighborhood_summary,
            "issue": {
                "has_issue": issue.has_issue,
                "issue_type": issue.issue_type,
                "severity": issue.severity,
                "summary": issue.summary,
                "guidance": issue.guidance,
                "affected_cluster_ids": issue.affected_cluster_ids,
                "affected_parent_ids": issue.affected_parent_ids,
                "confidence": issue.confidence,
            },
        }
        self.issues_by_neighborhood_id[neighborhood_id] = payload
        return payload

    def recluster_neighborhood(self, neighborhood_id: str, guidance: str = "") -> dict:
        """Perform localized reclustering for one neighborhood."""
        neighborhood = self._require_neighborhood(neighborhood_id)
        issue_payload = self.issues_by_neighborhood_id.get(neighborhood_id)
        if issue_payload is None:
            issue_payload = self.inspect_neighborhood(neighborhood_id)

        affected_comment_ids = self._affected_comment_ids(neighborhood)
        comments_payload = [
            {
                "comment_id": comment_id,
                "text": self.comments_by_id[comment_id].text,
                "current_signal": self._signal_payload(self.current_signals_by_comment_id[comment_id]),
                "current_assignment": self._assignment_payload(comment_id),
            }
            for comment_id in affected_comment_ids
        ]
        fallback_signals = {
            comment_id: self.current_signals_by_comment_id[comment_id]
            for comment_id in affected_comment_ids
        }
        supervisor_guidance = guidance or issue_payload["issue"].get("guidance", "")
        repaired_signals = self._local_reclusterer.repair(
            guidance=supervisor_guidance,
            neighborhood_json=dump_json(issue_payload["neighborhood"]),
            comments_json=dump_json(comments_payload),
            fallback_signals_by_comment_id=fallback_signals,
        )

        candidate_signals = dict(self.current_signals_by_comment_id)
        candidate_signals.update(repaired_signals)
        candidate_snapshot = self._snapshot_builder.build(
            list(self.comments_by_id.values()),
            candidate_signals,
        )

        changed_comment_ids = [
            comment_id
            for comment_id in affected_comment_ids
            if self._assignment_payload(comment_id, snapshot=self.snapshot)
            != self._assignment_payload(comment_id, snapshot=candidate_snapshot)
        ]

        self._patch_counter += 1
        patch_id = f"patch_{self._patch_counter}"
        patch = CandidatePatch(
            patch_id=patch_id,
            neighborhood_id=neighborhood_id,
            affected_comment_ids=affected_comment_ids,
            proposed_signals_by_comment_id=repaired_signals,
            candidate_snapshot=candidate_snapshot,
            before_summary=self._build_neighborhood_summary(
                neighborhood,
                snapshot=self.snapshot,
                include_vectors=True,
            ),
            after_summary=self._build_neighborhood_summary(
                neighborhood,
                snapshot=candidate_snapshot,
                include_vectors=True,
            ),
            changed_comment_ids=changed_comment_ids,
            guidance=supervisor_guidance,
        )
        self.patches_by_id[patch_id] = patch
        return {
            "patch_id": patch_id,
            "changed_comment_ids": changed_comment_ids,
            "guidance": supervisor_guidance,
            "before": self._strip_vectors(patch.before_summary),
            "after": self._strip_vectors(patch.after_summary),
        }

    def review_patch(self, patch_id: str) -> dict:
        """Run patch evaluation and cache the result."""
        patch = self._require_patch(patch_id)
        review = self._patch_evaluator.review(patch)
        patch.review = review
        return {
            "patch_id": patch.patch_id,
            "accept": review.accept,
            "confidence": review.confidence,
            "summary": review.summary,
            "objective_before": review.objective_before,
            "objective_after": review.objective_after,
            "changed_comment_ids": patch.changed_comment_ids,
        }

    def apply_patch(self, patch_id: str) -> dict:
        """Apply a reviewed patch if it looks useful."""
        patch = self._require_patch(patch_id)
        if patch.review is None:
            self.review_patch(patch_id)
        if patch.review is None or not patch.review.accept:
            return {
                "patch_id": patch.patch_id,
                "applied": False,
                "reason": patch.review.summary if patch.review else "Patch was not reviewed",
            }

        self.current_signals_by_comment_id.update(patch.proposed_signals_by_comment_id)
        self.snapshot = patch.candidate_snapshot
        self._refresh_neighborhoods()
        return {
            "patch_id": patch.patch_id,
            "applied": True,
            "changed_comment_ids": patch.changed_comment_ids,
            "summary": patch.review.summary,
        }

    def export_output_records(self) -> list[OutputRecord]:
        """Export the final user-facing assignments."""
        snapshot = self._require_snapshot()
        return [
            OutputRecord(
                comment_id=assignment.comment_id,
                specific_group=assignment.specific_group,
                parent_group=assignment.parent_group,
            )
            for assignment in sorted(
                snapshot.assignments_by_comment_id.values(),
                key=lambda item: int(item.comment_id) if str(item.comment_id).isdigit() else str(item.comment_id),
            )
        ]

    def export_debug_snapshot(self) -> dict:
        """Export a compact debug view for inspection."""
        snapshot = self._require_snapshot()
        return {
            "summary": self.summarize_state(),
            "specific_clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "specific_group": cluster.specific_group,
                    "specific_key": cluster.specific_key,
                    "parent_group": cluster.parent_group,
                    "parent_key": cluster.parent_key,
                    "size": len(cluster.member_comment_ids),
                    "sample_texts": cluster.sample_texts,
                }
                for cluster in snapshot.specific_clusters_by_id.values()
            ],
            "parent_clusters": [
                {
                    "parent_cluster_id": cluster.parent_cluster_id,
                    "parent_group": cluster.parent_group,
                    "parent_key": cluster.parent_key,
                    "size": len(cluster.member_comment_ids),
                    "child_specific_cluster_ids": cluster.child_specific_cluster_ids,
                }
                for cluster in snapshot.parent_clusters_by_id.values()
            ],
        }

    def _refresh_neighborhoods(self) -> None:
        snapshot = self._require_snapshot()
        neighborhoods = self._selector.select(snapshot)
        self.neighborhoods_by_id = {
            neighborhood.neighborhood_id: neighborhood for neighborhood in neighborhoods
        }
        self.issues_by_neighborhood_id = {}
        self.patches_by_id = {}

    def _build_neighborhood_summary(
        self,
        neighborhood: Neighborhood,
        *,
        snapshot: Snapshot | None = None,
        include_vectors: bool = False,
    ) -> dict:
        active_snapshot = snapshot or self._require_snapshot()
        specific_clusters = active_snapshot.specific_clusters_by_id
        parent_clusters = active_snapshot.parent_clusters_by_id
        cluster_payloads = []
        for cluster_id in neighborhood.cluster_ids:
            cluster = specific_clusters.get(cluster_id)
            if cluster is None:
                continue
            payload = {
                "cluster_id": cluster.cluster_id,
                "specific_group": cluster.specific_group,
                "specific_key": cluster.specific_key,
                "parent_group": cluster.parent_group,
                "parent_key": cluster.parent_key,
                "size": len(cluster.member_comment_ids),
                "sample_comment_ids": cluster.member_comment_ids[:3],
                "sample_texts": cluster.sample_texts,
            }
            if include_vectors:
                payload["vector"] = self._embeddings.embed_query(cluster.signature_text)
            cluster_payloads.append(payload)
        parent_payloads = []
        for parent_cluster_id in neighborhood.parent_cluster_ids:
            parent_cluster = parent_clusters.get(parent_cluster_id)
            if parent_cluster is None:
                continue
            parent_payloads.append(
                {
                    "parent_cluster_id": parent_cluster.parent_cluster_id,
                    "parent_group": parent_cluster.parent_group,
                    "parent_key": parent_cluster.parent_key,
                    "size": len(parent_cluster.member_comment_ids),
                    "child_specific_cluster_ids": parent_cluster.child_specific_cluster_ids,
                    "sample_texts": parent_cluster.sample_texts,
                }
            )
        return {
            "neighborhood_id": neighborhood.neighborhood_id,
            "reason": neighborhood.reason,
            "score": neighborhood.score,
            "clusters": cluster_payloads,
            "parent_clusters": parent_payloads,
        }

    @staticmethod
    def _strip_vectors(summary: dict) -> dict:
        stripped_clusters = []
        for cluster in summary.get("clusters", []):
            stripped = dict(cluster)
            stripped.pop("vector", None)
            stripped_clusters.append(stripped)
        result = dict(summary)
        result["clusters"] = stripped_clusters
        return result

    def _affected_comment_ids(self, neighborhood: Neighborhood) -> list[str]:
        snapshot = self._require_snapshot()
        comment_ids: list[str] = []
        for cluster_id in neighborhood.cluster_ids:
            cluster = snapshot.specific_clusters_by_id.get(cluster_id)
            if cluster is not None:
                comment_ids.extend(cluster.member_comment_ids)
        seen: set[str] = set()
        ordered: list[str] = []
        for comment_id in comment_ids:
            if comment_id not in seen:
                ordered.append(comment_id)
                seen.add(comment_id)
        return ordered

    def _signal_payload(self, signal: StructuredSignal) -> dict:
        return {
            "is_meaningful": signal.is_meaningful,
            "polarity": signal.polarity,
            "banking_area": signal.banking_area,
            "phenomenon": signal.phenomenon,
            "object_name": signal.object_name,
            "parent_focus": signal.parent_focus,
            "parent_key": signal.parent_key,
            "specific_focus": signal.specific_focus,
            "specific_key": signal.specific_key,
            "material_details": signal.material_details,
            "context_details": signal.context_details,
            "entities": signal.entities,
        }

    def _assignment_payload(self, comment_id: str, *, snapshot: Snapshot | None = None) -> dict:
        active_snapshot = snapshot or self._require_snapshot()
        assignment = active_snapshot.assignments_by_comment_id[comment_id]
        return {
            "specific_cluster_id": assignment.specific_cluster_id,
            "specific_group": assignment.specific_group,
            "specific_key": assignment.specific_key,
            "parent_cluster_id": assignment.parent_cluster_id,
            "parent_group": assignment.parent_group,
            "parent_key": assignment.parent_key,
        }

    def _require_snapshot(self) -> Snapshot:
        if self.snapshot is None:
            raise RuntimeError("Session has not been initialized")
        return self.snapshot

    def _require_neighborhood(self, neighborhood_id: str) -> Neighborhood:
        neighborhood = self.neighborhoods_by_id.get(neighborhood_id)
        if neighborhood is None:
            raise KeyError(f"Unknown neighborhood_id: {neighborhood_id}")
        return neighborhood

    def _require_patch(self, patch_id: str) -> CandidatePatch:
        patch = self.patches_by_id.get(patch_id)
        if patch is None:
            raise KeyError(f"Unknown patch_id: {patch_id}")
        return patch

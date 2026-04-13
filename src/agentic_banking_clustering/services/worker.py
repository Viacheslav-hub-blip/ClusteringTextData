"""Local and initial clustering worker for the agentic banking project."""

from __future__ import annotations

from collections import defaultdict
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import CommentAssignment, InputComment, StructuredSignal
from ..prompts import STRUCTURE_EXTRACTION_HUMAN, STRUCTURE_EXTRACTION_SYSTEM
from .utils import choose_weighted_label, clean_list, cosine_similarity, normalize_text, token_overlap

logger = logging.getLogger(__name__)


class BankingClusteringWorker:
    """Build a stable clustering snapshot from a batch of comments."""

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        *,
        extraction_batch_size: int = 50,
        neighbor_k: int = 6,
    ):
        self._embeddings = embeddings
        self._extraction_batch_size = extraction_batch_size
        self._neighbor_k = neighbor_k
        self._parser = JsonOutputParser()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", STRUCTURE_EXTRACTION_SYSTEM),
                    ("human", STRUCTURE_EXTRACTION_HUMAN),
                ]
            )
            | llm
            | self._parser
        )

    def run(
        self,
        raw_comments: list[dict],
        *,
        supervisor_feedback: str = "",
    ) -> list[CommentAssignment]:
        """Cluster comments and return comment-level assignments."""
        comments = self._validate(raw_comments)
        if not comments:
            return []
        signals = self._extract_batch(comments, supervisor_feedback=supervisor_feedback)
        return self._cluster(signals)

    @staticmethod
    def _validate(raw_comments: list[dict]) -> list[InputComment]:
        """Validate raw comments."""
        comments: list[InputComment] = []
        for index, raw in enumerate(raw_comments, start=1):
            comment_id = str(raw.get("comment_id", "")).strip() or str(index)
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            comments.append(InputComment(comment_id=comment_id, text=text))
        return comments

    def _extract_batch(
        self,
        comments: list[InputComment],
        *,
        supervisor_feedback: str,
    ) -> list[StructuredSignal]:
        """Extract structured signals for a batch of comments."""
        signals: list[StructuredSignal] = []
        for index in range(0, len(comments), self._extraction_batch_size):
            chunk = comments[index:index + self._extraction_batch_size]
            logger.info(
                "Agentic worker extraction: comments %d-%d of %d",
                index + 1,
                index + len(chunk),
                len(comments),
            )
            for comment in chunk:
                signals.append(
                    self._extract_one(comment, supervisor_feedback=supervisor_feedback)
                )
        return signals

    def _extract_one(
        self,
        comment: InputComment,
        *,
        supervisor_feedback: str,
    ) -> StructuredSignal:
        """Extract one structured signal."""
        try:
            raw = self._chain.invoke(
                {
                    "text": comment.text,
                    "supervisor_feedback": supervisor_feedback or "Нет дополнительного комментария",
                }
            )
            is_meaningful = bool(raw.get("is_meaningful", True))
            if not is_meaningful:
                return self._fallback_signal(comment)

            parent_focus = str(raw.get("parent_focus", "")).strip()
            specific_focus = str(raw.get("specific_focus", "")).strip()
            if not parent_focus or not specific_focus:
                return self._fallback_signal(comment)

            return StructuredSignal(
                comment_id=comment.comment_id,
                raw_text=comment.text,
                is_meaningful=True,
                polarity=str(raw.get("polarity", "neutral")).strip().lower() or "neutral",
                bank_area=str(raw.get("bank_area", "Не определено")).strip() or "Не определено",
                phenomenon=str(raw.get("phenomenon", "Не определено")).strip() or "Не определено",
                object_name=str(raw.get("object_name", "Не определено")).strip() or "Не определено",
                parent_focus=parent_focus,
                parent_key=normalize_text(str(raw.get("parent_key", parent_focus)).strip()) or "не определено",
                specific_focus=specific_focus,
                specific_key=normalize_text(str(raw.get("specific_key", specific_focus)).strip()) or "не определено",
                material_details=clean_list(raw.get("material_details", [])),
                context_details=clean_list(raw.get("context_details", [])),
                entities=clean_list(raw.get("entities", [])),
            )
        except Exception as exc:
            logger.error("Agentic worker extraction failed for %s: %s", comment.comment_id, exc)
            return self._fallback_signal(comment)

    @staticmethod
    def _fallback_signal(comment: InputComment) -> StructuredSignal:
        """Return a safe fallback signal."""
        return StructuredSignal(
            comment_id=comment.comment_id,
            raw_text=comment.text,
            is_meaningful=False,
            polarity="neutral",
            bank_area="Не определено",
            phenomenon="Не определено",
            object_name="Не определено",
            parent_focus="Не определено",
            parent_key="не определено",
            specific_focus="Не определено",
            specific_key="не определено",
            material_details=[],
            context_details=[],
            entities=[],
        )

    def _cluster(self, signals: list[StructuredSignal]) -> list[CommentAssignment]:
        """Cluster structured signals deterministically."""
        groups: dict[tuple[str, str], list[StructuredSignal]] = defaultdict(list)
        for signal in signals:
            groups[(signal.polarity, signal.specific_key)].append(signal)

        prototype_ids: list[str] = []
        prototype_signals: dict[str, list[StructuredSignal]] = {}
        for index, group_signals in enumerate(groups.values(), start=1):
            prototype_id = f"prototype_{index}"
            prototype_ids.append(prototype_id)
            prototype_signals[prototype_id] = group_signals

        parent = {prototype_id: prototype_id for prototype_id in prototype_ids}
        texts_by_prototype_id = {
            prototype_id: self._signature_text(group_signals[0])
            for prototype_id, group_signals in prototype_signals.items()
        }
        vectors_by_prototype_id = {
            prototype_id: vector
            for prototype_id, vector in zip(
                texts_by_prototype_id,
                self._embeddings.embed_documents(list(texts_by_prototype_id.values())),
                strict=True,
            )
        }

        if len(prototype_ids) > 1:
            vectorstore = FAISS.from_texts(
                list(texts_by_prototype_id.values()),
                self._embeddings,
                metadatas=[{"prototype_id": prototype_id} for prototype_id in texts_by_prototype_id],
            )
            for prototype_id, text in texts_by_prototype_id.items():
                query_vector = vectors_by_prototype_id[prototype_id]
                if hasattr(vectorstore, "similarity_search_with_score_by_vector"):
                    hits = vectorstore.similarity_search_with_score_by_vector(
                        query_vector,
                        k=min(len(prototype_ids), self._neighbor_k + 1),
                    )
                else:
                    hits = [
                        (document, 0.0)
                        for document in vectorstore.similarity_search_by_vector(
                            query_vector,
                            k=min(len(prototype_ids), self._neighbor_k + 1),
                        )
                    ]
                for document, _score in hits:
                    other_prototype_id = str(document.metadata.get("prototype_id", "")).strip()
                    if not other_prototype_id or other_prototype_id == prototype_id:
                        continue
                    if self._should_merge(
                        prototype_signals[prototype_id][0],
                        prototype_signals[other_prototype_id][0],
                        cosine_similarity(
                            vectors_by_prototype_id[prototype_id],
                            vectors_by_prototype_id[other_prototype_id],
                        ),
                    ):
                        self._union(parent, prototype_id, other_prototype_id)

        clusters_by_root: dict[str, list[str]] = defaultdict(list)
        for prototype_id in prototype_ids:
            clusters_by_root[self._find(parent, prototype_id)].append(prototype_id)

        assignments: list[CommentAssignment] = []
        parent_groups_by_cluster_root: dict[str, str] = {}
        specific_groups_by_cluster_root: dict[str, str] = {}
        cluster_parent_focuses: dict[str, list[tuple[str, int]]] = {}
        cluster_specific_focuses: dict[str, list[tuple[str, int]]] = {}
        cluster_representatives: dict[str, StructuredSignal] = {}

        for root, cluster_prototype_ids in clusters_by_root.items():
            parent_focus_candidates: list[tuple[str, int]] = []
            specific_focus_candidates: list[tuple[str, int]] = []
            representative_signal = prototype_signals[cluster_prototype_ids[0]][0]
            cluster_representatives[root] = representative_signal
            for prototype_id in cluster_prototype_ids:
                cluster_signals = prototype_signals[prototype_id]
                parent_focus_candidates.append(
                    (cluster_signals[0].parent_focus, len(cluster_signals))
                )
                specific_focus_candidates.append(
                    (cluster_signals[0].specific_focus, len(cluster_signals))
                )
            cluster_parent_focuses[root] = parent_focus_candidates
            cluster_specific_focuses[root] = specific_focus_candidates
            parent_groups_by_cluster_root[root] = choose_weighted_label(parent_focus_candidates)
            specific_groups_by_cluster_root[root] = choose_weighted_label(specific_focus_candidates)

        parent_roots = list(clusters_by_root)
        parent_parent = {root: root for root in parent_roots}
        if len(parent_roots) > 1:
            for left_index, left_root in enumerate(parent_roots):
                left_signal = cluster_representatives[left_root]
                for right_root in parent_roots[left_index + 1:]:
                    right_signal = cluster_representatives[right_root]
                    if self._should_merge_parent(left_signal, right_signal):
                        self._union(parent_parent, left_root, right_root)

        canonical_parent_label_by_root: dict[str, str] = {}
        merged_parent_candidates: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for root in parent_roots:
            merged_parent_candidates[self._find(parent_parent, root)].extend(
                cluster_parent_focuses[root]
            )
        for canonical_root, candidates in merged_parent_candidates.items():
            canonical_parent_label_by_root[canonical_root] = choose_weighted_label(candidates)

        for root, cluster_prototype_ids in clusters_by_root.items():
            canonical_parent_root = self._find(parent_parent, root)
            parent_group = canonical_parent_label_by_root.get(canonical_parent_root, "Не определено")
            specific_group = specific_groups_by_cluster_root[root]
            for prototype_id in cluster_prototype_ids:
                for signal in prototype_signals[prototype_id]:
                    assignments.append(
                        CommentAssignment(
                            comment_id=signal.comment_id,
                            text=signal.raw_text,
                            polarity=signal.polarity,
                            bank_area=signal.bank_area,
                            phenomenon=signal.phenomenon,
                            object_name=signal.object_name,
                            parent_focus=signal.parent_focus,
                            parent_key=signal.parent_key,
                            specific_focus=signal.specific_focus,
                            specific_key=signal.specific_key,
                            specific_group=specific_group,
                            parent_group=parent_group,
                            material_details=signal.material_details,
                            context_details=signal.context_details,
                            entities=signal.entities,
                        )
                    )

        return assignments

    @staticmethod
    def _signature_text(signal: StructuredSignal) -> str:
        """Build a stable semantic signature."""
        return " | ".join(
            [
                signal.polarity,
                signal.bank_area,
                signal.phenomenon,
                signal.object_name,
                signal.parent_focus,
                signal.parent_key,
                signal.specific_focus,
                signal.specific_key,
                " ".join(signal.material_details),
                " ".join(signal.entities),
            ]
        )

    @staticmethod
    def _should_merge(left: StructuredSignal, right: StructuredSignal, similarity: float) -> bool:
        """Conservative prototype merge rule."""
        if left.is_meaningful != right.is_meaningful:
            return False
        if not left.is_meaningful:
            return True
        if left.polarity != right.polarity:
            return False
        if left.specific_key == right.specific_key:
            return True
        return (
            similarity >= 0.90
            and token_overlap(left.specific_focus, right.specific_focus) >= 0.50
            and token_overlap(left.parent_focus, right.parent_focus) >= 0.50
        )

    @staticmethod
    def _should_merge_parent(left: StructuredSignal, right: StructuredSignal) -> bool:
        """Conservative parent merge rule."""
        if left.is_meaningful != right.is_meaningful:
            return False
        if not left.is_meaningful:
            return True
        if left.polarity != right.polarity:
            return False
        if left.parent_key == right.parent_key:
            return True
        return (
            token_overlap(left.parent_focus, right.parent_focus) >= 0.60
            and token_overlap(left.bank_area, right.bank_area) >= 0.50
        )

    @staticmethod
    def _find(parent: dict[str, str], node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    @classmethod
    def _union(cls, parent: dict[str, str], left_id: str, right_id: str) -> None:
        left_root = cls._find(parent, left_id)
        right_root = cls._find(parent, right_id)
        if left_root != right_root:
            parent[right_root] = left_root

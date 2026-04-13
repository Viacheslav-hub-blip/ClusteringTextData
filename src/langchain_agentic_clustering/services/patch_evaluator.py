"""Evaluation of localized reclustering patches."""

from __future__ import annotations

from itertools import combinations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import CandidatePatch, PatchReview
from ..prompts import PATCH_REVIEW_HUMAN, PATCH_REVIEW_SYSTEM
from .text_utils import cosine_similarity, dump_json, token_overlap


class PatchEvaluator:
    """Combine LLM judgment with a small objective duplication score."""

    def __init__(self, llm: BaseChatModel):
        self._parser = JsonOutputParser()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", PATCH_REVIEW_SYSTEM),
                    ("human", PATCH_REVIEW_HUMAN),
                ]
            )
            | llm
            | self._parser
        )

    def review(self, patch: CandidatePatch) -> PatchReview:
        """Review a candidate patch."""
        raw = self._chain.invoke(
            {
                "before_json": dump_json(self._strip_vectors(patch.before_summary)),
                "after_json": dump_json(self._strip_vectors(patch.after_summary)),
            }
        )
        objective_before = self._duplicate_score(patch.before_summary)
        objective_after = self._duplicate_score(patch.after_summary)
        llm_accept = bool(raw.get("accept", False))
        objective_improved = objective_after <= objective_before - 0.03
        accept = objective_improved or (llm_accept and objective_after <= objective_before + 0.01)
        return PatchReview(
            patch_id=patch.patch_id,
            accept=accept,
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            summary=str(raw.get("summary", "")).strip(),
            objective_before=objective_before,
            objective_after=objective_after,
        )

    @staticmethod
    def _duplicate_score(summary: dict) -> float:
        clusters = summary.get("clusters", [])
        if len(clusters) <= 1:
            return 0.0
        scores: list[float] = []
        for left, right in combinations(clusters, 2):
            label_overlap = max(
                token_overlap(left.get("specific_group", ""), right.get("specific_group", "")),
                token_overlap(left.get("parent_group", ""), right.get("parent_group", "")),
            )
            key_overlap = max(
                token_overlap(left.get("specific_key", ""), right.get("specific_key", "")),
                token_overlap(left.get("parent_key", ""), right.get("parent_key", "")),
            )
            vector_similarity = 0.0
            left_vector = left.get("vector", [])
            right_vector = right.get("vector", [])
            if left_vector and right_vector:
                vector_similarity = cosine_similarity(left_vector, right_vector)
            scores.append((0.5 * vector_similarity) + (0.3 * label_overlap) + (0.2 * key_overlap))
        return sum(scores) / len(scores)

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

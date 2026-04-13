"""Pair relation classification service."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import PairDecision, Prototype, RelationType
from ..prompts import RELATION_HUMAN, RELATION_SYSTEM

logger = logging.getLogger(__name__)


class PairRelationClassifier:
    """Determine semantic relation between prototype pairs."""

    def __init__(self, llm: BaseChatModel):
        self._parser = JsonOutputParser()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", RELATION_SYSTEM),
                    ("human", RELATION_HUMAN),
                ]
            )
            | llm
            | self._parser
        )
        self._cache: dict[tuple[str, str], PairDecision] = {}

    def classify_pair(self, left: Prototype, right: Prototype) -> PairDecision:
        """Classify one pair of prototypes."""
        cache_key = (left.prototype_id, right.prototype_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        left_frame = left.representative_frame
        right_frame = right.representative_frame

        try:
            raw = self._chain.invoke(
                {
                    "topic_a": left_frame.general_topic,
                    "parent_key_a": left_frame.parent_key,
                    "core_case_a": left_frame.core_case,
                    "exact_case_a": left_frame.exact_case,
                    "qualifiers_a": left_frame.key_qualifiers,
                    "context_details_a": left_frame.context_details,
                    "entities_a": left_frame.entities,
                    "canonical_key_a": left_frame.canonical_key,
                    "topic_b": right_frame.general_topic,
                    "parent_key_b": right_frame.parent_key,
                    "core_case_b": right_frame.core_case,
                    "exact_case_b": right_frame.exact_case,
                    "qualifiers_b": right_frame.key_qualifiers,
                    "context_details_b": right_frame.context_details,
                    "entities_b": right_frame.entities,
                    "canonical_key_b": right_frame.canonical_key,
                }
            )
            relation = RelationType(raw.get("relation", RelationType.DIFFERENT.value))
            decision = PairDecision(
                left_prototype_id=left.prototype_id,
                right_prototype_id=right.prototype_id,
                relation=relation,
                reason=str(raw.get("reason", "")).strip(),
            )
        except Exception as exc:
            logger.error(
                "Pair relation classification failed for (%s, %s): %s",
                left.prototype_id,
                right.prototype_id,
                exc,
            )
            decision = PairDecision(
                left_prototype_id=left.prototype_id,
                right_prototype_id=right.prototype_id,
                relation=RelationType.DIFFERENT,
                reason=f"Fallback to DIFFERENT due to error: {exc}",
            )

        self._cache[cache_key] = decision
        return decision

    def classify_candidates(
        self,
        prototypes: list[Prototype],
        candidate_map: dict[str, list[str]],
    ) -> list[PairDecision]:
        """Classify all candidate pairs returned by the retriever."""
        prototypes_by_id = {prototype.prototype_id: prototype for prototype in prototypes}
        decisions: list[PairDecision] = []
        processed_pairs: set[tuple[str, str]] = set()
        total_candidates = sum(len(candidate_ids) for candidate_ids in candidate_map.values())
        logger.info(
            "Pair relation classification: evaluating up to %d retrieved links",
            total_candidates,
        )

        for left_id, candidate_ids in candidate_map.items():
            for right_id in candidate_ids:
                ordered_pair = tuple(sorted((left_id, right_id)))
                if ordered_pair in processed_pairs:
                    continue
                processed_pairs.add(ordered_pair)

                left = prototypes_by_id[left_id]
                right = prototypes_by_id[right_id]
                decisions.append(self.classify_pair(left, right))

        logger.info(
            "Pair relation classification: completed %d unique pair decisions",
            len(decisions),
        )
        return decisions

"""Neighborhood critique with an LLM."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import CritiqueIssue, Neighborhood
from ..prompts import CRITIC_HUMAN, CRITIC_SYSTEM


class NeighborhoodCritic:
    """Ask the LLM to critique one local neighborhood."""

    def __init__(self, llm: BaseChatModel):
        self._parser = JsonOutputParser()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", CRITIC_SYSTEM),
                    ("human", CRITIC_HUMAN),
                ]
            )
            | llm
            | self._parser
        )

    def critique(self, neighborhood: Neighborhood, neighborhood_json: str) -> CritiqueIssue:
        """Return one critique issue object."""
        raw = self._chain.invoke({"neighborhood_json": neighborhood_json})
        return CritiqueIssue(
            neighborhood_id=neighborhood.neighborhood_id,
            has_issue=bool(raw.get("has_issue", False)),
            issue_type=str(raw.get("issue_type", "no_issue")).strip() or "no_issue",
            severity=max(1, min(5, int(raw.get("severity", 1)))),
            summary=str(raw.get("summary", "")).strip(),
            guidance=str(raw.get("guidance", "")).strip(),
            affected_cluster_ids=[str(item).strip() for item in raw.get("affected_cluster_ids", []) if str(item).strip()],
            affected_parent_ids=[str(item).strip() for item in raw.get("affected_parent_ids", []) if str(item).strip()],
            confidence=float(raw.get("confidence", 0.0) or 0.0),
        )

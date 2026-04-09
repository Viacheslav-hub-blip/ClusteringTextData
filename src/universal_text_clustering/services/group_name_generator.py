"""Group naming service."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import SemanticFrame
from ..prompts import (
    PARENT_GROUP_HUMAN,
    PARENT_GROUP_SYSTEM,
    SPECIFIC_GROUP_HUMAN,
    SPECIFIC_GROUP_SYSTEM,
)

logger = logging.getLogger(__name__)


class GroupNameGenerator:
    """Generate final specific and parent group labels."""

    def __init__(self, llm: BaseChatModel):
        parser = JsonOutputParser()
        self._specific_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SPECIFIC_GROUP_SYSTEM),
                    ("human", SPECIFIC_GROUP_HUMAN),
                ]
            )
            | llm
            | parser
        )
        self._parent_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", PARENT_GROUP_SYSTEM),
                    ("human", PARENT_GROUP_HUMAN),
                ]
            )
            | llm
            | parser
        )

    def generate_specific_group(self, frame: SemanticFrame) -> str:
        """Generate label for an exact specific cluster."""
        try:
            raw = self._specific_chain.invoke(
                {
                    "raw_text": frame.raw_text,
                    "general_topic": frame.general_topic,
                    "exact_case": frame.exact_case,
                    "key_qualifiers": frame.key_qualifiers,
                    "entities": frame.entities,
                    "canonical_key": frame.canonical_key,
                }
            )
            return str(raw.get("group_name", frame.general_topic or frame.exact_case)).strip()
        except Exception as exc:
            logger.error("Specific group naming failed: %s", exc)
            return frame.general_topic or frame.exact_case

    def generate_parent_group(
        self,
        child_cluster_descriptions: list[str],
        general_topics: list[str],
    ) -> str:
        """Generate label for a parent cluster."""
        try:
            raw = self._parent_chain.invoke(
                {
                    "child_clusters": "\n\n".join(child_cluster_descriptions),
                }
            )
            fallback = general_topics[0] if general_topics else child_cluster_descriptions[0]
            return str(raw.get("group_name", fallback)).strip()
        except Exception as exc:
            logger.error("Parent group naming failed: %s", exc)
            if general_topics:
                return general_topics[0]
            return child_cluster_descriptions[0]

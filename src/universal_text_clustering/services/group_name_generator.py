"""Group naming service."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import SemanticFrame
from ..prompts import (
    PARENT_GROUP_HUMAN,
    PARENT_GROUP_RECONCILIATION_HUMAN,
    PARENT_GROUP_RECONCILIATION_SYSTEM,
    PARENT_GROUP_SYSTEM,
    SPECIFIC_GROUP_RECONCILIATION_HUMAN,
    SPECIFIC_GROUP_RECONCILIATION_SYSTEM,
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
        self._specific_reconciliation_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SPECIFIC_GROUP_RECONCILIATION_SYSTEM),
                    ("human", SPECIFIC_GROUP_RECONCILIATION_HUMAN),
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
        self._parent_reconciliation_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", PARENT_GROUP_RECONCILIATION_SYSTEM),
                    ("human", PARENT_GROUP_RECONCILIATION_HUMAN),
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
                    "parent_key": frame.parent_key,
                    "core_case": frame.core_case,
                    "exact_case": frame.exact_case,
                    "key_qualifiers": frame.key_qualifiers,
                    "context_details": frame.context_details,
                    "entities": frame.entities,
                    "canonical_key": frame.canonical_key,
                }
            )
            fallback = (
                frame.core_case
                or frame.exact_case
                or frame.general_topic
                or "Не определено"
            )
            return str(raw.get("group_name") or fallback).strip()
        except Exception as exc:
            logger.error("Specific group naming failed: %s", exc)
            return (
                frame.core_case
                or frame.exact_case
                or frame.general_topic
                or "Не определено"
            )

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
            return str(raw.get("group_name") or fallback).strip()
        except Exception as exc:
            logger.error("Parent group naming failed: %s", exc)
            if general_topics:
                return general_topics[0]
            return (
                child_cluster_descriptions[0]
                if child_cluster_descriptions
                else "Не определено"
            )

    def reconcile_specific_groups(
        self,
        specific_cluster_descriptions: list[str],
    ) -> dict[str, str]:
        """Normalize synonymous specific group labels without a fixed category list."""
        if not specific_cluster_descriptions:
            return {}

        try:
            raw = self._specific_reconciliation_chain.invoke(
                {
                    "specific_clusters": "\n\n".join(specific_cluster_descriptions),
                }
            )
            items = raw.get("items", [])
            if not isinstance(items, list):
                return {}

            reconciled: dict[str, str] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                specific_cluster_id = str(item.get("specific_cluster_id", "")).strip()
                specific_group = str(item.get("specific_group", "")).strip()
                if specific_cluster_id and specific_group:
                    reconciled[specific_cluster_id] = specific_group
            return reconciled
        except Exception as exc:
            logger.error("Specific group reconciliation failed: %s", exc)
            return {}

    def reconcile_parent_groups(
        self,
        parent_cluster_descriptions: list[str],
    ) -> dict[str, str]:
        """Normalize synonymous parent group labels without a fixed category list."""
        if not parent_cluster_descriptions:
            return {}

        try:
            raw = self._parent_reconciliation_chain.invoke(
                {
                    "parent_clusters": "\n\n".join(parent_cluster_descriptions),
                }
            )
            items = raw.get("items", [])
            if not isinstance(items, list):
                return {}

            reconciled: dict[str, str] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                parent_cluster_id = str(item.get("parent_cluster_id", "")).strip()
                parent_group = str(item.get("parent_group", "")).strip()
                if parent_cluster_id and parent_group:
                    reconciled[parent_cluster_id] = parent_group
            return reconciled
        except Exception as exc:
            logger.error("Parent group reconciliation failed: %s", exc)
            return {}

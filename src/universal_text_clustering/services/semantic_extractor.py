"""Semantic extraction service."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import InputComment, SemanticFrame
from ..prompts import SEMANTIC_EXTRACTION_HUMAN, SEMANTIC_EXTRACTION_SYSTEM

logger = logging.getLogger(__name__)


class SemanticExtractor:
    """Extract semantic frames from raw texts using the LLM."""

    def __init__(self, llm: BaseChatModel):
        self._parser = JsonOutputParser()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SEMANTIC_EXTRACTION_SYSTEM),
                    ("human", SEMANTIC_EXTRACTION_HUMAN),
                ]
            )
            | llm
            | self._parser
        )

    @staticmethod
    def _as_clean_list(value: object) -> list[str]:
        """Return a clean list even if the LLM sends a scalar value."""
        if value is None:
            return []
        if isinstance(value, list):
            items = value
        else:
            items = [value]
        return [str(item).strip() for item in items if str(item).strip()]

    def extract_one(self, comment: InputComment) -> SemanticFrame:
        """Extract one semantic frame."""
        try:
            raw = self._chain.invoke({"text": comment.text})
            general_topic = str(raw.get("general_topic", "")).strip()
            exact_case = str(raw.get("exact_case", "")).strip()
            core_case = str(raw.get("core_case", exact_case)).strip()
            parent_key = str(raw.get("parent_key", general_topic)).strip().lower()
            return SemanticFrame(
                comment_id=comment.comment_id,
                raw_text=comment.text,
                general_topic=general_topic,
                parent_key=parent_key,
                core_case=core_case,
                exact_case=exact_case,
                key_qualifiers=self._as_clean_list(raw.get("key_qualifiers", [])),
                context_details=self._as_clean_list(raw.get("context_details", [])),
                entities=self._as_clean_list(raw.get("entities", [])),
                canonical_key=str(raw.get("canonical_key", core_case)).strip().lower(),
            )
        except Exception as exc:
            logger.error("Semantic extraction failed for %s: %s", comment.comment_id, exc)
            return SemanticFrame(
                comment_id=comment.comment_id,
                raw_text=comment.text,
                general_topic="Не определено",
                parent_key="не определено",
                core_case="Не определено",
                exact_case=comment.text,
                key_qualifiers=[],
                context_details=[],
                entities=[],
                canonical_key=comment.text.strip().lower(),
            )

    def extract_batch(
        self,
        comments: list[InputComment],
        batch_size: int = 50,
    ) -> list[SemanticFrame]:
        """Extract frames for a batch of comments."""
        frames: list[SemanticFrame] = []
        total = len(comments)
        for index in range(0, len(comments), batch_size):
            chunk = comments[index:index + batch_size]
            logger.info(
                "Semantic extraction: processing comments %d-%d of %d",
                index + 1,
                index + len(chunk),
                total,
            )
            for comment in chunk:
                frames.append(self.extract_one(comment))
        logger.info("Semantic extraction: completed %d comments", len(frames))
        return frames

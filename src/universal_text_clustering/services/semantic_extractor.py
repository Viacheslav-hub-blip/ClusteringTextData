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

    def extract_one(self, comment: InputComment) -> SemanticFrame:
        """Extract one semantic frame."""
        try:
            raw = self._chain.invoke({"text": comment.text})
            return SemanticFrame(
                comment_id=comment.comment_id,
                raw_text=comment.text,
                general_topic=str(raw.get("general_topic", "")).strip(),
                exact_case=str(raw.get("exact_case", "")).strip(),
                key_qualifiers=[
                    str(item).strip()
                    for item in raw.get("key_qualifiers", [])
                    if str(item).strip()
                ],
                entities=[
                    str(item).strip()
                    for item in raw.get("entities", [])
                    if str(item).strip()
                ],
                canonical_key=str(raw.get("canonical_key", "")).strip().lower(),
            )
        except Exception as exc:
            logger.error("Semantic extraction failed for %s: %s", comment.comment_id, exc)
            return SemanticFrame(
                comment_id=comment.comment_id,
                raw_text=comment.text,
                general_topic="Неизвестная тема",
                exact_case=comment.text,
                key_qualifiers=[],
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

"""Structured signal extraction using the LLM."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import InputComment, StructuredSignal
from ..prompts import STRUCTURE_EXTRACTION_HUMAN, STRUCTURE_EXTRACTION_SYSTEM
from .text_utils import clean_list, normalize_text

logger = logging.getLogger(__name__)


class StructureExtractor:
    """Extract stable structured signals from raw comments."""

    def __init__(self, llm: BaseChatModel):
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

    def extract_one(self, comment: InputComment) -> StructuredSignal:
        """Extract one structured signal."""
        try:
            raw = self._chain.invoke({"text": comment.text})
            is_meaningful = bool(raw.get("is_meaningful", True))
            polarity = str(raw.get("polarity", "neutral")).strip().lower()
            phenomenon = str(raw.get("phenomenon", "")).strip()
            subject = str(raw.get("subject", "")).strip()
            parent_focus = str(raw.get("parent_focus", "")).strip()
            specific_focus = str(raw.get("specific_focus", "")).strip()
            parent_key = normalize_text(str(raw.get("parent_key", parent_focus)).strip())
            specific_key = normalize_text(str(raw.get("specific_key", specific_focus)).strip())

            if not is_meaningful:
                return self._fallback_signal(comment, is_meaningful=False)

            if not specific_focus or not parent_focus:
                return self._fallback_signal(comment, is_meaningful=False)

            return StructuredSignal(
                comment_id=comment.comment_id,
                raw_text=comment.text,
                is_meaningful=True,
                polarity=polarity or "neutral",
                phenomenon=phenomenon or "Не определено",
                subject=subject or "Не определено",
                parent_focus=parent_focus,
                parent_key=parent_key or "не определено",
                specific_focus=specific_focus,
                specific_key=specific_key or normalize_text(specific_focus),
                material_details=clean_list(raw.get("material_details", [])),
                context_details=clean_list(raw.get("context_details", [])),
                entities=clean_list(raw.get("entities", [])),
            )
        except Exception as exc:
            logger.error("Structured extraction failed for %s: %s", comment.comment_id, exc)
            return self._fallback_signal(comment, is_meaningful=False)

    def extract_batch(
        self,
        comments: list[InputComment],
        batch_size: int = 50,
    ) -> list[StructuredSignal]:
        """Extract signals for a batch of comments."""
        signals: list[StructuredSignal] = []
        total = len(comments)
        for index in range(0, len(comments), batch_size):
            chunk = comments[index:index + batch_size]
            logger.info(
                "Structured extraction: processing comments %d-%d of %d",
                index + 1,
                index + len(chunk),
                total,
            )
            for comment in chunk:
                signals.append(self.extract_one(comment))
        logger.info("Structured extraction: completed %d comments", len(signals))
        return signals

    @staticmethod
    def _fallback_signal(
        comment: InputComment,
        *,
        is_meaningful: bool,
    ) -> StructuredSignal:
        """Build a safe fallback signal."""
        return StructuredSignal(
            comment_id=comment.comment_id,
            raw_text=comment.text,
            is_meaningful=is_meaningful,
            polarity="neutral",
            phenomenon="Не определено",
            subject="Не определено",
            parent_focus="Не определено",
            parent_key="не определено",
            specific_focus="Не определено",
            specific_key="не определено",
            material_details=[],
            context_details=[],
            entities=[],
        )

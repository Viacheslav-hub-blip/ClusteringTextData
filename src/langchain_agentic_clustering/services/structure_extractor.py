"""LLM-based extraction of structured banking signals."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import InputComment, StructuredSignal
from ..prompts import STRUCTURE_EXTRACTION_HUMAN, STRUCTURE_EXTRACTION_SYSTEM
from .text_utils import UNDEFINED_LABEL, clean_list, normalize_text

logger = logging.getLogger(__name__)


class StructureExtractor:
    """Extract structured signals from raw comments."""

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

    def extract_batch(
        self,
        comments: list[InputComment],
        *,
        batch_size: int = 50,
    ) -> list[StructuredSignal]:
        """Extract signals for all comments."""
        signals: list[StructuredSignal] = []
        total = len(comments)
        for index in range(0, total, batch_size):
            chunk = comments[index:index + batch_size]
            logger.info(
                "Structure extraction: comments %d-%d of %d",
                index + 1,
                index + len(chunk),
                total,
            )
            for comment in chunk:
                signals.append(self.extract_one(comment))
        return signals

    def extract_one(self, comment: InputComment) -> StructuredSignal:
        """Extract one structured signal."""
        try:
            raw = self._chain.invoke({"text": comment.text})
            return self._build_signal(comment, raw)
        except Exception as exc:
            logger.warning("Structure extraction failed for %s: %s", comment.comment_id, exc)
            return self._fallback_signal(comment)

    def _build_signal(self, comment: InputComment, raw: dict) -> StructuredSignal:
        is_meaningful = bool(raw.get("is_meaningful", True))

        if not is_meaningful:
            return self._fallback_signal(comment)

        polarity = str(raw.get("polarity", "neutral")).strip().lower() or "neutral"
        banking_area = str(raw.get("banking_area", UNDEFINED_LABEL)).strip() or UNDEFINED_LABEL
        phenomenon = str(raw.get("phenomenon", UNDEFINED_LABEL)).strip() or UNDEFINED_LABEL
        object_name = str(raw.get("object_name", UNDEFINED_LABEL)).strip() or UNDEFINED_LABEL
        parent_focus = str(raw.get("parent_focus", UNDEFINED_LABEL)).strip() or UNDEFINED_LABEL
        specific_focus = str(raw.get("specific_focus", UNDEFINED_LABEL)).strip() or UNDEFINED_LABEL
        parent_key = normalize_text(str(raw.get("parent_key", parent_focus)).strip()) or normalize_text(UNDEFINED_LABEL)
        specific_key = normalize_text(str(raw.get("specific_key", specific_focus)).strip()) or normalize_text(UNDEFINED_LABEL)

        if not parent_focus or not specific_focus:
            return self._fallback_signal(comment)

        return StructuredSignal(
            comment_id=comment.comment_id,
            raw_text=comment.text,
            is_meaningful=True,
            polarity=polarity,
            banking_area=banking_area,
            phenomenon=phenomenon,
            object_name=object_name,
            parent_focus=parent_focus,
            parent_key=parent_key,
            specific_focus=specific_focus,
            specific_key=specific_key,
            material_details=clean_list(raw.get("material_details", [])),
            context_details=clean_list(raw.get("context_details", [])),
            entities=clean_list(raw.get("entities", [])),
        )

    @staticmethod
    def _fallback_signal(comment: InputComment) -> StructuredSignal:
        """Return a safe fallback signal."""
        return StructuredSignal(
            comment_id=comment.comment_id,
            raw_text=comment.text,
            is_meaningful=False,
            polarity="neutral",
            banking_area=UNDEFINED_LABEL,
            phenomenon=UNDEFINED_LABEL,
            object_name=UNDEFINED_LABEL,
            parent_focus=UNDEFINED_LABEL,
            parent_key=normalize_text(UNDEFINED_LABEL),
            specific_focus=UNDEFINED_LABEL,
            specific_key=normalize_text(UNDEFINED_LABEL),
            material_details=[],
            context_details=[],
            entities=[],
        )

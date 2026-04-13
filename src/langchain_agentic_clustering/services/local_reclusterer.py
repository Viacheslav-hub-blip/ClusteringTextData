"""Localized reclustering for one suspicious neighborhood."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..models import StructuredSignal
from ..prompts import LOCAL_RECLUSTER_HUMAN, LOCAL_RECLUSTER_SYSTEM
from .text_utils import UNDEFINED_LABEL, clean_list, normalize_text


class LocalReclusterer:
    """Repair a small neighborhood instead of the full dataset."""

    def __init__(self, llm: BaseChatModel):
        self._parser = JsonOutputParser()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", LOCAL_RECLUSTER_SYSTEM),
                    ("human", LOCAL_RECLUSTER_HUMAN),
                ]
            )
            | llm
            | self._parser
        )

    def repair(
        self,
        *,
        guidance: str,
        neighborhood_json: str,
        comments_json: str,
        fallback_signals_by_comment_id: dict[str, StructuredSignal],
    ) -> dict[str, StructuredSignal]:
        """Return repaired signals for the local neighborhood."""
        raw = self._chain.invoke(
            {
                "guidance": guidance or "Сделай локальную перекластеризацию аккуратно и в банковском контексте.",
                "neighborhood_json": neighborhood_json,
                "comments_json": comments_json,
            }
        )
        items = raw if isinstance(raw, list) else []
        repaired: dict[str, StructuredSignal] = {}
        for item in items:
            comment_id = str(item.get("comment_id", "")).strip()
            fallback = fallback_signals_by_comment_id.get(comment_id)
            if not comment_id or fallback is None:
                continue
            is_meaningful = bool(item.get("is_meaningful", fallback.is_meaningful))
            if not is_meaningful:
                repaired[comment_id] = self._undefined_signal(fallback)
                continue
            repaired[comment_id] = StructuredSignal(
                comment_id=comment_id,
                raw_text=fallback.raw_text,
                is_meaningful=True,
                polarity=str(item.get("polarity", fallback.polarity)).strip().lower() or fallback.polarity,
                banking_area=str(item.get("banking_area", fallback.banking_area)).strip() or fallback.banking_area,
                phenomenon=str(item.get("phenomenon", fallback.phenomenon)).strip() or fallback.phenomenon,
                object_name=str(item.get("object_name", fallback.object_name)).strip() or fallback.object_name,
                parent_focus=str(item.get("parent_focus", fallback.parent_focus)).strip() or fallback.parent_focus,
                parent_key=normalize_text(str(item.get("parent_key", fallback.parent_key)).strip()) or fallback.parent_key,
                specific_focus=str(item.get("specific_focus", fallback.specific_focus)).strip() or fallback.specific_focus,
                specific_key=normalize_text(str(item.get("specific_key", fallback.specific_key)).strip()) or fallback.specific_key,
                material_details=clean_list(item.get("material_details", fallback.material_details)),
                context_details=clean_list(item.get("context_details", fallback.context_details)),
                entities=clean_list(item.get("entities", fallback.entities)),
            )

        for comment_id, fallback in fallback_signals_by_comment_id.items():
            repaired.setdefault(comment_id, fallback)

        return repaired

    @staticmethod
    def _undefined_signal(fallback: StructuredSignal) -> StructuredSignal:
        return StructuredSignal(
            comment_id=fallback.comment_id,
            raw_text=fallback.raw_text,
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

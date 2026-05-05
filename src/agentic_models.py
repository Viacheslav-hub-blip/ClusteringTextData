"""Pydantic-схемы агентской постобработки.

Файл содержит:
- ``ClusterAuditDecision`` — решение аудита кластера;
- ``PostProcessingGroupName`` — финальное название группы.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClusterAuditDecision(BaseModel):
    """Результат аудита одного кластера.

    Args:
        remove_comment_ids: Идентификаторы комментариев, которые нужно убрать из кластера.
        reason: Краткое объяснение решения аудита.

    Returns:
        Валидированная схема решения LLM для аудита кластера.
    """

    remove_comment_ids: list[str] = Field(
        default_factory=list,
        description="comment_id values that clearly do not belong to this cluster.",
    )
    reason: str = Field(description="Short explanation of the audit result.")


class PostProcessingGroupName(BaseModel):
    """Название группы после постобработки.

    Args:
        group_name: Короткое человекочитаемое название группы.

    Returns:
        Валидированная схема финального имени группы.
    """

    group_name: str = Field(description="Short human-readable final cluster name.")

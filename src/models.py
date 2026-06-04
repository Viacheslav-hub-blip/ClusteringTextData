"""Модели данных для упрощенного pipeline кластеризации.

Файл содержит:
- ``DecisionType`` — тип решения о группе комментария;
- ``InputComment`` — валидированный входной комментарий;
- ``NormalizationResult`` — результат локальной нормализации;
- ``StoredComment`` — сохраненный комментарий во внутреннем хранилище;
- ``CommentGroup`` — внутренняя группа комментариев;
- ``SimilarityHit`` — найденный похожий комментарий;
- ``CandidateGroup`` — группа-кандидат для LLM;
- ``PrimaryDecision`` — решение LLM о группе.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DecisionType(str, Enum):
    """Тип решения о назначении комментария в группу.

    Returns:
        Enum со значениями для существующей группы, новой группы и неопределенного комментария.
    """

    EXISTING_GROUP = "existing_group"
    NEW_GROUP = "new_group"
    UNDEFINED = "undefined"


@dataclass(slots=True)
class InputComment:
    """Валидированный входной комментарий.

    Args:
        comment_id: Идентификатор комментария.
        text: Исходный текст комментария.

    Returns:
        Объект входного комментария.
    """

    comment_id: str
    text: str


@dataclass(slots=True)
class NormalizationResult:
    """Результат локальной нормализации одного комментария.

    Args:
        normalized_text: Технически очищенный текст.
        is_meaningful: Признак содержательного комментария.
        reason: Краткое объяснение решения.

    Returns:
        Объект результата нормализации.
    """

    normalized_text: str
    is_meaningful: bool
    reason: str


@dataclass(slots=True)
class StoredComment:
    """Сохраненный комментарий во внутреннем хранилище.

    Args:
        comment_id: Идентификатор комментария.
        raw_text: Исходный текст.
        normalized_text: Технически очищенный текст.
        embedding: Векторное представление текста.
        group_id: Идентификатор назначенной группы.
        decision_type: Тип решения о группе.
        decision_reason: Объяснение решения.

    Returns:
        Объект сохраненного комментария.
    """

    comment_id: str
    raw_text: str
    normalized_text: str
    embedding: list[float] | None
    group_id: str
    decision_type: DecisionType
    decision_reason: str


@dataclass(slots=True)
class CommentGroup:
    """Группа комментариев во внутреннем хранилище.

    Args:
        group_id: Идентификатор группы.
        group_name: Человекочитаемое название группы.
        member_comment_ids: Идентификаторы комментариев группы.

    Returns:
        Объект группы комментариев.
    """

    group_id: str
    group_name: str = ""
    member_comment_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SimilarityHit:
    """Похожий комментарий, найденный поиском.

    Args:
        comment_id: Идентификатор найденного комментария.
        group_id: Идентификатор группы найденного комментария.
        similarity: Оценка похожести.

    Returns:
        Объект найденного совпадения.
    """

    comment_id: str
    group_id: str
    similarity: float


@dataclass(slots=True)
class CandidateGroup:
    """Группа-кандидат для передачи в LLM.

    Args:
        group_id: Идентификатор группы-кандидата.
        best_similarity: Максимальная похожесть среди найденных комментариев группы.
        representative_comment_ids: Идентификаторы примеров группы.

    Returns:
        Объект группы-кандидата.
    """

    group_id: str
    best_similarity: float
    representative_comment_ids: list[str]


@dataclass(slots=True)
class PrimaryDecision:
    """Решение LLM о назначении комментария в группу.

    Args:
        decision_type: Тип решения: существующая группа или новая группа.
        group_id: Идентификатор выбранной группы.
        reason: Объяснение решения.

    Returns:
        Объект решения LLM.
    """

    decision_type: DecisionType
    group_id: str
    reason: str

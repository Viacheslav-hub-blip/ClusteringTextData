"""Конфигурация prompt-шаблонов для pipeline кластеризации.

Файл содержит:
- ``PrimaryPromptConfig`` — конфигурацию system prompt выбора группы.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import PRIMARY_DECISION_SYSTEM


@dataclass(slots=True)
class PrimaryPromptConfig:
    """Prompt-конфигурация упрощенного pipeline.

    Args:
        primary_decision_system: System prompt для выбора существующей или новой группы.

    Returns:
        Экземпляр с system prompt для сборки LCEL-цепочки.
    """

    primary_decision_system: str

    @classmethod
    def default(cls) -> "PrimaryPromptConfig":
        """Создает конфигурацию prompt-ов из дефолтов библиотеки.

        Args:
            Входные аргументы отсутствуют.

        Returns:
            Конфигурация с system prompt из ``prompts.py``.
        """
        return cls(
            primary_decision_system=PRIMARY_DECISION_SYSTEM,
        )

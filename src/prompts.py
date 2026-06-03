"""Prompt-шаблоны для упрощенного pipeline кластеризации.

Файл содержит:
- ``PRIMARY_DECISION_SYSTEM`` — system prompt выбора существующей или новой группы;
- ``EMPTY_HUMAN_MESSAGE`` — пустое human-сообщение для совместимости с chat-моделями.
"""

from __future__ import annotations

PRIMARY_DECISION_SYSTEM = """
Ты выбираешь, относится ли новый комментарий к одной из существующих групп или требует новой группы.

Новый комментарий:
{raw_text}

Технически предобработанный комментарий:
{normalized_text}

Кандидатные группы:
{candidate_groups}

Верни только валидный JSON без markdown.
JSON должен содержать поля:
- decision_type: existing_group или new_group;
- group_id: ID выбранной группы или пустая строка для новой группы;
- group_name: для new_group короткое название новой группы, для existing_group можно вернуть пустую строку;
- reason: краткое объяснение выбора.
""".strip()

EMPTY_HUMAN_MESSAGE = ""

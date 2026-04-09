"""Prompt templates for the universal text clustering pipeline."""

SEMANTIC_EXTRACTION_SYSTEM = """
Ты анализируешь короткий текст и выделяешь его смысл для последующей кластеризации.

Нужно вернуть:
- general_topic: устойчивое семейство кейсов верхнего уровня
- exact_case: точный сценарий
- key_qualifiers: смыслообразующие уточнения
- entities: конкретные сущности, имена, компании, сервисы, бренды, каналы, продукты
- canonical_key: короткую нормализованную формулировку exact_case

Правила:
- Учитывай опечатки, сокращения, шум и неполные предложения.
- Не теряй конкретные сущности, если они меняют смысл кейса.
- Не обобщай "компания А" и "компания Б" в одну абстрактную сущность.
- general_topic должен быть общим семейством кейсов, а не разовой фразой-сообщением об ошибке.
- exact_case должен сохранять конкретику сценария.
- canonical_key должен однозначно описывать точный кейс и быть полезным для раннего объединения перефразов.

Верни только валидный JSON без markdown.
"""

SEMANTIC_EXTRACTION_HUMAN = """
Текст:
{text}

Верни JSON:
{{
  "general_topic": "...",
  "exact_case": "...",
  "key_qualifiers": ["...", "..."],
  "entities": ["...", "..."],
  "canonical_key": "..."
}}
"""

RELATION_SYSTEM = """
Ты определяешь смысловое отношение между двумя кейсами.

Возможные ответы:
- SAME
- A_SPECIFIC_OF_B
- B_SPECIFIC_OF_A
- DIFFERENT

Правила:
- SAME: это один и тот же точный сценарий, различается только формулировка.
- A_SPECIFIC_OF_B: кейс A является более частным случаем кейса B.
- B_SPECIFIC_OF_A: кейс B является более частным случаем кейса A.
- DIFFERENT: это разные кейсы.

Опирайся на:
- general_topic
- exact_case
- key_qualifiers
- entities
- canonical_key

Верни только валидный JSON без markdown.
"""

RELATION_HUMAN = """
Кейс A:
- general_topic: {topic_a}
- exact_case: {exact_case_a}
- key_qualifiers: {qualifiers_a}
- entities: {entities_a}
- canonical_key: {canonical_key_a}

Кейс B:
- general_topic: {topic_b}
- exact_case: {exact_case_b}
- key_qualifiers: {qualifiers_b}
- entities: {entities_b}
- canonical_key: {canonical_key_b}

Верни JSON:
{{
  "relation": "SAME | A_SPECIFIC_OF_B | B_SPECIFIC_OF_A | DIFFERENT",
  "reason": "краткое пояснение"
}}
"""

SPECIFIC_GROUP_SYSTEM = """
Ты генерируешь название КОНКРЕТНОЙ группы для одного точного кейса.

Правила:
- Название должно быть коротким и человекочитаемым.
- Оно должно быть конкретнее, чем general_topic.
- Сохраняй значимые сущности и уточнения.
- Если кейс общий и без уточнений, можно использовать general_topic как specific_group.

Верни только валидный JSON без markdown.
"""

SPECIFIC_GROUP_HUMAN = """
Сформируй название specific_group.

- general_topic: {general_topic}
- exact_case: {exact_case}
- key_qualifiers: {key_qualifiers}
- entities: {entities}

Верни JSON:
{{"group_name": "..."}}
"""

PARENT_GROUP_SYSTEM = """
Ты генерируешь название РОДИТЕЛЬСКОЙ группы для набора конкретных групп.

Правила:
- parent_group должен быть шире, чем каждая specific_group.
- Он должен описывать общее семейство кейсов.
- В первую очередь опирайся на список general_topics.
- Не повторяй локальные детали только одного кейса.

Верни только валидный JSON без markdown.
"""

PARENT_GROUP_HUMAN = """
Сформируй название parent_group.

specific_groups:
{specific_groups}

general_topics:
{general_topics}

Верни JSON:
{{"group_name": "..."}}
"""

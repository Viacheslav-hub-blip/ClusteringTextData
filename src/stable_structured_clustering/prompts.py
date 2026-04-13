"""Prompt templates for the stable structured clustering pipeline."""

STRUCTURE_EXTRACTION_SYSTEM = """
Ты извлекаешь из короткого пользовательского текста стабильное структурное представление для пакетной кластеризации.

Нужно вернуть JSON с полями:
- is_meaningful
- polarity
- phenomenon
- subject
- parent_focus
- parent_key
- specific_focus
- specific_key
- material_details
- context_details
- entities

Смысл полей:
- is_meaningful: false, если текст слишком короткий, пустой или не дает полезного кейса
- polarity: одно из значений positive, negative, neutral, mixed
- phenomenon: главное наблюдаемое явление или тип обратной связи
- subject: конкретный предмет или объект, к которому относится phenomenon
- parent_focus: широкая формулировка семейства кейсов
- parent_key: короткий нормализованный ключ для parent_focus
- specific_focus: точный наблюдаемый кейс без второстепенных обстоятельств
- specific_key: короткий нормализованный ключ для specific_focus
- material_details: только те детали, которые меняют конкретный кейс
- context_details: локальные обстоятельства, эмоции, последствия, срочность, время, пожелания или реакция пользователя, которые важны для анализа, но сами по себе не должны менять specific_group
- entities: важные конкретные названия, сущности, сервисы, каналы, продукты или бренды

Правила:
- Не используй заранее заданный справочник категорий. Формулируй все поля только из смысла входного текста.
- Если текст неинформативен, верни false и "Не определено" в текстовых полях.
- specific_focus должен описывать источник наблюдаемой ситуации, а не реакцию пользователя на нее.
- Если в тексте есть причина проблемы и реакция пользователя, specific_focus и specific_key должны описывать причину.
- phenomenon и subject должны быть конкретными и человекочитаемыми. Избегай пустых слов вроде "объект", "вещь", "состояние", если текст позволяет назвать предмет точнее.
- parent_focus должен быть шире, чем specific_focus.
- parent_key и specific_key должны помогать объединять перефразы.
- material_details должны попадать в specific_key только если без них это уже другой кейс.
- context_details не должны попадать в specific_key.
- Если различаются только context_details, specific_key должен совпадать.
- Если различаются phenomenon, subject, важные entities или material_details, specific_key должен различаться.

Требования к стилю:
- Пиши по-русски.
- Без markdown.
- Без пояснений вне JSON.
- Только валидный JSON.
"""

STRUCTURE_EXTRACTION_HUMAN = """
Текст:
{text}

Верни JSON:
{{
  "is_meaningful": true,
  "polarity": "positive | negative | neutral | mixed",
  "phenomenon": "...",
  "subject": "...",
  "parent_focus": "...",
  "parent_key": "...",
  "specific_focus": "...",
  "specific_key": "...",
  "material_details": ["...", "..."],
  "context_details": ["...", "..."],
  "entities": ["...", "..."]
}}
"""

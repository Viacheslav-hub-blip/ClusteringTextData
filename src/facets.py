"""Извлечение фасетов для кластеризации текстовых комментариев.

Файл содержит:
- ``DEFAULT_FACET_WEIGHTS`` — веса фасетов для мягкого скоринга похожести;
- ``CommentFacets`` — контейнер найденных фасетов комментария;
- ``extract_facets`` — извлечение бизнес-признаков из текста без LLM;
- ``score_facets_against_profile`` — расчет похожести фасетов комментария и профиля группы;
- ``format_facet_profile`` — компактное форматирование профиля группы для prompt-а.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_FACET_WEIGHTS: dict[str, float] = {
    "operation_type": 1.0,
    "product_or_service": 0.9,
    "merchant": 0.8,
    "company": 0.7,
    "place": 0.5,
    "time_period": 0.4,
    "city": 0.4,
    "channel": 0.35,
    "problem_type": 0.25,
    "unknown_entity": 0.2,
}

_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u00ab": '"', "\u00bb": '"',
    "\u2014": "-", "\u2013": "-",
})

_FACET_PATTERNS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "operation_type": [
        ("оплата", (r"\bоплат\w+", r"\bплат[её]ж\w*", r"\bпокупк\w+", r"\bзаплат\w+")),
        ("перевод", (r"\bперев[её]л\w*", r"\bперевод\w*", r"\bотправ\w+\s+деньг")),
        ("перевод через СБП", (r"\bсбп\b", r"систем[а-я]+\s+быстр[а-я]+\s+плат[её]ж")),
        ("снятие наличных", (r"\bсня\w+\s+налич", r"\bснять\s+деньг", r"\bвыдач\w+\s+налич")),
        ("списание", (r"\bсписан\w+", r"\bсписал\w+", r"\bснял[ио]сь\b")),
        ("пополнение", (r"\bпополн\w+", r"\bвнес\w+\s+деньг")),
        ("возврат", (r"\bвозврат\w*", r"\bвернуть\b", r"\bвернули\b")),
        ("подтверждение операции", (r"\bподтвержд\w+", r"\bкод\w+\s+подтвержд", r"\bсмс[- ]?код")),
        ("блокировка операции", (r"\bблокир\w+", r"\bзаблок\w+", r"\bотклонил\w+")),
    ],
    "product_or_service": [
        ("мобильная связь", (r"\bмтс\b", r"\bмегафон\b", r"\bmegafon\b", r"\bбилайн\b", r"\bbeeline\b", r"\bтеле2\b", r"\btele2\b", r"\bтелефон\w*", r"\bмобильн\w+\s+связ")),
        ("интернет", (r"\bинтернет\w*", r"\bпровайдер\w*", r"\bроутер\w*", r"\bдомашн\w+\s+интернет")),
        ("доставка еды", (r"\bдоставк\w+\s+ед", r"\bеда\b", r"\bяндекс\s*еда\b", r"\bdelivery\s*club\b", r"\bсамокат\b", r"\bлавк\w+")),
        ("подписки и цифровые сервисы", (r"\bподписк\w*", r"\bпродлен\w+", r"\bmovavi\b", r"\bwink\b", r"\bokko\b", r"\bкинопоиск\b", r"\bivi\b")),
        ("маркетплейсы", (r"\bozon\b", r"\bозон\b", r"\bwildberries\b", r"\bвайлдберр\w+", r"\bмаркетплейс\w*")),
        ("кредиты", (r"\bкредит\w*", r"\bпогашен\w+\s+кредит", r"\bежемесячн\w+\s+плат[её]ж")),
        ("карты", (r"\bкарт\w+", r"\bmir\s*pay\b", r"\bмир\s*п[еэ]й\b")),
        ("вклады", (r"\bвклад\w*", r"\bдепозит\w*")),
        ("коммунальные услуги", (r"\bжкх\b", r"\bкоммунал\w+", r"\bквартплат\w+")),
        ("наличные", (r"\bналичн\w*", r"\bбанкнот\w*")),
    ],
    "merchant": [
        ("Яндекс Еда", (r"\bяндекс\s*еда\b",)),
        ("Самокат", (r"\bсамокат\b",)),
        ("Delivery Club", (r"\bdelivery\s*club\b",)),
        ("Ozon", (r"\bozon\b", r"\bозон\b")),
        ("Wildberries", (r"\bwildberries\b", r"\bвайлдберр\w+")),
        ("Wink", (r"\bwink\b",)),
        ("Movavi", (r"\bmovavi\b",)),
    ],
    "company": [
        ("МТС", (r"\bмтс\b",)),
        ("Мегафон", (r"\bмегафон\b", r"\bmegafon\b")),
        ("Билайн", (r"\bбилайн\b", r"\bbeeline\b")),
        ("Tele2", (r"\bтеле2\b", r"\btele2\b")),
        ("Сбербанк", (r"\bсбер\w*", r"\bсбербанк\w*")),
    ],
    "place": [
        ("банкомат", (r"\bбанкомат\w*",)),
        ("офис банка", (r"\bофис\w*", r"\bотделени\w+")),
        ("терминал", (r"\bтерминал\w*",)),
        ("магазин", (r"\bмагазин\w*", r"\bторгов\w+\s+точ")),
        ("клиника", (r"\bклиник\w*", r"\bмедицин\w+\s+учреждени")),
    ],
    "time_period": [
        ("ночь", (r"\bноч\w*", r"\bночью\b")),
        ("утро", (r"\bутр\w*", r"\bутром\b")),
        ("день", (r"\bднем\b", r"\bдн[её]м\b")),
        ("вечер", (r"\bвечер\w*", r"\bвечером\b")),
        ("24 часа", (r"\b24\s*час", r"\bсутк\w+")),
        ("месяц и больше", (r"\bмесяц\w*", r"\bнедел\w+")),
    ],
    "city": [
        ("Москва", (r"\bмоскв\w*",)),
        ("Санкт-Петербург", (r"\bсанкт[- ]петербург\w*", r"\bспб\b")),
        ("Казань", (r"\bказан\w*",)),
        ("Екатеринбург", (r"\bекатеринбург\w*",)),
        ("Новосибирск", (r"\bновосибирск\w*",)),
    ],
    "channel": [
        ("мобильное приложение", (r"\bприложени\w*", r"\bличн\w+\s+кабинет")),
        ("SMS", (r"\bсмс\b", r"\bsms\b")),
        ("push", (r"\bпуш\b", r"\bpush\b")),
        ("оператор поддержки", (r"\bоператор\w*", r"\bподдержк\w*", r"\b900\b")),
        ("офис банка", (r"\bофис\w*", r"\bотделени\w+")),
        ("банкомат", (r"\bбанкомат\w*",)),
    ],
    "problem_type": [
        ("блокировка", (r"\bблокир\w+", r"\bзаблок\w+", r"\bотклонил\w+")),
        ("подтверждение", (r"\bподтвержд\w+", r"\bкод\w+\s+подтвержд")),
        ("не приходит код", (r"\bне\s+приход\w+.*\b(код|смс|sms)", r"\bкод\b.*\bне\s+приход")),
        ("задержка", (r"\bжд\w+", r"\bзадерж\w+", r"\bдолго\b", r"\bсутк\w+")),
        ("комиссия", (r"\bкомисс\w+", r"\bпроцент\w+")),
        ("техническая ошибка", (r"\bошибк\w+", r"\bбаг\w+", r"\bне\s+работ\w+", r"\bсбой\b")),
        ("несанкционированное списание", (r"\bбез\s+моего\s+разрешен", r"\bнесанкционирован", r"\bмошенник\w+")),
        ("неудобный процесс", (r"\bнеудоб\w+", r"\bсложн\w+", r"\bмного\s+времени")),
    ],
}

_UNKNOWN_ENTITY_RE = re.compile(r"\b[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9&.-]{2,}\b")
_SKIP_UNKNOWN_ENTITIES = {
    "Банк", "Сбер", "Сбербанк", "России", "РФ", "SMS", "СБП", "PIN", "Face", "ID",
    "Почему", "При", "Пытался", "Пыталась", "Списали", "Перевод", "Не",
    "После", "Если", "Хочу", "Нужно", "Банку",
}
_KNOWN_CANONICAL_VALUES = {
    canonical_value.lower()
    for entries in _FACET_PATTERNS.values()
    for canonical_value, _patterns in entries
}


@dataclass(slots=True)
class CommentFacets:
    """Набор фасетов, извлеченных из одного комментария.

    Args:
        values: Словарь, где ключ — тип фасета, а значение — список нормализованных значений.

    Returns:
        Экземпляр с фасетами комментария и методами форматирования.
    """

    values: dict[str, list[str]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Проверяет, есть ли в контейнере хотя бы один фасет.

        Args:
            Входные аргументы отсутствуют.

        Returns:
            ``True``, если фасеты не найдены, иначе ``False``.
        """
        return not any(self.values.values())

    def to_prompt_text(self) -> str:
        """Форматирует фасеты комментария для LLM prompt-а.

        Args:
            Входные аргументы отсутствуют.

        Returns:
            Многострочный текст с найденными фасетами или сообщение об их отсутствии.
        """
        if self.is_empty():
            return "Фасеты не найдены."
        return "\n".join(
            f"{facet_type}: {', '.join(values)}"
            for facet_type, values in self.values.items()
            if values
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Возвращает фасеты как обычный словарь.

        Args:
            Входные аргументы отсутствуют.

        Returns:
            Словарь со списками значений фасетов.
        """
        return {key: list(values) for key, values in self.values.items()}


def normalize_for_facets(value: str) -> str:
    """Нормализует текст для rule-based извлечения фасетов.

    Args:
        value: Исходный текст комментария.

    Returns:
        Строка в нижнем регистре с нормализованными пробелами, кавычками и буквой ``ё``.
    """
    text = str(value).translate(_QUOTE_MAP).lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_facets(value: str) -> CommentFacets:
    """Извлекает фасеты из комментария без обращения к LLM.

    Args:
        value: Исходный текст комментария.

    Returns:
        ``CommentFacets`` с найденными типами операций, продуктами, компаниями, каналами и проблемами.
    """
    normalized = normalize_for_facets(value)
    facets: dict[str, list[str]] = {}
    for facet_type, entries in _FACET_PATTERNS.items():
        for canonical_value, patterns in entries:
            if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
                _append_unique(facets, facet_type, canonical_value)

    for entity in _extract_unknown_entities(value):
        _append_unique(facets, "unknown_entity", entity)

    return CommentFacets(values=facets)


def score_facets_against_profile(
        facets: CommentFacets,
        profile: dict[str, dict[str, int]],
        *,
        weights: dict[str, float] | None = None,
) -> float:
    """Считает мягкую похожесть фасетов комментария и агрегированного профиля группы.

    Args:
        facets: Фасеты текущего комментария.
        profile: Профиль группы в формате ``тип фасета -> значение -> количество``.
        weights: Пользовательские веса фасетов или ``None`` для дефолтных весов.

    Returns:
        Число от ``0.0`` до ``1.0``, где ``1.0`` означает полное совпадение найденных фасетов.
    """
    if facets.is_empty() or not profile:
        return 0.0

    active_weights = weights or DEFAULT_FACET_WEIGHTS
    matched_weight = 0.0
    total_weight = 0.0

    for facet_type, query_values in facets.values.items():
        if not query_values:
            continue
        weight = active_weights.get(facet_type, 0.1)
        total_weight += weight
        group_values = profile.get(facet_type, {})
        if any(value in group_values for value in query_values):
            matched_weight += weight

    return matched_weight / total_weight if total_weight else 0.0


def format_facet_profile(profile: dict[str, dict[str, int]], *, max_values_per_type: int = 4) -> str:
    """Форматирует агрегированный профиль группы для prompt-а.

    Args:
        profile: Профиль группы в формате ``тип фасета -> значение -> количество``.
        max_values_per_type: Максимальное количество значений одного типа фасета в выводе.

    Returns:
        Компактная многострочная строка с самыми частыми фасетами группы.
    """
    if not profile:
        return "Профиль фасетов пуст."

    lines: list[str] = []
    for facet_type, counts in profile.items():
        if not counts:
            continue
        top_values = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:max_values_per_type]
        formatted_values = ", ".join(f"{value} ({count})" for value, count in top_values)
        lines.append(f"{facet_type}: {formatted_values}")
    return "\n".join(lines) if lines else "Профиль фасетов пуст."


def update_facet_profile(profile: dict[str, dict[str, int]], facets: CommentFacets, *, delta: int = 1) -> None:
    """Обновляет агрегированный профиль группы фасетами комментария.

    Args:
        profile: Изменяемый профиль группы.
        facets: Фасеты комментария, которые нужно добавить или вычесть.
        delta: Изменение счетчика. Для добавления используется ``1``, для удаления ``-1``.

    Returns:
        ``None``. Профиль изменяется на месте.
    """
    for facet_type, values in facets.values.items():
        if not values:
            continue
        counts = profile.setdefault(facet_type, {})
        for value in values:
            counts[value] = counts.get(value, 0) + delta
            if counts[value] <= 0:
                del counts[value]
        if not counts:
            del profile[facet_type]


def _append_unique(target: dict[str, list[str]], key: str, value: str) -> None:
    """Добавляет значение в список фасетов без дублей.

    Args:
        target: Изменяемый словарь фасетов.
        key: Тип фасета.
        value: Нормализованное значение фасета.

    Returns:
        ``None``. Словарь изменяется на месте.
    """
    values = target.setdefault(key, [])
    if value not in values:
        values.append(value)


def _extract_unknown_entities(value: str) -> Iterable[str]:
    """Извлекает неизвестные именованные сущности для сохранения редких категорий.

    Args:
        value: Исходный текст комментария.

    Returns:
        Итератор строк с сущностями, которые не входят в список служебных исключений.
    """
    seen: set[str] = set()
    for match in _UNKNOWN_ENTITY_RE.finditer(str(value)):
        entity = match.group(0).strip(".,;:!?()[]{}\"'")
        entity_key = entity.lower()
        is_probable_code_or_brand = entity.isupper() or bool(re.search(r"[A-Za-z0-9&.-]", entity))
        if match.start() == 0 and not is_probable_code_or_brand:
            continue
        if (
                entity in _SKIP_UNKNOWN_ENTITIES
                or entity_key in _KNOWN_CANONICAL_VALUES
                or entity_key in seen
        ):
            continue
        seen.add(entity_key)
        yield entity

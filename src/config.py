"""Конфигурация prompt-запросов для pipeline кластеризации.

Файл содержит:
- ``PrimaryPromptConfig`` — prompt-настройки базовой кластеризации;
- ``AgenticPromptConfig`` — prompt-настройки агентской постобработки;
- ``ClusteringPromptConfig`` — объединенная конфигурация всех prompt-запросов.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DOMAIN_CONTEXT = """
Ты работаешь с пользовательскими текстовыми комментариями. Нужно группировать записи по
смыслу проблемы, а не по совпадению отдельных слов.
""".strip()

CLUSTERING_POLICY_BLOCK = """
Правила кластеризации:
1. Объединяй комментарии только если совпадают объект жалобы, причина недовольства и ожидаемое изменение.
2. Не объединяй противоположные смыслы, даже если они используют одинаковые слова.
3. Не добавляй общий негатив без конкретного кейса в содержательные группы.
4. Не выдумывай факты, которых нет в исходном тексте.
5. При сомнении выбирай более узкую группу или новую группу.
""".strip()

NORMALIZATION_SYSTEM = f"""
Ты нормализуешь пользовательский комментарий перед кластеризацией.

{DOMAIN_CONTEXT}

Верни только валидный JSON без markdown.
JSON должен содержать поля:
- normalized_text: короткая смысловая формулировка комментария;
- is_meaningful: true, если комментарий содержит конкретный кейс;
- reason: краткое объяснение решения.
""".strip()

NORMALIZATION_HUMAN = """
Комментарий:
{text}
""".strip()

PRIMARY_DECISION_SYSTEM = f"""
Ты выбираешь, относится ли новый комментарий к одной из существующих групп или требует новой группы.

{DOMAIN_CONTEXT}

{CLUSTERING_POLICY_BLOCK}

Верни только валидный JSON без markdown.
JSON должен содержать поля:
- decision_type: existing_group или new_group;
- group_id: ID выбранной группы или пустая строка для новой группы;
- reason: краткое объяснение выбора.
""".strip()

PRIMARY_DECISION_HUMAN = """
Исходный комментарий:
{raw_text}

Нормализованный комментарий:
{normalized_text}

Кандидатные группы:
{candidate_groups}
""".strip()

GROUP_NAMING_SYSTEM = f"""
Ты генерируешь короткое название группы пользовательских комментариев.

{DOMAIN_CONTEXT}

Название должно быть конкретным, без лишних общих слов и без выдуманных деталей.
Верни только валидный JSON без markdown.
JSON должен содержать поле:
- group_name: короткое человекочитаемое название группы.
""".strip()

GROUP_NAMING_HUMAN = """
Примеры комментариев группы:
{group_examples}
""".strip()


@dataclass(slots=True)
class PrimaryPromptConfig:
    """Prompt-конфигурация базового pipeline.

    Args:
        normalization_system: System prompt для нормализации комментария.
        normalization_human: Human prompt для нормализации комментария.
        primary_decision_system: System prompt для выбора существующей или новой группы.
        primary_decision_human: Human prompt для выбора существующей или новой группы.
        group_naming_system: System prompt для генерации имени группы.
        group_naming_human: Human prompt для генерации имени группы.

    Returns:
        Экземпляр с шаблонами prompt-ов, которые используются при сборке LCEL-цепочек.
    """

    normalization_system: str
    normalization_human: str
    primary_decision_system: str
    primary_decision_human: str
    group_naming_system: str
    group_naming_human: str

    @classmethod
    def default(cls) -> "PrimaryPromptConfig":
        """Создает конфигурацию базовых prompt-ов из дефолтов библиотеки.

        Returns:
            Конфигурация с дефолтными prompt-шаблонами библиотеки.
        """
        return cls(
            normalization_system=NORMALIZATION_SYSTEM,
            normalization_human=NORMALIZATION_HUMAN,
            primary_decision_system=PRIMARY_DECISION_SYSTEM,
            primary_decision_human=PRIMARY_DECISION_HUMAN,
            group_naming_system=GROUP_NAMING_SYSTEM,
            group_naming_human=GROUP_NAMING_HUMAN,
        )


@dataclass(slots=True)
class AgenticPromptConfig:
    """Prompt-конфигурация агентской постобработки.

    Args:
        supervisor_system: System prompt supervisor-узла.
        supervisor_human: Human prompt supervisor-узла.
        route_unassigned_system: System prompt маршрутизации комментариев без группы.
        route_unassigned_human: Human prompt маршрутизации комментариев без группы.
        cluster_audit_system: System prompt аудита группы.
        cluster_audit_human: Human prompt аудита группы.
        group_naming_system: System prompt финального именования группы.
        group_naming_human: Human prompt финального именования группы.
        merge_groups_system: System prompt проверки объединения групп.
        merge_groups_human: Human prompt проверки объединения групп.

    Returns:
        Экземпляр с шаблонами prompt-ов для агентской постобработки.
    """

    supervisor_system: str | None = None
    supervisor_human: str | None = None
    route_unassigned_system: str | None = None
    route_unassigned_human: str | None = None
    cluster_audit_system: str | None = None
    cluster_audit_human: str | None = None
    group_naming_system: str | None = None
    group_naming_human: str | None = None
    merge_groups_system: str | None = None
    merge_groups_human: str | None = None


@dataclass(slots=True)
class ClusteringPromptConfig:
    """Объединенная prompt-конфигурация библиотеки.

    Args:
        primary: Prompt-конфигурация базового этапа кластеризации.
        agentic: Prompt-конфигурация агентской постобработки.

    Returns:
        Конфигурация, которую можно передать в pipeline при инициализации.
    """

    primary: PrimaryPromptConfig = field(default_factory=PrimaryPromptConfig.default)
    agentic: AgenticPromptConfig = field(default_factory=AgenticPromptConfig)

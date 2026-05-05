"""Агентная LLM-постобработка готовых текстовых кластеров.

Файл содержит:

- схемы структурированных решений LLM:
  - ``SupervisorDecision`` — выбор следующего агентного действия;
  - ``CommentRoutingDecision`` — решение для комментария без метки группы;
  - ``GroupMergeDecision`` — проверка безопасного объединения двух групп;

- класс ``AgenticPostProcessingPipeline``:
  - ``run`` / ``arun`` — синхронный и асинхронный запуск;
  - ``_build_initial_state`` — минимальная подготовка состояния без нормализации текста;
  - ``_supervisor_node`` — LLM-supervisor, который видит обзор групп и выбирает действие;
  - ``_merge_groups_node`` — объединение двух групп после дополнительной проверки примеров;
  - ``_audit_group_node`` — аудит одной группы и снятие метки с неподходящих комментариев;
  - ``_route_unassigned_node`` — попытка отнести комментарии без группы к существующим группам;
  - ``_finalize_node`` — финальное именование групп и сбор результата.

Ожидание к входу: пользователь уже подготовил тексты к кластеризации. Код не делает
языковую нормализацию, лемматизацию, чистку пунктуации или доменную фильтрацию.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .agentic_models import ClusterAuditDecision, PostProcessingGroupName
from .agentic_prompts import CLUSTER_AUDIT_HUMAN, CLUSTER_AUDIT_SYSTEM
from .config import AgenticPromptConfig, GROUP_NAMING_HUMAN, GROUP_NAMING_SYSTEM

logger = logging.getLogger(__name__)

Comment = dict[str, Any]
Group = dict[str, Any]
CommentsById = dict[str, Comment]
GroupsById = dict[str, Group]
NextStep = Literal["merge_groups", "audit_group", "route_unassigned", "finalize"]


class AgenticPostProcessingState(TypedDict, total=False):
    """Состояние LangGraph-графа постобработки."""

    comments_by_id: CommentsById
    groups_by_id: GroupsById
    comment_order: list[str]
    unassigned_queue: list[str]
    audit_queue: list[str]
    audit_attempts_by_group_id: dict[str, int]
    next_group_index: int
    next_step: NextStep
    action_payload: dict[str, Any]
    round_index: int
    no_change_rounds: int
    last_patch_summary: dict[str, Any]
    finish_reason: str
    final_result: dict[str, Any]



SUPERVISOR_SYSTEM = """
Ты — supervisor-модуль качества кластеризации пользовательских комментариев.

Твоя задача — проверить текущее состояние кластеризации и определить:
1. какие группы являются смыслово неоднородными;
2. какие комментарии попали в неправильные группы;
3. какие группы нужно разделить;
4. какие комментарии нужно вынести в новые или другие группы;
5. какие группы можно оставить без изменений.

Главный принцип:
одинаковые слова не означают одинаковый смысл.

Не считай группу корректной только потому, что комментарии содержат общие слова:
"подтверждение", "блокировка", "операция", "перевод", "деньги", "безопасность",
"сообщения", "негативный комментарий".

Группа считается однородной только если у комментариев совпадают три признака:
1. Объект жалобы — что именно затронуто.
2. Причина недовольства — почему клиент недоволен.
3. Ожидаемое изменение — чего клиент фактически хочет.

Обязательно разделяй разные смыслы:

1. Недостаточная защита и избыточная защита — разные группы.
Например:
- "мошенники совершили покупку, банк не остановил" — недостаточная защита.
- "банк слишком часто блокирует мои операции" — избыточная защита.

2. Разные проблемы с подтверждением — разные группы.
Например:
- нелогичные критерии подтверждения;
- подтверждение маленьких бытовых покупок;
- неудобный процесс подтверждения;
- долгое ожидание подтверждения;
- полный отказ клиента от подтверждений.

3. Проблема операции и проблема коммуникации — разные группы.
Например:
- "операция остановилась там, где не нужно" — проблема операции.
- "устал от ваших сообщений" — проблема коммуникаций.

4. Техническая проблема и антифрод-блокировка — разные группы.
Например:
- "не смог снять деньги с банкомата" — проблема банкомата.
- "банк блокирует снятие наличных" — блокировка доступа к деньгам.

5. Общий негатив без конкретного кейса нельзя добавлять в содержательные группы.
Например:
- "ужас";
- "всё может быть лучше";
- ".".

Если группа неоднородна, предложи разбиение на более узкие подгруппы.
Не пытайся сохранить исходную группу, если она слишком широкая.
Не выдумывай факты, которых нет в тексте.
При сомнении выбирай более безопасное решение: разделить группу или вынести комментарий отдельно.

Верни только структурированный ответ по схеме.
""".strip()


SUPERVISOR_HUMAN = """
Текущее состояние кластеризации:
{state_snapshot}

Задача:
Проверь группы на смысловую однородность.

Для каждой проблемной группы определи:
1. почему группа неоднородна;
2. какие комментарии отличаются по смыслу;
3. какие подгруппы нужно создать;
4. какие комментарии можно оставить в исходной группе;
5. какие комментарии нужно перенести в другие группы или новые группы.

Перед решением проверь:
- совпадает ли объект жалобы у комментариев внутри группы;
- совпадает ли причина недовольства;
- совпадает ли ожидаемое изменение;
- нет ли противоположных смыслов;
- нет ли комментариев, попавших в группу только из-за общих слов;
- нет ли общих негативных комментариев внутри содержательной группы.

Примеры правильной проверки:

Пример 1. Неоднородная группа:
Название группы:
"Проблемы с блокировками и безопасностью"

Комментарии:
- "мошенники совершили покупку на мои деньги, сбер не остановил их"
- "система защиты работает против клиента, частые необоснованные блокировки операций"
- "возникла проблема со списанием средств, которую решаю до сих пор"

Вывод:
Группа неоднородна.

Правильное разбиение:
- "Недостаточная защита от мошеннических операций"
- "Необоснованные блокировки и ограничения доступа к средствам"
- "Проблема со списанием средств"

Причина:
Комментарии используют близкие слова про деньги и безопасность, но описывают разные клиентские боли.


Пример 2. Неоднородная группа:
Название группы:
"Проблемы с подтверждением операций"

Комментарии:
- "операции с большей суммой прошли без проблем, а меньшую пришлось подтверждать"
- "на покупку еды не нужно подтверждение, так как суммы небольшие"
- "Я вообще не хотел бы никакого подтверждения"
- "прошли сутки, а подтверждения до сих пор нет"

Вывод:
Группа неоднородна.

Правильное разбиение:
- "Нелогичные критерии подтверждения операций"
- "Избыточное подтверждение небольших покупок"
- "Нежелание проходить подтверждение операций"
- "Долгое ожидание подтверждения операции"


Пример 3. Однородная группа:
Название группы:
"Долгое ожидание подтверждения операции"

Комментарии:
- "сутки на подтверждение - это придумка идиота, если мне нужно оплатить, значит это нужно сейчас"
- "прошли сутки, а подтверждения до сих пор нет"

Вывод:
Группа однородна.

Причина:
Оба комментария описывают одну проблему: подтверждение операции занимает слишком много времени.

{format_instructions}
""".strip()


ROUTE_UNASSIGNED_SYSTEM = """
Ты — модуль маршрутизации комментариев без группы.

Твоя задача — решить, можно ли безопасно отнести комментарий к одной из групп-кандидатов,
или нужно создать новую группу.

Главный принцип:
лучше создать новую узкую группу, чем ошибочно добавить комментарий в неподходящую.

Комментарий можно отнести к существующей группе только если совпадают три признака:
1. Объект жалобы.
2. Причина недовольства.
3. Ожидаемое изменение.

Не относить комментарий к группе только из-за совпадения общих слов:
"подтверждение", "блокировка", "операция", "перевод", "деньги", "безопасность",
"сообщения", "негативный комментарий".

Обязательно различай:

1. Недостаточная защита != избыточная защита.
- "банк не остановил мошенников" — недостаточная защита.
- "банк слишком часто блокирует меня" — избыточная защита.

2. Разные проблемы с подтверждением != одна группа.
- долгое подтверждение;
- отказ от любых подтверждений;
- подтверждение маленьких покупок;
- нелогичные критерии подтверждения;
- неудобный процесс подтверждения.

3. Проблема операции != проблема коммуникации.
- "операция остановилась" — операция.
- "устал от сообщений" — коммуникации.

4. Техническая проблема != антифрод-блокировка.
- "не смог снять деньги с банкомата" — банкомат/техническая проблема.
- "банк блокирует снятие наличных" — блокировка доступа к деньгам.

5. Общий негатив без конкретики не добавляй в содержательные группы.
- "ужас" — общая негативная оценка без конкретной причины.
- "." — неинформативный комментарий.
- "всё может быть лучше" — общий негатив без конкретного кейса.

При сомнении выбирай создание новой группы.
Не пытайся искусственно уменьшить количество групп.
Не выдумывай факты, которых нет в тексте.

Верни только структурированный ответ по схеме.
""".strip()


ROUTE_UNASSIGNED_HUMAN = """
Комментарий без группы:
{comment_card}

Группы-кандидаты:
{candidate_groups}

Задача:
Определи, можно ли безопасно отнести комментарий к одной из групп-кандидатов,
или нужно создать новую группу.

Перед решением проверь:
1. Совпадает ли объект жалобы?
2. Совпадает ли причина недовольства?
3. Совпадает ли ожидаемое изменение?
4. Нет ли противоположного смысла относительно группы-кандидата?
5. Не является ли совпадение только словесным, например по словам:
   "блокировка", "подтверждение", "операция", "деньги", "безопасность"?

Выбери существующую группу только если это тот же смысловой кейс.

Если подходящей группы нет, выбери создание новой группы.

Если комментарий слишком общий или неинформативный, не добавляй его в содержательную группу.

Примеры правильного решения:

Пример 1:
Комментарий:
"мошенники совершили покупку на мои деньги, сбер не остановил их"

Нельзя относить в группу:
"Необоснованные блокировки операций"

Правильный смысл:
"Недостаточная защита от мошеннических операций"

Причина:
Клиент жалуется не на лишнюю блокировку, а на то, что банк не остановил мошенников.


Пример 2:
Комментарий:
"Я вообще не хотел бы никакого подтверждения. Я сам в состоянии действовать безопасно."

Нельзя относить в группу:
"Долгое ожидание подтверждения операции"

Правильный смысл:
"Нежелание проходить подтверждение операций"

Причина:
Клиент не жалуется на длительность подтверждения. Он в принципе не хочет подтверждать операции.


Пример 3:
Комментарий:
"Я считаю что на покупку еды ненужно подтверждения так как суммы то небольшие. А на остальное конечно проверяйте."

Нельзя относить в группу:
"Нежелание проходить подтверждение операций"

Правильный смысл:
"Избыточное подтверждение небольших покупок"

Причина:
Клиент не против всех проверок. Он против подтверждения небольших покупок еды.


Пример 4:
Комментарий:
"устал от ваших сообщений"

Нельзя относить в группу:
"Проблемы с операциями и сообщениями"

Правильный смысл:
"Потребность в уменьшении количества сообщений"

Причина:
Комментарий про коммуникационную усталость, а не про остановку операций.


Пример 5:
Комментарий:
"возникла проблема со списанием средств, которую решаю до сих пор"

Нельзя относить в группу:
"Проблемы с блокировками и безопасностью"

Правильный смысл:
"Проблема со списанием средств"

Причина:
В комментарии нет явной блокировки, мошенничества или подтверждения.

{format_instructions}
""".strip()


MERGE_GROUPS_SYSTEM = """
Ты — модуль безопасного объединения текстовых групп.

Твоя задача — проверить, можно ли объединить две группы без потери смысла.

Главный принцип:
одинаковое или похожее название НЕ является достаточной причиной для объединения.

Объединять можно только группы, которые описывают один и тот же смысловой класс:
один объект жалобы, одна причина недовольства, одно ожидаемое изменение.

Разрешено объединять группы, если:
- комментарии из обеих групп описывают одну и ту же клиентскую боль;
- различия в формулировках не меняют смысл;
- различия в деталях не являются важным основанием для отдельной категории;
- после объединения можно дать одно точное название без потери смысла.

Запрещено объединять группы, если они отличаются по:
- объекту жалобы;
- причине недовольства;
- сценарию операции;
- намерению пользователя;
- каналу или сервису, если это важно для смысла;
- уровню детализации, если одна группа описывает общий кейс, а другая — отдельную специфичную проблему;
- направлению жалобы: недостаточная защита против избыточной защиты.

Не объединяй:

1. Недостаточную защиту и избыточные блокировки.
Например:
- "банк не остановил мошенников";
- "банк слишком часто блокирует мои операции".

2. Разные проблемы с подтверждением.
Например:
- "подтверждение занимает сутки";
- "не хочу никаких подтверждений";
- "маленькие покупки еды не нужно подтверждать";
- "большую сумму пропустили, меньшую заставили подтверждать".

3. Проблему сообщений и проблему операций.
Например:
- "устал от ваших сообщений";
- "операция остановилась там, где не нужно".

4. Банкоматную проблему и антифрод-блокировку снятия.
Например:
- "не смог снять деньги с банкомата";
- "банк блокирует снятие наличных".

5. Списание средств и блокировку средств.
Например:
- "проблема со списанием средств";
- "заморозили счет";
- "заблокировали операцию".

6. Общие негативные оценки и содержательные жалобы.
Например:
- "ужас";
- "сутки на подтверждение операции".

Примеры:

Пример 1. Можно объединить.

Группа A:
"Долгое ожидание подтверждения операции"
Комментарии:
"сутки на подтверждение - это придумка идиота, если мне нужно оплатить, значит это нужно сейчас"

Группа B:
"Подтверждение операции занимает слишком много времени"
Комментарии:
"прошли сутки, а подтверждения до сих пор нет!"

Решение:
should_merge=true

Причина:
Обе группы описывают одну проблему: подтверждение операции длится слишком долго.


Пример 2. Нельзя объединять.

Группа A:
"Недостаточная защита от мошеннических операций"
Комментарии:
"мошенники совершили покупку на мои деньги, сбер не остановил их"

Группа B:
"Необоснованные блокировки и ограничения доступа к средствам"
Комментарии:
"система защиты работает против клиента, частые и необоснованные блокировки операций"

Решение:
should_merge=false

Причина:
Смыслы противоположны. В первой группе банк недостаточно защитил клиента,
во второй — банк слишком агрессивно ограничивает клиента.


Пример 3. Нельзя объединять.

Группа A:
"Избыточное подтверждение небольших покупок"
Комментарии:
"на покупку еды не нужно подтверждение, так как суммы небольшие"

Группа B:
"Нежелание проходить подтверждение операций"
Комментарии:
"Я вообще не хотел бы никакого подтверждения"

Решение:
should_merge=false

Причина:
В первой группе клиент не против проверок в целом, но против подтверждения маленьких покупок.
Во второй группе клиент против любых подтверждений.


Пример 4. Нельзя объединять.

Группа A:
"Потребность в уменьшении количества сообщений"
Комментарии:
"устал от ваших сообщений"

Группа B:
"Избыточность скриптов робота и операторов"
Комментарии:
"пока робот не договорит, нельзя двигаться дальше. слишком много слов от операторов"

Решение:
should_merge=false

Причина:
Обе группы связаны с коммуникацией, но объект жалобы разный:
сообщения банка против длинных скриптов робота/оператора.


Пример 5. Можно объединить.

Группа A:
"Блокировка снятия наличных"
Комментарии:
"Срочно необходимо снять наличные, а банк блокирует и срываются договоренности"

Группа B:
"Блокировка снятия денег в отделении"
Комментарии:
"перевел себе деньги и попытался их снять в отделении Сбербанка, операцию заблокировали"

Решение:
should_merge=true

Причина:
Обе группы про блокировку доступа к наличным при попытке снять свои деньги.
Различие "отделение" можно сохранить в примерах, но общий смысл совпадает.

При сомнении выбери should_merge=false.
Не объединяй группы ради уменьшения их количества.
Не объединяй группы только потому, что название одной группы слишком широкое.
Если после объединения получится размытое название вроде "Проблемы с операциями", значит объединять нельзя.

Верни только структурированный ответ по схеме.
""".strip()


MERGE_GROUPS_HUMAN = """
Целевая группа:
{target_group_card}

Группа-кандидат на объединение:
{source_group_card}

Проверь, можно ли объединить эти две группы.

Перед решением проверь:
1. Совпадает ли объект жалобы?
2. Совпадает ли причина недовольства?
3. Совпадает ли ожидаемое изменение?
4. Не являются ли смыслы противоположными?
5. Не станет ли объединенная группа слишком общей?
6. Можно ли дать объединенной группе одно точное название без потери смысла?

Если хотя бы один важный признак отличается, не объединяй.

{format_instructions}
""".strip()

class SupervisorDecision(BaseModel):
    """Решение supervisor о следующем агентном действии."""

    action: NextStep = Field(description="Следующее действие графа.")
    target_group_id: str = Field(default="", description="Группа, в которую нужно объединить source_group_id.")
    source_group_id: str = Field(default="", description="Группа, которую нужно объединить в target_group_id.")
    audit_group_id: str = Field(default="", description="Группа, которую нужно проверить на чужеродные комментарии.")
    comment_ids: list[str] = Field(default_factory=list, description="Комментарии без группы для маршрутизации.")
    reason: str = Field(description="Краткое объяснение выбранного действия.")


class CommentRoutingDecision(BaseModel):
    """Решение о маршрутизации одного комментария без группы."""

    action: Literal["move_to_group", "create_new_group"] = Field(
        description="Перенести комментарий в существующую группу или создать новую группу."
    )
    target_group_id: str = Field(default="", description="ID целевой группы для action='move_to_group'.")
    reason: str = Field(description="Краткое объяснение решения.")


class GroupMergeDecision(BaseModel):
    """Решение о безопасном объединении двух групп."""

    should_merge: bool = Field(description="Можно ли объединить группы без потери смысла.")
    reason: str = Field(description="Краткое объяснение решения.")


class AgenticPostProcessingPipeline:
    """Гибкая агентная система для LLM-постобработки текстовых кластеров.

    В отличие от жёсткого pipeline, supervisor здесь сам выбирает следующее
    действие на каждом цикле. Он получает компактный снимок состояния и может:
    объединить похожие группы, проверить одну группу, маршрутизировать комментарии
    без метки или завершить работу.
    """

    def __init__(
            self,
            llm: BaseChatModel,
            *,
            text_fields: tuple[str, ...] = ("text", "normalized_text", "raw_text", "content", "body", "message"),
            comment_id_field: str = "comment_id",
            group_id_field: str = "group_id",
            group_name_field: str = "group_name",
            new_group_prefix: str = "group",
            supervisor_group_limit: int = 50,
            supervisor_examples_per_group: int = 2,
            supervisor_unassigned_limit: int = 20,
            max_examples_per_candidate_group: int = 3,
            route_batch_size: int = 10,
            audit_comment_limit: int = 80,
            merge_example_limit: int = 8,
            max_concurrent_llm_requests: int = 10,
            candidate_cluster_limit: int = 40,
            max_rounds: int = 100,
            max_no_change_rounds: int = 3,
            max_audit_passes_per_group: int = 2,
            merge_groups_by_final_name: bool = True,
            prompt_config: AgenticPromptConfig | None = None,
    ) -> None:
        """Инициализирует настройки, LLM-цепочки и LangGraph-граф.

        Args:
            llm: Chat-модель LangChain, которая используется supervisor-ом и worker-узлами.

            text_fields: Порядок полей, из которых берётся подготовленный текст комментария.
                Код выбирает первое непустое поле. Нормализация текста здесь не выполняется.
            comment_id_field: Имя входного поля с идентификатором комментария.
            group_id_field: Имя входного поля с идентификатором группы.
            group_name_field: Имя входного поля с названием группы.
            new_group_prefix: Префикс для новых групп, которые создаются при маршрутизации
                комментариев без подходящей существующей группы.

            supervisor_group_limit: Максимальное количество групп, которое показывается
                LLM-supervisor-у в общем снимке состояния. Ограничивает размер prompt-а.
            supervisor_examples_per_group: Количество примеров из каждой группы, которое
                показывается supervisor-у в обзорном снимке. Обычно достаточно 1-2 примеров.
            supervisor_unassigned_limit: Максимальное количество комментариев без метки группы,
                которое показывается supervisor-у за один обзор состояния.

            max_examples_per_candidate_group: Количество примеров из каждой группы-кандидата,
                которое передаётся в узел маршрутизации unassigned-комментариев.
            route_batch_size: Максимальное количество unassigned-комментариев, которое
                supervisor может отправить на маршрутизацию за один агентный шаг.
            audit_comment_limit: Максимальное количество комментариев из одной группы,
                которое передаётся LLM при аудите группы на чужеродные элементы.
            merge_example_limit: Максимальное количество примеров из каждой из двух групп,
                которое передаётся LLM при проверке безопасного объединения.

            max_concurrent_llm_requests: Максимальное количество параллельных LLM-запросов.
                Используется для защиты модели/API от слишком высокой нагрузки.
            candidate_cluster_limit: Максимальное количество групп-кандидатов, которое
                передаётся при попытке отнести unassigned-комментарий к существующей группе.

            max_rounds: Максимальное количество агентных циклов supervisor -> action.
                Защищает граф от бесконечного выполнения.
            max_no_change_rounds: Максимальное количество подряд идущих шагов без изменений.
                Если изменений нет слишком долго, граф завершает работу через finalize.
            max_audit_passes_per_group: Максимальное количество аудитов одной и той же группы.
                Защищает от повторной проверки группы без продуктивного результата.
            merge_groups_by_final_name: Нужно ли на этапе finalize дополнительно проверять
                группы с одинаковыми финальными названиями и объединять их только после
                отдельного LLM-подтверждения по примерам.
        """
        prompt_config = prompt_config or AgenticPromptConfig()
        self._text_fields = text_fields
        self._comment_id_field = comment_id_field
        self._group_id_field = group_id_field
        self._group_name_field = group_name_field
        self._new_group_prefix = new_group_prefix

        self._supervisor_group_limit = max(1, supervisor_group_limit)
        self._supervisor_examples_per_group = max(1, supervisor_examples_per_group)
        self._supervisor_unassigned_limit = max(1, supervisor_unassigned_limit)
        self._max_examples_per_candidate_group = max(1, max_examples_per_candidate_group)
        self._route_batch_size = max(1, route_batch_size)
        self._audit_comment_limit = max(1, audit_comment_limit)
        self._merge_example_limit = max(1, merge_example_limit)
        self._candidate_cluster_limit = max(1, candidate_cluster_limit)
        self._max_rounds = max(1, max_rounds)
        self._max_no_change_rounds = max(1, max_no_change_rounds)
        self._max_audit_passes_per_group = max(1, max_audit_passes_per_group)
        self._merge_groups_by_final_name = merge_groups_by_final_name

        self._llm_semaphore = asyncio.Semaphore(max(1, max_concurrent_llm_requests))

        self._supervisor_chain = self._build_chain(
            prompt_config.supervisor_system or SUPERVISOR_SYSTEM,
            prompt_config.supervisor_human or SUPERVISOR_HUMAN,
            SupervisorDecision,
            llm,
        )
        self._routing_chain = self._build_chain(
            prompt_config.route_unassigned_system or ROUTE_UNASSIGNED_SYSTEM,
            prompt_config.route_unassigned_human or ROUTE_UNASSIGNED_HUMAN,
            CommentRoutingDecision,
            llm,
        )
        self._audit_chain = self._build_chain(
            prompt_config.cluster_audit_system or CLUSTER_AUDIT_SYSTEM,
            prompt_config.cluster_audit_human or CLUSTER_AUDIT_HUMAN,
            ClusterAuditDecision,
            llm,
        )
        self._naming_chain = self._build_chain(
            prompt_config.group_naming_system or GROUP_NAMING_SYSTEM,
            prompt_config.group_naming_human or GROUP_NAMING_HUMAN,
            PostProcessingGroupName,
            llm,
        )
        self._merge_chain = self._build_chain(
            prompt_config.merge_groups_system or MERGE_GROUPS_SYSTEM,
            prompt_config.merge_groups_human or MERGE_GROUPS_HUMAN,
            GroupMergeDecision,
            llm,
        )

        self._graph = self._build_graph()

    def run(self, primary_result: dict[str, list[dict]]) -> dict[str, Any]:
        """Синхронно запускает постобработку."""
        return asyncio.run(self.arun(primary_result))

    async def arun(self, primary_result: dict[str, list[dict]]) -> dict[str, Any]:
        """Асинхронно запускает постобработку и возвращает финальный результат."""
        state = self._build_initial_state(primary_result)

        logger.info(
            "Agentic post-processing started: %d comments, %d groups",
            len(state["comments_by_id"]),
            len(state["groups_by_id"]),
        )

        final_state = await self._graph.ainvoke(state)

        logger.info(
            "Agentic post-processing finished: %d final groups, reason=%s",
            len(final_state.get("final_result", {}).get("groups", [])),
            final_state.get("finish_reason", ""),
        )
        return final_state["final_result"]

    @staticmethod
    def _build_chain(system_prompt: str, human_prompt: str, schema: type[BaseModel], llm: BaseChatModel) -> Any:
        """Собирает LCEL-цепочку: prompt -> LLM -> Pydantic parser.

        Инструкция для модели формируется на русском языке. Технические имена
        JSON-полей остаются как в Pydantic-схеме, потому что по ним parser
        валидирует ответ.
        """
        parser = PydanticOutputParser(pydantic_object=schema)
        schema_dict = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()
        format_instructions = (
            "Верни только валидный JSON без Markdown, пояснений и code fence. "
            "JSON должен строго соответствовать этой схеме. "
            "Названия полей и допустимые значения enum не переводи.            "
            f"JSON Schema:          {json.dumps(schema_dict, ensure_ascii=False, indent=2)}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        ).partial(format_instructions=format_instructions)
        return (prompt | llm | parser).with_retry(stop_after_attempt=2)

    def _build_graph(self) -> Any:
        """Создаёт агентный LangGraph-граф с LLM-supervisor."""
        graph = StateGraph(AgenticPostProcessingState)

        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("merge_groups", self._merge_groups_node)
        graph.add_node("audit_group", self._audit_group_node)
        graph.add_node("route_unassigned", self._route_unassigned_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "merge_groups": "merge_groups",
                "audit_group": "audit_group",
                "route_unassigned": "route_unassigned",
                "finalize": "finalize",
            },
        )
        graph.add_edge("merge_groups", "supervisor")
        graph.add_edge("audit_group", "supervisor")
        graph.add_edge("route_unassigned", "supervisor")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _build_initial_state(self, primary_result: dict[str, list[dict]]) -> AgenticPostProcessingState:
        """Создаёт начальное состояние без смысловой обработки текстов."""
        comments_by_id, comment_order = self._build_comments(primary_result.get("comments", []))
        groups_by_id = self._build_groups(primary_result.get("groups", []), comments_by_id, comment_order)

        state: AgenticPostProcessingState = {
            "comments_by_id": comments_by_id,
            "groups_by_id": groups_by_id,
            "comment_order": comment_order,
            "unassigned_queue": [],
            "audit_queue": [],
            "audit_attempts_by_group_id": {},
            "next_group_index": self._next_group_index(groups_by_id),
            "next_step": "finalize",
            "action_payload": {},
            "round_index": 0,
            "no_change_rounds": 0,
            "last_patch_summary": {},
            "finish_reason": "",
        }
        state.update(self._build_queue_update(state))
        return state

    def _build_comments(self, raw_comments: list[dict]) -> tuple[CommentsById, list[str]]:
        """Приводит входные комментарии к внутреннему формату."""
        comments_by_id: CommentsById = {}
        comment_order: list[str] = []

        for index, raw_comment in enumerate(raw_comments, start=1):
            comment = copy.deepcopy(raw_comment)
            comment_id = str(comment.get(self._comment_id_field, "")).strip() or str(index)
            group_id = str(comment.get(self._group_id_field, "")).strip()

            comment.update(
                {
                    "comment_id": comment_id,
                    "text": self._extract_text(comment),
                    "group_id": group_id,
                    "initial_group_id": group_id,
                    "postprocessing_trace": list(comment.get("postprocessing_trace", [])),
                }
            )

            comments_by_id[comment_id] = comment
            comment_order.append(comment_id)

        return comments_by_id, comment_order

    def _build_groups(
            self,
            raw_groups: list[dict],
            comments_by_id: CommentsById,
            comment_order: list[str],
    ) -> GroupsById:
        """Строит группы и синхронизирует их состав с group_id комментариев."""
        groups_by_id: GroupsById = {}

        for raw_group in raw_groups:
            group_id = str(raw_group.get(self._group_id_field, "")).strip()
            if not group_id:
                continue

            groups_by_id[group_id] = {
                "group_id": group_id,
                "group_name": str(raw_group.get(self._group_name_field, "")).strip(),
                "member_comment_ids": [],
            }

        for comment_id in comment_order:
            group_id = comments_by_id[comment_id]["group_id"]
            if not group_id:
                continue

            groups_by_id.setdefault(group_id, {"group_id": group_id, "group_name": "", "member_comment_ids": []})
            groups_by_id[group_id]["member_comment_ids"].append(comment_id)

        return {
            group_id: {**group, "member_comment_ids": list(dict.fromkeys(group["member_comment_ids"]))}
            for group_id, group in groups_by_id.items()
            if group["member_comment_ids"]
        }

    def _extract_text(self, comment: Comment) -> str:
        """Берёт уже подготовленный текст из первого непустого текстового поля."""
        for field_name in self._text_fields:
            value = comment.get(field_name)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    async def _supervisor_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        """LLM-supervisor выбирает следующее агентное действие."""
        queue_update = self._build_queue_update(state)
        state_for_snapshot: AgenticPostProcessingState = {**state, **queue_update}

        round_index = int(state.get("round_index", 0))
        no_change_rounds = int(state.get("no_change_rounds", 0))

        if round_index >= self._max_rounds:
            return {
                **queue_update,
                "next_step": "finalize",
                "action_payload": {},
                "finish_reason": f"Safety stop: max_rounds={self._max_rounds} reached",
            }

        if no_change_rounds >= self._max_no_change_rounds:
            return {
                **queue_update,
                "next_step": "finalize",
                "action_payload": {},
                "finish_reason": f"No-change stop: no productive progress for {no_change_rounds} rounds",
            }

        fallback = self._fallback_supervisor_decision(state_for_snapshot)
        decision = await self._ainvoke_chain(
            self._supervisor_chain,
            {"state_snapshot": self._build_supervisor_snapshot(state_for_snapshot)},
            fallback=fallback,
        )
        next_step, action_payload = self._validate_supervisor_decision(decision, state_for_snapshot, fallback)

        logger.info(
            "Supervisor selected action=%s, reason=%s, payload=%s",
            next_step,
            decision.reason,
            action_payload,
        )

        return {
            **queue_update,
            "next_step": next_step,
            "action_payload": action_payload,
            "finish_reason": decision.reason if next_step == "finalize" else state.get("finish_reason", ""),
        }

    @staticmethod
    def _route_from_supervisor(state: AgenticPostProcessingState) -> NextStep:
        """Возвращает имя следующего узла графа."""
        return state.get("next_step", "finalize")

    def _build_supervisor_snapshot(self, state: AgenticPostProcessingState) -> str:
        """Формирует русскоязычный обзор групп, который видит LLM-supervisor."""
        comments_by_id = state["comments_by_id"]
        groups_by_id = state["groups_by_id"]
        unassigned_ids = state.get("unassigned_queue", [])
        audit_group_ids = state.get("audit_queue", [])

        sorted_groups = sorted(
            groups_by_id.values(),
            key=lambda group: (-len(group.get("member_comment_ids", [])), group["group_id"]),
        )[: self._supervisor_group_limit]

        group_blocks = [
            self._format_group_card(
                group,
                comments_by_id,
                member_limit=self._supervisor_examples_per_group,
            )
            for group in sorted_groups
        ]

        unassigned_examples = [
            self._format_comment_card(comments_by_id[comment_id])
            for comment_id in unassigned_ids[: self._supervisor_unassigned_limit]
            if comment_id in comments_by_id
        ]

        return "\n".join(
            [
                f"Раунд: {state.get('round_index', 0)}",
                f"Раундов без изменений: {state.get('no_change_rounds', 0)}",
                f"Всего комментариев: {len(comments_by_id)}",
                f"Всего групп: {len(groups_by_id)}",
                f"Показано групп: {len(sorted_groups)} из {len(groups_by_id)}",
                f"Групп доступно для аудита: {len(audit_group_ids)}",
                f"Комментариев без метки группы: {len(unassigned_ids)}",
                f"Сводка последнего действия: {state.get('last_patch_summary', {})}",
                "",
                "ГРУППЫ:",
                "\n".join(group_blocks) if group_blocks else "Групп нет.",
                "",
                "ПРИМЕРЫ КОММЕНТАРИЕВ БЕЗ МЕТКИ ГРУППЫ:",
                "\n".join(unassigned_examples) if unassigned_examples else "Комментариев без метки группы нет.",
            ]
        )

    def _fallback_supervisor_decision(self, state: AgenticPostProcessingState) -> SupervisorDecision:
        """Детерминированное решение на случай сбоя supervisor LLM."""
        if state.get("unassigned_queue"):
            return SupervisorDecision(
                action="route_unassigned",
                comment_ids=state["unassigned_queue"][: self._route_batch_size],
                reason="Резервное решение: сначала обработать комментарии без метки группы",
            )

        if state.get("audit_queue"):
            return SupervisorDecision(
                action="audit_group",
                audit_group_id=state["audit_queue"][0],
                reason="Резервное решение: проверить самую крупную доступную группу",
            )

        return SupervisorDecision(action="finalize", reason="Резервное решение: доступных действий нет")

    def _validate_supervisor_decision(
            self,
            decision: SupervisorDecision,
            state: AgenticPostProcessingState,
            fallback: SupervisorDecision,
    ) -> tuple[NextStep, dict[str, Any]]:
        """Проверяет, что выбранное supervisor действие применимо к текущему состоянию."""
        groups_by_id = state["groups_by_id"]
        unassigned = set(state.get("unassigned_queue", []))
        audit_queue = set(state.get("audit_queue", []))

        if decision.action == "merge_groups":
            target_group_id = decision.target_group_id.strip()
            source_group_id = decision.source_group_id.strip()
            if (
                    target_group_id in groups_by_id
                    and source_group_id in groups_by_id
                    and target_group_id != source_group_id
            ):
                return "merge_groups", {
                    "target_group_id": target_group_id,
                    "source_group_id": source_group_id,
                    "reason": decision.reason,
                }

        if decision.action == "audit_group":
            audit_group_id = decision.audit_group_id.strip()
            if audit_group_id in audit_queue:
                return "audit_group", {"group_id": audit_group_id, "reason": decision.reason}

        if decision.action == "route_unassigned":
            comment_ids = [comment_id for comment_id in decision.comment_ids if comment_id in unassigned]
            if not comment_ids:
                comment_ids = list(state.get("unassigned_queue", []))[: self._route_batch_size]
            if comment_ids:
                return "route_unassigned", {"comment_ids": comment_ids[: self._route_batch_size],
                                            "reason": decision.reason}

        if decision.action == "finalize":
            return "finalize", {}

        if decision is fallback:
            return "finalize", {}

        return self._validate_supervisor_decision(fallback, state, fallback)

    async def _merge_groups_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        """Объединяет две выбранные supervisor группы после LLM-проверки примеров."""
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        payload = state.get("action_payload", {})
        target_group_id = str(payload.get("target_group_id", "")).strip()
        source_group_id = str(payload.get("source_group_id", "")).strip()

        processed_items = 0
        applied_changes = 0

        if target_group_id in groups_by_id and source_group_id in groups_by_id and target_group_id != source_group_id:
            processed_items = 1
            decision = await self._should_merge_groups(
                target_group=groups_by_id[target_group_id],
                source_group=groups_by_id[source_group_id],
                comments_by_id=comments_by_id,
            )

            if decision.should_merge:
                for comment_id in list(groups_by_id[source_group_id].get("member_comment_ids", [])):
                    moved = self._move_comment(
                        comments_by_id,
                        groups_by_id,
                        comment_id,
                        target_group_id,
                        f"Merged by supervisor after LLM check: {decision.reason}",
                    )
                    applied_changes += int(moved)

                groups_by_id.pop(source_group_id, None)
                logger.info("Groups merged: %s -> %s", source_group_id, target_group_id)
            else:
                logger.info("Groups merge rejected: %s -> %s, reason=%s", source_group_id, target_group_id,
                            decision.reason)

        return self._build_action_update(
            state,
            applied_changes=applied_changes,
            processed_items=processed_items,
            comments_by_id=comments_by_id,
            groups_by_id=groups_by_id,
            last_patch_summary={
                "step": "merge_groups",
                "target_group_id": target_group_id,
                "source_group_id": source_group_id,
                "processed": processed_items,
                "changes": applied_changes,
            },
        )

    async def _audit_group_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        """Проверяет одну выбранную группу и снимает метку с неподходящих комментариев."""
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        audit_attempts_by_group_id = dict(state.get("audit_attempts_by_group_id", {}))
        group_id = str(state.get("action_payload", {}).get("group_id", "")).strip()

        processed_items = 0
        applied_changes = 0

        if group_id in groups_by_id:
            processed_items = 1
            audit_attempts_by_group_id[group_id] = audit_attempts_by_group_id.get(group_id, 0) + 1
            decision = await self._audit_one_group(group_id, groups_by_id, comments_by_id)

            group = groups_by_id.get(group_id)
            visible_comment_ids = set(
                group.get("member_comment_ids", [])[: self._audit_comment_limit]) if group else set()
            removable_ids = [
                comment_id
                for comment_id in decision.remove_comment_ids
                if comment_id in visible_comment_ids and group and comment_id in group.get("member_comment_ids", [])
            ]

            for comment_id in removable_ids:
                self._unassign_comment(comments_by_id, groups_by_id, comment_id, decision.reason)

            applied_changes = len(removable_ids)
            logger.info("Group audited: %s, removed=%d", group_id, applied_changes)

        return self._build_action_update(
            state,
            applied_changes=applied_changes,
            processed_items=processed_items,
            comments_by_id=comments_by_id,
            groups_by_id=groups_by_id,
            audit_attempts_by_group_id=audit_attempts_by_group_id,
            last_patch_summary={
                "step": "audit_group",
                "group_id": group_id,
                "processed": processed_items,
                "changes": applied_changes,
            },
        )

    async def _audit_one_group(
            self,
            group_id: str,
            groups_by_id: GroupsById,
            comments_by_id: CommentsById,
    ) -> ClusterAuditDecision:
        """Получает LLM-решение о комментариях, которые не подходят группе."""
        return await self._ainvoke_chain(
            self._audit_chain,
            {"group_card": self._format_group_card(groups_by_id[group_id], comments_by_id,
                                                   member_limit=self._audit_comment_limit)},
            fallback=ClusterAuditDecision(remove_comment_ids=[],
                                          reason="Резервное решение: группа выглядит однородной"),
        )

    async def _route_unassigned_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        """Пытается соотнести комментарии без метки группы с существующими группами."""
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        next_group_index = int(state.get("next_group_index", 1))
        comment_ids = list(state.get("action_payload", {}).get("comment_ids", []))[: self._route_batch_size]

        tasks = [
            self._decide_unassigned_route(
                comment=comments_by_id[comment_id],
                candidate_groups=self._build_cluster_candidates(
                    comments_by_id=comments_by_id,
                    groups_by_id=groups_by_id,
                    limit=self._candidate_cluster_limit,
                ),
            )
            for comment_id in comment_ids
            if comment_id in comments_by_id and not comments_by_id[comment_id].get("group_id")
        ]
        decisions = await asyncio.gather(*tasks) if tasks else []

        processed_items = 0
        applied_changes = 0

        for comment_id, decision in zip(comment_ids, decisions, strict=False):
            if comment_id not in comments_by_id or comments_by_id[comment_id].get("group_id"):
                continue

            processed_items += 1
            target_group_id = decision.target_group_id.strip()

            if decision.action != "move_to_group" or target_group_id not in groups_by_id:
                target_group_id, next_group_index = self._create_group(groups_by_id, next_group_index)

            self._assign_comment_to_group(comments_by_id, groups_by_id, comment_id, target_group_id, decision.reason)
            applied_changes += 1

        return self._build_action_update(
            state,
            applied_changes=applied_changes,
            processed_items=processed_items,
            comments_by_id=comments_by_id,
            groups_by_id=groups_by_id,
            next_group_index=next_group_index,
            last_patch_summary={
                "step": "route_unassigned",
                "comment_ids": comment_ids,
                "processed": processed_items,
                "changes": applied_changes,
            },
        )

    async def _decide_unassigned_route(self, *, comment: Comment,
                                       candidate_groups: list[dict[str, Any]]) -> CommentRoutingDecision:
        """Получает LLM-решение о маршрутизации одного unassigned-комментария."""
        if not candidate_groups:
            return CommentRoutingDecision(
                action="create_new_group",
                target_group_id="",
                reason="Групп-кандидатов нет",
            )

        return await self._ainvoke_chain(
            self._routing_chain,
            {
                "comment_card": self._format_comment_card(comment),
                "candidate_groups": self._format_candidate_groups(candidate_groups),
            },
            fallback=CommentRoutingDecision(
                action="create_new_group",
                target_group_id="",
                reason="Резервное решение: нет безопасной существующей группы",
            ),
        )

    async def _finalize_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        """Именует группы, осторожно объединяет группы с одинаковыми названиями и возвращает результат."""
        comments_by_id, groups_by_id = self._copy_cluster_state(state)

        await self._rename_groups(groups_by_id, comments_by_id)

        if self._merge_groups_by_final_name:
            await self._merge_groups_by_name(groups_by_id, comments_by_id)

        return {"final_result": self._build_final_result(state, comments_by_id, groups_by_id)}

    async def _rename_groups(self, groups_by_id: GroupsById, comments_by_id: CommentsById) -> None:
        """Генерирует финальное название для каждой группы."""

        async def rename_one(group_id: str, group: Group) -> tuple[str, str]:
            fallback_name = self._fallback_group_name(group, comments_by_id)
            decision = await self._ainvoke_chain(
                self._naming_chain,
                {"group_examples": self._format_group_examples(group, comments_by_id, limit=self._merge_example_limit)},
                fallback=PostProcessingGroupName(group_name=fallback_name),
            )
            return group_id, str(decision.group_name).strip() or fallback_name

        tasks = [rename_one(group_id, group) for group_id, group in groups_by_id.items() if
                 group.get("member_comment_ids")]

        for group_id, group_name in await asyncio.gather(*tasks) if tasks else []:
            if group_id in groups_by_id:
                groups_by_id[group_id]["group_name"] = group_name

    async def _merge_groups_by_name(self, groups_by_id: GroupsById, comments_by_id: CommentsById) -> None:
        """Проверяет группы с одинаковым названием и объединяет только подтверждённые пары."""
        group_ids_by_name: dict[str, list[str]] = {}

        for source_group_id in sorted(list(groups_by_id)):
            source_group = groups_by_id.get(source_group_id)
            if not source_group or not source_group.get("member_comment_ids"):
                groups_by_id.pop(source_group_id, None)
                continue

            group_name_key = str(source_group.get("group_name", "")).strip().casefold()
            if not group_name_key:
                continue

            merged = False
            for target_group_id in group_ids_by_name.get(group_name_key, []):
                if target_group_id not in groups_by_id:
                    continue

                decision = await self._should_merge_groups(
                    target_group=groups_by_id[target_group_id],
                    source_group=source_group,
                    comments_by_id=comments_by_id,
                )
                if not decision.should_merge:
                    continue

                for comment_id in list(source_group.get("member_comment_ids", [])):
                    self._move_comment(
                        comments_by_id,
                        groups_by_id,
                        comment_id,
                        target_group_id,
                        f"Merged by final-name check: {decision.reason}",
                    )

                groups_by_id.pop(source_group_id, None)
                merged = True
                break

            if not merged and source_group_id in groups_by_id:
                group_ids_by_name.setdefault(group_name_key, []).append(source_group_id)

    async def _should_merge_groups(
            self,
            *,
            target_group: Group,
            source_group: Group,
            comments_by_id: CommentsById,
    ) -> GroupMergeDecision:
        """Проверяет через LLM, можно ли объединить две группы."""
        return await self._ainvoke_chain(
            self._merge_chain,
            {
                "target_group_card": self._format_group_card(target_group, comments_by_id,
                                                             member_limit=self._merge_example_limit),
                "source_group_card": self._format_group_card(source_group, comments_by_id,
                                                             member_limit=self._merge_example_limit),
            },
            fallback=GroupMergeDecision(
                should_merge=False,
                reason="Резервное решение: объединение отклонено, чтобы избежать ложного объединения",
            ),
        )

    async def _ainvoke_chain(self, chain: Any, payload: dict[str, Any], *, fallback: Any) -> Any:
        """Вызывает LLM-цепочку с ограничением параллелизма и fallback."""
        try:
            async with self._llm_semaphore:
                return await chain.ainvoke(payload)
        except Exception as exc:
            logger.exception("LLM step failed, using fallback: %s", exc)
            return fallback

    @staticmethod
    def _copy_cluster_state(state: AgenticPostProcessingState) -> tuple[CommentsById, GroupsById]:
        """Копирует изменяемую часть состояния."""
        return copy.deepcopy(state["comments_by_id"]), copy.deepcopy(state["groups_by_id"])

    def _build_action_update(
            self,
            state: AgenticPostProcessingState,
            *,
            applied_changes: int,
            processed_items: int,
            comments_by_id: CommentsById,
            groups_by_id: GroupsById,
            last_patch_summary: dict[str, Any],
            audit_attempts_by_group_id: dict[str, int] | None = None,
            next_group_index: int | None = None,
    ) -> dict[str, Any]:
        """Формирует patch состояния после агентного действия."""
        next_state: AgenticPostProcessingState = {
            **state,
            "comments_by_id": comments_by_id,
            "groups_by_id": groups_by_id,
            "audit_attempts_by_group_id": audit_attempts_by_group_id
            if audit_attempts_by_group_id is not None
            else dict(state.get("audit_attempts_by_group_id", {})),
            "next_group_index": next_group_index if next_group_index is not None else int(
                state.get("next_group_index", 1)),
            "round_index": int(state.get("round_index", 0)) + 1,
            "no_change_rounds": int(state.get("no_change_rounds", 0)) + 1 if applied_changes == 0 else 0,
            "last_patch_summary": last_patch_summary,
            "action_payload": {},
        }
        next_state.update(self._build_queue_update(next_state))

        return {
            "comments_by_id": next_state["comments_by_id"],
            "groups_by_id": next_state["groups_by_id"],
            "unassigned_queue": next_state["unassigned_queue"],
            "audit_queue": next_state["audit_queue"],
            "audit_attempts_by_group_id": next_state["audit_attempts_by_group_id"],
            "next_group_index": next_state["next_group_index"],
            "round_index": next_state["round_index"],
            "no_change_rounds": next_state["no_change_rounds"],
            "last_patch_summary": next_state["last_patch_summary"],
            "action_payload": next_state["action_payload"],
        }

    def _build_queue_update(self, state: AgenticPostProcessingState) -> dict[str, list[str]]:
        """Пересчитывает очереди unassigned и auditable-групп."""
        return {
            "unassigned_queue": self._unassigned_comment_ids(state),
            "audit_queue": self._auditable_group_ids(state),
        }

    @staticmethod
    def _unassigned_comment_ids(state: AgenticPostProcessingState) -> list[str]:
        """Возвращает комментарии без метки группы в исходном порядке."""
        comments_by_id = state["comments_by_id"]
        return [
            comment_id
            for comment_id in state.get("comment_order", list(comments_by_id))
            if comment_id in comments_by_id and not str(comments_by_id[comment_id].get("group_id", "")).strip()
        ]

    def _auditable_group_ids(self, state: AgenticPostProcessingState) -> list[str]:
        """Возвращает группы, которые ещё можно аудировать."""
        groups_by_id = state["groups_by_id"]
        attempts = dict(state.get("audit_attempts_by_group_id", {}))
        return [
            group["group_id"]
            for group in
            sorted(groups_by_id.values(), key=lambda item: (-len(item.get("member_comment_ids", [])), item["group_id"]))
            if len(group.get("member_comment_ids", [])) > 1
               and attempts.get(group["group_id"], 0) < self._max_audit_passes_per_group
        ]

    def _next_group_index(self, groups_by_id: GroupsById) -> int:
        """Вычисляет следующий свободный индекс для новой группы."""
        pattern = re.compile(rf"{re.escape(self._new_group_prefix)}_(\d+)$")
        indexes = [int(match.group(1)) for group_id in groups_by_id if (match := pattern.fullmatch(group_id))]
        return max(indexes, default=0) + 1

    def _create_group(self, groups_by_id: GroupsById, next_group_index: int) -> tuple[str, int]:
        """Создаёт новую группу."""
        group_id = f"{self._new_group_prefix}_{next_group_index:04d}"
        while group_id in groups_by_id:
            next_group_index += 1
            group_id = f"{self._new_group_prefix}_{next_group_index:04d}"

        groups_by_id[group_id] = {"group_id": group_id, "group_name": "", "member_comment_ids": []}
        return group_id, next_group_index + 1

    def _build_cluster_candidates(
            self,
            *,
            comments_by_id: CommentsById,
            groups_by_id: GroupsById,
            limit: int,
    ) -> list[dict[str, Any]]:
        """Собирает группы-кандидаты для маршрутизации комментария."""
        candidates: list[dict[str, Any]] = []

        for group in sorted(groups_by_id.values(),
                            key=lambda item: (-len(item.get("member_comment_ids", [])), item["group_id"])):
            member_ids = list(group.get("member_comment_ids", []))
            if not member_ids:
                continue

            candidates.append(
                {
                    "group_id": group["group_id"],
                    "group_name": str(group.get("group_name", "")).strip() or "Без названия",
                    "size": len(member_ids),
                    "examples": [
                        {"comment_id": comment_id, "text": str(comment.get("text", ""))}
                        for comment_id in member_ids[: self._max_examples_per_candidate_group]
                        if (comment := comments_by_id.get(comment_id))
                    ],
                }
            )

            if len(candidates) >= limit:
                break

        return candidates

    @staticmethod
    def _format_comment_card(comment: Comment) -> str:
        """Формирует карточку одного комментария для prompt-а."""
        return "\n".join(
            [
                f"comment_id (идентификатор комментария): {comment['comment_id']}",
                f"текст: {_truncate_text(comment.get('text', ''), limit=500)}",
            ]
        )

    def _format_group_card(self, group: Group, comments_by_id: CommentsById, *, member_limit: int) -> str:
        """Формирует карточку группы с размером, названием и примерами."""
        member_ids = list(group.get("member_comment_ids", []))[:member_limit]
        examples = [
            f"- {comment_id}: {_truncate_text(comment.get('text', ''), limit=300)}"
            for comment_id in member_ids
            if (comment := comments_by_id.get(comment_id))
        ]
        return "\n".join(
            [
                f"group_id (идентификатор группы): {group['group_id']}",
                f"group_name (название группы): {str(group.get('group_name', '')).strip() or 'Без названия'}",
                f"размер группы: {len(group.get('member_comment_ids', []))}",
                "примеры:",
                *(examples or ["- примеров нет"]),
            ]
        )

    @staticmethod
    def _format_candidate_groups(candidate_groups: list[dict[str, Any]]) -> str:
        """Форматирует группы-кандидаты для prompt-а."""
        if not candidate_groups:
            return "Групп-кандидатов нет."

        chunks: list[str] = []
        for candidate in candidate_groups:
            examples = [
                f"- {example['comment_id']}: {_truncate_text(example['text'], limit=300)}"
                for example in candidate.get("examples", [])
            ]
            chunks.append(
                "\n".join(
                    [
                        f"group_id (идентификатор группы): {candidate['group_id']}",
                        f"group_name (название группы): {candidate['group_name']}",
                        f"размер группы: {candidate['size']}",
                        "примеры:",
                        *(examples or ["- примеров нет"]),
                    ]
                )
            )
        return "\n\n".join(chunks)

    def _format_group_examples(self, group: Group, comments_by_id: CommentsById, *, limit: int) -> str:
        """Форматирует примеры группы для генерации названия."""
        examples = [
            f"- comment_id: {comment_id} | text: {_truncate_text(comment.get('text', ''), limit=300)}"
            for comment_id in list(group.get("member_comment_ids", []))[:limit]
            if (comment := comments_by_id.get(comment_id))
        ]
        return "\n".join(examples) if examples else "Примеров нет."

    def _fallback_group_name(self, group: Group, comments_by_id: CommentsById) -> str:
        """Возвращает безопасное название группы без LLM."""
        first_comment_id = next(iter(group.get("member_comment_ids", [])), "")
        first_comment = comments_by_id.get(first_comment_id, {})
        return _truncate_text(first_comment.get("text", "") or "Без названия", limit=80)

    def _move_comment(
            self,
            comments_by_id: CommentsById,
            groups_by_id: GroupsById,
            comment_id: str,
            target_group_id: str,
            reason: str,
    ) -> bool:
        """Перемещает комментарий из текущей группы в целевую."""
        comment = comments_by_id.get(comment_id)
        if not comment:
            return False

        source_group_id = str(comment.get("group_id", "")).strip()
        if not source_group_id or source_group_id == target_group_id or target_group_id not in groups_by_id:
            return False

        self._remove_comment_from_group(groups_by_id, source_group_id, comment_id)
        self._assign_comment_to_group(comments_by_id, groups_by_id, comment_id, target_group_id, reason)

        if source_group_id in groups_by_id and not groups_by_id[source_group_id]["member_comment_ids"]:
            groups_by_id.pop(source_group_id, None)

        return True

    def _unassign_comment(self, comments_by_id: CommentsById, groups_by_id: GroupsById, comment_id: str,
                          reason: str) -> None:
        """Снимает с комментария метку группы."""
        comment = comments_by_id.get(comment_id)
        if not comment:
            return

        source_group_id = str(comment.get("group_id", "")).strip()
        if source_group_id:
            self._remove_comment_from_group(groups_by_id, source_group_id, comment_id)
            if source_group_id in groups_by_id and not groups_by_id[source_group_id]["member_comment_ids"]:
                groups_by_id.pop(source_group_id, None)

        comment["group_id"] = ""
        comment["postprocessing_reason"] = reason
        comment.setdefault("postprocessing_trace", []).append(
            f"{source_group_id or 'unassigned'} -> unassigned: {reason}")

    @staticmethod
    def _assign_comment_to_group(
            comments_by_id: CommentsById,
            groups_by_id: GroupsById,
            comment_id: str,
            target_group_id: str,
            reason: str,
    ) -> None:
        """Назначает комментарий в группу и пишет trace изменения."""
        comment = comments_by_id[comment_id]
        previous_group_id = comment.get("group_id") or "unassigned"

        groups_by_id.setdefault(target_group_id,
                                {"group_id": target_group_id, "group_name": "", "member_comment_ids": []})
        groups_by_id[target_group_id]["member_comment_ids"].append(comment_id)
        groups_by_id[target_group_id]["member_comment_ids"] = list(
            dict.fromkeys(groups_by_id[target_group_id]["member_comment_ids"]))

        comment["group_id"] = target_group_id
        comment["postprocessing_reason"] = reason
        comment.setdefault("postprocessing_trace", []).append(f"{previous_group_id} -> {target_group_id}: {reason}")

    @staticmethod
    def _remove_comment_from_group(groups_by_id: GroupsById, group_id: str, comment_id: str) -> None:
        """Удаляет комментарий из списка участников группы."""
        group = groups_by_id.get(group_id)
        if not group:
            return

        group["member_comment_ids"] = [
            existing_comment_id
            for existing_comment_id in group.get("member_comment_ids", [])
            if existing_comment_id != comment_id
        ]

    @staticmethod
    def _build_final_result(
            state: AgenticPostProcessingState,
            comments_by_id: CommentsById,
            groups_by_id: GroupsById,
    ) -> dict[str, Any]:
        """Собирает финальный результат в исходном порядке комментариев.

        Args:
            state: Текущее состояние графа постобработки.
            comments_by_id: Комментарии, сгруппированные по идентификатору комментария.
            groups_by_id: Финальные группы, сгруппированные по техническому идентификатору.

        Returns:
            Словарь с публичным результатом. Комментарии не содержат embeddings и технических
            номеров групп, вместо них возвращается человекочитаемое название группы.
        """
        def build_public_comment(comment: Comment) -> Comment:
            """Преобразует внутренний комментарий в публичную строку результата.

            Args:
                comment: Внутренний словарь комментария с техническими полями.

            Returns:
                Словарь комментария без embeddings и технических ID группы, но с ``group_name``.
            """
            public_comment = copy.deepcopy(comment)
            group_id = str(public_comment.get("group_id", "")).strip()
            group = groups_by_id.get(group_id, {})
            public_comment["group_name"] = str(group.get("group_name", "")).strip() or "Без названия"
            public_comment.pop("embedding", None)
            public_comment.pop("group_id", None)
            public_comment.pop("initial_group_id", None)
            return public_comment

        return {
            "comments": [
                build_public_comment(comments_by_id[comment_id])
                for comment_id in state["comment_order"]
                if comment_id in comments_by_id
            ],
            "groups": [
                {"group_id": group_id, "group_name": str(group.get("group_name", "")).strip() or "Без названия"}
                for group_id, group in sorted(groups_by_id.items())
                if group.get("member_comment_ids")
            ],
        }


def _truncate_text(value: object, *, limit: int) -> str:
    """Обрезает длинный текст для prompt-а без языковой нормализации."""
    text = str(value)
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}..."

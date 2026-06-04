"""Упрощенный pipeline кластеризации комментариев через FAISS, BM25 и LLM.

Файл содержит:
- ``run_coroutine_sync`` — безопасный синхронный запуск coroutine из IDE и Jupyter;
- ``render_progress_bar`` — форматирование прогресса для консоли;
- ``preprocess_text`` — минимальная техническая предобработка текста;
- ``tokenize_for_bm25`` — токенизация текста для BM25;
- ``truncate_text`` — сокращение длинных строк для prompt-ов;
- ``SimpleHybridMemoryStore`` — in-memory хранилище комментариев, групп, FAISS и BM25;
- ``SimpleGroupDecisionEngine`` — LLM-решение о выборе существующей или новой группы;
- ``SimpleFaissBM25LLMClusteringPipeline`` — основной pipeline кластеризации.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from .config import PrimaryPromptConfig
from .prompts import EMPTY_HUMAN_MESSAGE

logger = logging.getLogger(__name__)

_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u00ab": '"', "\u00bb": '"',
    "\u2014": "-", "\u2013": "-",
})


def run_coroutine_sync(coro: Any) -> Any:
    """Запускает coroutine из обычной IDE и из окружений с активным event loop.

    Args:
        coro: Coroutine-объект, который нужно выполнить синхронно.

    Returns:
        Результат выполнения coroutine.

    Raises:
        BaseException: Пробрасывает исключение, возникшее внутри coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        """Выполняет coroutine в отдельном потоке с собственным event loop.

        Args:
            Входные аргументы отсутствуют, coroutine берется из внешней области видимости.

        Returns:
            ``None``. Результат или исключение сохраняются во внешнем словаре.
        """
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")


def render_progress_bar(current: int, total: int, *, width: int = 24) -> str:
    """Рендерит ASCII прогресс-бар для консольного вывода.

    Args:
        current: Количество уже обработанных элементов.
        total: Общее количество элементов.
        width: Ширина визуальной части прогресс-бара.

    Returns:
        Строка с прогресс-баром и счетчиком элементов.
    """
    total = max(total, 1)
    current = max(0, min(current, total))
    filled = int(width * current / total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {current}/{total}"


def preprocess_text(value: str) -> str:
    """Выполняет минимальную техническую предобработку комментария.

    Args:
        value: Исходный текст комментария.

    Returns:
        Текст в нижнем регистре со схлопнутыми пробелами, повторяющейся пунктуацией и ``ё``.
    """
    text = str(value).translate(_QUOTE_MAP).lower().replace("ё", "е")
    text = re.sub(r"([!?.,;:])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_bm25(value: str) -> list[str]:
    """Разбивает текст на токены для BM25-поиска.

    Args:
        value: Предобработанный или исходный текст комментария.

    Returns:
        Список токенов из букв и цифр.
    """
    return re.findall(r"[0-9a-zа-я]+", preprocess_text(value), flags=re.IGNORECASE)


def truncate_text(value: str, limit: int = 700) -> str:
    """Сокращает длинный текст для компактного prompt-а.

    Args:
        value: Исходная строка.
        limit: Максимальная длина результата.

    Returns:
        Строка не длиннее ``limit`` символов.
    """
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


@dataclass(slots=True)
class SimpleStoredComment:
    """Сохраненный комментарий с исходными полями и назначенной группой.

    Args:
        comment_id: Внутренний идентификатор комментария.
        source: Исходная строка данных в виде словаря.
        raw_text: Исходный текст комментария.
        processed_text: Предобработанный текст комментария.
        embedding: Векторное представление комментария.
        group_id: Идентификатор назначенной группы.

    Returns:
        Экземпляр комментария, сохраненный в памяти pipeline.
    """

    comment_id: str
    source: dict[str, Any]
    raw_text: str
    processed_text: str
    embedding: list[float]
    group_id: str


@dataclass(slots=True)
class SimpleCommentGroup:
    """Группа комментариев, созданная pipeline.

    Args:
        group_id: Внутренний идентификатор группы.
        group_name: Человекочитаемое название группы.
        member_comment_ids: Идентификаторы комментариев, входящих в группу.

    Returns:
        Экземпляр группы комментариев.
    """

    group_id: str
    group_name: str
    member_comment_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SimpleCandidateGroup:
    """Группа-кандидат для передачи в LLM.

    Args:
        group_id: Идентификатор группы-кандидата.
        group_name: Текущее название группы.
        score: Объединенный retrieval-score по FAISS и BM25.
        representative_comment_ids: Идентификаторы примеров из группы.

    Returns:
        Экземпляр группы-кандидата с примерами для prompt-а.
    """

    group_id: str
    group_name: str
    score: float
    representative_comment_ids: list[str]


@dataclass(slots=True)
class SimpleGroupDecision:
    """Решение LLM о группе для одного комментария.

    Args:
        decision_type: ``existing_group`` или ``new_group``.
        group_id: Идентификатор существующей группы или пустая строка.
        group_name: Название новой группы или уточненное название существующей.
        reason: Краткое объяснение решения.

    Returns:
        Экземпляр решения маршрутизации комментария.
    """

    decision_type: str
    group_id: str
    group_name: str
    reason: str


class SimpleHybridMemoryStore:
    """Хранит комментарии, группы, FAISS-индекс и BM25-корпус в памяти.

    Args:
        embeddings: LangChain embedding-модель, необходимая FAISS при создании индекса.
        bm25_top_k: Количество документов, извлекаемых через BM25.
        faiss_top_k: Количество документов, извлекаемых через FAISS.

    Returns:
        Экземпляр in-memory хранилища для одного запуска pipeline.
    """

    def __init__(self, embeddings: Embeddings, *, bm25_top_k: int, faiss_top_k: int) -> None:
        self._embeddings = embeddings
        self._bm25_top_k = bm25_top_k
        self._faiss_top_k = faiss_top_k
        self._ordered_ids: list[str] = []
        self._comments: dict[str, SimpleStoredComment] = {}
        self._groups: dict[str, SimpleCommentGroup] = {}
        self._vectorstore: FAISS | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_tokens: list[list[str]] = []
        self._next_group_index = 1
        self._bm25_dirty = True

    def create_group(self, group_name: str) -> SimpleCommentGroup:
        """Создает новую группу с последовательным идентификатором.

        Args:
            group_name: Название новой группы.

        Returns:
            Созданная группа комментариев.
        """
        group_id = f"group_{self._next_group_index:04d}"
        self._next_group_index += 1
        group = SimpleCommentGroup(group_id=group_id, group_name=group_name.strip() or group_id)
        self._groups[group_id] = group
        return group

    def add_comment(self, comment: SimpleStoredComment) -> None:
        """Добавляет комментарий в память, группу, FAISS и BM25-корпус.

        Args:
            comment: Подготовленный комментарий с назначенной группой.

        Returns:
            ``None``.
        """
        self._ordered_ids.append(comment.comment_id)
        self._comments[comment.comment_id] = comment
        self._groups[comment.group_id].member_comment_ids.append(comment.comment_id)
        self._index_comment(comment)
        self._bm25_tokens.append(tokenize_for_bm25(comment.processed_text))
        self._bm25_dirty = True

    def get_comment(self, comment_id: str) -> SimpleStoredComment:
        """Возвращает комментарий по идентификатору.

        Args:
            comment_id: Идентификатор комментария.

        Returns:
            Сохраненный комментарий.
        """
        return self._comments[comment_id]

    def get_group(self, group_id: str) -> SimpleCommentGroup | None:
        """Возвращает группу по идентификатору.

        Args:
            group_id: Идентификатор группы.

        Returns:
            Группа комментариев или ``None``, если группа не найдена.
        """
        return self._groups.get(group_id)

    def all_groups(self) -> list[SimpleCommentGroup]:
        """Возвращает все группы в порядке создания.

        Args:
            Входные аргументы отсутствуют.

        Returns:
            Список групп комментариев.
        """
        return sorted(self._groups.values(), key=lambda group: group.group_id)

    def search_candidates(
            self,
            *,
            query_text: str,
            query_embedding: list[float],
            candidate_group_limit: int,
            max_examples_per_group: int,
            exclude_group_ids: set[str] | None = None,
    ) -> list[SimpleCandidateGroup]:
        """Ищет группы-кандидаты через FAISS и BM25.

        Args:
            query_text: Предобработанный текст текущего комментария.
            query_embedding: Вектор текущего комментария.
            candidate_group_limit: Максимальное количество групп-кандидатов.
            max_examples_per_group: Максимальное количество примеров на одну группу.
            exclude_group_ids: Идентификаторы групп, которые нужно исключить из выдачи.

        Returns:
            Список групп-кандидатов для LLM.
        """
        if not self._ordered_ids:
            return []

        excluded_groups = exclude_group_ids or set()
        group_scores: dict[str, float] = {}
        group_examples: dict[str, list[str]] = {}

        ranked_sources = (
            self._search_faiss(query_embedding),
            self._search_bm25(query_text),
        )
        for ranked_comment_ids in ranked_sources:
            self._accumulate_candidate_scores(
                ranked_comment_ids=ranked_comment_ids,
                group_scores=group_scores,
                group_examples=group_examples,
                max_examples_per_group=max_examples_per_group,
                exclude_group_ids=excluded_groups,
            )

        candidates: list[SimpleCandidateGroup] = []
        for group_id in sorted(group_scores, key=group_scores.get, reverse=True)[:candidate_group_limit]:
            group = self._groups[group_id]
            candidates.append(SimpleCandidateGroup(
                group_id=group_id,
                group_name=group.group_name,
                score=group_scores[group_id],
                representative_comment_ids=group_examples.get(group_id, [])[:max_examples_per_group],
            ))
        return candidates

    def move_comments_to_group(self, *, comment_ids: list[str], target_group_id: str) -> None:
        """Переносит комментарии из текущих групп в целевую группу.

        Args:
            comment_ids: Идентификаторы комментариев, которые нужно перенести.
            target_group_id: Идентификатор группы, в которую нужно перенести комментарии.

        Returns:
            ``None``.

        Raises:
            ValueError: Если целевая группа не найдена.
        """
        if target_group_id not in self._groups:
            raise ValueError(f"Целевая группа не найдена: {target_group_id}.")

        source_group_ids: set[str] = set()
        for comment_id in comment_ids:
            comment = self._comments[comment_id]
            if comment.group_id == target_group_id:
                continue
            source_group_ids.add(comment.group_id)
            source_group = self._groups[comment.group_id]
            if comment_id in source_group.member_comment_ids:
                source_group.member_comment_ids.remove(comment_id)
            comment.group_id = target_group_id
            self._groups[target_group_id].member_comment_ids.append(comment_id)

        for group_id in source_group_ids:
            group = self._groups.get(group_id)
            if group is not None and not group.member_comment_ids:
                del self._groups[group_id]

    def _accumulate_candidate_scores(
            self,
            *,
            ranked_comment_ids: list[tuple[str, float]],
            group_scores: dict[str, float],
            group_examples: dict[str, list[str]],
            max_examples_per_group: int,
            exclude_group_ids: set[str],
    ) -> None:
        """Добавляет scores одного retrieval-источника в общий рейтинг групп.

        Args:
            ranked_comment_ids: Ранжированный список комментариев одного источника retrieval.
            group_scores: Накопленные scores групп.
            group_examples: Накопленные примеры комментариев по группам.
            max_examples_per_group: Максимальное количество примеров на одну группу.
            exclude_group_ids: Идентификаторы групп, исключенных из выдачи.

        Returns:
            ``None``.
        """
        for rank, (comment_id, source_weight) in enumerate(ranked_comment_ids, start=1):
            comment = self._comments.get(comment_id)
            if comment is None or comment.group_id in exclude_group_ids:
                continue
            group_scores[comment.group_id] = group_scores.get(comment.group_id, 0.0) + source_weight / rank
            examples = group_examples.setdefault(comment.group_id, [])
            if len(examples) < max_examples_per_group and comment_id not in examples:
                examples.append(comment_id)

    def rows_with_group_name(self) -> list[dict[str, Any]]:
        """Собирает строки результата в формате исходные поля плюс ``group_name``.

        Args:
            Входные аргументы отсутствуют.

        Returns:
            Список словарей, где в конце добавлено поле ``group_name``.
        """
        rows: list[dict[str, Any]] = []
        for comment_id in self._ordered_ids:
            comment = self._comments[comment_id]
            group = self._groups[comment.group_id]
            row = dict(comment.source)
            row["group_name"] = group.group_name
            rows.append(row)
        return rows

    def group_outputs(self) -> list[dict[str, Any]]:
        """Собирает техническую информацию о группах.

        Args:
            Входные аргументы отсутствуют.

        Returns:
            Список словарей с идентификатором, названием и размером группы.
        """
        return [
            {
                "group_id": group.group_id,
                "group_name": group.group_name,
                "comment_count": len(group.member_comment_ids),
            }
            for group in self.all_groups()
        ]

    def _search_faiss(self, query_embedding: list[float]) -> list[tuple[str, float]]:
        """Ищет похожие комментарии в FAISS.

        Args:
            query_embedding: Вектор текущего комментария.

        Returns:
            Список пар ``comment_id`` и веса источника retrieval.
        """
        if self._vectorstore is None:
            return []
        try:
            docs = self._vectorstore.similarity_search_by_vector(query_embedding, k=self._faiss_top_k)
        except Exception as exc:
            logger.error("FAISS-поиск завершился с ошибкой: %s", exc)
            return []
        return [(str(doc.metadata.get("comment_id", "")), 1.0) for doc in docs if doc.metadata.get("comment_id")]

    def _search_bm25(self, query_text: str) -> list[tuple[str, float]]:
        """Ищет похожие комментарии в BM25.

        Args:
            query_text: Предобработанный текст текущего комментария.

        Returns:
            Список пар ``comment_id`` и веса источника retrieval.
        """
        tokens = tokenize_for_bm25(query_text)
        if not tokens or not self._bm25_tokens:
            return []
        if self._bm25_dirty or self._bm25 is None:
            self._bm25 = BM25Okapi(self._bm25_tokens)
            self._bm25_dirty = False
        scores = self._bm25.get_scores(tokens)
        ranked_indexes = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        result: list[tuple[str, float]] = []
        for index in ranked_indexes[:self._bm25_top_k]:
            if scores[index] <= 0:
                continue
            result.append((self._ordered_ids[index], 0.8))
        return result

    def _index_comment(self, comment: SimpleStoredComment) -> None:
        """Добавляет комментарий в FAISS-индекс.

        Args:
            comment: Комментарий с готовым embedding.

        Returns:
            ``None``.
        """
        text_embedding = [(comment.processed_text, comment.embedding)]
        metadata = [{"comment_id": comment.comment_id, "group_id": comment.group_id}]
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_embeddings(
                text_embedding,
                self._embeddings,
                metadatas=metadata,
                ids=[comment.comment_id],
                normalize_L2=True,
                distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
            )
            return
        self._vectorstore.add_embeddings(text_embedding, metadatas=metadata, ids=[comment.comment_id])


class SimpleGroupDecisionEngine:
    """Запрашивает у LLM решение о группе для одного комментария.

    Args:
        llm: Chat-модель LangChain.
        llm_semaphore: Semaphore для ограничения параллельных LLM-запросов.
        max_retries: Количество повторных попыток LLM-вызова после первой ошибки.
        prompt_config: Prompt-конфигурация выбора группы.

    Returns:
        Экземпляр decision engine для маршрутизации комментариев.
    """

    def __init__(
            self,
            llm: BaseChatModel,
            *,
            llm_semaphore: asyncio.Semaphore,
            max_retries: int = 1,
            prompt_config: PrimaryPromptConfig | None = None,
    ) -> None:
        prompt_config = prompt_config or PrimaryPromptConfig.default()
        self._chain = (
                ChatPromptTemplate.from_messages([
                    ("system", prompt_config.primary_decision_system),
                    ("human", EMPTY_HUMAN_MESSAGE),
                ])
                | llm
                | JsonOutputParser()
        )
        self._sem = llm_semaphore
        self._max_retries = max(0, max_retries)

    async def achoose_group(
            self,
            *,
            raw_text: str,
            processed_text: str,
            candidate_groups_text: str,
            candidate_group_ids: set[str],
    ) -> SimpleGroupDecision:
        """Возвращает решение LLM о существующей или новой группе.

        Args:
            raw_text: Исходный текст комментария.
            processed_text: Предобработанный текст комментария.
            candidate_groups_text: Текстовое описание групп-кандидатов.
            candidate_group_ids: Допустимые идентификаторы групп-кандидатов.

        Returns:
            Валидированное решение LLM.

        Raises:
            RuntimeError: Если LLM-вызов не сработал после всех попыток.
            ValueError: Если LLM вернула невалидный тип решения, несуществующий ``group_id``
                или не вернула название для новой группы.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = await self._invoke_once(
                    raw_text=raw_text,
                    processed_text=processed_text,
                    candidate_groups_text=candidate_groups_text,
                )
                return self._parse_decision(raw, candidate_group_ids)
            except Exception as exc:
                last_error = exc
                logger.error(
                    "LLM-решение о группе не прошло проверку, попытка %d из %d: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
        raise RuntimeError("LLM-решение о группе не сработало после всех retry-попыток.") from last_error

    @staticmethod
    def _parse_decision(raw: dict[str, Any], candidate_group_ids: set[str]) -> SimpleGroupDecision:
        """Валидирует JSON-ответ LLM и преобразует его в решение о группе.

        Args:
            raw: JSON-словарь, полученный от LLM.
            candidate_group_ids: Допустимые идентификаторы групп-кандидатов.

        Returns:
            Валидированное решение о группе.

        Raises:
            ValueError: Если LLM вернула невалидный тип решения, недопустимый ``group_id``
                или пустое название новой группы.
        """
        if not isinstance(raw, dict):
            raise ValueError(f"LLM вернула не JSON-объект: {type(raw).__name__}.")

        decision_type = str(raw.get("decision_type", "")).strip().lower()
        group_id = str(raw.get("group_id", "")).strip()
        group_name = str(raw.get("group_name", "") or raw.get("new_group_name", "")).strip()
        reason = str(raw.get("reason", "")).strip()

        if decision_type == "existing_group":
            if group_id not in candidate_group_ids:
                raise ValueError(
                    f"LLM вернула недопустимый group_id '{group_id}'. "
                    f"Допустимые группы: {sorted(candidate_group_ids)}"
                )
            return SimpleGroupDecision(
                decision_type="existing_group",
                group_id=group_id,
                group_name=group_name,
                reason=reason,
            )

        if decision_type == "new_group":
            if not group_name:
                raise ValueError("LLM выбрала new_group, но не вернула поле group_name.")
            return SimpleGroupDecision(
                decision_type="new_group",
                group_id="",
                group_name=group_name,
                reason=reason,
            )

        raise ValueError(f"LLM вернула недопустимый decision_type: '{decision_type}'.")

    async def _invoke_once(
            self,
            *,
            raw_text: str,
            processed_text: str,
            candidate_groups_text: str,
    ) -> dict[str, Any]:
        """Один раз вызывает LLM-цепочку выбора группы.

        Args:
            raw_text: Исходный текст комментария.
            processed_text: Предобработанный текст комментария.
            candidate_groups_text: Текстовое описание групп-кандидатов.

        Returns:
            JSON-словарь, разобранный ``JsonOutputParser``.
        """
        async with self._sem:
            return await self._chain.ainvoke({
                "raw_text": raw_text,
                "normalized_text": processed_text,
                "candidate_groups": candidate_groups_text,
            })


class SimpleFaissBM25LLMClusteringPipeline:
    """Кластеризует комментарии через минимальную предобработку, FAISS, BM25 и LLM.

    Args:
        llm: Chat-модель LangChain для выбора группы.
        embeddings: Embedding-модель LangChain для построения векторов и FAISS.
        text_field: Поле исходной строки, из которого берется текст комментария.
        faiss_top_k: Количество ближайших комментариев из FAISS.
        bm25_top_k: Количество ближайших комментариев из BM25.
        candidate_group_limit: Максимальное количество групп-кандидатов для LLM.
        max_examples_per_candidate_group: Максимальное количество примеров на группу-кандидат.
        merge_small_groups: Если ``True``, запускает второй проход слияния маленьких групп.
        small_group_max_size: Максимальный размер группы, которую можно рассматривать для слияния.
        merge_candidate_group_limit: Максимальное количество групп-кандидатов для второго прохода.
        max_concurrent_llm_requests: Лимит параллельных LLM-запросов.
        max_llm_retries: Количество повторных попыток LLM-вызова после первой ошибки.
        max_embedding_retries: Количество повторных попыток batch embedding-вызова после первой ошибки.
        prompt_config: Prompt-конфигурация LLM-решения.
        show_progress: Если ``True``, печатает прогресс обработки.

    Returns:
        Экземпляр pipeline, который возвращает исходные строки с добавленным ``group_name``.
    """

    def __init__(
            self,
            llm: BaseChatModel,
            embeddings: Embeddings,
            *,
            text_field: str = "text",
            faiss_top_k: int = 80,
            bm25_top_k: int = 80,
            candidate_group_limit: int = 30,
            max_examples_per_candidate_group: int = 8,
            merge_small_groups: bool = True,
            small_group_max_size: int = 5,
            merge_candidate_group_limit: int = 40,
            max_concurrent_llm_requests: int = 3,
            max_llm_retries: int = 1,
            max_embedding_retries: int = 1,
            prompt_config: PrimaryPromptConfig | None = None,
            show_progress: bool = True,
    ) -> None:
        self._text_field = text_field
        self._candidate_group_limit = candidate_group_limit
        self._max_examples = max_examples_per_candidate_group
        self._merge_small_groups = merge_small_groups
        self._small_group_max_size = max(1, small_group_max_size)
        self._merge_candidate_group_limit = max(1, merge_candidate_group_limit)
        self._embeddings = embeddings
        self._llm_sem = asyncio.Semaphore(max_concurrent_llm_requests)
        self._max_embedding_retries = max(0, max_embedding_retries)
        self._decision_engine = SimpleGroupDecisionEngine(
            llm,
            llm_semaphore=self._llm_sem,
            max_retries=max_llm_retries,
            prompt_config=prompt_config,
        )
        self._store = SimpleHybridMemoryStore(
            embeddings,
            bm25_top_k=bm25_top_k,
            faiss_top_k=faiss_top_k,
        )
        self._show_progress = show_progress

    def run(self, raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Синхронно запускает pipeline из IDE, скрипта или Jupyter Notebook.

        Args:
            raw_rows: Список исходных строк данных с непустым текстовым полем.

        Returns:
            Словарь с ключами ``rows`` и ``groups``.
        """
        return run_coroutine_sync(self.arun(raw_rows))

    async def arun(self, raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Асинхронно запускает pipeline.

        Args:
            raw_rows: Список исходных строк данных с непустым текстовым полем.

        Returns:
            Словарь с исходными строками и добавленным полем ``group_name``.
        """
        prepared = self._prepare_rows(raw_rows)
        total = len(prepared)
        self._print("Построение embeddings", 0, total)
        embeddings = await self._build_embeddings([row["processed_text"] for row in prepared])
        self._print("Построение embeddings", total, total)

        self._print("Кластеризация", 0, total)
        step = max(1, total // 10)
        for index, (row, embedding) in enumerate(zip(prepared, embeddings, strict=True), start=1):
            await self._process_row(row=row, embedding=embedding)
            if index == 1 or index == total or index % step == 0:
                self._print("Кластеризация", index, total)

        if self._merge_small_groups:
            self._print("Слияние маленьких групп", 0, total)
            await self._merge_small_candidate_groups()
            self._print("Слияние маленьких групп", total, total)

        self._print("Готово", total, total)
        return {
            "rows": self._store.rows_with_group_name(),
            "groups": self._store.group_outputs(),
        }

    def to_dataframe(self, result: dict[str, Any]) -> Any:
        """Преобразует результат pipeline в pandas DataFrame.

        Args:
            result: Результат метода ``run`` или ``arun``.

        Returns:
            ``pandas.DataFrame`` со всеми исходными полями и ``group_name``.

        Raises:
            ImportError: Если пакет ``pandas`` не установлен в окружении.
        """
        import pandas as pd

        return pd.DataFrame(result["rows"])

    def save_excel(self, result: dict[str, Any], output_path: str) -> None:
        """Сохраняет результат pipeline в Excel-файл.

        Args:
            result: Результат метода ``run`` или ``arun``.
            output_path: Путь к итоговому ``.xlsx`` файлу.

        Returns:
            ``None``.
        """
        from openpyxl import Workbook

        rows = result["rows"]
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "clusters"
        if not rows:
            workbook.save(output_path)
            return

        headers = list(rows[0].keys())
        sheet.append(headers)
        for row in rows:
            sheet.append([row.get(header, "") for header in headers])
        workbook.save(output_path)

    def _prepare_rows(self, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Готовит строки к обработке без проверки пустых комментариев.

        Args:
            raw_rows: Список исходных строк данных.

        Returns:
            Список строк с внутренним ID, исходным текстом и предобработанным текстом.
        """
        prepared: list[dict[str, Any]] = []
        for index, row in enumerate(raw_rows, start=1):
            raw_text = str(row.get(self._text_field, "")).strip()
            prepared.append({
                "comment_id": str(row.get("comment_id", "")).strip() or str(index),
                "source": dict(row),
                "raw_text": raw_text,
                "processed_text": preprocess_text(raw_text),
            })
        return prepared

    async def _build_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Строит embeddings для всех комментариев.

        Args:
            texts: Список предобработанных текстов.

        Returns:
            Список embedding-векторов в том же порядке.

        Raises:
            RuntimeError: Если batch embedding-вызов не сработал после всех попыток.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_embedding_retries + 1):
            try:
                return [list(vector) for vector in await self._embeddings.aembed_documents(texts)]
            except Exception as exc:
                last_error = exc
                logger.error(
                    "Batch embedding завершился с ошибкой, попытка %d из %d: %s",
                    attempt + 1,
                    self._max_embedding_retries + 1,
                    exc,
                )
        raise RuntimeError("Batch embedding не сработал после всех retry-попыток.") from last_error

    async def _process_row(self, *, row: dict[str, Any], embedding: list[float]) -> None:
        """Назначает одной строке существующую или новую группу.

        Args:
            row: Подготовленная строка с исходным и предобработанным текстом.
            embedding: Embedding-вектор текущего комментария.

        Returns:
            ``None``.
        """
        candidates = self._store.search_candidates(
            query_text=row["processed_text"],
            query_embedding=embedding,
            candidate_group_limit=self._candidate_group_limit,
            max_examples_per_group=self._max_examples,
        )
        decision = await self._decision_engine.achoose_group(
            raw_text=row["raw_text"],
            processed_text=row["processed_text"],
            candidate_groups_text=self._format_candidates(candidates),
            candidate_group_ids={candidate.group_id for candidate in candidates},
        )

        if decision.decision_type == "existing_group":
            group = self._store.get_group(decision.group_id)
            if group is None:
                raise ValueError(f"LLM выбрала несуществующую группу: {decision.group_id}.")
            group_id = decision.group_id
        else:
            group_id = self._create_group(decision)

        self._store.add_comment(SimpleStoredComment(
            comment_id=row["comment_id"],
            source=row["source"],
            raw_text=row["raw_text"],
            processed_text=row["processed_text"],
            embedding=embedding,
            group_id=group_id,
        ))

    def _create_group(self, decision: SimpleGroupDecision) -> str:
        """Создает новую группу и возвращает ее идентификатор.

        Args:
            decision: Решение LLM с возможным названием новой группы.

        Returns:
            Идентификатор созданной группы.

        Raises:
            ValueError: Если LLM не вернула название новой группы.
        """
        if not decision.group_name:
            raise ValueError("Для новой группы LLM должна вернуть непустое поле group_name.")
        group = self._store.create_group(decision.group_name)
        return group.group_id

    def _format_candidates(self, candidates: list[SimpleCandidateGroup]) -> str:
        """Форматирует группы-кандидаты для LLM prompt-а.

        Args:
            candidates: Список групп-кандидатов из FAISS и BM25.

        Returns:
            Текстовое описание кандидатов с примерами комментариев.
        """
        if not candidates:
            return "Кандидатных групп нет."

        lines: list[str] = []
        for candidate in candidates:
            lines.append(
                f"group_id: {candidate.group_id} | "
                f"group_name: {truncate_text(candidate.group_name, 120)} | "
                f"retrieval_score: {candidate.score:.3f}"
            )
            for index, comment_id in enumerate(candidate.representative_comment_ids, start=1):
                comment = self._store.get_comment(comment_id)
                lines.append(f"  пример_{index}: {truncate_text(comment.raw_text)}")
            lines.append("")
        return "\n".join(lines).strip()

    async def _merge_small_candidate_groups(self) -> None:
        """Запускает второй проход слияния маленьких групп с похожими кандидатами.

        Args:
            Входные аргументы отсутствуют.

        Returns:
            ``None``. Комментарии маленьких групп переносятся в существующие группы,
            если LLM подтверждает совпадение основного смысла.
        """
        group_ids = [group.group_id for group in self._store.all_groups()]
        for group_id in group_ids:
            group = self._store.get_group(group_id)
            if group is None or len(group.member_comment_ids) > self._small_group_max_size:
                continue

            candidates = self._search_merge_candidates(group)
            if not candidates:
                continue

            merge_text = self._format_group_for_merge(group)
            decision = await self._decision_engine.achoose_group(
                raw_text=merge_text,
                processed_text=preprocess_text(merge_text),
                candidate_groups_text=self._format_candidates(candidates),
                candidate_group_ids={candidate.group_id for candidate in candidates},
            )
            if decision.decision_type != "existing_group":
                continue

            target_group = self._store.get_group(decision.group_id)
            current_group = self._store.get_group(group_id)
            if target_group is None or current_group is None or target_group.group_id == current_group.group_id:
                continue
            self._store.move_comments_to_group(
                comment_ids=list(current_group.member_comment_ids),
                target_group_id=target_group.group_id,
            )

    def _search_merge_candidates(self, group: SimpleCommentGroup) -> list[SimpleCandidateGroup]:
        """Ищет группы-кандидаты для слияния маленькой группы.

        Args:
            group: Маленькая группа, которую нужно проверить на возможное слияние.

        Returns:
            Список групп-кандидатов, отсортированных по объединенному retrieval-score.
        """
        candidates_by_group_id: dict[str, SimpleCandidateGroup] = {}
        for comment_id in group.member_comment_ids:
            comment = self._store.get_comment(comment_id)
            candidates = self._store.search_candidates(
                query_text=f"{group.group_name} {comment.processed_text}",
                query_embedding=comment.embedding,
                candidate_group_limit=self._merge_candidate_group_limit,
                max_examples_per_group=self._max_examples,
                exclude_group_ids={group.group_id},
            )
            for candidate in candidates:
                existing = candidates_by_group_id.get(candidate.group_id)
                if existing is None:
                    candidates_by_group_id[candidate.group_id] = candidate
                    continue
                existing.score += candidate.score
                for representative_id in candidate.representative_comment_ids:
                    if (
                            representative_id not in existing.representative_comment_ids
                            and len(existing.representative_comment_ids) < self._max_examples
                    ):
                        existing.representative_comment_ids.append(representative_id)

        return sorted(
            candidates_by_group_id.values(),
            key=lambda candidate: candidate.score,
            reverse=True,
        )[:self._merge_candidate_group_limit]

    def _format_group_for_merge(self, group: SimpleCommentGroup) -> str:
        """Форматирует маленькую группу как текущий объект для LLM-сравнения.

        Args:
            group: Маленькая группа, которую нужно проверить на слияние.

        Returns:
            Текстовое описание группы с названием и примерами комментариев.
        """
        lines = [
            "Проверь, нужно ли присоединить эту маленькую группу к одной из групп-кандидатов.",
            f"Название маленькой группы: {truncate_text(group.group_name, 180)}",
            "Комментарии маленькой группы:",
        ]
        for index, comment_id in enumerate(group.member_comment_ids, start=1):
            comment = self._store.get_comment(comment_id)
            lines.append(f"пример_{index}: {truncate_text(comment.raw_text)}")
        return "\n".join(lines)

    def _print(self, stage: str, current: int, total: int) -> None:
        """Печатает прогресс текущего этапа.

        Args:
            stage: Название этапа обработки.
            current: Количество обработанных элементов.
            total: Общее количество элементов.

        Returns:
            ``None``.
        """
        if self._show_progress:
            print(f"\r{stage}: {render_progress_bar(current, total)}".ljust(80))

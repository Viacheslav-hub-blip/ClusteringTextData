"""Упрощенный pipeline кластеризации комментариев через embeddings, FAISS/BM25 и LLM.

Файл содержит:
- ``render_progress_bar`` — форматирование прогресса для консольного запуска;
- ``technical_normalize`` — быстрая техническая нормализация текста без LLM;
- ``normalize_for_match`` — нормализация строки для точного сравнения;
- ``parse_decision_type`` — преобразование строкового решения LLM в enum;
- ``truncate_text`` — сокращение длинного текста для prompt-ов и fallback-названий;
- ``CommentNormalizer`` — локальная проверка и нормализация комментария без LLM-вызова;
- ``GroupDecisionEngine`` — LLM-выбор существующей или новой группы;
- ``GroupNameGenerator`` — опциональный LLM-нейминг групп;
- ``CommentMemoryStore`` — in-memory хранилище комментариев, групп, FAISS и BM25;
- ``IncrementalMVPClusteringPipeline`` — основной упрощенный pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import re

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever

from .async_utils import run_coroutine_sync
from .config import PrimaryPromptConfig
from .models import (
    CandidateGroup,
    CommentGroup,
    DecisionType,
    InputComment,
    NormalizationResult,
    PrimaryDecision,
    SimilarityHit,
    StoredComment,
)

logger = logging.getLogger(__name__)

# Таблица замены типографских символов на ASCII-аналоги
_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u00ab": '"', "\u00bb": '"',
    "\u2014": "-", "\u2013": "-",
})


def render_progress_bar(current: int, total: int, *, width: int = 24) -> str:
    """Рендерит ASCII прогресс-бар для вывода в консоль."""
    total = max(total, 1)
    current = max(0, min(current, total))
    filled = int(width * current / total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {current}/{total}"


def technical_normalize(value: str) -> str:
    """Техническая нормализация текста: замена типографики и схлопывание пробелов."""
    return re.sub(r"\s+", " ", str(value).translate(_QUOTE_MAP)).strip()


def normalize_for_match(value: str) -> str:
    """Агрессивная нормализация для сравнения строк: lowercase, только буквы/цифры/дефис."""
    value = technical_normalize(value).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zа-я\s-]+", " ", value, flags=re.IGNORECASE)).strip()


def parse_decision_type(value: object) -> DecisionType | None:
    """Конвертирует строку решения LLM в enum DecisionType, при неизвестном значении возвращает None."""
    normalized = normalize_for_match(str(value)).replace(" ", "_")
    for member in DecisionType:
        if normalized == member.value:
            return member
    return None


def truncate_text(value: str, limit: int = 10000) -> str:
    """Обрезает длинный текст для промптов и fallback-имён."""
    value = technical_normalize(value)
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


class CommentNormalizer:
    """Быстрый локальный нормализатор комментариев без LLM-вызовов."""

    def __init__(
            self,
            llm: BaseChatModel | None = None,
            *,
            min_meaningful_length: int = 3,
            llm_semaphore: asyncio.Semaphore | None = None,
            prompt_config: PrimaryPromptConfig | None = None,
    ) -> None:
        _ = llm, llm_semaphore, prompt_config
        self._min_meaningful_length = min_meaningful_length

    def _is_noise(self, text: str) -> bool:
        """Проверяет, является ли текст явным мусором по длине после нормализации."""
        return len(normalize_for_match(text).replace(" ", "")) < self._min_meaningful_length

    async def anormalize(self, text: str) -> NormalizationResult:
        """Возвращает технически очищенный текст и признак осмысленности.

        Args:
            text: Исходный текст комментария.

        Returns:
            Результат локальной нормализации без обращения к LLM.
        """
        normalized_text = technical_normalize(text)
        is_meaningful = bool(normalized_text) and not self._is_noise(normalized_text)
        return NormalizationResult(
            normalized_text=normalized_text,
            is_meaningful=is_meaningful,
            reason=(
                "Комментарий содержит осмысленный кейс"
                if is_meaningful
                else "Комментарий пустой, шумный или бессодержательный"
            ),
        )


class GroupDecisionEngine:
    """LLM для маршрутизации комментария: существующая группа или новая."""

    def __init__(
            self,
            llm: BaseChatModel,
            *,
            llm_semaphore: asyncio.Semaphore,
            prompt_config: PrimaryPromptConfig | None = None,
    ):
        prompt_config = prompt_config or PrimaryPromptConfig.default()
        self._chain = (
                ChatPromptTemplate.from_messages([
                    ("system", prompt_config.primary_decision_system),
                    ("human", prompt_config.primary_decision_human),
                ])
                | llm
                | JsonOutputParser()
        )
        self._sem = llm_semaphore

    async def achoose_group(
            self,
            *,
            raw_text: str,
            normalized_text: str,
            candidate_groups_text: str,
            candidate_group_ids: set[str],
            fallback: PrimaryDecision,
    ) -> PrimaryDecision:
        """Запрашивает у LLM решение о группе комментария, при ошибке возвращает fallback."""
        if not candidate_group_ids:
            return PrimaryDecision(
                decision_type=DecisionType.NEW_GROUP,
                group_id="",
                reason="Среди уже обработанных комментариев нет кандидатов для сравнения",
            )
        try:
            async with self._sem:
                raw = await self._chain.ainvoke({
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "candidate_groups": candidate_groups_text,
                })

            decision_type = parse_decision_type(raw.get("decision_type"))
            group_id = str(raw.get("group_id", "")).strip()
            reason = str(raw.get("reason", "")).strip() or fallback.reason

            if decision_type == DecisionType.EXISTING_GROUP and group_id in candidate_group_ids:
                return PrimaryDecision(decision_type=DecisionType.EXISTING_GROUP, group_id=group_id, reason=reason)
            if decision_type == DecisionType.NEW_GROUP:
                return PrimaryDecision(decision_type=DecisionType.NEW_GROUP, group_id="", reason=reason)

        except Exception as exc:
            logger.error("Решение о группе завершилось с ошибкой, применяется fallback: %s", exc)

        return fallback


class GroupNameGenerator:
    """Генератор коротких имён групп на основе LLM с fallback."""

    def __init__(
            self,
            llm: BaseChatModel,
            *,
            llm_semaphore: asyncio.Semaphore,
            prompt_config: PrimaryPromptConfig | None = None,
    ):
        prompt_config = prompt_config or PrimaryPromptConfig.default()
        self._chain = (
                ChatPromptTemplate.from_messages([
                    ("system", prompt_config.group_naming_system),
                    ("human", prompt_config.group_naming_human),
                ])
                | llm
                | JsonOutputParser()
        )
        self._sem = llm_semaphore

    async def agenerate_name(self, examples_text: str, fallback_name: str) -> str:
        """Генерирует короткое название группы по примерам комментариев."""
        try:
            async with self._sem:
                raw = await self._chain.ainvoke({"group_examples": examples_text})
            if group_name := technical_normalize(raw.get("group_name", "")):
                return group_name
        except Exception as exc:
            logger.error("Нейминг группы завершился с ошибкой, применяется fallback: %s", exc)
        return fallback_name or "Не определено"


class CommentMemoryStore:
    """In-memory хранилище комментариев и групп с гибридным поиском FAISS + BM25."""

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings
        self._ordered_ids: list[str] = []
        self._comments: dict[str, StoredComment] = {}
        self._groups: dict[str, CommentGroup] = {}
        self._vectorstore: FAISS | None = None
        self._bm25: BM25Retriever | None = None
        self._hybrid: EnsembleRetriever | None = None
        self._dirty = True
        self._next_group_index = 1

    def create_group(self) -> CommentGroup:
        """Создаёт новую пустую группу с уникальным последовательным ID."""
        group_id = f"group_{self._next_group_index:04d}"
        self._next_group_index += 1
        group = CommentGroup(group_id=group_id)
        self._groups[group_id] = group
        return group

    def add_comment(self, comment: StoredComment) -> None:
        """Сохраняет комментарий, добавляет в группу и индексирует в FAISS если осмысленный."""
        self._ordered_ids.append(comment.comment_id)
        self._comments[comment.comment_id] = comment

        if comment.group_id:
            self._groups.setdefault(comment.group_id, CommentGroup(group_id=comment.group_id)).member_comment_ids.append(comment.comment_id)

        if comment.decision_type != DecisionType.UNDEFINED and comment.group_id and comment.embedding and comment.normalized_text:
            self._index_comment(comment)
            self._dirty = True

    def get_comment(self, comment_id: str) -> StoredComment:
        """Возвращает комментарий по ID."""
        return self._comments[comment_id]

    def get_group_comments(self, group_id: str) -> list[StoredComment]:
        """Возвращает все комментарии группы в порядке добавления."""
        group = self._groups.get(group_id)
        return [self._comments[cid] for cid in group.member_comment_ids] if group else []

    def all_groups(self) -> list[CommentGroup]:
        """Возвращает все группы в порядке их ID."""
        return sorted(self._groups.values(), key=lambda g: g.group_id)

    def indexed_count(self) -> int:
        """Возвращает количество документов в FAISS-индексе."""
        return len(self._vectorstore.index_to_docstore_id) if self._vectorstore else 0

    def unique_group_comments(self, group_id: str) -> list[StoredComment]:
        """Возвращает уникальные комментарии группы без нормализованных дубликатов."""
        seen: set[str] = set()
        result: list[StoredComment] = []
        for comment in self.get_group_comments(group_id):
            key = normalize_for_match(comment.normalized_text)
            if key and key not in seen:
                seen.add(key)
                result.append(comment)
        return result

    def merge_groups_by_name(self) -> None:
        """Сливает группы с одинаковыми нормализованными именами."""
        canonical: dict[str, CommentGroup] = {}
        for group in self.all_groups():
            key = normalize_for_match(group.group_name)
            if not key:
                continue
            if key not in canonical:
                canonical[key] = group
                continue
            target = canonical[key]
            for cid in group.member_comment_ids:
                if cid not in target.member_comment_ids:
                    target.member_comment_ids.append(cid)
                if stored := self._comments.get(cid):
                    stored.group_id = target.group_id
            self._groups.pop(group.group_id, None)

    def comment_outputs(
            self,
            *,
            include_embeddings: bool = False,
            include_group_id: bool = False,
    ) -> list[dict]:
        """Сериализует комментарии в порядке обработки.

        Args:
            include_embeddings: Если ``True``, добавляет техническое поле ``embedding``.
            include_group_id: Если ``True``, добавляет технический идентификатор группы ``group_id``.

        Returns:
            Список словарей с данными комментариев. По умолчанию результат не содержит embeddings,
            а вместо технического номера группы содержит человекочитаемое поле ``group_name``.
        """
        comments = []
        for comment in (self._comments[cid] for cid in self._ordered_ids):
            group = self._groups.get(comment.group_id)
            output = {
                "comment_id": comment.comment_id,
                "raw_text": comment.raw_text,
                "normalized_text": comment.normalized_text,
                "group_name": (group.group_name if group else "") or "Не определено",
                "decision_type": comment.decision_type.value,
                "decision_reason": comment.decision_reason,
            }
            if include_embeddings:
                output["embedding"] = comment.embedding
            if include_group_id:
                output["group_id"] = comment.group_id
            comments.append(output)
        return comments

    def group_outputs(self) -> list[dict]:
        """Сериализует все группы в порядке их ID."""
        return [
            {"group_id": g.group_id, "group_name": g.group_name or "Не определено"}
            for g in self.all_groups()
        ]

    async def asearch_similar(
            self,
            query_text: str,
            top_k: int,
            *,
            max_hits_per_group: int | None = None,
    ) -> list[SimilarityHit]:
        """Гибридный поиск похожих комментариев через FAISS + BM25."""
        if not self._vectorstore or not query_text.strip() or top_k <= 0:
            return []

        retriever = self._ensure_retriever(top_k=top_k)
        if retriever is None:
            return []

        try:
            documents = await retriever.ainvoke(query_text)
        except Exception as exc:
            logger.error("Гибридный поиск завершился с ошибкой: %s", exc)
            return []

        hits: list[SimilarityHit] = []
        seen: set[str] = set()
        total = max(len(documents), 1)

        for rank, doc in enumerate(documents, start=1):
            cid = str(doc.metadata.get("comment_id", "")).strip()
            if not cid or cid in seen:
                continue
            stored = self._comments.get(cid)
            if not stored or not stored.group_id or stored.decision_type == DecisionType.UNDEFINED:
                continue
            seen.add(cid)
            hits.append(SimilarityHit(comment_id=cid, group_id=stored.group_id, similarity=1.0 - (rank - 1) / total))

        if max_hits_per_group is None:
            return hits[:top_k]

        result, counts = [], {}
        for hit in hits:
            if counts.get(hit.group_id, 0) >= max_hits_per_group:
                continue
            result.append(hit)
            counts[hit.group_id] = counts.get(hit.group_id, 0) + 1
            if len(result) >= top_k:
                break
        return result

    def _ensure_retriever(self, *, top_k: int) -> EnsembleRetriever | None:
        """Лениво создаёт или обновляет гибридный retriever при изменении индекса."""
        if not self._vectorstore:
            return None

        if self._hybrid and not self._dirty:
            self._bm25.k = top_k
            for r in self._hybrid.retrievers:
                if hasattr(r, "search_kwargs"):
                    r.search_kwargs.update({"k": top_k, "fetch_k": self.indexed_count()})
            return self._hybrid

        docs = [
            Document(
                page_content=c.normalized_text,
                metadata={"comment_id": c.comment_id, "group_id": c.group_id},
            )
            for c in (self._comments[cid] for cid in self._ordered_ids)
            if c.decision_type != DecisionType.UNDEFINED and c.group_id and c.normalized_text
        ]
        if not docs:
            return None

        dense = self._vectorstore.as_retriever(search_kwargs={"k": top_k, "fetch_k": self.indexed_count()})
        self._bm25 = BM25Retriever.from_documents(docs, k=top_k)
        self._hybrid = EnsembleRetriever(retrievers=[dense, self._bm25], weights=[0.6, 0.4], id_key="comment_id")
        self._dirty = False
        return self._hybrid

    def _index_comment(self, comment: StoredComment) -> None:
        """Добавляет комментарий в FAISS-индекс, создавая его при первом вызове."""
        text_emb = [(comment.normalized_text, comment.embedding or [])]
        meta = [{"comment_id": comment.comment_id, "group_id": comment.group_id}]

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_embeddings(
                text_emb, self._embeddings,
                metadatas=meta, ids=[comment.comment_id],
                normalize_L2=True, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        else:
            self._vectorstore.add_embeddings(text_emb, metadatas=meta, ids=[comment.comment_id])


class IncrementalMVPClusteringPipeline:
    """Упрощенный инкрементальный pipeline кластеризации комментариев.

    Args:
        llm: Chat-модель LangChain для выбора группы и опционального нейминга.
        embeddings: Embedding-модель LangChain для поиска похожих комментариев.
        retrieval_top_k: Количество похожих комментариев для поиска кандидатов.
        max_examples_per_candidate_group: Максимальное число примеров одной группы в prompt-е.
        min_meaningful_length: Минимальная длина содержательного комментария после очистки.
        primary_similarity_threshold: Порог fallback-назначения в ближайшую группу.
        max_concurrent_llm_requests: Лимит параллельных LLM-вызовов.
        max_concurrent_embedding_requests: Лимит параллельных embedding-вызовов.
        generate_group_names: Если ``True``, запускает LLM-нейминг групп после кластеризации.
        merge_same_name_groups: Если ``True``, объединяет группы с одинаковыми названиями.
        show_progress: Если ``True``, печатает прогресс в консоль.
        prompt_config: Prompt-конфигурация LLM-решений.

    Returns:
        Экземпляр pipeline, который возвращает словарь с комментариями и группами.
    """

    def __init__(
            self,
            llm: BaseChatModel,
            embeddings: Embeddings,
            *,
            retrieval_top_k: int = 12,
            max_examples_per_candidate_group: int = 3,
            min_meaningful_length: int = 3,
            primary_similarity_threshold: float = 0.5,
            max_concurrent_llm_requests: int = 3,
            max_concurrent_embedding_requests: int = 3,
            generate_group_names: bool = True,
            merge_same_name_groups: bool = False,
            show_progress: bool = False,
            prompt_config: PrimaryPromptConfig | None = None,
    ) -> None:
        prompt_config = prompt_config or PrimaryPromptConfig.default()
        self._embeddings = embeddings
        self._llm_sem = asyncio.Semaphore(max_concurrent_llm_requests)
        self._emb_sem = asyncio.Semaphore(max_concurrent_embedding_requests)
        self._normalizer = CommentNormalizer(
            llm,
            min_meaningful_length=min_meaningful_length,
            llm_semaphore=self._llm_sem,
            prompt_config=prompt_config,
        )
        self._decision_engine = GroupDecisionEngine(
            llm,
            llm_semaphore=self._llm_sem,
            prompt_config=prompt_config,
        )
        self._name_generator = GroupNameGenerator(
            llm,
            llm_semaphore=self._llm_sem,
            prompt_config=prompt_config,
        )
        self._store = CommentMemoryStore(embeddings)
        self._top_k = retrieval_top_k
        self._max_examples = max_examples_per_candidate_group
        self._threshold = primary_similarity_threshold
        self._generate_names = generate_group_names
        self._merge_same_name_groups = merge_same_name_groups
        self._show_progress = show_progress

    def run(self, raw_comments: list[dict]) -> dict[str, list[dict]]:
        """Синхронно запускает pipeline и возвращает результат.

        Args:
            raw_comments: Список словарей с полями ``comment_id`` и ``text``.

        Returns:
            Словарь с ключами ``comments`` и ``groups``.
        """
        return run_coroutine_sync(self.arun(raw_comments))

    async def arun(self, raw_comments: list[dict]) -> dict[str, list[dict]]:
        """Асинхронно запускает pipeline без LLM-нормализации и agentic-проходов.

        Args:
            raw_comments: Список словарей с полями ``comment_id`` и ``text``.

        Returns:
            Словарь с ключами ``comments`` и ``groups``.
        """
        self._store = CommentMemoryStore(self._embeddings)
        comments = self._validate(raw_comments)
        total = len(comments)
        logger.info("Упрощенный pipeline запущен: %d комментариев", total)

        self._print("Подготовка embeddings", 0, total)
        prepared = await asyncio.gather(*(self._prepare_comment(c) for c in comments))
        self._print("Подготовка embeddings", total, total)

        self._print("Кластеризация", 0, total)
        step = max(1, total // 10)
        for i, (comment, norm, emb) in enumerate(prepared, start=1):
            await self._process_comment(comment, norm, emb)
            if i == 1 or i == total or i % step == 0:
                self._print("Кластеризация", i, total)

        groups = self._store.all_groups()
        if self._generate_names:
            self._print("Нейминг групп", 0, len(groups))
            await self._generate_group_names(groups)
            self._print("Нейминг групп", len(groups), len(groups))
        else:
            self._assign_fallback_group_names(groups)

        if self._merge_same_name_groups:
            self._store.merge_groups_by_name()

        self._print("Готово", total, total)

        return self._build_output()

    async def arun_internal(self, raw_comments: list[dict]) -> dict[str, list[dict]]:
        """Запускает pipeline и возвращает технический результат для внутренних этапов.

        Args:
            raw_comments: Список словарей с исходными комментариями.

        Returns:
            Словарь с ``comments`` и ``groups``, где комментарии содержат ``embedding`` и ``group_id``.
        """
        await self.arun(raw_comments)
        return self._build_output(include_embeddings=True, include_group_id=True)

    def _build_output(
            self,
            *,
            include_embeddings: bool = False,
            include_group_id: bool = False,
    ) -> dict[str, list[dict]]:
        """Собирает результат pipeline в публичном или техническом формате.

        Args:
            include_embeddings: Если ``True``, включает embeddings в комментарии.
            include_group_id: Если ``True``, включает технические ID групп в комментарии.

        Returns:
            Словарь с ключами ``comments`` и ``groups``.
        """
        return {
            "comments": self._store.comment_outputs(
                include_embeddings=include_embeddings,
                include_group_id=include_group_id,
            ),
            "groups": self._store.group_outputs(),
        }

    @staticmethod
    def _validate(raw_comments: list[dict]) -> list[InputComment]:
        """Конвертирует сырые словари во входные модели, подставляя порядковый номер если нет ID."""
        return [
            InputComment(
                comment_id=str(raw.get("comment_id", "")).strip() or str(i),
                text=str(raw.get("text", "")).strip(),
            )
            for i, raw in enumerate(raw_comments, start=1)
        ]

    async def _prepare_comment(
            self, comment: InputComment
    ) -> tuple[InputComment, NormalizationResult, list[float] | None]:
        """Очищает комментарий и генерирует embedding, если комментарий содержательный.

        Args:
            comment: Входной комментарий.

        Returns:
            Кортеж из комментария, результата локальной нормализации и embedding-вектора.
        """
        norm = await self._normalizer.anormalize(comment.text)
        emb = await self._build_embedding(norm.normalized_text) if norm.is_meaningful else None
        return comment, norm, emb

    async def _process_comment(
            self,
            comment: InputComment,
            norm: NormalizationResult,
            embedding: list[float] | None,
    ) -> None:
        """Кластеризует один комментарий: ищет похожие, спрашивает LLM, сохраняет."""
        if not norm.is_meaningful:
            self._store.add_comment(StoredComment(
                comment_id=comment.comment_id, raw_text=comment.text,
                normalized_text=norm.normalized_text, embedding=None,
                group_id="", decision_type=DecisionType.UNDEFINED,
                decision_reason=norm.reason,
            ))
            return

        hits = await self._store.asearch_similar(
            query_text=norm.normalized_text,
            top_k=self._top_k,
            max_hits_per_group=self._max_examples,
        )
        candidates = self._build_candidates(hits)
        fallback = self._fallback_decision(norm.normalized_text, candidates)
        decision = await self._decision_engine.achoose_group(
            raw_text=comment.text,
            normalized_text=norm.normalized_text,
            candidate_groups_text=self._format_candidates(candidates),
            candidate_group_ids={c.group_id for c in candidates},
            fallback=fallback,
        )

        group_id = (
            decision.group_id
            if decision.decision_type == DecisionType.EXISTING_GROUP and decision.group_id
            else self._store.create_group().group_id
        )
        self._store.add_comment(StoredComment(
            comment_id=comment.comment_id, raw_text=comment.text,
            normalized_text=norm.normalized_text, embedding=embedding,
            group_id=group_id, decision_type=decision.decision_type,
            decision_reason=decision.reason,
        ))

    async def _build_embedding(self, text: str) -> list[float] | None:
        """Генерирует векторное представление текста с ограничением параллелизма."""
        try:
            async with self._emb_sem:
                return list(await self._embeddings.aembed_query(text))
        except Exception as exc:
            logger.error("Генерация эмбеддинга завершилась с ошибкой: %s", exc)
            return None

    def _build_candidates(self, hits: list[SimilarityHit]) -> list[CandidateGroup]:
        """Группирует hits по group_id, берёт максимальный score каждой группы."""
        scores: dict[str, float] = {}
        hit_ids: dict[str, list[str]] = {}
        for hit in hits:
            scores[hit.group_id] = max(scores.get(hit.group_id, float("-inf")), hit.similarity)
            hit_ids.setdefault(hit.group_id, []).append(hit.comment_id)

        return [
            CandidateGroup(
                group_id=gid,
                best_similarity=scores[gid],
                representative_comment_ids=list(dict.fromkeys(hit_ids[gid])),
            )
            for gid in sorted(scores, key=lambda g: scores[g], reverse=True)
        ]

    def _fallback_decision(self, normalized_text: str, candidates: list[CandidateGroup]) -> PrimaryDecision:
        """Детерминированное решение о группе без LLM: точное совпадение, порог или новая группа."""
        if not candidates:
            return PrimaryDecision(
                decision_type=DecisionType.NEW_GROUP, group_id="",
                reason="Подходящая существующая группа не найдена среди уже обработанных комментариев",
            )
        key = normalize_for_match(normalized_text)
        best = candidates[0]
        reps = [self._store.get_comment(cid) for cid in best.representative_comment_ids]

        if any(normalize_for_match(c.normalized_text) == key for c in reps):
            return PrimaryDecision(
                decision_type=DecisionType.EXISTING_GROUP, group_id=best.group_id,
                reason="Есть точное совпадение с уже обработанным комментарием этой группы",
            )
        if best.best_similarity >= self._threshold:
            return PrimaryDecision(
                decision_type=DecisionType.EXISTING_GROUP, group_id=best.group_id,
                reason="Лучший кандидат достаточно близок по retrieval similarity",
            )
        return PrimaryDecision(
            decision_type=DecisionType.NEW_GROUP, group_id="",
            reason="Похожие комментарии есть, но уверенного совпадения с существующей группой нет",
        )

    async def _generate_group_names(self, groups: list[CommentGroup]) -> None:
        """Параллельно генерирует имена для всех групп."""
        completed = 0
        total = len(groups)
        step = max(1, total // 10)
        lock = asyncio.Lock()

        async def name_one(group: CommentGroup) -> None:
            nonlocal completed
            reps = self._store.unique_group_comments(group.group_id)
            group.group_name = await self._name_generator.agenerate_name(
                examples_text=self._format_examples(reps),
                fallback_name=truncate_text(reps[0].normalized_text or reps[0].raw_text, 80) if reps else "Не определено",
            )
            async with lock:
                completed += 1
                if completed == 1 or completed == total or completed % step == 0:
                    self._print("Нейминг групп", completed, total)

        await asyncio.gather(*(name_one(g) for g in groups))

    def _assign_fallback_group_names(self, groups: list[CommentGroup]) -> None:
        """Заполняет названия групп без LLM по первому уникальному комментарию.

        Args:
            groups: Группы, которым нужно назначить fallback-названия.

        Returns:
            ``None``. Группы изменяются на месте.
        """
        for group in groups:
            reps = self._store.unique_group_comments(group.group_id)
            group.group_name = (
                truncate_text(reps[0].normalized_text or reps[0].raw_text, 80)
                if reps
                else "Не определено"
            )

    def _format_candidates(self, candidates: list[CandidateGroup]) -> str:
        """Форматирует кандидатные группы в текст для промпта LLM."""
        if not candidates:
            return "Кандидатных групп нет."
        lines: list[str] = []
        for c in candidates:
            lines.append(f"group_id: {c.group_id} | best_similarity: {c.best_similarity:.3f}")
            for i, cid in enumerate(c.representative_comment_ids, start=1):
                rep = self._store.get_comment(cid)
                lines.append(f"  пример_{i}: raw_text={truncate_text(rep.raw_text)} | normalized_text={truncate_text(rep.normalized_text)}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_examples(comments: list[StoredComment]) -> str:
        """Форматирует примеры комментариев группы для промпта нейминга."""
        if not comments:
            return "Примеров нет."
        return "\n".join(
            f"- comment_id: {c.comment_id} | raw_text: {truncate_text(c.raw_text)} | normalized_text: {truncate_text(c.normalized_text)}"
            for c in comments
        )

    def _print(self, stage: str, current: int, total: int) -> None:
        """Выводит строку прогресса для текущего этапа pipeline.

        Args:
            stage: Название текущего этапа.
            current: Количество обработанных элементов.
            total: Общее количество элементов.

        Returns:
            ``None``. При выключенном ``show_progress`` ничего не выводит.
        """
        if not self._show_progress:
            return
        print(f"\r{stage}: {render_progress_bar(current, total)}".ljust(80))

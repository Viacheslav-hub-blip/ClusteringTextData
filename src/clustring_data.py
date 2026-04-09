"""
Production pipeline для семантической группировки комментариев пользователей.

Архитектура:
1. Приём и валидация входных данных
2. Техническая нормализация
3. LLM-извлечение смыслового представления
4. Безопасное сокращение числа объектов (early deduplication)
5. Поиск кандидатов (embeddings + векторный поиск)
6. LLM-определение отношения между комментариями
7. Формирование конкретных групп
8. Формирование родительских групп
9. Присвоение итоговых меток
"""

import re
import uuid
import logging
from typing import Optional
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

try:
    from .prompts import (
        NAMING_HUMAN_PARENT,
        NAMING_HUMAN_SPECIFIC,
        NAMING_SYSTEM,
        RELATION_HUMAN,
        RELATION_SYSTEM,
        SEMANTIC_EXTRACTION_HUMAN,
        SEMANTIC_EXTRACTION_SYSTEM,
    )
except ImportError:
    from prompts import (
        NAMING_HUMAN_PARENT,
        NAMING_HUMAN_SPECIFIC,
        NAMING_SYSTEM,
        RELATION_HUMAN,
        RELATION_SYSTEM,
        SEMANTIC_EXTRACTION_HUMAN,
        SEMANTIC_EXTRACTION_SYSTEM,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("comment_grouping_pipeline")


# ===========================================================================
# 1. ВНУТРЕННИЕ МОДЕЛИ ДАННЫХ
# ===========================================================================

class InputComment(BaseModel):
    """Входной комментарий от пользователя."""
    comment_id: str
    text: str


class SemanticRepresentation(BaseModel):
    """
    Внутреннее смысловое представление комментария.
    Используется только внутри pipeline.
    """
    comment_id: str
    normalized_text: str
    general_topic: str = Field(description="Общий класс проблемы")
    exact_case: str = Field(description="Точный пользовательский кейс")
    key_qualifiers: list[str] = Field(
        description="Смыслообразующие уточнения"
    )
    canonical_key: str = Field(
        description="Нормализованный ключ для раннего объединения"
    )


class PairRelation(BaseModel):
    """Результат LLM-сопоставления двух семантических представлений."""
    relation: str = Field(
        description="SAME | SPECIFIC_OF | A_SPECIFIC_OF_B | B_SPECIFIC_OF_A | DIFFERENT"
    )
    reason: str = Field(description="Внутреннее пояснение для дебага")


@dataclass
class ClusterNode:
    """Внутренний узел кластера конкретной группы."""
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    specific_group_name: str = ""
    parent_group_name: str = ""
    member_ids: list[str] = field(default_factory=list)
    prototype_repr: Optional[SemanticRepresentation] = None


class OutputRecord(BaseModel):
    """Единственный обязательный внешний результат для одного комментария."""
    comment_id: str
    specific_group: str
    parent_group: str


# ===========================================================================
# 2. ЭТАП 1 + 2 — ВАЛИДАЦИЯ И НОРМАЛИЗАЦИЯ
# ===========================================================================

class InputProcessor:
    """
    Этап 1: Приём и валидация входных данных.
    Этап 2: Техническая нормализация без изменения смысла.
    """

    # Таблица нормализации кавычек и дефисов
    _QUOTE_MAP = str.maketrans({
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u00ab": '"', "\u00bb": '"',
        "\u2014": "-", "\u2013": "-",
    })

    def validate_and_normalize(
            self, raw_comments: list[dict]
    ) -> list[InputComment]:
        """
        Принимает сырой список словарей, возвращает
        валидированные и нормализованные InputComment.
        Невалидные записи логируются и пропускаются.
        """
        result: list[InputComment] = []

        for raw in raw_comments:
            try:
                comment_id = str(raw.get("comment_id", "")).strip()
                text = str(raw.get("text", "")).strip()

                if not comment_id:
                    logger.warning(
                        "Пропущена запись без comment_id: %s", raw
                    )
                    continue

                if not text:
                    logger.warning(
                        "Пропущен пустой комментарий id=%s", comment_id
                    )
                    continue

                normalized = self._normalize_text(text)
                result.append(
                    InputComment(comment_id=comment_id, text=normalized)
                )

            except Exception as exc:
                logger.error(
                    "Ошибка валидации записи %s: %s", raw, exc
                )

        logger.info(
            "InputProcessor: принято %d / %d комментариев",
            len(result), len(raw_comments),
        )
        return result

    def _normalize_text(self, text: str) -> str:
        """
        Лёгкая техническая нормализация:
        - trim
        - схлопывание пробелов
        - нормализация кавычек и дефисов
        НЕ изменяет смысл текста.
        """
        text = text.translate(self._QUOTE_MAP)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class SemanticExtractor:
    """Этап 3: LLM-извлечение смыслового представления."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._parser = JsonOutputParser()
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SEMANTIC_EXTRACTION_SYSTEM),
            ("human", SEMANTIC_EXTRACTION_HUMAN),
        ])
        self._chain = self._prompt | self._llm | self._parser

    def extract_batch(
            self, comments: list[InputComment], batch_size: int = 50
    ) -> list[SemanticRepresentation]:
        """
        Извлекает смысловые представления для всего batch.
        Обрабатывает по batch_size за раз, чтобы не перегружать API.
        Ошибка на одном комментарии не роняет весь batch.
        """
        results: list[SemanticRepresentation] = []

        for i in range(0, len(comments), batch_size):
            chunk = comments[i: i + batch_size]
            logger.info(
                "SemanticExtractor: обрабатывается чанк %d-%d",
                i, i + len(chunk),
            )
            for comment in chunk:
                try:
                    raw = self._chain.invoke({"text": comment.text})
                    repr_ = SemanticRepresentation(
                        comment_id=comment.comment_id,
                        normalized_text=comment.text,
                        general_topic=raw.get("general_topic", ""),
                        exact_case=raw.get("exact_case", ""),
                        key_qualifiers=raw.get("key_qualifiers", []),
                        canonical_key=raw.get(
                            "canonical_key", ""
                        ).lower().strip(),
                    )
                    results.append(repr_)
                except Exception as exc:
                    logger.error(
                        "SemanticExtractor: ошибка для id=%s: %s",
                        comment.comment_id, exc,
                    )
                    # Fallback: создаём минимальное представление
                    results.append(
                        SemanticRepresentation(
                            comment_id=comment.comment_id,
                            normalized_text=comment.text,
                            general_topic="Неизвестная тема",
                            exact_case=comment.text,
                            key_qualifiers=[],
                            canonical_key=comment.text.lower().strip(),
                        )
                    )
        return results


# ===========================================================================
# 4. ЭТАП 4 — БЕЗОПАСНОЕ РАННЕЕ ОБЪЕДИНЕНИЕ
# ===========================================================================

@dataclass
class PrototypeGroup:
    """
    Внутренний прототип группы после раннего объединения.
    Содержит один repr как представитель группы и список всех id.
    """
    prototype: SemanticRepresentation
    member_ids: list[str]


class EarlyDeduplicator:
    """
    Этап 4: Безопасное сокращение числа объектов.
    Объединяет только те комментарии, у которых canonical_key идентичен.
    Это покрывает чисто лингвистические перефразы.
    """

    def deduplicate(
            self, representations: list[SemanticRepresentation]
    ) -> list[PrototypeGroup]:
        """
        Группирует по canonical_key.
        Возвращает список прототипов — по одному на уникальный ключ.
        """
        groups: dict[str, PrototypeGroup] = {}

        for repr_ in representations:
            key = repr_.canonical_key
            if key not in groups:
                groups[key] = PrototypeGroup(
                    prototype=repr_,
                    member_ids=[repr_.comment_id],
                )
            else:
                groups[key].member_ids.append(repr_.comment_id)

        result = list(groups.values())
        logger.info(
            "EarlyDeduplicator: %d представлений → %d прототипов",
            len(representations), len(result),
        )
        return result


# ===========================================================================
# 5. ЭТАП 5 — ПОИСК КАНДИДАТОВ (EMBEDDINGS + FAISS)
# ===========================================================================

class CandidateRetriever:
    """
    Этап 5: Поиск кандидатов на сравнение через векторный индекс.
    Результат — только кандидаты для проверки LLM, не финальная группировка.
    """

    def __init__(
            self,
            embeddings: Embeddings,
            top_k: int = 10,
    ):
        self._embeddings = embeddings
        self._top_k = top_k

    def build_index_and_retrieve(
            self, prototypes: list[PrototypeGroup]
    ) -> dict[str, list[str]]:
        """
        Строит FAISS-индекс по exact_case всех прототипов.
        Для каждого прототипа возвращает top_k ближайших кандидатов
        (по canonical_key, исключая самого себя).

        Возвращает: {canonical_key → [canonical_key кандидатов]}
        """
        if len(prototypes) <= 1:
            return {}

        docs = []
        for pg in prototypes:
            text_for_index = (
                f"{pg.prototype.exact_case} "
                f"{' '.join(pg.prototype.key_qualifiers)}"
            )
            docs.append(
                Document(
                    page_content=text_for_index,
                    metadata={"canonical_key": pg.prototype.canonical_key},
                )
            )

        vectorstore = FAISS.from_documents(docs, self._embeddings)

        candidates: dict[str, list[str]] = {}
        for pg in prototypes:
            query = (
                f"{pg.prototype.exact_case} "
                f"{' '.join(pg.prototype.key_qualifiers)}"
            )
            # +1 потому что сам прототип попадает в результаты
            hits = vectorstore.similarity_search(query, k=self._top_k + 1)
            own_key = pg.prototype.canonical_key
            found_keys = [
                h.metadata["canonical_key"]
                for h in hits
                if h.metadata["canonical_key"] != own_key
            ]
            candidates[own_key] = found_keys

        logger.info(
            "CandidateRetriever: индекс из %d прототипов построен",
            len(prototypes),
        )
        return candidates


class PairRelationClassifier:
    """Этап 6: LLM-сопоставление пар прототипов."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._parser = JsonOutputParser()
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", RELATION_SYSTEM),
            ("human", RELATION_HUMAN),
        ])
        self._chain = self._prompt | self._llm | self._parser
        # Кэш, чтобы не сравнивать одну пару дважды
        self._cache: dict[tuple[str, str], PairRelation] = {}

    def classify_pair(
            self,
            repr_a: SemanticRepresentation,
            repr_b: SemanticRepresentation,
    ) -> PairRelation:
        """Классифицирует отношение между двумя смысловыми представлениями."""
        cache_key = tuple(
            sorted([repr_a.canonical_key, repr_b.canonical_key])
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            raw = self._chain.invoke({
                "exact_case_a": repr_a.exact_case,
                "qualifiers_a": repr_a.key_qualifiers,
                "topic_a": repr_a.general_topic,
                "canonical_key_a": repr_a.canonical_key,
                "exact_case_b": repr_b.exact_case,
                "qualifiers_b": repr_b.key_qualifiers,
                "topic_b": repr_b.general_topic,
                "canonical_key_b": repr_b.canonical_key,
            })
            relation = PairRelation(
                relation=raw.get("relation", "DIFFERENT"),
                reason=raw.get("reason", ""),
            )
        except Exception as exc:
            logger.error(
                "PairRelationClassifier: ошибка для пары (%s, %s): %s",
                repr_a.canonical_key, repr_b.canonical_key, exc,
            )
            relation = PairRelation(
                relation="DIFFERENT",
                reason=f"Ошибка классификации: {exc}",
            )

        self._cache[cache_key] = relation
        return relation


class GroupNameGenerator:
    """Генерирует читаемые названия групп через LLM."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._parser = JsonOutputParser()

        self._specific_prompt = ChatPromptTemplate.from_messages([
            ("system", NAMING_SYSTEM),
            ("human", NAMING_HUMAN_SPECIFIC),
        ])
        self._parent_prompt = ChatPromptTemplate.from_messages([
            ("system", NAMING_SYSTEM),
            ("human", NAMING_HUMAN_PARENT),
        ])

        self._specific_chain = (
                self._specific_prompt | self._llm | self._parser
        )
        self._parent_chain = (
                self._parent_prompt | self._llm | self._parser
        )

    def generate_specific_name(
            self, repr_: SemanticRepresentation
    ) -> str:
        """Генерирует название конкретной группы."""
        try:
            raw = self._specific_chain.invoke({
                "general_topic": repr_.general_topic,
                "exact_case": repr_.exact_case,
                "key_qualifiers": repr_.key_qualifiers,
            })
            return raw.get("group_name", repr_.exact_case)
        except Exception as exc:
            logger.error(
                "GroupNameGenerator (specific): ошибка: %s", exc
            )
            return repr_.exact_case

    def generate_parent_name(
            self,
            specific_group_names: list[str],
            general_topics: list[str],
    ) -> str:
        """Генерирует название родительской группы."""
        try:
            raw = self._parent_chain.invoke({
                "specific_groups": "\n".join(
                    f"- {n}" for n in specific_group_names
                ),
                "general_topics": "\n".join(
                    f"- {topic}" for topic in general_topics if topic
                ),
            })
            return raw.get("group_name", specific_group_names[0])
        except Exception as exc:
            logger.error(
                "GroupNameGenerator (parent): ошибка: %s", exc
            )
            return specific_group_names[0] if specific_group_names else "Прочее"


class ClusterBuilder:
    """
    Этапы 7 и 8: Формирование конкретных и родительских групп.

    Логика:
    - Каждый прототип изначально — отдельный кластер.
    - Пары с SAME объединяются в один кластер.
    - Пары с SPECIFIC_OF → более конкретный остаётся отдельным кластером,
      но связывается с более общим как с родителем.
    - Пары с DIFFERENT → остаются разными кластерами.
    """

    def __init__(
            self,
            classifier: PairRelationClassifier,
            name_generator: GroupNameGenerator,
    ):
        self._classifier = classifier
        self._name_generator = name_generator

    def build_clusters(
            self,
            prototypes: list[PrototypeGroup],
            candidates: dict[str, list[str]],
    ) -> list[ClusterNode]:
        """
        Строит итоговые кластеры.

        Алгоритм:
        1. Каждый прототип → отдельный кластер.
        2. Для каждой пары кандидатов вызываем LLM-классификацию.
        3. SAME → объединяем кластеры (union-find).
        4. SPECIFIC_OF → запоминаем связь «частное→общее».
        5. Генерируем названия для конкретных групп.
        6. Формируем родительские группы из SPECIFIC_OF-связей.
        """
        # Индекс прототипов по canonical_key
        proto_index: dict[str, PrototypeGroup] = {
            pg.prototype.canonical_key: pg for pg in prototypes
        }

        # Union-Find для SAME-объединений
        parent_uf: dict[str, str] = {
            pg.prototype.canonical_key: pg.prototype.canonical_key
            for pg in prototypes
        }

        def find(x: str) -> str:
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent_uf[ry] = rx

        # Связи SPECIFIC_OF: {конкретный_root → общий_root}
        specific_of_links: dict[str, str] = {}

        # Шаг 2: Классифицируем пары кандидатов
        processed_pairs: set[tuple[str, str]] = set()

        for pg in prototypes:
            own_key = pg.prototype.canonical_key
            cands = candidates.get(own_key, [])

            for cand_key in cands:
                if cand_key not in proto_index:
                    continue
                pair = tuple(sorted([own_key, cand_key]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                repr_a = proto_index[own_key].prototype
                repr_b = proto_index[cand_key].prototype

                relation = self._classifier.classify_pair(repr_a, repr_b)
                logger.debug(
                    "Пара (%s, %s) → %s | %s",
                    own_key, cand_key, relation.relation, relation.reason,
                )

                if relation.relation == "SAME":
                    union(own_key, cand_key)

                elif relation.relation == "SPECIFIC_OF":
                    # Определяем, какой из двух является частным случаем.
                    # Эвристика: у кого больше key_qualifiers → тот конкретнее
                    if len(repr_a.key_qualifiers) >= len(repr_b.key_qualifiers):
                        specific_of_links[own_key] = cand_key
                    else:
                        specific_of_links[cand_key] = own_key

                elif relation.relation == "A_SPECIFIC_OF_B":
                    specific_of_links[own_key] = cand_key

                elif relation.relation == "B_SPECIFIC_OF_A":
                    specific_of_links[cand_key] = own_key

        # Шаг 3: Собираем кластеры по root из Union-Find
        cluster_map: dict[str, ClusterNode] = {}

        for pg in prototypes:
            key = pg.prototype.canonical_key
            root = find(key)

            if root not in cluster_map:
                cluster_map[root] = ClusterNode(
                    prototype_repr=proto_index[root].prototype,
                    member_ids=[],
                )
            # Добавляем всех членов прототипа
            cluster_map[root].member_ids.extend(pg.member_ids)

        # Шаг 4: Генерируем названия конкретных групп
        logger.info(
            "ClusterBuilder: генерация названий для %d кластеров",
            len(cluster_map),
        )
        for root, cluster in cluster_map.items():
            cluster.specific_group_name = (
                self._name_generator.generate_specific_name(
                    cluster.prototype_repr
                )
            )
            cluster.parent_group_name = self._name_generator.generate_parent_name(
                specific_group_names=[cluster.specific_group_name],
                general_topics=[cluster.prototype_repr.general_topic],
            )

        # Шаг 5: Формируем родительские группы
        # Нормализуем SPECIFIC_OF-ссылки к root
        parent_groups: dict[str, str] = {}  # root → parent_root

        for specific_key, general_key in specific_of_links.items():
            specific_root = find(specific_key)
            general_root = find(general_key)
            if specific_root != general_root:
                parent_groups[specific_root] = general_root

        # Группируем кластеры по общему родителю для генерации имени
        parent_name_inputs: dict[str, list[str]] = {}

        for root in cluster_map:
            if root in parent_groups:
                parent_root = parent_groups[root]
                if parent_root not in parent_name_inputs:
                    parent_name_inputs[parent_root] = []
                parent_name_inputs[parent_root].append(
                    cluster_map[root].specific_group_name
                )

        # Генерируем названия родительских групп
        generated_parent_names: dict[str, str] = {}

        for parent_root, children_names in parent_name_inputs.items():
            if parent_root in cluster_map:
                # Если родитель сам является кластером —
                # включаем его имя в список для генерации
                children_names_with_parent = (
                        [cluster_map[parent_root].specific_group_name]
                        + children_names
                )
                parent_topics = [
                    cluster_map[parent_root].prototype_repr.general_topic,
                ]
                for root, parent_root_candidate in parent_groups.items():
                    if parent_root_candidate == parent_root:
                        parent_topics.append(
                            cluster_map[root].prototype_repr.general_topic
                        )
                generated_parent_names[parent_root] = (
                    self._name_generator.generate_parent_name(
                        specific_group_names=children_names_with_parent,
                        general_topics=parent_topics,
                    )
                )
            else:
                parent_topics = [
                    cluster_map[root].prototype_repr.general_topic
                    for root, parent_root_candidate in parent_groups.items()
                    if parent_root_candidate == parent_root
                ]
                generated_parent_names[parent_root] = (
                    self._name_generator.generate_parent_name(
                        specific_group_names=children_names,
                        general_topics=parent_topics,
                    )
                )

        # Назначаем parent_group каждому кластеру
        for root, cluster in cluster_map.items():
            if root in parent_groups:
                parent_root = parent_groups[root]
                if parent_root in generated_parent_names:
                    cluster.parent_group_name = (
                        generated_parent_names[parent_root]
                    )
                elif parent_root in cluster_map and cluster_map[parent_root].parent_group_name:
                    cluster.parent_group_name = (
                        cluster_map[parent_root].parent_group_name
                    )

        return list(cluster_map.values())


# ===========================================================================
# 9. ЭТАП 9 — ПРИСВОЕНИЕ ИТОГОВЫХ МЕТОК
# ===========================================================================

class LabelAssigner:
    """Этап 9: Присваивает итоговые метки каждому comment_id."""

    def assign(
            self, clusters: list[ClusterNode]
    ) -> dict[str, OutputRecord]:
        """
        Возвращает словарь {comment_id → OutputRecord}.
        """
        result: dict[str, OutputRecord] = {}

        for cluster in clusters:
            for cid in cluster.member_ids:
                result[cid] = OutputRecord(
                    comment_id=cid,
                    specific_group=cluster.specific_group_name,
                    parent_group=cluster.parent_group_name,
                )

        return result


# ===========================================================================
# ГЛАВНЫЙ PIPELINE
# ===========================================================================

class CommentGroupingPipeline:
    """
    Production pipeline для семантической группировки комментариев.

    Использование:
        pipeline = CommentGroupingPipeline(
            llm=llm,
            embeddings=embeddings,
        )
        results = pipeline.run(raw_comments)
    """

    def __init__(
            self,
            llm: BaseChatModel,
            embeddings: Embeddings,
            candidate_top_k: int = 10,
            extraction_batch_size: int = 50,
    ):
        self._input_processor = InputProcessor()
        self._semantic_extractor = SemanticExtractor(llm)
        self._early_deduplicator = EarlyDeduplicator()
        self._candidate_retriever = CandidateRetriever(
            embeddings, top_k=candidate_top_k
        )
        self._pair_classifier = PairRelationClassifier(llm)
        self._name_generator = GroupNameGenerator(llm)
        self._cluster_builder = ClusterBuilder(
            self._pair_classifier, self._name_generator
        )
        self._label_assigner = LabelAssigner()
        self._extraction_batch_size = extraction_batch_size

    def run(self, raw_comments: list[dict]) -> list[dict]:
        """
        Главный метод pipeline.

        Args:
            raw_comments: список словарей с comment_id и text.

        Returns:
            Список словарей с comment_id, specific_group, parent_group.
        """
        logger.info(
            "Pipeline: старт. Входных комментариев: %d", len(raw_comments)
        )

        # Этап 1+2: Валидация и нормализация
        comments = self._input_processor.validate_and_normalize(raw_comments)
        if not comments:
            logger.warning("Pipeline: нет валидных комментариев для обработки.")
            return []

        # Этап 3: LLM-извлечение смысла
        logger.info("Pipeline: Этап 3 — извлечение смысловых представлений")
        representations = self._semantic_extractor.extract_batch(
            comments, batch_size=self._extraction_batch_size
        )

        # Этап 4: Раннее объединение перефразов
        logger.info("Pipeline: Этап 4 — раннее объединение")
        prototypes = self._early_deduplicator.deduplicate(representations)

        # Этап 5: Поиск кандидатов
        logger.info("Pipeline: Этап 5 — поиск кандидатов")
        candidates = self._candidate_retriever.build_index_and_retrieve(
            prototypes
        )

        # Этапы 6+7+8: Сопоставление и формирование групп
        logger.info(
            "Pipeline: Этапы 6-8 — классификация пар и формирование групп"
        )
        clusters = self._cluster_builder.build_clusters(prototypes, candidates)

        # Этап 9: Присвоение меток
        logger.info("Pipeline: Этап 9 — присвоение итоговых меток")
        label_map = self._label_assigner.assign(clusters)

        # Формируем финальный результат в исходном порядке
        output: list[dict] = []
        for comment in comments:
            if comment.comment_id in label_map:
                output.append(
                    label_map[comment.comment_id].model_dump()
                )
            else:
                logger.error(
                    "Pipeline: не найдена метка для id=%s",
                    comment.comment_id,
                )
                output.append({
                    "comment_id": comment.comment_id,
                    "specific_group": "Ошибка классификации",
                    "parent_group": "Ошибка классификации",
                })

        logger.info(
            "Pipeline: завершён. Обработано %d комментариев.", len(output)
        )
        return output



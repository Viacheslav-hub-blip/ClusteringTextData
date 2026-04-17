"""Incremental MVP pipeline: normalize, retrieve, decide, verify, save, and name groups."""

from __future__ import annotations

import asyncio
import logging
import re

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .models import (
    CandidateGroup,
    CommentGroup,
    DecisionType,
    InputComment,
    NormalizationResult,
    PrimaryDecision,
    SimilarityHit,
    StoredComment,
    VerificationDecision,
)
from .prompts import (
    GROUP_NAMING_HUMAN,
    GROUP_NAMING_SYSTEM,
    NORMALIZATION_HUMAN,
    NORMALIZATION_SYSTEM,
    PRIMARY_DECISION_HUMAN,
    PRIMARY_DECISION_SYSTEM,
    VERIFICATION_HUMAN,
    VERIFICATION_SYSTEM,
)

logger = logging.getLogger(__name__)

_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2014": "-",
        "\u2013": "-",
    }
)


def render_progress_bar(current: int, total: int, *, width: int = 24) -> str:
    """Render a compact ASCII progress bar for console output."""
    total = max(total, 1)
    current = max(0, min(current, total))
    filled = int(width * current / total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {current}/{total}"


def technical_normalize_text(value: str) -> str:
    """Apply safe technical normalization without changing the meaning."""
    value = str(value).translate(_QUOTE_MAP)
    return re.sub(r"\s+", " ", value).strip()


def normalize_for_match(value: str) -> str:
    """Normalize text for stable string comparison."""
    value = technical_normalize_text(value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я\s-]+", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def coerce_bool(value: object, default: bool = False) -> bool:
    """Convert LLM output to bool without trusting Python truthiness of strings."""
    if isinstance(value, bool):
        return value
    normalized = normalize_for_match(str(value))
    if normalized in {"true", "1", "yes", "да"}:
        return True
    if normalized in {"false", "0", "no", "нет"}:
        return False
    return default


def parse_decision_type(value: object) -> DecisionType | None:
    """Convert the LLM decision string into a known enum value."""
    normalized = normalize_for_match(str(value)).replace(" ", "_")
    if normalized == DecisionType.EXISTING_GROUP.value:
        return DecisionType.EXISTING_GROUP
    if normalized == DecisionType.NEW_GROUP.value:
        return DecisionType.NEW_GROUP
    if normalized == DecisionType.UNDEFINED.value:
        return DecisionType.UNDEFINED
    return None


def truncate_text(value: str, limit: int = 120) -> str:
    """Trim long text for prompts and fallback names."""
    value = technical_normalize_text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


class CommentNormalizer:
    """LLM-based comment normalizer with a small local fallback."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        min_meaningful_length: int = 3,
        llm_semaphore: asyncio.Semaphore | None = None,
    ):
        parser = JsonOutputParser()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", NORMALIZATION_SYSTEM),
                    ("human", NORMALIZATION_HUMAN),
                ]
            )
            | llm
            | parser
        )
        self._min_meaningful_length = min_meaningful_length
        self._llm_semaphore = llm_semaphore

    def normalize(self, text: str) -> NormalizationResult:
        return asyncio.run(self.anormalize(text))

    async def anormalize(self, text: str) -> NormalizationResult:
        """Normalize a comment and decide if it is meaningful."""
        try:
            raw = await self._ainvoke_chain({"text": text})
            normalized_text = technical_normalize_text(raw.get("normalized_text", text))
            if not normalized_text:
                normalized_text = technical_normalize_text(text)
            is_meaningful = coerce_bool(raw.get("is_meaningful"), default=True)
            if self._is_clearly_noise(normalized_text):
                is_meaningful = False
            reason = str(raw.get("reason", "")).strip()
            if not reason:
                reason = (
                    "Комментарий содержит осмысленный кейс"
                    if is_meaningful
                    else "Комментарий пустой, шумный или бессодержательный"
                )
            return NormalizationResult(
                normalized_text=normalized_text,
                is_meaningful=is_meaningful,
                reason=reason,
            )
        except Exception as exc:
            logger.error("Normalization failed, using fallback: %s", exc)
            return self._fallback(text)

    async def _ainvoke_chain(self, payload: dict) -> dict:
        if self._llm_semaphore is None:
            return await self._chain.ainvoke(payload)
        async with self._llm_semaphore:
            return await self._chain.ainvoke(payload)

    def _fallback(self, text: str) -> NormalizationResult:
        normalized_text = technical_normalize_text(text)
        is_meaningful = not self._is_clearly_noise(normalized_text)
        reason = (
            "Комментарий содержит осмысленный кейс"
            if is_meaningful
            else "Комментарий пустой, шумный или бессодержательный"
        )
        return NormalizationResult(
            normalized_text=normalized_text,
            is_meaningful=is_meaningful,
            reason=reason,
        )

    def _is_clearly_noise(self, text: str) -> bool:
        normalized = normalize_for_match(text)
        return len(normalized.replace(" ", "")) < self._min_meaningful_length


class GroupDecisionEngine:
    """LLM steps for primary routing and verification."""

    def __init__(self, llm: BaseChatModel, *, llm_semaphore: asyncio.Semaphore | None = None):
        parser = JsonOutputParser()
        self._primary_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", PRIMARY_DECISION_SYSTEM),
                    ("human", PRIMARY_DECISION_HUMAN),
                ]
            )
            | llm
            | parser
        )
        self._verification_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", VERIFICATION_SYSTEM),
                    ("human", VERIFICATION_HUMAN),
                ]
            )
            | llm
            | parser
        )
        self._llm_semaphore = llm_semaphore

    def choose_group(
        self,
        *,
        raw_text: str,
        normalized_text: str,
        candidate_groups_text: str,
        candidate_group_ids: set[str],
        fallback: PrimaryDecision,
    ) -> PrimaryDecision:
        return asyncio.run(
            self.achoose_group(
                raw_text=raw_text,
                normalized_text=normalized_text,
                candidate_groups_text=candidate_groups_text,
                candidate_group_ids=candidate_group_ids,
                fallback=fallback,
            )
        )

    async def achoose_group(
        self,
        *,
        raw_text: str,
        normalized_text: str,
        candidate_groups_text: str,
        candidate_group_ids: set[str],
        fallback: PrimaryDecision,
    ) -> PrimaryDecision:
        """Ask the LLM to choose an existing group or create a new one."""
        if not candidate_group_ids:
            return PrimaryDecision(
                decision_type=DecisionType.NEW_GROUP,
                group_id="",
                reason="Среди уже обработанных комментариев нет кандидатов для сравнения",
            )
        try:
            raw = await self._ainvoke_chain(
                self._primary_chain,
                {
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "candidate_groups": candidate_groups_text,
                }
            )
            decision_type = parse_decision_type(raw.get("decision_type"))
            group_id = str(raw.get("group_id", "")).strip()
            reason = str(raw.get("reason", "")).strip() or fallback.reason
            if decision_type == DecisionType.EXISTING_GROUP and group_id in candidate_group_ids:
                return PrimaryDecision(
                    decision_type=DecisionType.EXISTING_GROUP,
                    group_id=group_id,
                    reason=reason,
                )
            if decision_type == DecisionType.NEW_GROUP:
                return PrimaryDecision(
                    decision_type=DecisionType.NEW_GROUP,
                    group_id="",
                    reason=reason,
                )
        except Exception as exc:
            logger.error("Primary decision failed, using fallback: %s", exc)
        return fallback

    def verify_group(
        self,
        *,
        raw_text: str,
        normalized_text: str,
        group_id: str,
        group_examples_text: str,
        fallback: VerificationDecision,
    ) -> VerificationDecision:
        return asyncio.run(
            self.averify_group(
                raw_text=raw_text,
                normalized_text=normalized_text,
                group_id=group_id,
                group_examples_text=group_examples_text,
                fallback=fallback,
            )
        )

    async def averify_group(
        self,
        *,
        raw_text: str,
        normalized_text: str,
        group_id: str,
        group_examples_text: str,
        fallback: VerificationDecision,
    ) -> VerificationDecision:
        """Ask the LLM to verify the chosen existing group."""
        try:
            raw = await self._ainvoke_chain(
                self._verification_chain,
                {
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    "group_id": group_id,
                    "group_examples": group_examples_text,
                }
            )
            passed = coerce_bool(raw.get("fits_group"), default=fallback.passed)
            reason = str(raw.get("reason", "")).strip() or fallback.reason
            return VerificationDecision(passed=passed, reason=reason)
        except Exception as exc:
            logger.error("Verification failed, using fallback: %s", exc)
            return fallback

    async def _ainvoke_chain(self, chain, payload: dict) -> dict:
        if self._llm_semaphore is None:
            return await chain.ainvoke(payload)
        async with self._llm_semaphore:
            return await chain.ainvoke(payload)


class GroupNameGenerator:
    """LLM-based group naming with a lightweight fallback."""

    def __init__(self, llm: BaseChatModel, *, llm_semaphore: asyncio.Semaphore | None = None):
        parser = JsonOutputParser()
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", GROUP_NAMING_SYSTEM),
                    ("human", GROUP_NAMING_HUMAN),
                ]
            )
            | llm
            | parser
        )
        self._llm_semaphore = llm_semaphore

    def generate_name(self, examples_text: str, fallback_name: str) -> str:
        return asyncio.run(self.agenerate_name(examples_text, fallback_name))

    async def agenerate_name(self, examples_text: str, fallback_name: str) -> str:
        """Generate a short group name."""
        try:
            if self._llm_semaphore is None:
                raw = await self._chain.ainvoke({"group_examples": examples_text})
            else:
                async with self._llm_semaphore:
                    raw = await self._chain.ainvoke({"group_examples": examples_text})
            group_name = technical_normalize_text(raw.get("group_name", ""))
            if group_name:
                return group_name
        except Exception as exc:
            logger.error("Group naming failed, using fallback: %s", exc)
        return fallback_name or "Не определено"


class CommentMemoryStore:
    """In-memory storage for processed comments, groups, and vector search."""

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings
        self._ordered_comment_ids: list[str] = []
        self._comments_by_id: dict[str, StoredComment] = {}
        self._groups_by_id: dict[str, CommentGroup] = {}
        self._vectorstore: FAISS | None = None
        self._next_group_index = 1

    def create_group(self) -> CommentGroup:
        """Create an empty group with a stable sequential id."""
        group_id = f"group_{self._next_group_index:04d}"
        self._next_group_index += 1
        group = CommentGroup(group_id=group_id)
        self._groups_by_id[group_id] = group
        return group

    def add_comment(self, comment: StoredComment) -> None:
        """Persist a comment and index it if it is eligible for retrieval."""
        self._ordered_comment_ids.append(comment.comment_id)
        self._comments_by_id[comment.comment_id] = comment

        if comment.group_id:
            group = self._groups_by_id.setdefault(comment.group_id, CommentGroup(group_id=comment.group_id))
            group.member_comment_ids.append(comment.comment_id)

        if (
            comment.decision_type != DecisionType.UNDEFINED
            and comment.group_id
            and comment.embedding
            and comment.normalized_text
        ):
            self._index_comment(comment)

    def get_comment(self, comment_id: str) -> StoredComment:
        """Return one stored comment."""
        return self._comments_by_id[comment_id]

    def get_group_comments(self, group_id: str) -> list[StoredComment]:
        """Return all stored comments in a group."""
        group = self._groups_by_id.get(group_id)
        if not group:
            return []
        return [self._comments_by_id[comment_id] for comment_id in group.member_comment_ids]

    def all_groups(self) -> list[CommentGroup]:
        """Return all groups in creation order."""
        return sorted(self._groups_by_id.values(), key=lambda group: group.group_id)

    def indexed_comments_count(self) -> int:
        """Return the number of comments currently stored in the vector index."""
        if self._vectorstore is None:
            return 0
        return len(self._vectorstore.index_to_docstore_id)

    def merge_groups_by_name(self) -> None:
        """Merge groups that ended up with the same normalized final name."""
        canonical_group_by_name: dict[str, CommentGroup] = {}
        for group in self.all_groups():
            normalized_name = normalize_for_match(group.group_name)
            if not normalized_name:
                continue

            canonical_group = canonical_group_by_name.get(normalized_name)
            if canonical_group is None:
                canonical_group_by_name[normalized_name] = group
                continue

            for comment_id in group.member_comment_ids:
                if comment_id not in canonical_group.member_comment_ids:
                    canonical_group.member_comment_ids.append(comment_id)
                stored_comment = self._comments_by_id.get(comment_id)
                if stored_comment:
                    stored_comment.group_id = canonical_group.group_id

            if len(group.member_comment_ids) > len(canonical_group.member_comment_ids) and group.group_name:
                canonical_group.group_name = group.group_name

            self._groups_by_id.pop(group.group_id, None)

    def comment_outputs(self) -> list[dict]:
        """Serialize stored comment records."""
        return [
            {
                "comment_id": comment.comment_id,
                "raw_text": comment.raw_text,
                "normalized_text": comment.normalized_text,
                "embedding": comment.embedding,
                "group_id": comment.group_id,
                "decision_type": comment.decision_type.value,
                "decision_reason": comment.decision_reason,
                "verification_passed": comment.verification_passed,
            }
            for comment in (self._comments_by_id[comment_id] for comment_id in self._ordered_comment_ids)
        ]

    def group_outputs(self) -> list[dict]:
        """Serialize stored group records."""
        return [
            {
                "group_id": group.group_id,
                "group_name": group.group_name or "Не определено",
            }
            for group in self.all_groups()
        ]

    def search_similar(
        self,
        embedding: list[float],
        top_k: int,
        *,
        filter_metadata: dict[str, str] | None = None,
    ) -> list[SimilarityHit]:
        """Search similar already-processed comments using FAISS scores directly."""
        if not self._vectorstore or not embedding:
            return []

        try:
            hits = self._vectorstore.similarity_search_with_score_by_vector(
                embedding,
                k=top_k,
                filter=filter_metadata,
                fetch_k=max(top_k * 3, top_k),
            )
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            return []

        similarities: list[SimilarityHit] = []
        seen_comment_ids: set[str] = set()
        for document, score in hits:
            comment_id = str(document.metadata.get("comment_id", "")).strip()
            if not comment_id or comment_id in seen_comment_ids:
                continue
            stored = self._comments_by_id.get(comment_id)
            if not stored or not stored.group_id or stored.decision_type == DecisionType.UNDEFINED:
                continue
            seen_comment_ids.add(comment_id)
            similarities.append(
                SimilarityHit(
                    comment_id=comment_id,
                    group_id=stored.group_id,
                    similarity=float(score),
                )
            )
        similarities.sort(key=lambda item: item.similarity, reverse=True)
        return similarities

    def select_group_representatives(
        self,
        group_id: str,
        *,
        limit: int,
        preferred_comment_ids: list[str] | None = None,
    ) -> list[StoredComment]:
        """Choose representative comments, prioritizing retrieved hits when available."""
        unique_comments = self._unique_group_comments(group_id)
        if len(unique_comments) <= limit and not preferred_comment_ids:
            return unique_comments

        comments_by_id = {comment.comment_id: comment for comment in unique_comments}
        selected: list[StoredComment] = []
        seen_comment_ids: set[str] = set()
        seen_normalized: set[str] = set()

        def add_comment(comment: StoredComment | None) -> None:
            if comment is None or len(selected) >= limit:
                return
            normalized_key = normalize_for_match(comment.normalized_text)
            if not normalized_key or comment.comment_id in seen_comment_ids or normalized_key in seen_normalized:
                return
            selected.append(comment)
            seen_comment_ids.add(comment.comment_id)
            seen_normalized.add(normalized_key)

        for comment_id in preferred_comment_ids or []:
            add_comment(comments_by_id.get(comment_id))

        for comment in reversed(self.get_group_comments(group_id)):
            add_comment(comment)

        for comment in unique_comments:
            add_comment(comment)

        return selected

    def unique_group_comments(self, group_id: str) -> list[StoredComment]:
        """Return all unique comments inside a group."""
        return self._unique_group_comments(group_id)

    def _unique_group_comments(self, group_id: str) -> list[StoredComment]:
        """Drop exact normalized duplicates inside one group."""
        unique: list[StoredComment] = []
        seen_normalized: set[str] = set()
        for comment in self.get_group_comments(group_id):
            normalized_key = normalize_for_match(comment.normalized_text)
            if not normalized_key or normalized_key in seen_normalized:
                continue
            seen_normalized.add(normalized_key)
            unique.append(comment)
        return unique

    def _index_comment(self, comment: StoredComment) -> None:
        text_embeddings = [(comment.normalized_text, comment.embedding or [])]
        metadatas = [{"comment_id": comment.comment_id, "group_id": comment.group_id}]
        ids = [comment.comment_id]
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_embeddings(
                text_embeddings,
                self._embeddings,
                metadatas=metadatas,
                ids=ids,
                normalize_L2=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
            return

        self._vectorstore.add_embeddings(
            text_embeddings,
            metadatas=metadatas,
            ids=ids,
        )


class IncrementalMVPClusteringPipeline:
    """Incremental retrieval-plus-LLM clustering pipeline described in the MVP spec."""

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        *,
        retrieval_top_k: int = 12,
        max_candidate_groups: int = 5,
        representatives_per_group: int = 3,
        verification_sample_size: int = 6,
        group_naming_sample_size: int = 12,
        min_meaningful_length: int = 3,
        primary_similarity_threshold: float = 0.92,
        verification_similarity_threshold: float = 0.90,
        max_concurrent_llm_requests: int = 10,
        max_concurrent_embedding_requests: int = 10,
    ):
        self._embeddings = embeddings
        self._llm_semaphore = asyncio.Semaphore(max_concurrent_llm_requests)
        self._embedding_semaphore = asyncio.Semaphore(max_concurrent_embedding_requests)
        self._normalizer = CommentNormalizer(
            llm,
            min_meaningful_length=min_meaningful_length,
            llm_semaphore=self._llm_semaphore,
        )
        self._decision_engine = GroupDecisionEngine(
            llm,
            llm_semaphore=self._llm_semaphore,
        )
        self._name_generator = GroupNameGenerator(
            llm,
            llm_semaphore=self._llm_semaphore,
        )
        self._store = CommentMemoryStore(embeddings)
        self._retrieval_top_k = retrieval_top_k
        self._max_candidate_groups = max_candidate_groups
        self._representatives_per_group = representatives_per_group
        self._verification_sample_size = verification_sample_size
        self._group_naming_sample_size = group_naming_sample_size
        self._primary_similarity_threshold = primary_similarity_threshold
        self._verification_similarity_threshold = verification_similarity_threshold
        self._max_concurrent_llm_requests = max_concurrent_llm_requests
        self._max_concurrent_embedding_requests = max_concurrent_embedding_requests

    def run(self, raw_comments: list[dict]) -> dict[str, list[dict]]:
        return asyncio.run(self.arun(raw_comments))

    async def arun(self, raw_comments: list[dict]) -> dict[str, list[dict]]:
        """Process comments incrementally and return comments plus final groups."""
        comments = self._validate(raw_comments)
        logger.info("Incremental MVP pipeline started: %d comments", len(comments))
        self._print_stage("Подготовка комментариев", 0, len(comments))

        prepared_comments = await asyncio.gather(
            *(self._prepare_comment(comment) for comment in comments)
        )
        self._print_stage("Подготовка комментариев", len(comments), len(comments))

        total_comments = len(prepared_comments)
        progress_step = self._progress_step(total_comments)
        self._print_stage("Инкрементальная кластеризация", 0, total_comments)
        for index, (comment, normalization, embedding) in enumerate(prepared_comments, start=1):
            await self._process_comment(comment, normalization, embedding)
            if index == 1 or index == total_comments or index % progress_step == 0:
                self._print_stage("Инкрементальная кластеризация", index, total_comments)

        self._print_stage("Нейминг групп", 0, len(self._store.all_groups()))
        await self._generate_group_names()
        self._print_stage("Нейминг групп", len(self._store.all_groups()), len(self._store.all_groups()))
        self._print_message("Слияние групп с одинаковыми именами")
        self._store.merge_groups_by_name()
        self._print_message("Пайплайн завершен")

        return {
            "comments": self._store.comment_outputs(),
            "groups": self._store.group_outputs(),
        }

    @staticmethod
    def _validate(raw_comments: list[dict]) -> list[InputComment]:
        comments: list[InputComment] = []
        for index, raw in enumerate(raw_comments, start=1):
            comment_id = str(raw.get("comment_id", "")).strip() or str(index)
            text = str(raw.get("text", "")).strip()
            comments.append(InputComment(comment_id=comment_id, text=text))
        return comments

    async def _prepare_comment(
        self,
        comment: InputComment,
    ) -> tuple[InputComment, NormalizationResult, list[float] | None]:
        normalization = await self._normalizer.anormalize(comment.text)
        embedding = None
        if normalization.is_meaningful:
            embedding = await self._build_embedding(normalization.normalized_text)
        return comment, normalization, embedding

    async def _process_comment(
        self,
        comment: InputComment,
        normalization: NormalizationResult,
        embedding: list[float] | None,
    ) -> None:

        if not normalization.is_meaningful:
            self._store.add_comment(
                StoredComment(
                    comment_id=comment.comment_id,
                    raw_text=comment.text,
                    normalized_text=normalization.normalized_text,
                    embedding=None,
                    group_id="",
                    decision_type=DecisionType.UNDEFINED,
                    decision_reason=normalization.reason,
                    verification_passed=False,
                )
            )
            return

        hits = self._store.search_similar(
            embedding=embedding or [],
            top_k=self._store.indexed_comments_count(),
        )
        candidate_groups = self._build_candidate_groups(hits)
        fallback_primary = self._fallback_primary_decision(
            normalized_text=normalization.normalized_text,
            candidate_groups=candidate_groups,
        )
        primary_decision = await self._decision_engine.achoose_group(
            raw_text=comment.text,
            normalized_text=normalization.normalized_text,
            candidate_groups_text=self._format_candidate_groups(candidate_groups),
            candidate_group_ids={candidate.group_id for candidate in candidate_groups},
            fallback=fallback_primary,
        )

        if primary_decision.decision_type == DecisionType.EXISTING_GROUP and primary_decision.group_id:
            verification = await self._verify_existing_group(
                comment=comment,
                normalized_text=normalization.normalized_text,
                embedding=embedding,
                decision=primary_decision,
            )
            if verification.passed:
                self._store.add_comment(
                    StoredComment(
                        comment_id=comment.comment_id,
                        raw_text=comment.text,
                        normalized_text=normalization.normalized_text,
                        embedding=embedding,
                        group_id=primary_decision.group_id,
                        decision_type=DecisionType.EXISTING_GROUP,
                        decision_reason=verification.reason or primary_decision.reason,
                        verification_passed=True,
                    )
                )
                return

            new_group = self._store.create_group()
            self._store.add_comment(
                StoredComment(
                    comment_id=comment.comment_id,
                    raw_text=comment.text,
                    normalized_text=normalization.normalized_text,
                    embedding=embedding,
                    group_id=new_group.group_id,
                    decision_type=DecisionType.NEW_GROUP,
                    decision_reason=(
                        f"{primary_decision.reason}. "
                        f"Повторная проверка отклонила существующую группу: {verification.reason}"
                    ).strip(),
                    verification_passed=False,
                )
            )
            return

        new_group = self._store.create_group()
        self._store.add_comment(
            StoredComment(
                comment_id=comment.comment_id,
                raw_text=comment.text,
                normalized_text=normalization.normalized_text,
                embedding=embedding,
                group_id=new_group.group_id,
                decision_type=DecisionType.NEW_GROUP,
                decision_reason=primary_decision.reason,
                verification_passed=False,
            )
        )

    async def _build_embedding(self, normalized_text: str) -> list[float] | None:
        try:
            async with self._embedding_semaphore:
                return list(await self._embeddings.aembed_query(normalized_text))
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            return None

    def _build_candidate_groups(self, hits: list[SimilarityHit]) -> list[CandidateGroup]:
        group_scores: dict[str, float] = {}
        group_hit_ids: dict[str, list[str]] = {}

        for hit in hits:
            group_scores[hit.group_id] = max(group_scores.get(hit.group_id, float("-inf")), hit.similarity)
            group_hit_ids.setdefault(hit.group_id, []).append(hit.comment_id)

        ordered_group_ids = sorted(
            group_scores,
            key=lambda group_id: group_scores[group_id],
            reverse=True,
        )

        candidates: list[CandidateGroup] = []
        for group_id in ordered_group_ids:
            representatives = self._store.unique_group_comments(group_id)
            candidates.append(
                CandidateGroup(
                    group_id=group_id,
                    best_similarity=group_scores[group_id],
                    representative_comment_ids=[comment.comment_id for comment in representatives],
                )
            )
        return candidates

    def _fallback_primary_decision(
        self,
        *,
        normalized_text: str,
        candidate_groups: list[CandidateGroup],
    ) -> PrimaryDecision:
        if not candidate_groups:
            return PrimaryDecision(
                decision_type=DecisionType.NEW_GROUP,
                group_id="",
                reason="Подходящая существующая группа не найдена среди уже обработанных комментариев",
            )

        normalized_key = normalize_for_match(normalized_text)
        best_candidate = candidate_groups[0]
        representative_comments = [
            self._store.get_comment(comment_id)
            for comment_id in best_candidate.representative_comment_ids
        ]

        if any(normalize_for_match(comment.normalized_text) == normalized_key for comment in representative_comments):
            return PrimaryDecision(
                decision_type=DecisionType.EXISTING_GROUP,
                group_id=best_candidate.group_id,
                reason="Есть точное совпадение с уже обработанным комментарием этой группы",
            )

        if best_candidate.best_similarity >= self._primary_similarity_threshold:
            return PrimaryDecision(
                decision_type=DecisionType.EXISTING_GROUP,
                group_id=best_candidate.group_id,
                reason="Лучший кандидат достаточно близок по retrieval similarity",
            )

        return PrimaryDecision(
            decision_type=DecisionType.NEW_GROUP,
            group_id="",
            reason="Похожие комментарии есть, но уверенного совпадения с существующей группой нет",
        )

    async def _verify_existing_group(
        self,
        *,
        comment: InputComment,
        normalized_text: str,
        embedding: list[float] | None,
        decision: PrimaryDecision,
    ) -> VerificationDecision:
        group_comments = self._store.get_group_comments(decision.group_id)
        if not group_comments:
            return VerificationDecision(
                passed=False,
                reason="Выбранная группа не содержит сохраненных комментариев для проверки",
            )

        group_hits = self._store.search_similar(
            embedding=embedding or [],
            top_k=self._store.indexed_comments_count(),
            filter_metadata={"group_id": decision.group_id},
        )
        representatives = self._store.unique_group_comments(decision.group_id)
        fallback = self._fallback_verification(
            normalized_text=normalized_text,
            representatives=representatives,
            group_hits=group_hits,
        )
        return await self._decision_engine.averify_group(
            raw_text=comment.text,
            normalized_text=normalized_text,
            group_id=decision.group_id,
            group_examples_text=self._format_group_examples(representatives),
            fallback=fallback,
        )

    def _fallback_verification(
        self,
        *,
        normalized_text: str,
        representatives: list[StoredComment],
        group_hits: list[SimilarityHit],
    ) -> VerificationDecision:
        normalized_key = normalize_for_match(normalized_text)
        if any(normalize_for_match(comment.normalized_text) == normalized_key for comment in representatives):
            return VerificationDecision(
                passed=True,
                reason="В группе есть комментарий с тем же нормализованным смыслом",
            )

        best_similarity = max((hit.similarity for hit in group_hits), default=0.0)
        passed = best_similarity >= self._verification_similarity_threshold
        reason = (
            "Комментарий соответствует проверяемой группе"
            if passed
            else "Комментарий недостаточно похож на примеры проверяемой группы"
        )
        return VerificationDecision(passed=passed, reason=reason)

    async def _generate_group_names(self) -> None:
        groups = self._store.all_groups()
        if not groups:
            return

        completed = 0
        total = len(groups)
        progress_step = self._progress_step(total)
        progress_lock = asyncio.Lock()

        async def assign_group_name(group: CommentGroup) -> None:
            nonlocal completed
            representatives = self._store.unique_group_comments(group.group_id)
            examples_text = self._format_group_examples(representatives)
            fallback_name = self._fallback_group_name(representatives)
            group.group_name = await self._name_generator.agenerate_name(
                examples_text=examples_text,
                fallback_name=fallback_name,
            )
            async with progress_lock:
                completed += 1
                if completed == 1 or completed == total or completed % progress_step == 0:
                    self._print_stage("Нейминг групп", completed, total)

        await asyncio.gather(*(assign_group_name(group) for group in groups))

    @staticmethod
    def _fallback_group_name(representatives: list[StoredComment]) -> str:
        if not representatives:
            return "Не определено"
        first = representatives[0]
        return truncate_text(first.normalized_text or first.raw_text, limit=80) or "Не определено"

    def _format_candidate_groups(self, candidate_groups: list[CandidateGroup]) -> str:
        if not candidate_groups:
            return "Кандидатных групп нет."

        lines: list[str] = []
        for candidate in candidate_groups:
            lines.append(
                f"group_id: {candidate.group_id} | best_similarity: {candidate.best_similarity:.3f}"
            )
            representatives = [
                self._store.get_comment(comment_id)
                for comment_id in candidate.representative_comment_ids
            ]
            for index, representative in enumerate(representatives, start=1):
                lines.append(
                    (
                        f"  example_{index}: raw_text={truncate_text(representative.raw_text)} | "
                        f"normalized_text={truncate_text(representative.normalized_text)}"
                    )
                )
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_group_examples(comments: list[StoredComment]) -> str:
        if not comments:
            return "Примеров нет."
        return "\n".join(
            [
                (
                    f"- comment_id: {comment.comment_id} | "
                    f"raw_text: {truncate_text(comment.raw_text)} | "
                    f"normalized_text: {truncate_text(comment.normalized_text)}"
                )
                for comment in comments
            ]
        )

    @staticmethod
    def _progress_step(total: int) -> int:
        return max(1, total // 10)

    @staticmethod
    def _print_message(message: str) -> None:
        print(f"\r{message}".ljust(80))

    @staticmethod
    def _print_stage(stage: str, current: int, total: int) -> None:
        print(f"\r{stage}: {render_progress_bar(current, total)}".ljust(80))

"""Incremental MVP pipeline: normalize, retrieve, decide, verify, save, and name groups."""

from __future__ import annotations

import logging
import math
import re

from langchain_community.vectorstores import FAISS
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
_NON_INFORMATIVE_TOKENS = {
    "а",
    "ага",
    "ок",
    "окей",
    "угу",
    "мм",
    "эм",
    "хм",
    "спс",
    "спасибо",
    "понял",
    "понятно",
    "ясно",
    "норм",
    "нормально",
    "класс",
    "жесть",
    "мда",
    "лол",
    "test",
    "testing",
}


def technical_normalize_text(value: str) -> str:
    """Apply safe technical normalization without changing the meaning."""
    value = str(value).translate(_QUOTE_MAP)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_for_match(value: str) -> str:
    """Normalize text for comparisons and lightweight heuristics."""
    value = technical_normalize_text(value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я\s-]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def token_overlap(left: str, right: str) -> float:
    """Compute token overlap relative to the smaller token set."""
    left_tokens = set(normalize_for_match(left).split())
    right_tokens = set(normalize_for_match(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    """Compute cosine similarity between two vectors."""
    if not left or not right:
        return 0.0
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


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
    """LLM-based comment normalizer with a local fallback."""

    def __init__(self, llm: BaseChatModel):
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

    def normalize(self, text: str) -> NormalizationResult:
        """Normalize a comment and decide if it is meaningful."""
        try:
            raw = self._chain.invoke({"text": text})
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
                    else "Комментарий пустой, мусорный или бессодержательный"
                )
            return NormalizationResult(
                normalized_text=normalized_text,
                is_meaningful=is_meaningful,
                reason=reason,
            )
        except Exception as exc:
            logger.error("Normalization failed, using fallback: %s", exc)
            return self._fallback(text)

    def _fallback(self, text: str) -> NormalizationResult:
        normalized_text = technical_normalize_text(text)
        is_meaningful = not self._is_clearly_noise(normalized_text)
        reason = (
            "Комментарий содержит осмысленный кейс"
            if is_meaningful
            else "Комментарий пустой, мусорный или бессодержательный"
        )
        return NormalizationResult(
            normalized_text=normalized_text,
            is_meaningful=is_meaningful,
            reason=reason,
        )

    @staticmethod
    def _is_clearly_noise(text: str) -> bool:
        normalized = normalize_for_match(text)
        if not normalized:
            return True
        tokens = normalized.split()
        if not tokens:
            return True
        if len("".join(tokens)) < 3:
            return True
        if len(tokens) <= 2 and all(token in _NON_INFORMATIVE_TOKENS for token in tokens):
            return True
        if re.fullmatch(r"[\W_]+", text or ""):
            return True
        return False


class GroupDecisionEngine:
    """LLM steps for primary routing and verification."""

    def __init__(self, llm: BaseChatModel):
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

    def choose_group(
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
            raw = self._primary_chain.invoke(
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
        """Ask the LLM to verify the chosen existing group."""
        try:
            raw = self._verification_chain.invoke(
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


class GroupNameGenerator:
    """LLM-based group naming with a lightweight fallback."""

    def __init__(self, llm: BaseChatModel):
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

    def generate_name(self, examples_text: str, fallback_name: str) -> str:
        """Generate a short group name."""
        try:
            raw = self._chain.invoke({"group_examples": examples_text})
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

    def merge_groups_by_name(self) -> None:
        """Merge groups that ended up with the same normalized final name."""
        groups = self.all_groups()
        if len(groups) < 2:
            return

        canonical_group_by_name: dict[str, CommentGroup] = {}
        for group in groups:
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

            if (
                len(group.member_comment_ids) > len(canonical_group.member_comment_ids)
                and group.group_name
            ):
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

    def search_similar(self, embedding: list[float], top_k: int) -> list[SimilarityHit]:
        """Search similar already-processed comments."""
        if not self._vectorstore or not embedding:
            return []

        try:
            hits = self._vectorstore.similarity_search_with_score_by_vector(
                embedding,
                k=top_k,
            )
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            return []

        similarities: list[SimilarityHit] = []
        seen_comment_ids: set[str] = set()
        for document, _score in hits:
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
                    similarity=cosine_similarity(embedding, stored.embedding),
                )
            )
        similarities.sort(key=lambda item: item.similarity, reverse=True)
        return similarities

    def select_group_representatives(
        self,
        group_id: str,
        *,
        limit: int,
        query_embedding: list[float] | None = None,
    ) -> list[StoredComment]:
        """Choose representative comments with a balance of relevance and diversity."""
        comments = self._unique_group_comments(group_id)
        if len(comments) <= limit:
            return comments

        if query_embedding:
            comments.sort(
                key=lambda comment: cosine_similarity(query_embedding, comment.embedding),
                reverse=True,
            )

        pool_size = min(len(comments), max(limit * 3, limit))
        pool = comments[:pool_size]
        selected: list[StoredComment] = []
        remaining = pool[:]

        if remaining:
            if query_embedding:
                first = max(
                    remaining,
                    key=lambda comment: cosine_similarity(query_embedding, comment.embedding),
                )
            else:
                first = remaining[0]
            selected.append(first)
            remaining.remove(first)

        while remaining and len(selected) < limit:
            next_comment = max(
                remaining,
                key=lambda comment: self._representative_score(
                    comment=comment,
                    selected=selected,
                    query_embedding=query_embedding,
                ),
            )
            selected.append(next_comment)
            remaining.remove(next_comment)

        return selected

    def _unique_group_comments(self, group_id: str) -> list[StoredComment]:
        """Drop exact normalized duplicates inside one group."""
        unique: list[StoredComment] = []
        seen_normalized: set[str] = set()
        for comment in self.get_group_comments(group_id):
            normalized_key = normalize_for_match(comment.normalized_text)
            if normalized_key in seen_normalized:
                continue
            seen_normalized.add(normalized_key)
            unique.append(comment)
        return unique

    @staticmethod
    def _representative_score(
        *,
        comment: StoredComment,
        selected: list[StoredComment],
        query_embedding: list[float] | None,
    ) -> tuple[float, float, float]:
        query_similarity = cosine_similarity(query_embedding, comment.embedding) if query_embedding else 0.0
        if selected and comment.embedding:
            max_similarity_to_selected = max(
                cosine_similarity(comment.embedding, selected_comment.embedding)
                for selected_comment in selected
            )
        else:
            max_similarity_to_selected = 0.0
        novelty = 1.0 - max_similarity_to_selected
        return novelty, query_similarity, -len(comment.normalized_text)

    def _index_comment(self, comment: StoredComment) -> None:
        text_embeddings = [(comment.normalized_text, comment.embedding or [])]
        metadatas = [{"comment_id": comment.comment_id}]
        ids = [comment.comment_id]
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_embeddings(
                text_embeddings,
                self._embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
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
    ):
        self._embeddings = embeddings
        self._normalizer = CommentNormalizer(llm)
        self._decision_engine = GroupDecisionEngine(llm)
        self._name_generator = GroupNameGenerator(llm)
        self._store = CommentMemoryStore(embeddings)
        self._retrieval_top_k = retrieval_top_k
        self._max_candidate_groups = max_candidate_groups
        self._representatives_per_group = representatives_per_group
        self._verification_sample_size = verification_sample_size
        self._group_naming_sample_size = group_naming_sample_size

    def run(self, raw_comments: list[dict]) -> dict[str, list[dict]]:
        """Process comments incrementally and return comments plus final groups."""
        comments = self._validate(raw_comments)
        logger.info("Incremental MVP pipeline started: %d comments", len(comments))

        for comment in comments:
            self._process_comment(comment)

        self._generate_group_names()
        self._store.merge_groups_by_name()

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

    def _process_comment(self, comment: InputComment) -> None:
        logger.info("Processing comment %s", comment.comment_id)
        normalization = self._normalizer.normalize(comment.text)

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

        embedding = self._build_embedding(normalization.normalized_text)
        hits = self._store.search_similar(
            embedding=embedding or [],
            top_k=self._retrieval_top_k,
        )
        candidate_groups = self._build_candidate_groups(hits, embedding)
        fallback_primary = self._fallback_primary_decision(
            normalized_text=normalization.normalized_text,
            candidate_groups=candidate_groups,
        )
        primary_decision = self._decision_engine.choose_group(
            raw_text=comment.text,
            normalized_text=normalization.normalized_text,
            candidate_groups_text=self._format_candidate_groups(candidate_groups),
            candidate_group_ids={candidate.group_id for candidate in candidate_groups},
            fallback=fallback_primary,
        )

        if primary_decision.decision_type == DecisionType.EXISTING_GROUP and primary_decision.group_id:
            verification = self._verify_existing_group(
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

    def _build_embedding(self, normalized_text: str) -> list[float] | None:
        try:
            return list(self._embeddings.embed_query(normalized_text))
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            return None

    def _build_candidate_groups(
        self,
        hits: list[SimilarityHit],
        embedding: list[float] | None,
    ) -> list[CandidateGroup]:
        group_scores: dict[str, float] = {}
        for hit in hits:
            current_best = group_scores.get(hit.group_id)
            if current_best is None or hit.similarity > current_best:
                group_scores[hit.group_id] = hit.similarity

        ordered_group_ids = sorted(
            group_scores,
            key=lambda group_id: group_scores[group_id],
            reverse=True,
        )[: self._max_candidate_groups]

        candidates: list[CandidateGroup] = []
        for group_id in ordered_group_ids:
            representatives = self._store.select_group_representatives(
                group_id,
                limit=self._representatives_per_group,
                query_embedding=embedding,
            )
            candidates.append(
                CandidateGroup(
                    group_id=group_id,
                    best_similarity=group_scores[group_id],
                    representative_comment_ids=[
                        representative.comment_id for representative in representatives
                    ],
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

        for representative in representative_comments:
            if normalize_for_match(representative.normalized_text) == normalized_key:
                return PrimaryDecision(
                    decision_type=DecisionType.EXISTING_GROUP,
                    group_id=best_candidate.group_id,
                    reason="Есть точное совпадение с уже обработанным комментарием этой группы",
                )

        best_overlap = max(
            (
                token_overlap(normalized_text, representative.normalized_text)
                for representative in representative_comments
            ),
            default=0.0,
        )
        if best_candidate.best_similarity >= 0.94 or (
            best_candidate.best_similarity >= 0.90 and best_overlap >= 0.75
        ):
            return PrimaryDecision(
                decision_type=DecisionType.EXISTING_GROUP,
                group_id=best_candidate.group_id,
                reason="Лучший кандидат достаточно близок по embedding и текстовому совпадению",
            )

        return PrimaryDecision(
            decision_type=DecisionType.NEW_GROUP,
            group_id="",
            reason="Похожие комментарии есть, но уверенного совпадения с существующей группой нет",
        )

    def _verify_existing_group(
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

        sample_size = len(group_comments)
        if len(group_comments) >= 5:
            sample_size = max(5, min(self._verification_sample_size, len(group_comments)))
        representatives = self._store.select_group_representatives(
            decision.group_id,
            limit=sample_size,
            query_embedding=embedding,
        )
        fallback = self._fallback_verification(
            normalized_text=normalized_text,
            embedding=embedding,
            representatives=representatives,
        )
        return self._decision_engine.verify_group(
            raw_text=comment.text,
            normalized_text=normalized_text,
            group_id=decision.group_id,
            group_examples_text=self._format_group_examples(representatives),
            fallback=fallback,
        )

    @staticmethod
    def _fallback_verification(
        *,
        normalized_text: str,
        embedding: list[float] | None,
        representatives: list[StoredComment],
    ) -> VerificationDecision:
        normalized_key = normalize_for_match(normalized_text)
        if any(
            normalize_for_match(representative.normalized_text) == normalized_key
            for representative in representatives
        ):
            return VerificationDecision(
                passed=True,
                reason="В группе есть комментарий с тем же нормализованным смыслом",
            )

        similarities = [
            cosine_similarity(embedding, representative.embedding)
            for representative in representatives
        ]
        max_similarity = max(similarities, default=0.0)
        mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        overlap = max(
            (
                token_overlap(normalized_text, representative.normalized_text)
                for representative in representatives
            ),
            default=0.0,
        )

        passed = max_similarity >= 0.94 or (mean_similarity >= 0.88 and overlap >= 0.70)
        reason = (
            "Комментарий соответствует проверяемой группе"
            if passed
            else "Комментарий недостаточно похож на примеры проверяемой группы"
        )
        return VerificationDecision(passed=passed, reason=reason)

    def _generate_group_names(self) -> None:
        for group in self._store.all_groups():
            representatives = self._store.select_group_representatives(
                group.group_id,
                limit=min(self._group_naming_sample_size, max(1, len(group.member_comment_ids))),
                query_embedding=None,
            )
            examples_text = self._format_group_examples(representatives)
            fallback_name = self._fallback_group_name(representatives)
            group.group_name = self._name_generator.generate_name(
                examples_text=examples_text,
                fallback_name=fallback_name,
            )

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

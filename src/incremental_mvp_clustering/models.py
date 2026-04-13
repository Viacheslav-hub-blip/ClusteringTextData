"""Core models for the incremental MVP clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DecisionType(str, Enum):
    """Final decision type for a processed comment."""

    EXISTING_GROUP = "existing_group"
    NEW_GROUP = "new_group"
    UNDEFINED = "undefined"


@dataclass(slots=True)
class InputComment:
    """Validated input comment."""

    comment_id: str
    text: str


@dataclass(slots=True)
class NormalizationResult:
    """Normalization output for one comment."""

    normalized_text: str
    is_meaningful: bool
    reason: str


@dataclass(slots=True)
class StoredComment:
    """Persisted comment record."""

    comment_id: str
    raw_text: str
    normalized_text: str
    embedding: list[float] | None
    group_id: str
    decision_type: DecisionType
    decision_reason: str
    verification_passed: bool


@dataclass(slots=True)
class CommentGroup:
    """Persisted group record."""

    group_id: str
    group_name: str = ""
    member_comment_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SimilarityHit:
    """Retrieved processed comment from the vector store."""

    comment_id: str
    group_id: str
    similarity: float


@dataclass(slots=True)
class CandidateGroup:
    """Candidate group shown to the primary LLM decision."""

    group_id: str
    best_similarity: float
    representative_comment_ids: list[str]


@dataclass(slots=True)
class PrimaryDecision:
    """Primary decision returned by the first LLM step."""

    decision_type: DecisionType
    group_id: str
    reason: str


@dataclass(slots=True)
class VerificationDecision:
    """Verification result for an existing-group assignment."""

    passed: bool
    reason: str

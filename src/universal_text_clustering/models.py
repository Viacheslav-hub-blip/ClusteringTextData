"""Core models for the universal text clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RelationType(str, Enum):
    """Possible semantic relations between two prototypes."""

    SAME = "SAME"
    A_SPECIFIC_OF_B = "A_SPECIFIC_OF_B"
    B_SPECIFIC_OF_A = "B_SPECIFIC_OF_A"
    DIFFERENT = "DIFFERENT"


@dataclass(slots=True)
class InputComment:
    """Validated input comment."""

    comment_id: str
    text: str


@dataclass(slots=True)
class SemanticFrame:
    """Internal semantic representation extracted by the LLM."""

    comment_id: str
    raw_text: str
    general_topic: str
    exact_case: str
    key_qualifiers: list[str]
    entities: list[str]
    canonical_key: str


@dataclass(slots=True)
class Prototype:
    """Representative of one exact case after early deduplication."""

    prototype_id: str
    canonical_key: str
    representative_frame: SemanticFrame
    member_comment_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidatePair:
    """Candidate pair for semantic relation classification."""

    left_prototype_id: str
    right_prototype_id: str


@dataclass(slots=True)
class PairDecision:
    """LLM decision for a pair of prototypes."""

    left_prototype_id: str
    right_prototype_id: str
    relation: RelationType
    reason: str


@dataclass(slots=True)
class SpecificCluster:
    """Exact-case cluster."""

    specific_cluster_id: str
    prototype_ids: list[str]
    member_comment_ids: list[str]
    representative_prototype_id: str
    specific_group: str = ""
    parent_cluster_id: Optional[str] = None


@dataclass(slots=True)
class ParentCluster:
    """Higher-level parent cluster built over specific clusters."""

    parent_cluster_id: str
    child_specific_cluster_ids: list[str]
    representative_topic: str
    parent_group: str = ""


@dataclass(slots=True)
class OutputRecord:
    """Final output for one comment."""

    comment_id: str
    specific_group: str
    parent_group: str

"""Core models for the LangChain agentic clustering project."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InputComment:
    """One validated input comment."""

    comment_id: str
    text: str


@dataclass(slots=True)
class StructuredSignal:
    """Structured interpretation of a comment in banking context."""

    comment_id: str
    raw_text: str
    is_meaningful: bool
    polarity: str
    banking_area: str
    phenomenon: str
    object_name: str
    parent_focus: str
    parent_key: str
    specific_focus: str
    specific_key: str
    material_details: list[str]
    context_details: list[str]
    entities: list[str]


@dataclass(slots=True)
class SpecificClusterSummary:
    """One specific cluster in the current snapshot."""

    cluster_id: str
    specific_group: str
    specific_key: str
    parent_group: str
    parent_key: str
    polarity: str
    member_comment_ids: list[str]
    sample_texts: list[str]
    signature_text: str
    parent_cluster_id: str = ""


@dataclass(slots=True)
class ParentClusterSummary:
    """One parent cluster in the current snapshot."""

    parent_cluster_id: str
    parent_group: str
    parent_key: str
    child_specific_cluster_ids: list[str]
    member_comment_ids: list[str]
    sample_texts: list[str]
    signature_text: str


@dataclass(slots=True)
class CommentAssignment:
    """Final assignment for one comment."""

    comment_id: str
    specific_cluster_id: str
    specific_group: str
    specific_key: str
    parent_cluster_id: str
    parent_group: str
    parent_key: str


@dataclass(slots=True)
class Snapshot:
    """Whole current clustering state."""

    signals_by_comment_id: dict[str, StructuredSignal]
    assignments_by_comment_id: dict[str, CommentAssignment]
    specific_clusters_by_id: dict[str, SpecificClusterSummary]
    parent_clusters_by_id: dict[str, ParentClusterSummary]


@dataclass(slots=True)
class Neighborhood:
    """A small local area that may need repair."""

    neighborhood_id: str
    cluster_ids: list[str]
    reason: str
    score: float
    sample_comment_ids: list[str] = field(default_factory=list)
    parent_cluster_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CritiqueIssue:
    """LLM critique for one neighborhood."""

    neighborhood_id: str
    has_issue: bool
    issue_type: str
    severity: int
    summary: str
    guidance: str
    affected_cluster_ids: list[str]
    affected_parent_ids: list[str]
    confidence: float


@dataclass(slots=True)
class PatchReview:
    """Decision about whether a candidate patch is better."""

    patch_id: str
    accept: bool
    confidence: float
    summary: str
    objective_before: float
    objective_after: float


@dataclass(slots=True)
class CandidatePatch:
    """Localized reclustering candidate."""

    patch_id: str
    neighborhood_id: str
    affected_comment_ids: list[str]
    proposed_signals_by_comment_id: dict[str, StructuredSignal]
    candidate_snapshot: Snapshot
    before_summary: dict
    after_summary: dict
    changed_comment_ids: list[str]
    guidance: str
    review: PatchReview | None = None


@dataclass(slots=True)
class OutputRecord:
    """Output row for the final JSON response."""

    comment_id: str
    specific_group: str
    parent_group: str

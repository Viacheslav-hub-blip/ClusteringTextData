"""Core models for the agentic banking clustering project."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InputComment:
    """Validated input comment."""

    comment_id: str
    text: str


@dataclass(slots=True)
class StructuredSignal:
    """Structured banking-aware signal extracted from a comment."""

    comment_id: str
    raw_text: str
    is_meaningful: bool
    polarity: str
    bank_area: str
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
class CommentAssignment:
    """Final assignment for one comment after a worker pass."""

    comment_id: str
    text: str
    polarity: str
    bank_area: str
    phenomenon: str
    object_name: str
    parent_focus: str
    parent_key: str
    specific_focus: str
    specific_key: str
    specific_group: str
    parent_group: str
    material_details: list[str]
    context_details: list[str]
    entities: list[str]


@dataclass(slots=True)
class SpecificClusterSummary:
    """Summary of one specific cluster in the current snapshot."""

    cluster_id: str
    specific_group: str
    parent_group: str
    parent_key: str
    specific_key: str
    bank_area: str
    polarity: str
    size: int
    member_comment_ids: list[str]
    representative_texts: list[str]
    summary_text: str


@dataclass(slots=True)
class ParentClusterSummary:
    """Summary of one parent cluster in the current snapshot."""

    parent_cluster_id: str
    parent_group: str
    size: int
    child_cluster_ids: list[str]
    representative_texts: list[str]
    summary_text: str


@dataclass(slots=True)
class Snapshot:
    """Current full clustering snapshot."""

    assignments_by_id: dict[str, CommentAssignment]
    specific_clusters_by_id: dict[str, SpecificClusterSummary]
    parent_clusters_by_id: dict[str, ParentClusterSummary]


@dataclass(slots=True)
class NeighborhoodCandidate:
    """Suspicious local region that may need repair."""

    neighborhood_id: str
    cluster_ids: list[str]
    rationale: str
    summary: str


@dataclass(slots=True)
class PatchProposal:
    """Candidate local patch produced by re-clustering a neighborhood."""

    patch_id: str
    neighborhood_id: str
    cluster_ids: list[str]
    comment_ids: list[str]
    updated_assignments_by_id: dict[str, CommentAssignment]
    diff_summary: str
    evaluation_summary: str = ""
    accepted: bool = False

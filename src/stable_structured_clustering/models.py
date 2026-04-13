"""Core models for the stable structured clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InputComment:
    """Validated input comment."""

    comment_id: str
    text: str


@dataclass(slots=True)
class StructuredSignal:
    """Structured interpretation of a comment."""

    comment_id: str
    raw_text: str
    is_meaningful: bool
    polarity: str
    phenomenon: str
    subject: str
    parent_focus: str
    parent_key: str
    specific_focus: str
    specific_key: str
    material_details: list[str]
    context_details: list[str]
    entities: list[str]


@dataclass(slots=True)
class SpecificPrototype:
    """Prototype created from an exact specific key match."""

    prototype_id: str
    representative_signal: StructuredSignal
    member_comment_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SpecificCluster:
    """Stable specific cluster."""

    specific_cluster_id: str
    prototype_ids: list[str]
    member_comment_ids: list[str]
    representative_prototype_id: str
    specific_group: str = ""
    parent_cluster_id: str = ""


@dataclass(slots=True)
class ParentCluster:
    """Stable parent cluster."""

    parent_cluster_id: str
    child_specific_cluster_ids: list[str]
    member_comment_ids: list[str]
    representative_specific_cluster_id: str
    parent_group: str = ""


@dataclass(slots=True)
class OutputRecord:
    """Final output for one comment."""

    comment_id: str
    specific_group: str
    parent_group: str

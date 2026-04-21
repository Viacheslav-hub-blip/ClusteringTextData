"""Structured models for the agentic post-processing pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


SupervisorStep = Literal["route_unassigned", "resolve_singletons", "audit_group", "finalize"]
PlannerStep = Literal["resolve_singletons", "audit_group", "route_unassigned", "finish"]
SingletonActionType = Literal["move_to_group", "keep_current_group"]
UnassignedActionType = Literal["move_to_group", "create_new_group"]


class PlannerDecision(BaseModel):
    """Legacy planner choice for compatibility with old prompt code."""

    next_step: PlannerStep = Field(description="Next post-processing step.")
    reason: str = Field(description="Short explanation for the selected step.")


class SingletonResolutionDecision(BaseModel):
    """Decision for one singleton cluster."""

    action: SingletonActionType = Field(description="Move the singleton or keep it as a valid separate cluster.")
    target_group_id: str = Field(
        default="",
        description="Existing group_id when action=move_to_group.",
    )
    reason: str = Field(description="Short explanation of the decision.")


class ClusterAuditDecision(BaseModel):
    """Audit result for one cluster."""

    remove_comment_ids: list[str] = Field(
        default_factory=list,
        description="comment_id values that clearly do not belong to this cluster.",
    )
    reason: str = Field(description="Short explanation of the audit result.")


class UnassignedRoutingDecision(BaseModel):
    """Decision for one temporarily unassigned comment."""

    action: UnassignedActionType = Field(description="Move to an existing cluster or create a new cluster.")
    target_group_id: str = Field(
        default="",
        description="Existing group_id when action=move_to_group.",
    )
    reason: str = Field(description="Short explanation of the decision.")


class PostProcessingGroupName(BaseModel):
    """Generated group name after post-processing."""

    group_name: str = Field(description="Short human-readable final cluster name.")


class AgenticPostProcessingState(TypedDict, total=False):
    """LangGraph state for the stateless post-processing workflow."""

    comments_by_id: dict[str, dict[str, Any]]
    groups_by_id: dict[str, dict[str, Any]]
    comment_order: list[str]
    singleton_queue: list[str]
    audit_queue: list[str]
    unassigned_queue: list[str]
    accepted_singleton_group_ids: list[str]
    audit_attempts_by_group_id: dict[str, int]
    next_group_index: int
    next_step: SupervisorStep
    round_index: int
    no_change_rounds: int
    last_patch_summary: dict[str, Any]
    finish_reason: str
    final_result: dict[str, Any]

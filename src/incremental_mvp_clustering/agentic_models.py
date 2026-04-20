"""Structured models for the agentic post-processing pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


PlannerStep = Literal["resolve_singletons", "audit_group", "route_unassigned", "finish"]
SingletonActionType = Literal["move_to_group", "keep_current_group"]
UnassignedActionType = Literal["move_to_group", "create_new_group"]


class PlannerDecision(BaseModel):
    """Planner choice for the next graph step."""

    next_step: PlannerStep = Field(description="Следующий шаг post-processing pipeline.")
    reason: str = Field(description="Краткое объяснение, почему выбран именно этот шаг.")


class SingletonResolutionDecision(BaseModel):
    """Decision for one singleton cluster."""

    action: SingletonActionType = Field(description="Переместить комментарий или оставить singleton-кластер как есть.")
    target_group_id: str = Field(
        default="",
        description="Точный group_id существующего кластера, если выбран action=move_to_group.",
    )
    reason: str = Field(description="Краткое объяснение решения.")


class ClusterAuditDecision(BaseModel):
    """Audit result for one cluster."""

    remove_comment_ids: list[str] = Field(
        default_factory=list,
        description="Список comment_id, которые не должны находиться в этом кластере.",
    )
    reason: str = Field(description="Краткое объяснение результата аудита.")


class UnassignedRoutingDecision(BaseModel):
    """Decision for one temporarily unassigned comment."""

    action: UnassignedActionType = Field(
        description="Либо переместить в существующий кластер, либо создать новый кластер.",
    )
    target_group_id: str = Field(
        default="",
        description="Точный group_id существующего кластера, если выбран action=move_to_group.",
    )
    reason: str = Field(description="Краткое объяснение решения.")


class PostProcessingGroupName(BaseModel):
    """Generated group name after post-processing."""

    group_name: str = Field(description="Короткое человекочитаемое название итогового кластера.")


class AgenticPostProcessingState(TypedDict, total=False):
    """LangGraph state for the post-processing workflow."""

    comments_by_id: dict[str, dict[str, Any]]
    groups_by_id: dict[str, dict[str, Any]]
    comment_order: list[str]
    pending_audit_group_ids: list[str]
    planner_decision: dict[str, Any]
    next_group_index: int
    no_change_rounds: int
    final_result: dict[str, Any]

"""Agentic post-processing pipeline over primary clustering results."""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from .agentic_models import (
    AgenticPostProcessingState,
    ClusterAuditDecision,
    PlannerDecision,
    PostProcessingGroupName,
    SingletonResolutionDecision,
    UnassignedRoutingDecision,
)
from .agentic_prompts import (
    CLUSTER_AUDIT_HUMAN,
    CLUSTER_AUDIT_SYSTEM,
    POST_REVIEW_HUMAN,
    POST_REVIEW_SYSTEM,
    SINGLETON_RESOLUTION_HUMAN,
    SINGLETON_RESOLUTION_SYSTEM,
    UNASSIGNED_ROUTING_HUMAN,
    UNASSIGNED_ROUTING_SYSTEM,
)
from .prompts import GROUP_NAMING_HUMAN, GROUP_NAMING_SYSTEM

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
_GROUP_ID_RE = re.compile(r"group_(\d+)$")


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).translate(_QUOTE_MAP)).strip()


def _normalize_key(value: object) -> str:
    value = _clean_text(value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я\s-]+", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _truncate_text(value: object, limit: int = 140) -> str:
    value = _clean_text(value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


class AgenticPostProcessingPipeline:
    """LangGraph reconciliation layer over the primary clustering output."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        max_examples_per_candidate_group: int = 3,
        planner_preview_group_limit: int = 20,
        audit_batch_size: int = 10,
        max_concurrent_llm_requests: int = 10,
    ):
        self._max_examples_per_candidate_group = max_examples_per_candidate_group
        self._planner_preview_group_limit = planner_preview_group_limit
        self._audit_batch_size = max(1, audit_batch_size)
        self._llm_semaphore = asyncio.Semaphore(max_concurrent_llm_requests)

        self._planner_chain = self._build_chain(POST_REVIEW_SYSTEM, POST_REVIEW_HUMAN, PlannerDecision, llm)
        self._singleton_chain = self._build_chain(
            SINGLETON_RESOLUTION_SYSTEM,
            SINGLETON_RESOLUTION_HUMAN,
            SingletonResolutionDecision,
            llm,
        )
        self._audit_chain = self._build_chain(CLUSTER_AUDIT_SYSTEM, CLUSTER_AUDIT_HUMAN, ClusterAuditDecision, llm)
        self._unassigned_chain = self._build_chain(
            UNASSIGNED_ROUTING_SYSTEM,
            UNASSIGNED_ROUTING_HUMAN,
            UnassignedRoutingDecision,
            llm,
        )
        self._naming_chain = self._build_chain(
            GROUP_NAMING_SYSTEM,
            GROUP_NAMING_HUMAN,
            PostProcessingGroupName,
            llm,
        )
        self._graph = self._build_graph()

    def run(self, primary_result: dict[str, list[dict]]) -> dict[str, Any]:
        return asyncio.run(self.arun(primary_result))

    async def arun(self, primary_result: dict[str, list[dict]]) -> dict[str, Any]:
        state = self._build_initial_state(primary_result)
        logger.info(
            "Agentic post-processing started: %d comments, %d groups",
            len(state["comments_by_id"]),
            len(state["groups_by_id"]),
        )
        final_state = await self._graph.ainvoke(state)
        logger.info(
            "Agentic post-processing finished: %d final groups",
            len(final_state.get("final_result", {}).get("groups", [])),
        )
        return final_state["final_result"]

    @staticmethod
    def _build_chain(system_prompt: str, human_prompt: str, schema: type, llm: BaseChatModel) -> Any:
        parser = PydanticOutputParser(pydantic_object=schema)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        return prompt | llm | parser

    def _build_graph(self) -> Any:
        graph = StateGraph(AgenticPostProcessingState)
        graph.add_node("review", self._review_node)
        graph.add_node("resolve_singletons", self._resolve_singletons_node)
        graph.add_node("audit_group", self._audit_groups_batch_node)
        graph.add_node("route_unassigned", self._route_unassigned_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "review")
        graph.add_conditional_edges(
            "review",
            self._route_from_review,
            {
                "resolve_singletons": "resolve_singletons",
                "audit_group": "audit_group",
                "route_unassigned": "route_unassigned",
                "finalize": "finalize",
            },
        )
        graph.add_edge("resolve_singletons", "review")
        graph.add_edge("audit_group", "review")
        graph.add_edge("route_unassigned", "review")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _build_initial_state(self, primary_result: dict[str, list[dict]]) -> AgenticPostProcessingState:
        comments_by_id: dict[str, dict[str, Any]] = {}
        comment_order: list[str] = []
        for index, raw_comment in enumerate(primary_result.get("comments", []), start=1):
            comment = copy.deepcopy(raw_comment)
            comment_id = str(comment.get("comment_id", "")).strip() or str(index)
            comment.update(
                {
                    "comment_id": comment_id,
                    "raw_text": _clean_text(comment.get("raw_text", "")),
                    "normalized_text": _clean_text(comment.get("normalized_text", "")),
                    "group_id": str(comment.get("group_id", "")).strip(),
                    "initial_group_id": str(comment.get("group_id", "")).strip(),
                    "postprocessing_trace": [],
                }
            )
            comments_by_id[comment_id] = comment
            comment_order.append(comment_id)

        groups_by_id = self._build_groups_by_id(primary_result.get("groups", []), comments_by_id, comment_order)
        return {
            "comments_by_id": comments_by_id,
            "groups_by_id": groups_by_id,
            "comment_order": comment_order,
            "pending_audit_group_ids": self._initial_pending_audit_groups(groups_by_id),
            "planner_decision": {},
            "next_group_index": self._next_group_index(groups_by_id),
            "no_change_rounds": 0,
        }

    @staticmethod
    def _build_groups_by_id(
        raw_groups: list[dict],
        comments_by_id: dict[str, dict[str, Any]],
        comment_order: list[str],
    ) -> dict[str, dict[str, Any]]:
        groups_by_id: dict[str, dict[str, Any]] = {}
        for raw_group in raw_groups:
            group_id = str(raw_group.get("group_id", "")).strip()
            if group_id:
                groups_by_id[group_id] = {
                    "group_id": group_id,
                    "group_name": _clean_text(raw_group.get("group_name", "")),
                    "member_comment_ids": [],
                }

        for comment_id in comment_order:
            group_id = comments_by_id[comment_id]["group_id"]
            if not group_id:
                continue
            groups_by_id.setdefault(group_id, {"group_id": group_id, "group_name": "", "member_comment_ids": []})
            groups_by_id[group_id]["member_comment_ids"].append(comment_id)

        return {
            group_id: {**group, "member_comment_ids": list(dict.fromkeys(group["member_comment_ids"]))}
            for group_id, group in groups_by_id.items()
            if group["member_comment_ids"]
        }

    async def _review_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        fallback = self._planner_fallback(state)
        if fallback.next_step == "finish" and not self._needs_llm_review(state):
            decision = fallback
        else:
            decision = await self._ainvoke_chain(
                self._planner_chain,
                {"state_summary": self._build_state_summary(state)},
                fallback=fallback,
            )

        planner_decision = decision.model_dump()
        logger.info(
            "Agentic review selected step=%s, singletons=%d, unassigned=%d, audit_queue=%d",
            planner_decision["next_step"],
            len(self._singleton_group_ids(state)),
            len(self._unassigned_comment_ids(state)),
            len(self._pending_audit_group_ids(state)),
        )
        return {"planner_decision": planner_decision}

    def _route_from_review(self, state: AgenticPostProcessingState) -> str:
        next_step = str(state.get("planner_decision", {}).get("next_step", "finish")).strip()
        if self._unassigned_comment_ids(state) and next_step in {"route_unassigned", "finish"}:
            return "route_unassigned"
        if next_step == "resolve_singletons" and self._singleton_group_ids(state):
            return "resolve_singletons"
        if next_step == "audit_group" and self._pending_audit_group_ids(state):
            return "audit_group"
        if next_step == "route_unassigned" and self._unassigned_comment_ids(state):
            return "route_unassigned"
        return "finalize"

    async def _resolve_singletons_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        pending_audit_group_ids = list(state.get("pending_audit_group_ids", []))
        change_count = 0
        singleton_group_ids = self._singleton_group_ids({"groups_by_id": groups_by_id})
        logger.info("Agentic step resolve_singletons started: %d singleton clusters", len(singleton_group_ids))

        for group_id in singleton_group_ids:
            group = groups_by_id.get(group_id)
            if not group or len(group.get("member_comment_ids", [])) != 1:
                continue
            comment_id = group["member_comment_ids"][0]
            comment = comments_by_id.get(comment_id)
            if not comment:
                continue

            decision = await self._decide_singleton(
                comment=comment,
                group=group,
                candidate_groups=self._build_all_cluster_candidates(
                    comments_by_id=comments_by_id,
                    groups_by_id=groups_by_id,
                    exclude_group_ids={group_id},
                ),
            )
            target_group_id = decision.target_group_id.strip()
            if decision.action != "move_to_group" or target_group_id not in groups_by_id:
                continue

            if self._move_comment(comments_by_id, groups_by_id, comment_id, target_group_id, decision.reason):
                change_count += 1
                logger.info("Singleton comment %s moved: %s -> %s", comment_id, group_id, target_group_id)
                pending_audit_group_ids = self._mark_group_for_audit(
                    self._remove_group_from_audit_queue(pending_audit_group_ids, group_id),
                    groups_by_id,
                    target_group_id,
                )

        return self._build_action_update(
            state,
            change_count,
            comments_by_id,
            groups_by_id,
            pending_audit_group_ids,
        )

    async def _audit_groups_batch_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        pending_audit_group_ids = list(state.get("pending_audit_group_ids", []))
        target_group_ids = self._pending_audit_group_ids(
            {"groups_by_id": groups_by_id, "pending_audit_group_ids": pending_audit_group_ids}
        )[: self._audit_batch_size]
        logger.info(
            "Agentic step audit_group started: auditing %d/%d pending clusters",
            len(target_group_ids),
            len(self._pending_audit_group_ids({"groups_by_id": groups_by_id, "pending_audit_group_ids": pending_audit_group_ids})),
        )

        if not target_group_ids:
            return self._build_action_update(state, 0, comments_by_id, groups_by_id, pending_audit_group_ids)

        audit_results = await asyncio.gather(
            *(self._audit_one_group(group_id, groups_by_id, comments_by_id) for group_id in target_group_ids)
        )

        change_count = 0
        for group_id, decision in audit_results:
            group = groups_by_id.get(group_id)
            if not group:
                pending_audit_group_ids = self._remove_group_from_audit_queue(pending_audit_group_ids, group_id)
                continue

            removable_ids = [
                comment_id
                for comment_id in decision.remove_comment_ids
                if comment_id in group.get("member_comment_ids", [])
            ]
            for comment_id in removable_ids:
                self._unassign_comment(comments_by_id, groups_by_id, comment_id, decision.reason)
                change_count += 1
            if removable_ids:
                logger.info("Audit cluster %s removed %d comments", group_id, len(removable_ids))

            if removable_ids and group_id in groups_by_id and len(groups_by_id[group_id]["member_comment_ids"]) > 1:
                pending_audit_group_ids = self._mark_group_for_audit(pending_audit_group_ids, groups_by_id, group_id)
            else:
                pending_audit_group_ids = self._remove_group_from_audit_queue(pending_audit_group_ids, group_id)

        return self._build_action_update(
            state,
            change_count,
            comments_by_id,
            groups_by_id,
            pending_audit_group_ids,
        )

    async def _audit_one_group(
        self,
        group_id: str,
        groups_by_id: dict[str, dict[str, Any]],
        comments_by_id: dict[str, dict[str, Any]],
    ) -> tuple[str, ClusterAuditDecision]:
        decision = await self._ainvoke_chain(
            self._audit_chain,
            {"group_card": self._format_group_card(groups_by_id[group_id], comments_by_id, include_all_members=True)},
            fallback=ClusterAuditDecision(remove_comment_ids=[], reason="Cluster looks consistent"),
        )
        return group_id, decision

    async def _route_unassigned_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        pending_audit_group_ids = list(state.get("pending_audit_group_ids", []))
        next_group_index = int(state.get("next_group_index", 1))
        change_count = 0
        unassigned_comment_ids = self._unassigned_comment_ids({"comments_by_id": comments_by_id})
        logger.info("Agentic step route_unassigned started: %d comments", len(unassigned_comment_ids))

        for comment_id in unassigned_comment_ids:
            comment = comments_by_id[comment_id]
            decision = await self._decide_unassigned(
                comment=comment,
                candidate_groups=self._build_all_cluster_candidates(
                    comments_by_id=comments_by_id,
                    groups_by_id=groups_by_id,
                    exclude_group_ids=set(),
                ),
            )
            target_group_id = decision.target_group_id.strip()
            if decision.action != "move_to_group" or target_group_id not in groups_by_id:
                target_group_id, next_group_index = self._create_group(groups_by_id, next_group_index)

            self._assign_comment_to_group(comments_by_id, groups_by_id, comment_id, target_group_id, decision.reason)
            logger.info("Unassigned comment %s routed to %s", comment_id, target_group_id)
            pending_audit_group_ids = self._mark_group_for_audit(pending_audit_group_ids, groups_by_id, target_group_id)
            change_count += 1

        return self._build_action_update(
            state,
            change_count,
            comments_by_id,
            groups_by_id,
            pending_audit_group_ids,
            next_group_index=next_group_index,
        )

    async def _finalize_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        logger.info("Agentic step finalize started: naming %d groups", len(groups_by_id))
        await self._rename_groups(groups_by_id, comments_by_id)
        self._merge_groups_by_name(groups_by_id, comments_by_id)
        logger.info("Agentic step finalize finished: %d groups after merge", len(groups_by_id))
        return {"final_result": self._build_final_result(state, comments_by_id, groups_by_id)}

    async def _decide_singleton(
        self,
        *,
        comment: dict[str, Any],
        group: dict[str, Any],
        candidate_groups: list[dict[str, Any]],
    ) -> SingletonResolutionDecision:
        if not candidate_groups:
            return SingletonResolutionDecision(
                action="keep_current_group",
                target_group_id="",
                reason="No existing candidate clusters",
            )
        return await self._ainvoke_chain(
            self._singleton_chain,
            {
                "singleton_cluster": self._format_group_card(
                    group,
                    {comment["comment_id"]: comment},
                    include_all_members=True,
                ),
                "candidate_groups": self._format_candidate_groups(candidate_groups),
            },
            fallback=SingletonResolutionDecision(
                action="keep_current_group",
                target_group_id="",
                reason="No confident existing cluster match",
            ),
        )

    async def _decide_unassigned(
        self,
        *,
        comment: dict[str, Any],
        candidate_groups: list[dict[str, Any]],
    ) -> UnassignedRoutingDecision:
        if not candidate_groups:
            return UnassignedRoutingDecision(
                action="create_new_group",
                target_group_id="",
                reason="No existing candidate clusters",
            )
        return await self._ainvoke_chain(
            self._unassigned_chain,
            {
                "comment_card": self._format_comment_card(comment),
                "candidate_groups": self._format_candidate_groups(candidate_groups),
            },
            fallback=UnassignedRoutingDecision(
                action="create_new_group",
                target_group_id="",
                reason="No confident existing cluster match",
            ),
        )

    async def _rename_groups(
        self,
        groups_by_id: dict[str, dict[str, Any]],
        comments_by_id: dict[str, dict[str, Any]],
    ) -> None:
        async def rename_group(group_id: str, group: dict[str, Any]) -> tuple[str, str]:
            members = self._unique_group_comments(group, comments_by_id)
            fallback_name = self._fallback_group_name(members)
            decision = await self._ainvoke_chain(
                self._naming_chain,
                {"group_examples": self._format_group_examples(members)},
                fallback=PostProcessingGroupName(group_name=fallback_name),
            )
            return group_id, _clean_text(decision.group_name) or fallback_name

        tasks = [
            rename_group(group_id, group)
            for group_id, group in groups_by_id.items()
            if group.get("member_comment_ids")
        ]
        for group_id, group_name in await asyncio.gather(*tasks) if tasks else []:
            if group_id in groups_by_id:
                groups_by_id[group_id]["group_name"] = group_name

    async def _ainvoke_chain(self, chain: Any, payload: dict[str, Any], *, fallback: Any) -> Any:
        try:
            async with self._llm_semaphore:
                return await chain.ainvoke(payload)
        except Exception as exc:
            logger.error("Agentic post-processing chain failed, using fallback: %s", exc)
            return fallback

    @staticmethod
    def _copy_cluster_state(
        state: AgenticPostProcessingState,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        return copy.deepcopy(state["comments_by_id"]), copy.deepcopy(state["groups_by_id"])

    def _build_state_summary(self, state: AgenticPostProcessingState) -> str:
        singleton_group_ids = self._singleton_group_ids(state)
        unassigned_comment_ids = self._unassigned_comment_ids(state)
        pending_audit_group_ids = self._pending_audit_group_ids(state)
        return "\n".join(
            [
                f"Total comments: {len(state['comments_by_id'])}",
                f"Non-empty clusters: {len(state['groups_by_id'])}",
                f"Singleton clusters: {len(singleton_group_ids)}",
                f"Unassigned comments: {len(unassigned_comment_ids)}",
                f"Clusters pending audit: {len(pending_audit_group_ids)}",
                f"No-change rounds: {state.get('no_change_rounds', 0)}",
                "",
                "Singleton cluster examples:",
                self._preview_groups(state, singleton_group_ids),
                "",
                "Clusters pending audit:",
                self._preview_groups(state, pending_audit_group_ids),
                "",
                "Unassigned comments:",
                self._preview_unassigned(state, unassigned_comment_ids),
            ]
        )

    def _preview_groups(self, state: AgenticPostProcessingState, group_ids: list[str]) -> str:
        if not group_ids:
            return "- none"
        previews: list[str] = []
        for group_id in group_ids[: self._planner_preview_group_limit]:
            group = state["groups_by_id"].get(group_id)
            if not group:
                continue
            member_ids = group.get("member_comment_ids", [])
            sample = ""
            if member_ids:
                sample_comment = state["comments_by_id"][member_ids[0]]
                sample = _truncate_text(sample_comment.get("normalized_text") or sample_comment.get("raw_text") or "")
            previews.append(f"- {group_id} | {group.get('group_name') or 'Not named'} | size={len(member_ids)} | {sample}")
        return "\n".join(previews) or "- none"

    def _preview_unassigned(self, state: AgenticPostProcessingState, comment_ids: list[str]) -> str:
        if not comment_ids:
            return "- none"
        return "\n".join(
            f"- {comment_id} | {_truncate_text(state['comments_by_id'][comment_id].get('normalized_text') or state['comments_by_id'][comment_id].get('raw_text') or '')}"
            for comment_id in comment_ids[: self._planner_preview_group_limit]
        )

    def _planner_fallback(self, state: AgenticPostProcessingState) -> PlannerDecision:
        if self._unassigned_comment_ids(state):
            return PlannerDecision(next_step="route_unassigned", reason="There are unassigned comments")
        if self._singleton_group_ids(state) and state.get("no_change_rounds", 0) == 0:
            return PlannerDecision(next_step="resolve_singletons", reason="There are singleton clusters to check")
        if self._pending_audit_group_ids(state):
            return PlannerDecision(next_step="audit_group", reason="There are clusters pending audit")
        return PlannerDecision(next_step="finish", reason="No obvious remaining clustering issues")

    def _needs_llm_review(self, state: AgenticPostProcessingState) -> bool:
        return bool(
            self._unassigned_comment_ids(state)
            or self._singleton_group_ids(state)
            or self._pending_audit_group_ids(state)
        )

    def _build_action_update(
        self,
        state: AgenticPostProcessingState,
        change_count: int,
        comments_by_id: dict[str, dict[str, Any]],
        groups_by_id: dict[str, dict[str, Any]],
        pending_audit_group_ids: list[str],
        *,
        next_group_index: int | None = None,
    ) -> dict[str, Any]:
        logger.info("Agentic action finished: %d changes", change_count)
        return {
            "comments_by_id": comments_by_id,
            "groups_by_id": groups_by_id,
            "pending_audit_group_ids": self._pending_audit_group_ids(
                {"groups_by_id": groups_by_id, "pending_audit_group_ids": pending_audit_group_ids}
            ),
            "no_change_rounds": int(state.get("no_change_rounds", 0)) + 1 if change_count == 0 else 0,
            "next_group_index": next_group_index if next_group_index is not None else int(state.get("next_group_index", 1)),
        }

    def _move_comment(
        self,
        comments_by_id: dict[str, dict[str, Any]],
        groups_by_id: dict[str, dict[str, Any]],
        comment_id: str,
        target_group_id: str,
        reason: str,
    ) -> bool:
        comment = comments_by_id.get(comment_id)
        if not comment:
            return False
        source_group_id = str(comment.get("group_id", "")).strip()
        if not source_group_id or source_group_id == target_group_id or target_group_id not in groups_by_id:
            return False
        self._remove_comment_from_group(groups_by_id, source_group_id, comment_id)
        self._assign_comment_to_group(comments_by_id, groups_by_id, comment_id, target_group_id, reason)
        comment.setdefault("postprocessing_trace", []).append(f"{source_group_id} -> {target_group_id}: {reason}")
        if source_group_id in groups_by_id and not groups_by_id[source_group_id]["member_comment_ids"]:
            groups_by_id.pop(source_group_id, None)
        return True

    def _unassign_comment(
        self,
        comments_by_id: dict[str, dict[str, Any]],
        groups_by_id: dict[str, dict[str, Any]],
        comment_id: str,
        reason: str,
    ) -> None:
        comment = comments_by_id.get(comment_id)
        if not comment:
            return
        source_group_id = str(comment.get("group_id", "")).strip()
        if source_group_id:
            self._remove_comment_from_group(groups_by_id, source_group_id, comment_id)
            if source_group_id in groups_by_id and not groups_by_id[source_group_id]["member_comment_ids"]:
                groups_by_id.pop(source_group_id, None)
        comment["group_id"] = ""
        comment["postprocessing_reason"] = reason
        comment.setdefault("postprocessing_trace", []).append(f"{source_group_id or 'unassigned'} -> unassigned: {reason}")

    @staticmethod
    def _assign_comment_to_group(
        comments_by_id: dict[str, dict[str, Any]],
        groups_by_id: dict[str, dict[str, Any]],
        comment_id: str,
        target_group_id: str,
        reason: str,
    ) -> None:
        comment = comments_by_id[comment_id]
        groups_by_id.setdefault(target_group_id, {"group_id": target_group_id, "group_name": "", "member_comment_ids": []})
        groups_by_id[target_group_id]["member_comment_ids"].append(comment_id)
        groups_by_id[target_group_id]["member_comment_ids"] = list(dict.fromkeys(groups_by_id[target_group_id]["member_comment_ids"]))
        previous_group_id = comment.get("group_id") or "unassigned"
        comment["group_id"] = target_group_id
        comment["postprocessing_reason"] = reason
        comment.setdefault("postprocessing_trace", []).append(f"{previous_group_id} -> {target_group_id}: {reason}")

    @staticmethod
    def _create_group(groups_by_id: dict[str, dict[str, Any]], next_group_index: int) -> tuple[str, int]:
        group_id = f"group_{next_group_index:04d}"
        groups_by_id[group_id] = {"group_id": group_id, "group_name": "", "member_comment_ids": []}
        return group_id, next_group_index + 1

    @staticmethod
    def _remove_comment_from_group(groups_by_id: dict[str, dict[str, Any]], group_id: str, comment_id: str) -> None:
        group = groups_by_id.get(group_id)
        if group:
            group["member_comment_ids"] = [
                existing_comment_id
                for existing_comment_id in group.get("member_comment_ids", [])
                if existing_comment_id != comment_id
            ]

    def _mark_group_for_audit(
        self,
        pending_audit_group_ids: list[str],
        groups_by_id: dict[str, dict[str, Any]],
        group_id: str,
    ) -> list[str]:
        if group_id not in groups_by_id or len(groups_by_id[group_id].get("member_comment_ids", [])) <= 1:
            return self._remove_group_from_audit_queue(pending_audit_group_ids, group_id)
        queue = [existing_group_id for existing_group_id in pending_audit_group_ids if existing_group_id != group_id]
        return [group_id, *queue]

    @staticmethod
    def _remove_group_from_audit_queue(pending_audit_group_ids: list[str], group_id: str) -> list[str]:
        return [existing_group_id for existing_group_id in pending_audit_group_ids if existing_group_id != group_id]

    @staticmethod
    def _initial_pending_audit_groups(groups_by_id: dict[str, dict[str, Any]]) -> list[str]:
        return [
            group["group_id"]
            for group in sorted(groups_by_id.values(), key=lambda item: (-len(item.get("member_comment_ids", [])), item["group_id"]))
            if len(group.get("member_comment_ids", [])) > 1
        ]

    @staticmethod
    def _pending_audit_group_ids(state: AgenticPostProcessingState) -> list[str]:
        groups_by_id = state["groups_by_id"]
        seen: set[str] = set()
        group_ids: list[str] = []
        for group_id in state.get("pending_audit_group_ids", []):
            if group_id in groups_by_id and len(groups_by_id[group_id].get("member_comment_ids", [])) > 1 and group_id not in seen:
                group_ids.append(group_id)
                seen.add(group_id)
        return group_ids

    @staticmethod
    def _singleton_group_ids(state: AgenticPostProcessingState) -> list[str]:
        return [
            group_id
            for group_id, group in sorted(state["groups_by_id"].items())
            if len(group.get("member_comment_ids", [])) == 1
        ]

    @staticmethod
    def _unassigned_comment_ids(state: AgenticPostProcessingState) -> list[str]:
        return [
            comment_id
            for comment_id, comment in state["comments_by_id"].items()
            if not str(comment.get("group_id", "")).strip() and comment.get("decision_type") != "undefined"
        ]

    @staticmethod
    def _next_group_index(groups_by_id: dict[str, dict[str, Any]]) -> int:
        indexes = [
            int(match.group(1))
            for group_id in groups_by_id
            if (match := _GROUP_ID_RE.fullmatch(group_id))
        ]
        return max(indexes, default=0) + 1

    @staticmethod
    def _format_comment_card(comment: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"comment_id: {comment['comment_id']}",
                f"normalized_text: {_clean_text(comment.get('normalized_text') or comment.get('raw_text') or '')}",
                f"raw_text: {_truncate_text(comment.get('raw_text', ''), limit=240)}",
            ]
        )

    def _format_group_card(self, group: dict[str, Any], comments_by_id: dict[str, dict[str, Any]], *, include_all_members: bool) -> str:
        member_ids = list(group.get("member_comment_ids", []))
        if not include_all_members:
            member_ids = member_ids[: self._max_examples_per_candidate_group]
        members = [
            f"- {comment_id} | normalized: {_clean_text(comment.get('normalized_text') or comment.get('raw_text') or '')} | raw: {_truncate_text(comment.get('raw_text', ''), limit=180)}"
            for comment_id in member_ids
            if (comment := comments_by_id.get(comment_id))
        ]
        return "\n".join(
            [
                f"group_id: {group['group_id']}",
                f"group_name: {_clean_text(group.get('group_name', '')) or 'Not named'}",
                f"size: {len(group.get('member_comment_ids', []))}",
                "members:",
                *members,
            ]
        )

    def _build_all_cluster_candidates(
        self,
        *,
        comments_by_id: dict[str, dict[str, Any]],
        groups_by_id: dict[str, dict[str, Any]],
        exclude_group_ids: set[str],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for group in sorted(groups_by_id.values(), key=lambda item: (-len(item.get("member_comment_ids", [])), item["group_id"])):
            group_id = str(group.get("group_id", "")).strip()
            member_ids = list(group.get("member_comment_ids", []))
            if not group_id or group_id in exclude_group_ids or not member_ids:
                continue
            candidates.append(
                {
                    "group_id": group_id,
                    "group_name": _clean_text(group.get("group_name", "")) or "Not named",
                    "size": len(member_ids),
                    "examples": [
                        {
                            "comment_id": comment_id,
                            "normalized_text": _clean_text(comment.get("normalized_text") or comment.get("raw_text") or ""),
                            "raw_text": _truncate_text(comment.get("raw_text", "")),
                        }
                        for comment_id in member_ids[: self._max_examples_per_candidate_group]
                        if (comment := comments_by_id.get(comment_id))
                    ],
                }
            )
        return candidates

    @staticmethod
    def _format_candidate_groups(candidate_groups: list[dict[str, Any]]) -> str:
        if not candidate_groups:
            return "No candidate clusters."
        chunks: list[str] = []
        for candidate in candidate_groups:
            examples = [
                f"- {example['comment_id']} | normalized: {example['normalized_text']} | raw: {example['raw_text']}"
                for example in candidate.get("examples", [])
            ]
            chunks.append(
                "\n".join(
                    [
                        f"group_id: {candidate['group_id']}",
                        f"group_name: {candidate['group_name']}",
                        f"size: {candidate['size']}",
                        "examples:",
                        *(examples or ["- no examples"]),
                    ]
                )
            )
        return "\n\n".join(chunks)

    @staticmethod
    def _unique_group_comments(group: dict[str, Any], comments_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        unique_comments: list[dict[str, Any]] = []
        seen: set[str] = set()
        for comment_id in group.get("member_comment_ids", []):
            comment = comments_by_id.get(comment_id)
            if not comment:
                continue
            normalized_key = _normalize_key(comment.get("normalized_text") or comment.get("raw_text") or "")
            if normalized_key and normalized_key in seen:
                continue
            if normalized_key:
                seen.add(normalized_key)
            unique_comments.append(comment)
        return unique_comments

    @staticmethod
    def _format_group_examples(comments: list[dict[str, Any]]) -> str:
        if not comments:
            return "No examples."
        return "\n".join(
            f"- comment_id: {comment['comment_id']} | normalized_text: {_clean_text(comment.get('normalized_text') or comment.get('raw_text') or '')} | raw_text: {_truncate_text(comment.get('raw_text', ''), limit=180)}"
            for comment in comments
        )

    @staticmethod
    def _fallback_group_name(comments: list[dict[str, Any]]) -> str:
        if not comments:
            return "Not named"
        return _truncate_text(comments[0].get("normalized_text") or comments[0].get("raw_text") or "Not named", limit=60)

    def _merge_groups_by_name(
        self,
        groups_by_id: dict[str, dict[str, Any]],
        comments_by_id: dict[str, dict[str, Any]],
    ) -> None:
        canonical_by_name: dict[str, str] = {}
        for group_id in sorted(list(groups_by_id)):
            group = groups_by_id.get(group_id)
            if not group or not group.get("member_comment_ids"):
                groups_by_id.pop(group_id, None)
                continue
            normalized_name = _normalize_key(group.get("group_name", ""))
            if not normalized_name:
                continue
            canonical_group_id = canonical_by_name.setdefault(normalized_name, group_id)
            if canonical_group_id == group_id or canonical_group_id not in groups_by_id:
                continue
            for comment_id in list(group.get("member_comment_ids", [])):
                self._move_comment(
                    comments_by_id,
                    groups_by_id,
                    comment_id,
                    canonical_group_id,
                    "Merged because final group names matched",
                )
            groups_by_id.pop(group_id, None)

    @staticmethod
    def _build_final_result(
        state: AgenticPostProcessingState,
        comments_by_id: dict[str, dict[str, Any]],
        groups_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "comments": [
                comments_by_id[comment_id]
                for comment_id in state["comment_order"]
                if comment_id in comments_by_id
            ],
            "groups": [
                {"group_id": group_id, "group_name": _clean_text(group.get("group_name", "")) or "Not named"}
                for group_id, group in sorted(groups_by_id.items())
                if group.get("member_comment_ids")
            ],
        }

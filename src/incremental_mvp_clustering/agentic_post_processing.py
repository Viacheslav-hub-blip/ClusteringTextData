"""Stateless agentic post-processing pipeline over primary clustering results."""

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
    SupervisorStep,
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
    """LangGraph reconciliation layer with deterministic supervisor and stateless workers."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        max_examples_per_candidate_group: int = 3,
        planner_preview_group_limit: int = 20,
        audit_batch_size: int = 10,
        max_concurrent_llm_requests: int = 10,
        candidate_cluster_limit: int = 40,
        audit_comment_limit: int = 80,
        max_rounds: int = 100,
        max_no_change_rounds: int = 2,
        max_audit_passes_per_group: int = 2,
    ):
        self._max_examples_per_candidate_group = max(1, max_examples_per_candidate_group)
        self._planner_preview_group_limit = max(1, planner_preview_group_limit)
        self._audit_batch_size = max(1, audit_batch_size)
        self._candidate_cluster_limit = max(1, candidate_cluster_limit)
        self._audit_comment_limit = max(1, audit_comment_limit)
        self._max_rounds = max(1, max_rounds)
        self._max_no_change_rounds = max(1, max_no_change_rounds)
        self._max_audit_passes_per_group = max(1, max_audit_passes_per_group)
        self._llm_semaphore = asyncio.Semaphore(max_concurrent_llm_requests)

        self._router_chain = self._build_chain(POST_REVIEW_SYSTEM, POST_REVIEW_HUMAN, PlannerDecision, llm)
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
            "Agentic post-processing finished: %d final groups, reason=%s",
            len(final_state.get("final_result", {}).get("groups", [])),
            final_state.get("finish_reason", ""),
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
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("route_unassigned", self._route_unassigned_node)
        graph.add_node("resolve_singletons", self._resolve_singletons_node)
        graph.add_node("audit_group", self._audit_groups_batch_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "route_unassigned": "route_unassigned",
                "resolve_singletons": "resolve_singletons",
                "audit_group": "audit_group",
                "finalize": "finalize",
            },
        )
        graph.add_edge("route_unassigned", "supervisor")
        graph.add_edge("resolve_singletons", "supervisor")
        graph.add_edge("audit_group", "supervisor")
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
                    "postprocessing_trace": list(comment.get("postprocessing_trace", [])),
                }
            )
            comments_by_id[comment_id] = comment
            comment_order.append(comment_id)

        groups_by_id = self._build_groups_by_id(primary_result.get("groups", []), comments_by_id, comment_order)
        state: AgenticPostProcessingState = {
            "comments_by_id": comments_by_id,
            "groups_by_id": groups_by_id,
            "comment_order": comment_order,
            "accepted_singleton_group_ids": [],
            "audit_queue": self._initial_audit_queue(groups_by_id),
            "audit_attempts_by_group_id": {},
            "next_group_index": self._next_group_index(groups_by_id),
            "next_step": "finalize",
            "round_index": 0,
            "no_change_rounds": 0,
            "last_patch_summary": {},
            "finish_reason": "",
        }
        state.update(self._build_queue_update(state))
        return state

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

    async def _supervisor_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        queue_update = self._build_queue_update(state)
        unassigned_queue = queue_update["unassigned_queue"]
        singleton_queue = queue_update["singleton_queue"]
        audit_queue = queue_update["audit_queue"]
        round_index = int(state.get("round_index", 0))
        no_change_rounds = int(state.get("no_change_rounds", 0))
        finish_reason = ""

        if round_index >= self._max_rounds:
            next_step: SupervisorStep = "finalize"
            finish_reason = f"Safety stop: max_rounds={self._max_rounds} reached"
        elif not unassigned_queue and no_change_rounds >= self._max_no_change_rounds:
            next_step = "finalize"
            finish_reason = f"No-change stop: no productive progress for {no_change_rounds} rounds"
            queue_update["accepted_singleton_group_ids"] = self._merge_unique(
                state.get("accepted_singleton_group_ids", []),
                singleton_queue,
            )
            queue_update["singleton_queue"] = []
        elif not unassigned_queue and not singleton_queue and not audit_queue:
            next_step = "finalize"
            finish_reason = "Normal stop: all queues are empty or accepted"
        else:
            fallback_step = self._fallback_router_step(unassigned_queue, singleton_queue, audit_queue)
            decision = await self._ainvoke_chain(
                self._router_chain,
                {"state_summary": self._build_router_summary(state, queue_update)},
                fallback=PlannerDecision(next_step=self._to_planner_step(fallback_step), reason="Fallback router step"),
            )
            next_step = self._validate_router_step(
                decision.next_step,
                fallback_step=fallback_step,
                unassigned_queue=unassigned_queue,
                singleton_queue=singleton_queue,
                audit_queue=audit_queue,
            )
            if next_step == "finalize":
                finish_reason = f"LLM router selected finish: {decision.reason}"

        logger.info(
            "LLM supervisor selected step=%s, round=%d, unassigned=%d, singleton=%d, audit=%d, no_change=%d",
            next_step,
            round_index,
            len(unassigned_queue),
            len(queue_update["singleton_queue"]),
            len(audit_queue),
            no_change_rounds,
        )
        return {
            **queue_update,
            "next_step": next_step,
            "finish_reason": finish_reason or state.get("finish_reason", ""),
        }

    @staticmethod
    def _route_from_supervisor(state: AgenticPostProcessingState) -> SupervisorStep:
        return state.get("next_step", "finalize")

    def _build_router_summary(self, state: AgenticPostProcessingState, queues: dict[str, list[str]]) -> str:
        """Compact router context without cluster texts or worker history."""
        return "\n".join(
            [
                f"Total comments: {len(state['comments_by_id'])}",
                f"Non-empty clusters: {len(state['groups_by_id'])}",
                f"Unassigned comments: {len(queues['unassigned_queue'])}",
                f"Singleton clusters pending decision: {len(queues['singleton_queue'])}",
                f"Accepted singleton clusters: {len(queues['accepted_singleton_group_ids'])}",
                f"Clusters pending audit: {len(queues['audit_queue'])}",
                f"Round index: {state.get('round_index', 0)}",
                f"No-change rounds: {state.get('no_change_rounds', 0)}",
                f"Max rounds: {self._max_rounds}",
                f"Max no-change rounds: {self._max_no_change_rounds}",
                f"Audit batch size: {self._audit_batch_size}",
                f"Last patch summary: {state.get('last_patch_summary', {})}",
                "",
                "Router constraints:",
                "- If unassigned comments exist, choose route_unassigned unless there is a stronger reason to audit first.",
                "- resolve_singletons is allowed when singleton_queue is non-empty.",
                "- audit_group is allowed when audit_queue is non-empty.",
                "- finish is allowed only when queues are empty/accepted or safety guards indicate stop.",
                "- Do not request text-level context; workers receive local context later.",
            ]
        )

    @staticmethod
    def _fallback_router_step(
        unassigned_queue: list[str],
        singleton_queue: list[str],
        audit_queue: list[str],
    ) -> SupervisorStep:
        if unassigned_queue:
            return "route_unassigned"
        if singleton_queue:
            return "resolve_singletons"
        if audit_queue:
            return "audit_group"
        return "finalize"

    @staticmethod
    def _to_planner_step(step: SupervisorStep) -> str:
        return "finish" if step == "finalize" else step

    @staticmethod
    def _validate_router_step(
        requested_step: str,
        *,
        fallback_step: SupervisorStep,
        unassigned_queue: list[str],
        singleton_queue: list[str],
        audit_queue: list[str],
    ) -> SupervisorStep:
        if requested_step == "finish":
            return "finalize" if not (unassigned_queue or singleton_queue or audit_queue) else fallback_step
        if requested_step == "route_unassigned":
            return "route_unassigned" if unassigned_queue else fallback_step
        if requested_step == "resolve_singletons":
            return "resolve_singletons" if singleton_queue else fallback_step
        if requested_step == "audit_group":
            return "audit_group" if audit_queue else fallback_step
        return fallback_step

    async def _resolve_singletons_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        accepted_singleton_group_ids = list(state.get("accepted_singleton_group_ids", []))
        audit_queue = list(state.get("audit_queue", []))
        singleton_queue = list(state.get("singleton_queue", []))
        logger.info("Agentic step resolve_singletons started: %d stateless workers", len(singleton_queue))

        tasks = [
            self._decide_singleton(
                comment=comments_by_id[group["member_comment_ids"][0]],
                group=group,
                candidate_groups=self._build_cluster_candidates(
                    comments_by_id=comments_by_id,
                    groups_by_id=groups_by_id,
                    exclude_group_ids={group_id},
                    limit=self._candidate_cluster_limit,
                ),
            )
            for group_id in singleton_queue
            if (group := groups_by_id.get(group_id))
            and len(group.get("member_comment_ids", [])) == 1
            and group["member_comment_ids"][0] in comments_by_id
        ]
        decisions = await asyncio.gather(*tasks) if tasks else []

        applied_changes = 0
        processed_items = 0
        for group_id, decision in decisions:
            group = groups_by_id.get(group_id)
            if not group or len(group.get("member_comment_ids", [])) != 1:
                continue
            comment_id = group["member_comment_ids"][0]
            target_group_id = decision.target_group_id.strip()
            processed_items += 1

            if decision.action == "move_to_group" and target_group_id in groups_by_id:
                if self._move_comment(comments_by_id, groups_by_id, comment_id, target_group_id, decision.reason):
                    applied_changes += 1
                    audit_queue = self._mark_group_for_audit(audit_queue, state, groups_by_id, target_group_id)
                    logger.info("Singleton comment %s moved: %s -> %s", comment_id, group_id, target_group_id)
                    continue

            accepted_singleton_group_ids = self._merge_unique(accepted_singleton_group_ids, [group_id])
            logger.info("Singleton cluster %s accepted as separate cluster", group_id)

        return self._build_action_update(
            state,
            applied_changes=applied_changes,
            processed_items=processed_items,
            comments_by_id=comments_by_id,
            groups_by_id=groups_by_id,
            audit_queue=audit_queue,
            accepted_singleton_group_ids=accepted_singleton_group_ids,
            last_patch_summary={
                "step": "resolve_singletons",
                "workers": len(tasks),
                "processed": processed_items,
                "changes": applied_changes,
            },
        )

    async def _audit_groups_batch_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        audit_queue = list(state.get("audit_queue", []))
        audit_attempts_by_group_id = dict(state.get("audit_attempts_by_group_id", {}))
        target_group_ids = self._valid_audit_queue(state, groups_by_id)[: self._audit_batch_size]
        logger.info(
            "Agentic step audit_group started: %d/%d stateless workers",
            len(target_group_ids),
            len(self._valid_audit_queue(state, groups_by_id)),
        )

        tasks = [self._audit_one_group(group_id, groups_by_id, comments_by_id) for group_id in target_group_ids]
        audit_results = await asyncio.gather(*tasks) if tasks else []

        applied_changes = 0
        processed_items = 0
        for group_id, decision, visible_comment_ids in audit_results:
            processed_items += 1
            audit_attempts_by_group_id[group_id] = audit_attempts_by_group_id.get(group_id, 0) + 1
            group = groups_by_id.get(group_id)
            if not group:
                audit_queue = self._remove_group_from_queue(audit_queue, group_id)
                continue

            removable_ids = [
                comment_id
                for comment_id in decision.remove_comment_ids
                if comment_id in visible_comment_ids and comment_id in group.get("member_comment_ids", [])
            ]
            for comment_id in removable_ids:
                self._unassign_comment(comments_by_id, groups_by_id, comment_id, decision.reason)
                applied_changes += 1
            if removable_ids:
                logger.info("Audit cluster %s removed %d comments", group_id, len(removable_ids))

            audit_queue = self._remove_group_from_queue(audit_queue, group_id)
            if removable_ids and group_id in groups_by_id and len(groups_by_id[group_id]["member_comment_ids"]) > 1:
                audit_queue = self._mark_group_for_audit(audit_queue, state, groups_by_id, group_id, audit_attempts_by_group_id)

        return self._build_action_update(
            state,
            applied_changes=applied_changes,
            processed_items=processed_items,
            comments_by_id=comments_by_id,
            groups_by_id=groups_by_id,
            audit_queue=audit_queue,
            audit_attempts_by_group_id=audit_attempts_by_group_id,
            last_patch_summary={
                "step": "audit_group",
                "workers": len(tasks),
                "processed": processed_items,
                "changes": applied_changes,
            },
        )

    async def _route_unassigned_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        audit_queue = list(state.get("audit_queue", []))
        next_group_index = int(state.get("next_group_index", 1))
        unassigned_queue = list(state.get("unassigned_queue", []))
        logger.info("Agentic step route_unassigned started: %d stateless workers", len(unassigned_queue))

        tasks = [
            self._decide_unassigned(
                comment=comments_by_id[comment_id],
                candidate_groups=self._build_cluster_candidates(
                    comments_by_id=comments_by_id,
                    groups_by_id=groups_by_id,
                    exclude_group_ids=set(),
                    limit=self._candidate_cluster_limit,
                ),
            )
            for comment_id in unassigned_queue
            if comment_id in comments_by_id
        ]
        decisions = await asyncio.gather(*tasks) if tasks else []

        applied_changes = 0
        processed_items = 0
        for comment_id, decision in decisions:
            if comment_id not in comments_by_id or comments_by_id[comment_id].get("group_id"):
                continue
            processed_items += 1
            target_group_id = decision.target_group_id.strip()
            if decision.action != "move_to_group" or target_group_id not in groups_by_id:
                target_group_id, next_group_index = self._create_group(groups_by_id, next_group_index)

            self._assign_comment_to_group(comments_by_id, groups_by_id, comment_id, target_group_id, decision.reason)
            audit_queue = self._mark_group_for_audit(audit_queue, state, groups_by_id, target_group_id)
            applied_changes += 1
            logger.info("Unassigned comment %s routed to %s", comment_id, target_group_id)

        return self._build_action_update(
            state,
            applied_changes=applied_changes,
            processed_items=processed_items,
            comments_by_id=comments_by_id,
            groups_by_id=groups_by_id,
            audit_queue=audit_queue,
            next_group_index=next_group_index,
            last_patch_summary={
                "step": "route_unassigned",
                "workers": len(tasks),
                "processed": processed_items,
                "changes": applied_changes,
            },
        )

    async def _finalize_node(self, state: AgenticPostProcessingState) -> dict[str, Any]:
        comments_by_id, groups_by_id = self._copy_cluster_state(state)
        logger.info(
            "Agentic step finalize started: naming %d groups, reason=%s",
            len(groups_by_id),
            state.get("finish_reason", ""),
        )
        await self._rename_groups(groups_by_id, comments_by_id)
        self._merge_groups_by_name(groups_by_id, comments_by_id)
        final_result = self._build_final_result(state, comments_by_id, groups_by_id)
        logger.info("Agentic step finalize finished: %d groups after merge", len(final_result["groups"]))
        return {"final_result": final_result}

    async def _decide_singleton(
        self,
        *,
        comment: dict[str, Any],
        group: dict[str, Any],
        candidate_groups: list[dict[str, Any]],
    ) -> tuple[str, SingletonResolutionDecision]:
        group_id = str(group["group_id"])
        if not candidate_groups:
            return group_id, SingletonResolutionDecision(
                action="keep_current_group",
                target_group_id="",
                reason="No existing candidate clusters",
            )
        decision = await self._ainvoke_chain(
            self._singleton_chain,
            {
                "singleton_cluster": self._format_group_card(
                    group,
                    {comment["comment_id"]: comment},
                    member_limit=1,
                ),
                "candidate_groups": self._format_candidate_groups(candidate_groups),
            },
            fallback=SingletonResolutionDecision(
                action="keep_current_group",
                target_group_id="",
                reason="Fallback: singleton kept separate",
            ),
        )
        return group_id, decision

    async def _audit_one_group(
        self,
        group_id: str,
        groups_by_id: dict[str, dict[str, Any]],
        comments_by_id: dict[str, dict[str, Any]],
    ) -> tuple[str, ClusterAuditDecision, set[str]]:
        group = groups_by_id[group_id]
        visible_comment_ids = set(group.get("member_comment_ids", [])[: self._audit_comment_limit])
        decision = await self._ainvoke_chain(
            self._audit_chain,
            {"group_card": self._format_group_card(group, comments_by_id, member_limit=self._audit_comment_limit)},
            fallback=ClusterAuditDecision(remove_comment_ids=[], reason="Fallback: cluster looks consistent"),
        )
        return group_id, decision, visible_comment_ids

    async def _decide_unassigned(
        self,
        *,
        comment: dict[str, Any],
        candidate_groups: list[dict[str, Any]],
    ) -> tuple[str, UnassignedRoutingDecision]:
        comment_id = str(comment["comment_id"])
        if not candidate_groups:
            return comment_id, UnassignedRoutingDecision(
                action="create_new_group",
                target_group_id="",
                reason="No existing candidate clusters",
            )
        decision = await self._ainvoke_chain(
            self._unassigned_chain,
            {
                "comment_card": self._format_comment_card(comment),
                "candidate_groups": self._format_candidate_groups(candidate_groups),
            },
            fallback=UnassignedRoutingDecision(
                action="create_new_group",
                target_group_id="",
                reason="Fallback: no safe existing cluster",
            ),
        )
        return comment_id, decision

    async def _rename_groups(
        self,
        groups_by_id: dict[str, dict[str, Any]],
        comments_by_id: dict[str, dict[str, Any]],
    ) -> None:
        async def rename_one(group_id: str, group: dict[str, Any]) -> tuple[str, str]:
            members = self._unique_group_comments(group, comments_by_id)
            fallback_name = self._fallback_group_name(members)
            decision = await self._ainvoke_chain(
                self._naming_chain,
                {"group_examples": self._format_group_examples(members)},
                fallback=PostProcessingGroupName(group_name=fallback_name),
            )
            return group_id, _clean_text(decision.group_name) or fallback_name

        tasks = [
            rename_one(group_id, group)
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
            logger.error("Agentic post-processing worker failed, using fallback: %s", exc)
            return fallback

    @staticmethod
    def _copy_cluster_state(
        state: AgenticPostProcessingState,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        return copy.deepcopy(state["comments_by_id"]), copy.deepcopy(state["groups_by_id"])

    def _build_action_update(
        self,
        state: AgenticPostProcessingState,
        *,
        applied_changes: int,
        processed_items: int,
        comments_by_id: dict[str, dict[str, Any]],
        groups_by_id: dict[str, dict[str, Any]],
        audit_queue: list[str],
        accepted_singleton_group_ids: list[str] | None = None,
        audit_attempts_by_group_id: dict[str, int] | None = None,
        next_group_index: int | None = None,
        last_patch_summary: dict[str, Any],
    ) -> dict[str, Any]:
        next_state: AgenticPostProcessingState = {
            **state,
            "comments_by_id": comments_by_id,
            "groups_by_id": groups_by_id,
            "audit_queue": audit_queue,
            "accepted_singleton_group_ids": accepted_singleton_group_ids
            if accepted_singleton_group_ids is not None
            else list(state.get("accepted_singleton_group_ids", [])),
            "audit_attempts_by_group_id": audit_attempts_by_group_id
            if audit_attempts_by_group_id is not None
            else dict(state.get("audit_attempts_by_group_id", {})),
            "next_group_index": next_group_index
            if next_group_index is not None
            else int(state.get("next_group_index", 1)),
            "round_index": int(state.get("round_index", 0)) + 1,
            "no_change_rounds": int(state.get("no_change_rounds", 0)) + 1
            if processed_items == 0 and applied_changes == 0
            else 0,
            "last_patch_summary": last_patch_summary,
        }
        next_state.update(self._build_queue_update(next_state))
        logger.info(
            "Agentic action finished: step=%s, workers=%s, processed=%s, changes=%s",
            last_patch_summary.get("step"),
            last_patch_summary.get("workers"),
            processed_items,
            applied_changes,
        )
        return {
            "comments_by_id": next_state["comments_by_id"],
            "groups_by_id": next_state["groups_by_id"],
            "singleton_queue": next_state["singleton_queue"],
            "audit_queue": next_state["audit_queue"],
            "unassigned_queue": next_state["unassigned_queue"],
            "accepted_singleton_group_ids": next_state["accepted_singleton_group_ids"],
            "audit_attempts_by_group_id": next_state["audit_attempts_by_group_id"],
            "next_group_index": next_state["next_group_index"],
            "round_index": next_state["round_index"],
            "no_change_rounds": next_state["no_change_rounds"],
            "last_patch_summary": next_state["last_patch_summary"],
        }

    def _build_queue_update(self, state: AgenticPostProcessingState) -> dict[str, list[str]]:
        groups_by_id = state["groups_by_id"]
        comments_by_id = state["comments_by_id"]
        accepted_singleton_group_ids = [
            group_id
            for group_id in state.get("accepted_singleton_group_ids", [])
            if group_id in groups_by_id and len(groups_by_id[group_id].get("member_comment_ids", [])) == 1
        ]
        accepted = set(accepted_singleton_group_ids)
        audit_attempts = dict(state.get("audit_attempts_by_group_id", {}))
        return {
            "unassigned_queue": self._unassigned_comment_ids(state),
            "singleton_queue": [
                group_id
                for group_id, group in sorted(groups_by_id.items())
                if len(group.get("member_comment_ids", [])) == 1 and group_id not in accepted
            ],
            "audit_queue": self._valid_audit_queue(state, groups_by_id, audit_attempts),
            "accepted_singleton_group_ids": accepted_singleton_group_ids,
        }

    def _valid_audit_queue(
        self,
        state: AgenticPostProcessingState,
        groups_by_id: dict[str, dict[str, Any]],
        audit_attempts: dict[str, int] | None = None,
    ) -> list[str]:
        audit_attempts = audit_attempts if audit_attempts is not None else dict(state.get("audit_attempts_by_group_id", {}))
        seen: set[str] = set()
        group_ids: list[str] = []
        for group_id in state.get("audit_queue", []):
            if group_id in seen:
                continue
            if group_id not in groups_by_id or len(groups_by_id[group_id].get("member_comment_ids", [])) <= 1:
                continue
            if audit_attempts.get(group_id, 0) >= self._max_audit_passes_per_group:
                continue
            group_ids.append(group_id)
            seen.add(group_id)
        return group_ids

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
        while group_id in groups_by_id:
            next_group_index += 1
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
        audit_queue: list[str],
        state: AgenticPostProcessingState,
        groups_by_id: dict[str, dict[str, Any]],
        group_id: str,
        audit_attempts_by_group_id: dict[str, int] | None = None,
    ) -> list[str]:
        attempts = audit_attempts_by_group_id if audit_attempts_by_group_id is not None else dict(state.get("audit_attempts_by_group_id", {}))
        if (
            group_id not in groups_by_id
            or len(groups_by_id[group_id].get("member_comment_ids", [])) <= 1
            or attempts.get(group_id, 0) >= self._max_audit_passes_per_group
        ):
            return self._remove_group_from_queue(audit_queue, group_id)
        queue = [existing_group_id for existing_group_id in audit_queue if existing_group_id != group_id]
        return [group_id, *queue]

    @staticmethod
    def _remove_group_from_queue(queue: list[str], group_id: str) -> list[str]:
        return [existing_group_id for existing_group_id in queue if existing_group_id != group_id]

    @staticmethod
    def _merge_unique(existing: list[str], additions: list[str]) -> list[str]:
        return list(dict.fromkeys([*existing, *additions]))

    @staticmethod
    def _initial_audit_queue(groups_by_id: dict[str, dict[str, Any]]) -> list[str]:
        return [
            group["group_id"]
            for group in sorted(groups_by_id.values(), key=lambda item: (-len(item.get("member_comment_ids", [])), item["group_id"]))
            if len(group.get("member_comment_ids", [])) > 1
        ]

    @staticmethod
    def _unassigned_comment_ids(state: AgenticPostProcessingState) -> list[str]:
        comments_by_id = state["comments_by_id"]
        return [
            comment_id
            for comment_id in state.get("comment_order", list(comments_by_id))
            if comment_id in comments_by_id
            and not str(comments_by_id[comment_id].get("group_id", "")).strip()
            and str(comments_by_id[comment_id].get("decision_type", "")).lower() not in {"undefined", "decisiontype.undefined"}
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

    def _format_group_card(
        self,
        group: dict[str, Any],
        comments_by_id: dict[str, dict[str, Any]],
        *,
        member_limit: int,
    ) -> str:
        member_ids = list(group.get("member_comment_ids", []))[:member_limit]
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
                f"shown_members: {len(members)}",
                "members:",
                *members,
            ]
        )

    def _build_cluster_candidates(
        self,
        *,
        comments_by_id: dict[str, dict[str, Any]],
        groups_by_id: dict[str, dict[str, Any]],
        exclude_group_ids: set[str],
        limit: int,
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
            if len(candidates) >= limit:
                break
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

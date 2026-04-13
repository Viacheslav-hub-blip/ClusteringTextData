"""LangChain agent orchestration for local cluster repair."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool

from .prompts import AGENT_SYSTEM_PROMPT
from .services.session import AgenticClusteringSession
from .services.text_utils import dump_json


class LangChainAgentOrchestrator:
    """Run a bounded local-repair pass through LangChain tools."""

    def __init__(
        self,
        *,
        llm,
        session: AgenticClusteringSession,
        max_repairs: int = 3,
        recursion_limit: int = 40,
    ):
        self._llm = llm
        self._session = session
        self._max_repairs = max_repairs
        self._recursion_limit = recursion_limit
        self._agent = create_agent(
            model=llm,
            tools=self._build_tools(),
            system_prompt=AGENT_SYSTEM_PROMPT,
            name="banking_clustering_supervisor",
        )

    def run(self) -> dict:
        """Run the agent and return its final state."""
        task = (
            "Сделай один ограниченный проход локального улучшения кластеризации. "
            f"Максимум полезных примененных patch: {self._max_repairs}. "
            "Работай только через инструменты и только локально."
        )
        return self._agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": self._recursion_limit},
        )

    def _build_tools(self):
        session = self._session

        @tool
        def summarize_state() -> str:
            """Return a compact summary of the current clustering state."""
            return dump_json(session.summarize_state())

        @tool
        def list_neighborhoods(limit: int = 5) -> str:
            """List suspicious local neighborhoods that may need localized reclustering."""
            return dump_json(session.list_neighborhoods(limit=limit))

        @tool
        def inspect_neighborhood(neighborhood_id: str) -> str:
            """Inspect a neighborhood and get a critique with local repair guidance."""
            return dump_json(session.inspect_neighborhood(neighborhood_id))

        @tool
        def recluster_neighborhood(neighborhood_id: str, guidance: str = "") -> str:
            """Run local reclustering for one neighborhood and create a candidate patch."""
            return dump_json(session.recluster_neighborhood(neighborhood_id, guidance=guidance))

        @tool
        def review_patch(patch_id: str) -> str:
            """Review a candidate patch before applying it."""
            return dump_json(session.review_patch(patch_id))

        @tool
        def apply_patch(patch_id: str) -> str:
            """Apply a reviewed patch to the current snapshot if it improves local quality."""
            return dump_json(session.apply_patch(patch_id))

        return [
            summarize_state,
            list_neighborhoods,
            inspect_neighborhood,
            recluster_neighborhood,
            review_patch,
            apply_patch,
        ]

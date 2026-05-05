"""Публичные pipeline-классы библиотеки.

Файл содержит:
- ``VectorLLMClusteringPipeline`` — базовая кластеризация через векторы и LLM;
- ``VectorLLMAgenticClusteringPipeline`` — базовая кластеризация и агентская постобработка.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .agentic_post_processing import AgenticPostProcessingPipeline
from .config import AgenticPromptConfig, ClusteringPromptConfig, PrimaryPromptConfig
from .pipeline import IncrementalMVPClusteringPipeline


class VectorLLMClusteringPipeline(IncrementalMVPClusteringPipeline):
    """Pipeline кластеризации только через векторный поиск и LLM.

    Args:
        llm: Chat-модель LangChain для нормализации, выбора группы и именования.
        embeddings: Embedding-модель LangChain для построения векторного индекса.
        prompt_config: Prompt-конфигурация базового этапа.
        kwargs: Остальные параметры ``IncrementalMVPClusteringPipeline``.

    Returns:
        Экземпляр pipeline, который возвращает словарь с ``comments`` и ``groups``.
    """

    def __init__(
            self,
            llm: BaseChatModel,
            embeddings: Embeddings,
            *,
            prompt_config: PrimaryPromptConfig | None = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, embeddings=embeddings, prompt_config=prompt_config, **kwargs)


class VectorLLMAgenticClusteringPipeline:
    """Pipeline кластеризации через векторы и LLM с агентской постобработкой.

    Args:
        llm: Chat-модель LangChain для базовой кластеризации и постобработки.
        embeddings: Embedding-модель LangChain для построения векторного индекса.
        prompt_config: Объединенная prompt-конфигурация базового и агентского этапов.
        primary_kwargs: Параметры базового pipeline.
        agentic_kwargs: Параметры агентской постобработки.

    Returns:
        Экземпляр pipeline, который возвращает финальный словарь с ``comments``,
        ``groups`` и служебной информацией постобработки.
    """

    def __init__(
            self,
            llm: BaseChatModel,
            embeddings: Embeddings,
            *,
            prompt_config: ClusteringPromptConfig | None = None,
            primary_kwargs: dict[str, Any] | None = None,
            agentic_kwargs: dict[str, Any] | None = None,
    ) -> None:
        prompt_config = prompt_config or ClusteringPromptConfig()
        primary_kwargs = dict(primary_kwargs or {})
        agentic_kwargs = dict(agentic_kwargs or {})

        self._primary_pipeline = VectorLLMClusteringPipeline(
            llm=llm,
            embeddings=embeddings,
            prompt_config=prompt_config.primary,
            **primary_kwargs,
        )
        self._agentic_pipeline = AgenticPostProcessingPipeline(
            llm=llm,
            prompt_config=prompt_config.agentic,
            **agentic_kwargs,
        )

    def run(self, raw_comments: list[dict]) -> dict[str, Any]:
        """Синхронно запускает базовую кластеризацию и агентскую постобработку.

        Args:
            raw_comments: Список словарей с полями ``comment_id`` и ``text``.

        Returns:
            Финальный результат кластеризации.
        """
        return asyncio.run(self.arun(raw_comments))

    async def arun(self, raw_comments: list[dict]) -> dict[str, Any]:
        """Асинхронно запускает базовую кластеризацию и агентскую постобработку.

        Args:
            raw_comments: Список словарей с полями ``comment_id`` и ``text``.

        Returns:
            Финальный результат кластеризации.
        """
        primary_result = await self._primary_pipeline.arun(raw_comments)
        return await self._agentic_pipeline.arun(primary_result)


__all__ = [
    "AgenticPromptConfig",
    "ClusteringPromptConfig",
    "PrimaryPromptConfig",
    "VectorLLMAgenticClusteringPipeline",
    "VectorLLMClusteringPipeline",
]

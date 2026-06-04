"""Публичные pipeline-классы библиотеки.

Файл содержит:
- ``VectorLLMClusteringPipeline`` — упрощенная кластеризация через embeddings,
  FAISS/BM25 и LLM-решение о группе.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import PrimaryPromptConfig
from .pipeline import IncrementalMVPClusteringPipeline


class VectorLLMClusteringPipeline(IncrementalMVPClusteringPipeline):
    """Публичный pipeline кластеризации комментариев.

    Args:
        llm: Chat-модель LangChain для выбора группы и опционального нейминга.
        embeddings: Embedding-модель LangChain для построения векторов.
        prompt_config: Prompt-конфигурация LLM-решений.
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


__all__ = [
    "PrimaryPromptConfig",
    "VectorLLMClusteringPipeline",
]

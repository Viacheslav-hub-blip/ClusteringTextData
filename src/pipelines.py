"""Публичные pipeline-классы библиотеки.

Файл содержит:
- ``VectorLLMClusteringPipeline`` — упрощенная кластеризация через FAISS, BM25 и LLM;
- ``SimpleFaissBM25LLMClusteringPipeline`` — явное имя той же рабочей реализации.
"""

from __future__ import annotations

from .config import PrimaryPromptConfig
from .pipeline import SimpleFaissBM25LLMClusteringPipeline


class VectorLLMClusteringPipeline(SimpleFaissBM25LLMClusteringPipeline):
    """Публичный pipeline кластеризации комментариев через FAISS, BM25 и LLM.

    Args:
        llm: Chat-модель LangChain для выбора группы.
        embeddings: Embedding-модель LangChain для построения векторов и FAISS.
        prompt_config: Prompt-конфигурация выбора группы.
        kwargs: Остальные параметры ``SimpleFaissBM25LLMClusteringPipeline``.

    Returns:
        Экземпляр pipeline, который возвращает исходные строки с добавленным ``group_name``.
    """


__all__ = [
    "PrimaryPromptConfig",
    "SimpleFaissBM25LLMClusteringPipeline",
    "VectorLLMClusteringPipeline",
]

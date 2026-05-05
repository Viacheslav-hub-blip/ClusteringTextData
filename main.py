"""Примеры запуска библиотеки из файла ``main.py``.

Файл содержит:
- ``build_llm`` — точка подключения пользовательской LLM-модели;
- ``build_embeddings`` — точка подключения пользовательской embedding-модели;
- ``load_comments`` — загрузка комментариев из JSON-файла или demo-списка;
- ``run_vector_llm_example`` — запуск базового pipeline;
- ``run_agentic_example`` — запуск полного pipeline с постобработкой;
- ``main`` — точка входа для запуска из IDE или файла.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from clusteringtextdata import (
    ClusteringPromptConfig,
    PrimaryPromptConfig,
    VectorLLMAgenticClusteringPipeline,
    VectorLLMClusteringPipeline,
)


DEMO_COMMENTS: list[dict[str, str]] = [
    {"comment_id": "1", "text": "Не приходит код подтверждения перевода"},
    {"comment_id": "2", "text": "Слишком долго жду подтверждение операции"},
    {"comment_id": "3", "text": "Банк слишком часто блокирует обычные покупки"},
    {"comment_id": "4", "text": "Постоянные блокировки карт мешают оплате"},
]

RUN_MODE = "vector"
INPUT_PATH: str | None = None
OUTPUT_PATH: str | None = None


def build_llm() -> BaseChatModel:
    """Создает пользовательский экземпляр LLM.

    Args:
        Входные аргументы отсутствуют.

    Returns:
        Объект, совместимый с ``langchain_core.language_models.BaseChatModel``.

    Raises:
        NotImplementedError: Если пользователь не подключил свою модель.
    """
    raise NotImplementedError(
        "Подключите свою LLM-модель в функции build_llm(). "
        "Например, создайте экземпляр LangChain BaseChatModel и верните его."
    )


def build_embeddings() -> Embeddings:
    """Создает пользовательский экземпляр embedding-модели.

    Args:
        Входные аргументы отсутствуют.

    Returns:
        Объект, совместимый с ``langchain_core.embeddings.Embeddings``.

    Raises:
        NotImplementedError: Если пользователь не подключил свою embedding-модель.
    """
    raise NotImplementedError(
        "Подключите свою embedding-модель в функции build_embeddings(). "
        "Например, создайте экземпляр LangChain Embeddings и верните его."
    )


def load_comments(input_path: str | None) -> list[dict[str, Any]]:
    """Загружает комментарии из JSON-файла или возвращает demo-набор.

    Args:
        input_path: Путь к JSON-файлу со списком комментариев или ``None``.

    Returns:
        Список словарей комментариев с полями ``comment_id`` и ``text``.

    Raises:
        ValueError: Если JSON имеет неверную структуру.
    """
    if input_path is None:
        return list(DEMO_COMMENTS)

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Входной JSON должен содержать список комментариев.")
    return data


def run_vector_llm_example(comments: list[dict[str, Any]]) -> dict[str, Any]:
    """Запускает базовый pipeline кластеризации.

    Args:
        comments: Список комментариев для обработки.

    Returns:
        Результат базовой кластеризации.
    """
    llm = build_llm()
    embeddings = build_embeddings()

    prompt_config = PrimaryPromptConfig.default()
    pipeline = VectorLLMClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
        retrieval_top_k=12,
        max_examples_per_candidate_group=3,
        primary_similarity_threshold=0.5,
        prompt_config=prompt_config,
    )
    return pipeline.run(comments)


def run_agentic_example(comments: list[dict[str, Any]]) -> dict[str, Any]:
    """Запускает полный pipeline с агентской постобработкой.

    Args:
        comments: Список комментариев для обработки.

    Returns:
        Финальный результат кластеризации после постобработки.
    """
    llm = build_llm()
    embeddings = build_embeddings()

    pipeline = VectorLLMAgenticClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
        prompt_config=ClusteringPromptConfig(),
        primary_kwargs={
            "retrieval_top_k": 12,
            "max_examples_per_candidate_group": 3,
            "primary_similarity_threshold": 0.5,
        },
        agentic_kwargs={
            "max_rounds": 100,
            "candidate_cluster_limit": 40,
            "merge_groups_by_final_name": True,
        },
    )
    return pipeline.run(comments)


def main() -> None:
    """Запускает пример библиотеки напрямую из файла.

    Args:
        Входные аргументы отсутствуют. Настройки задаются константами в начале файла.

    Returns:
        ``None``. Функция печатает результат и при необходимости сохраняет его в JSON.
    """
    mode = RUN_MODE.strip().lower()
    if mode not in {"vector", "agentic"}:
        raise ValueError("RUN_MODE должен быть равен 'vector' или 'agentic'.")

    comments = load_comments(INPUT_PATH)

    if mode == "vector":
        result = run_vector_llm_example(comments)
    else:
        result = run_agentic_example(comments)

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)

    if OUTPUT_PATH:
        Path(OUTPUT_PATH).write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()

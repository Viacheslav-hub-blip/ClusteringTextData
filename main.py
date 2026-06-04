"""Примеры запуска библиотеки из файла ``main.py``.

Файл содержит:
- ``build_llm`` — точка подключения пользовательской LLM-модели;
- ``build_embeddings`` — точка подключения пользовательской embedding-модели;
- ``load_comments`` — загрузка комментариев из JSON-файла или demo-списка;
- ``run_vector_llm_example`` — запуск pipeline через FAISS, BM25 и LLM;
- ``main`` — точка входа для запуска из IDE или файла.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from clusteringtextdata import (
    PrimaryPromptConfig,
    VectorLLMClusteringPipeline,
)


DEMO_COMMENTS: list[dict[str, str]] = [
    {"comment_id": "1", "text": "Не приходит код подтверждения перевода"},
    {"comment_id": "2", "text": "Слишком долго жду подтверждение операции"},
    {"comment_id": "3", "text": "Банк слишком часто блокирует обычные покупки"},
    {"comment_id": "4", "text": "Постоянные блокировки карт мешают оплате"},
]

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
    """Запускает pipeline кластеризации через FAISS, BM25 и LLM.

    Args:
        comments: Список комментариев для обработки.

    Returns:
        Результат кластеризации со строками ``rows`` и описанием групп ``groups``.
    """
    llm = build_llm()
    embeddings = build_embeddings()

    prompt_config = PrimaryPromptConfig.default()
    pipeline = VectorLLMClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
        faiss_top_k=80,
        bm25_top_k=80,
        candidate_group_limit=30,
        max_examples_per_candidate_group=8,
        merge_small_groups=True,
        small_group_max_size=5,
        merge_candidate_group_limit=40,
        prompt_config=prompt_config,
    )
    return pipeline.run(comments)


def main() -> None:
    """Запускает пример библиотеки напрямую из файла.

    Args:
        Входные аргументы отсутствуют. Настройки задаются константами в начале файла.

    Returns:
        ``None``. Функция печатает результат и при необходимости сохраняет его в JSON.
    """
    comments = load_comments(INPUT_PATH)
    result = run_vector_llm_example(comments)

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)

    if OUTPUT_PATH:
        Path(OUTPUT_PATH).write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()

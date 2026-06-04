"""Пример простого запуска библиотеки из IDE.

Файл содержит:
- ``build_llm`` — точка подключения пользовательской LLM-модели;
- ``build_embeddings`` — точка подключения пользовательской embedding-модели;
- ``main`` — пример запуска ``cluster_text_data``.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from clusteringtextdata import cluster_text_data


INPUT_PATH: str | None = None
OUTPUT_PATH: str | None = None

DEMO_COMMENTS: list[dict[str, str]] = [
    {"comment_id": "1", "text": "Не приходит код подтверждения перевода"},
    {"comment_id": "2", "text": "Слишком долго жду подтверждение операции"},
    {"comment_id": "3", "text": "Банк слишком часто блокирует обычные покупки"},
    {"comment_id": "4", "text": "Постоянные блокировки карт мешают оплате"},
]


def build_llm() -> BaseChatModel:
    """Создает пользовательский экземпляр LLM.

    Returns:
        Объект, совместимый с ``langchain_core.language_models.BaseChatModel``.

    Raises:
        NotImplementedError: Если пользователь не подключил свою модель.
    """
    raise NotImplementedError(
        "Подключите свою LLM-модель в функции build_llm(). "
        "Не вставляйте API-ключи в код библиотеки."
    )


def build_embeddings() -> Embeddings:
    """Создает пользовательский экземпляр embedding-модели.

    Returns:
        Объект, совместимый с ``langchain_core.embeddings.Embeddings``.

    Raises:
        NotImplementedError: Если пользователь не подключил embedding-модель.
    """
    raise NotImplementedError(
        "Подключите свою embedding-модель в функции build_embeddings(). "
        "Не вставляйте API-ключи в код библиотеки."
    )


def main() -> None:
    """Запускает простой пример кластеризации.

    Returns:
        ``None``. Функция печатает таблицу результата или сохраняет ее в файл.
    """
    llm = build_llm()
    embeddings = build_embeddings()
    data = INPUT_PATH or DEMO_COMMENTS

    result = cluster_text_data(
        data,
        llm=llm,
        embeddings=embeddings,
        output_path=OUTPUT_PATH,
        generate_group_names=True,
        merge_same_name_groups=False,
        show_progress=True,
    )
    print(result)


if __name__ == "__main__":
    main()

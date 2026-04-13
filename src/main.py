"""Application entrypoint for running the incremental MVP clustering demo."""

from __future__ import annotations

import json
import logging

try:
    from .incremental_mvp_clustering import IncrementalMVPClusteringPipeline
    from .model import embeddings, model as llm
except ImportError:
    from incremental_mvp_clustering import IncrementalMVPClusteringPipeline
    from model import embeddings, model as llm


DEMO_COMMENTS = [
    {"comment_id": "c1", "text": "Не могу перевести деньги"},
    {"comment_id": "c2", "text": "Перевод денег не проходит"},
    {"comment_id": "c3", "text": "Не проходит перевод между своими счетами"},
    {"comment_id": "c4", "text": "Перевод в компанию А блокируется"},
    {"comment_id": "c5", "text": "Не могу перевести в компанию Б"},
    {"comment_id": "c6", "text": "Проблема с переводами по СБП"},
    {"comment_id": "c7", "text": "Не могу войти в приложение"},
    {"comment_id": "c8", "text": "Приложение не открывается"},
    {"comment_id": "c9", "text": "Вход в приложение через Face ID не работает"},
    {"comment_id": "c10", "text": "..."},
]


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def build_console_output(result: dict) -> dict:
    """Return a console-friendly copy without embeddings."""
    comments = [
        {key: value for key, value in comment.items() if key != "embedding"}
        for comment in result.get("comments", [])
    ]
    return {
        "comments": comments,
        "groups": result.get("groups", []),
    }


def main() -> None:
    """Create the MVP pipeline and run the bundled demo dataset."""
    configure_logging()

    pipeline = IncrementalMVPClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
    )
    result = pipeline.run(DEMO_COMMENTS)
    print(json.dumps(build_console_output(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

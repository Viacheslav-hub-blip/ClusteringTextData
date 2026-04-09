"""Entrypoint for the universal text clustering pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

try:
    from .pipeline import UniversalTextClusteringPipeline
except ImportError:
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from src.universal_text_clustering.pipeline import UniversalTextClusteringPipeline

try:
    from ..model import embeddings, model as llm
except ImportError:
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from src.model import embeddings, model as llm


DEMO_COMMENTS = [
    {"comment_id": "c1", "text": "Не могу перевести деньги"},
    {"comment_id": "c2", "text": "Не проходит перевод между своими счетами"},
    {"comment_id": "c3", "text": "Перевод в компанию А блокируется"},
    {"comment_id": "c4", "text": "Перевод денег не проходит"},
    {"comment_id": "c5", "text": "Не могу перевести в компанию Б"},
    {"comment_id": "c6", "text": "Проблема с переводами по СБП"},
    {"comment_id": "c7", "text": "не могу войти в приложение"},
    {"comment_id": "c8", "text": "приложение не открывается"},
    {"comment_id": "c9", "text": "вход в приложение через Face ID не работает"},
]


def configure_logging() -> None:
    """Enable console logs for pipeline progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def main() -> None:
    """Run the universal pipeline on demo data."""
    configure_logging()
    pipeline = UniversalTextClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
        extraction_batch_size=50,
        dense_top_k=10,
    )
    result = pipeline.run(DEMO_COMMENTS)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

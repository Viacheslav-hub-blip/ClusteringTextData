"""CLI entrypoint for the stable structured clustering pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from ..model import embeddings, model as llm
except ImportError:
    from src.model import embeddings, model as llm

from .pipeline import StableStructuredClusteringPipeline
from .services.excel_loader import load_comments_from_excel

DEMO_COMMENTS = [
    {"comment_id": "1", "text": "Сообщения приходят слишком часто, хочется отключить пуши"},
    {"comment_id": "2", "text": "Постоянные сообщения мешают, пользы от них мало"},
    {"comment_id": "3", "text": "Заблокировали оплату без причины"},
    {"comment_id": "4", "text": "Оплата прошла быстро и без проблем"},
]


def configure_logging() -> None:
    """Configure package logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the stable structured clustering pipeline",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        help="Path to an Excel file with comment_id/comment columns",
    )
    parser.add_argument(
        "--text-column",
        default="comment",
        help="Name of the text column in the Excel file",
    )
    parser.add_argument(
        "--id-column",
        default="comment_id",
        help="Name of the id column in the Excel file",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for loaded rows",
    )
    return parser.parse_args()


def main() -> None:
    """Run the stable structured clustering pipeline."""
    configure_logging()
    args = parse_args()

    if args.excel:
        comments = load_comments_from_excel(
            args.excel,
            text_column=args.text_column,
            id_column=args.id_column,
            sheet_name=args.sheet,
            limit=args.limit,
        )
    else:
        comments = DEMO_COMMENTS

    pipeline = StableStructuredClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
        extraction_batch_size=50,
        candidate_top_k=8,
    )
    result = pipeline.run(comments)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

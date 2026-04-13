"""CLI entrypoint for the incremental MVP clustering pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__:
    from ..model import embeddings, model as llm
    from .pipeline import IncrementalMVPClusteringPipeline
else:
    CURRENT_FILE = Path(__file__).resolve()
    SRC_DIR = CURRENT_FILE.parents[1]
    PROJECT_ROOT = CURRENT_FILE.parents[2]
    for path in (str(PROJECT_ROOT), str(SRC_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from src.incremental_mvp_clustering.pipeline import IncrementalMVPClusteringPipeline
    from src.model import embeddings, model as llm

DEMO_COMMENTS = [
    {"comment_id": "1", "text": "Не могу перевести деньги"},
    {"comment_id": "2", "text": "Перевод денег не проходит"},
    {"comment_id": "3", "text": "Не проходит перевод между своими счетами"},
    {"comment_id": "4", "text": "Не могу войти в приложение"},
    {"comment_id": "5", "text": "Приложение не открывается"},
    {"comment_id": "6", "text": "..."},
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
        description="Run the incremental MVP clustering pipeline",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Path to a JSON file with a list of {comment_id, text} objects",
    )
    return parser.parse_args()


def load_comments(json_path: Path | None) -> list[dict]:
    """Load comments from JSON or use demo data."""
    if not json_path:
        return DEMO_COMMENTS
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        msg = "Expected a JSON array of comment objects"
        raise ValueError(msg)
    return payload


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
    """Run the incremental MVP clustering demo."""
    configure_logging()
    args = parse_args()
    comments = load_comments(args.json)
    pipeline = IncrementalMVPClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
    )
    df = pd.read_excel(r"C:\Users\Slav4ik\PycharmProjects\ClusteringTextData\data\comments_1000.xlsx")
    df = df.iloc[:10]
    # print(df)
    df_comments = [
        {"comment_id": str(row["comment_id"]), "text": row["comment"]} for idx, row in df.iterrows()
    ]
    result = pipeline.run(df_comments)
    print(json.dumps(build_console_output(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""CLI entrypoint for the incremental MVP clustering pipeline."""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path

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
    comments_by_group_id: dict[str, list[dict]] = {}
    for comment in comments:
        group_id = str(comment.get("group_id", "")).strip()
        if not group_id:
            continue
        comments_by_group_id.setdefault(group_id, []).append(
            {
                "comment_id": comment["comment_id"],
                "raw_text": comment["raw_text"],
                "normalized_text": comment["normalized_text"],
            }
        )

    groups = []
    for group in result.get("groups", []):
        group_id = group.get("group_id", "")
        groups.append(
            {
                **group,
                "member_comments": comments_by_group_id.get(group_id, []),
            }
        )

    return {
        "comments": comments,
        "groups": groups,
    }


async def amain() -> None:
    """Run the incremental MVP clustering demo."""
    configure_logging()
    args = parse_args()
    comments = load_comments(args.json)
    pipeline = IncrementalMVPClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
    )
    result = await pipeline.arun(comments)
    print(json.dumps(build_console_output(result), ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()

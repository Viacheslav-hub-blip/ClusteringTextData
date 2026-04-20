"""Entrypoint for running the incremental MVP clustering pipeline on the full demo dataset."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

if __package__:
    from ..model import embeddings, model as llm
    from .agentic_post_processing import AgenticPostProcessingPipeline
    from .pipeline import IncrementalMVPClusteringPipeline
else:
    CURRENT_FILE = Path(__file__).resolve()
    SRC_DIR = CURRENT_FILE.parents[1]
    PROJECT_ROOT = CURRENT_FILE.parents[2]
    for path in (str(PROJECT_ROOT), str(SRC_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from src.incremental_mvp_clustering.agentic_post_processing import AgenticPostProcessingPipeline
    from src.incremental_mvp_clustering.pipeline import IncrementalMVPClusteringPipeline
    from src.model import embeddings, model as llm

DEMO_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "comments_1000_utf8.csv"


def configure_logging() -> None:
    """Configure package logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_comments() -> list[dict]:
    """Load all comments from the demo CSV file."""
    comments: list[dict[str, str]] = []
    with DEMO_DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            comment_id = str(row.get("comment_id", "")).strip()
            text = str(row.get("comment", "")).strip()
            if not comment_id or not text:
                continue
            comments.append({"comment_id": comment_id, "text": text})
    return comments


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

    output = {
        "comments": comments,
        "groups": groups,
    }
    if "post_processing" in result:
        output["post_processing"] = result["post_processing"]
    return output


async def amain() -> None:
    """Run the incremental MVP clustering pipeline on the demo CSV."""
    configure_logging()
    comments = load_comments()
    comments = comments[:20]
    primary_pipeline = IncrementalMVPClusteringPipeline(
        llm=llm,
        embeddings=embeddings,
    )
    logging.info("Primary clustering started: %d comments", len(comments))
    primary_result = await primary_pipeline.arun(comments)
    logging.info(
        "Primary clustering finished: %d comments, %d groups",
        len(primary_result.get("comments", [])),
        len(primary_result.get("groups", [])),
    )
    logging.info("Agentic post-processing started")
    post_processing_pipeline = AgenticPostProcessingPipeline(
        llm=llm,
        audit_batch_size=2
    )
    refined_result = await post_processing_pipeline.arun(primary_result)
    logging.info(
        "Agentic post-processing finished: %d final groups",
        len(refined_result.get("groups", [])),
    )
    print(json.dumps(build_console_output(refined_result), ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()

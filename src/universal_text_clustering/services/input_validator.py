"""Input validation service."""

from __future__ import annotations

import logging

from ..models import InputComment

logger = logging.getLogger(__name__)


class InputValidator:
    """Validate raw input records."""

    def validate(self, raw_comments: list[dict]) -> list[InputComment]:
        """Validate comment ids and non-empty text."""
        validated: list[InputComment] = []

        for raw in raw_comments:
            comment_id = str(raw.get("comment_id", "")).strip()
            text = str(raw.get("text", "")).strip()

            if not comment_id:
                logger.warning("Skipped record without comment_id: %s", raw)
                continue

            if not text:
                logger.warning("Skipped empty comment: %s", comment_id)
                continue

            validated.append(InputComment(comment_id=comment_id, text=text))

        return validated

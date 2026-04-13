"""Input validation for the stable structured clustering pipeline."""

from __future__ import annotations

import logging

from ..models import InputComment

logger = logging.getLogger(__name__)


class InputValidator:
    """Validate raw input comment records."""

    def validate(self, raw_comments: list[dict]) -> list[InputComment]:
        """Validate comment ids and non-empty text."""
        validated: list[InputComment] = []

        for index, raw in enumerate(raw_comments, start=1):
            comment_id = str(raw.get("comment_id", "")).strip() or str(index)
            text = str(raw.get("text", "")).strip()
            if not text:
                logger.warning("Skipped empty comment: %s", comment_id)
                continue
            validated.append(InputComment(comment_id=comment_id, text=text))

        return validated

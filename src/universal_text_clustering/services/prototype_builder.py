"""Prototype building service."""

from __future__ import annotations

from collections import defaultdict

from ..models import Prototype, SemanticFrame


class PrototypeBuilder:
    """Build prototypes by exact canonical key match."""

    def build(
        self,
        frames: list[SemanticFrame],
    ) -> tuple[list[Prototype], dict[str, str]]:
        """Group frames by canonical key and return prototypes plus comment mapping."""
        groups: dict[str, list[SemanticFrame]] = defaultdict(list)
        for frame in frames:
            key = frame.canonical_key or frame.raw_text.strip().lower()
            groups[key].append(frame)

        prototypes: list[Prototype] = []
        comment_to_prototype: dict[str, str] = {}

        for index, (canonical_key, group_frames) in enumerate(groups.items(), start=1):
            representative = group_frames[0]
            prototype_id = f"prototype_{index}"
            member_comment_ids = [frame.comment_id for frame in group_frames]
            prototype = Prototype(
                prototype_id=prototype_id,
                canonical_key=canonical_key,
                representative_frame=representative,
                member_comment_ids=member_comment_ids,
            )
            prototypes.append(prototype)
            for comment_id in member_comment_ids:
                comment_to_prototype[comment_id] = prototype_id

        return prototypes, comment_to_prototype

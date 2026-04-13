"""Prototype builder for exact specific-key matches."""

from __future__ import annotations

from collections import defaultdict

from ..models import SpecificPrototype, StructuredSignal


class PrototypeBuilder:
    """Build exact prototypes from identical specific keys."""

    def build(
        self,
        signals: list[StructuredSignal],
    ) -> tuple[list[SpecificPrototype], dict[str, str]]:
        """Return prototypes plus comment-to-prototype mapping."""
        groups: dict[tuple[str, str], list[StructuredSignal]] = defaultdict(list)
        for signal in signals:
            key = (signal.polarity, signal.specific_key)
            groups[key].append(signal)

        prototypes: list[SpecificPrototype] = []
        comment_to_prototype: dict[str, str] = {}

        for index, group_signals in enumerate(groups.values(), start=1):
            prototype_id = f"specific_prototype_{index}"
            representative_signal = group_signals[0]
            member_comment_ids = [signal.comment_id for signal in group_signals]
            prototypes.append(
                SpecificPrototype(
                    prototype_id=prototype_id,
                    representative_signal=representative_signal,
                    member_comment_ids=member_comment_ids,
                )
            )
            for comment_id in member_comment_ids:
                comment_to_prototype[comment_id] = prototype_id

        return prototypes, comment_to_prototype

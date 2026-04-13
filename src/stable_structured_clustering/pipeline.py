"""Stable structured clustering pipeline orchestration."""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .models import OutputRecord, ParentCluster, SpecificCluster, SpecificPrototype
from .services.input_validator import InputValidator
from .services.label_selector import LabelSelector
from .services.output_builder import OutputBuilder
from .services.parent_cluster_builder import ParentClusterBuilder
from .services.prototype_builder import PrototypeBuilder
from .services.specific_cluster_builder import SpecificClusterBuilder
from .services.structure_extractor import StructureExtractor

logger = logging.getLogger(__name__)


class StableStructuredClusteringPipeline:
    """Batch-oriented stable clustering based on structured semantic signals."""

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        extraction_batch_size: int = 50,
        candidate_top_k: int = 8,
    ):
        self._validator = InputValidator()
        self._extractor = StructureExtractor(llm)
        self._prototype_builder = PrototypeBuilder()
        self._specific_cluster_builder = SpecificClusterBuilder(
            embeddings=embeddings,
            top_k=candidate_top_k,
        )
        self._parent_cluster_builder = ParentClusterBuilder(
            embeddings=embeddings,
            top_k=candidate_top_k,
        )
        self._label_selector = LabelSelector()
        self._output_builder = OutputBuilder()
        self._extraction_batch_size = extraction_batch_size

    def run(self, raw_comments: list[dict]) -> list[dict]:
        """Run the stable batch clustering pipeline."""
        logger.info("Stable pipeline started: received %d raw comments", len(raw_comments))
        comments = self._validator.validate(raw_comments)
        if not comments:
            logger.warning("Stable pipeline stopped: no valid comments")
            return []

        logger.info("Stable stage 1/5: structured extraction")
        signals = self._extractor.extract_batch(
            comments,
            batch_size=self._extraction_batch_size,
        )

        logger.info("Stable stage 2/5: exact prototypes")
        prototypes, comment_to_prototype = self._prototype_builder.build(signals)
        prototypes_by_id: dict[str, SpecificPrototype] = {
            prototype.prototype_id: prototype for prototype in prototypes
        }

        logger.info("Stable stage 3/5: specific clustering")
        specific_clusters, prototype_to_specific_cluster = self._specific_cluster_builder.build(
            prototypes
        )
        specific_clusters_by_id: dict[str, SpecificCluster] = {
            cluster.specific_cluster_id: cluster for cluster in specific_clusters
        }

        logger.info("Stable stage 4/5: parent clustering")
        parent_clusters_by_id: dict[str, ParentCluster] = self._parent_cluster_builder.build(
            specific_clusters,
            prototypes_by_id,
        )

        logger.info("Stable stage 5/5: label selection and output assignment")
        self._label_selector.assign_labels(
            specific_clusters=specific_clusters,
            parent_clusters_by_id=parent_clusters_by_id,
            prototypes_by_id=prototypes_by_id,
        )
        output_records = self._output_builder.build(
            comment_to_prototype=comment_to_prototype,
            prototype_to_specific_cluster=prototype_to_specific_cluster,
            specific_clusters_by_id=specific_clusters_by_id,
            parent_clusters_by_id=parent_clusters_by_id,
        )
        outputs_by_id: dict[str, OutputRecord] = {
            output.comment_id: output for output in output_records
        }

        return [
            {
                "comment_id": comment.comment_id,
                "specific_group": outputs_by_id[comment.comment_id].specific_group,
                "parent_group": outputs_by_id[comment.comment_id].parent_group,
            }
            for comment in comments
            if comment.comment_id in outputs_by_id
        ]

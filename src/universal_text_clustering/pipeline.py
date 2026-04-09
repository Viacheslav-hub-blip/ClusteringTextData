"""Universal text clustering pipeline orchestration."""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .models import OutputRecord, ParentCluster, Prototype, SpecificCluster
from .services.dense_retriever import DensePrototypeRetriever
from .services.group_name_generator import GroupNameGenerator
from .services.input_validator import InputValidator
from .services.label_assigner import LabelAssigner
from .services.parent_cluster_builder import ParentClusterBuilder
from .services.prototype_builder import PrototypeBuilder
from .services.relation_classifier import PairRelationClassifier
from .services.semantic_extractor import SemanticExtractor
from .services.specific_cluster_builder import SpecificClusterBuilder

logger = logging.getLogger(__name__)


class UniversalTextClusteringPipeline:
    """Universal LLM-first text clustering pipeline with LangChain dense retrieval."""

    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        extraction_batch_size: int = 50,
        dense_top_k: int = 10,
    ):
        self._validator = InputValidator()
        self._extractor = SemanticExtractor(llm)
        self._prototype_builder = PrototypeBuilder()
        self._retriever = DensePrototypeRetriever(embeddings=embeddings, top_k=dense_top_k)
        self._relation_classifier = PairRelationClassifier(llm)
        self._specific_cluster_builder = SpecificClusterBuilder()
        self._parent_cluster_builder = ParentClusterBuilder()
        self._name_generator = GroupNameGenerator(llm)
        self._label_assigner = LabelAssigner()
        self._extraction_batch_size = extraction_batch_size

    def run(self, raw_comments: list[dict]) -> list[dict]:
        """Run the full universal clustering pipeline."""
        logger.info("Pipeline started: received %d raw comments", len(raw_comments))
        comments = self._validator.validate(raw_comments)
        if not comments:
            logger.warning("Pipeline stopped: no valid comments")
            return []
        logger.info("Validation completed: %d comments accepted", len(comments))

        logger.info("Stage 1/6: semantic extraction")
        frames = self._extractor.extract_batch(
            comments,
            batch_size=self._extraction_batch_size,
        )

        logger.info("Stage 2/6: prototype building")
        prototypes, comment_to_prototype = self._prototype_builder.build(frames)
        prototypes_by_id: dict[str, Prototype] = {
            prototype.prototype_id: prototype for prototype in prototypes
        }
        logger.info(
            "Prototype building completed: %d comments -> %d prototypes",
            len(comments),
            len(prototypes),
        )

        logger.info("Stage 3/6: dense retrieval")
        if len(prototypes) > 1:
            self._retriever.build_index(prototypes)
            candidate_map = self._retriever.retrieve_all(prototypes)
            logger.info(
                "Dense retrieval completed for %d prototypes",
                len(candidate_map),
            )
            logger.info("Stage 4/6: pair relation classification")
            decisions = self._relation_classifier.classify_candidates(prototypes, candidate_map)
        else:
            decisions = []
            logger.info("Skipped retrieval and pair classification: only one prototype")

        logger.info("Stage 5/6: specific and parent clustering")
        specific_clusters, prototype_to_specific_cluster = (
            self._specific_cluster_builder.build(prototypes, decisions)
        )
        specific_clusters_by_id: dict[str, SpecificCluster] = {
            cluster.specific_cluster_id: cluster for cluster in specific_clusters
        }

        for cluster in specific_clusters:
            frame = prototypes_by_id[cluster.representative_prototype_id].representative_frame
            cluster.specific_group = self._name_generator.generate_specific_group(frame)

        parent_links = self._parent_cluster_builder.build_parent_links(
            decisions,
            prototype_to_specific_cluster,
        )
        parent_clusters = self._parent_cluster_builder.build_parent_clusters(
            specific_clusters=specific_clusters,
            parent_links=parent_links,
            prototypes_by_id=prototypes_by_id,
        )
        parent_clusters_by_id: dict[str, ParentCluster] = {
            cluster.parent_cluster_id: cluster for cluster in parent_clusters
        }

        logger.info("Stage 6/6: naming and label assignment")
        for parent_cluster in parent_clusters:
            child_specific_groups: list[str] = []
            child_general_topics: list[str] = []

            for specific_cluster_id in parent_cluster.child_specific_cluster_ids:
                specific_cluster = specific_clusters_by_id[specific_cluster_id]
                child_specific_groups.append(specific_cluster.specific_group)
                frame = prototypes_by_id[specific_cluster.representative_prototype_id].representative_frame
                child_general_topics.append(frame.general_topic)

            parent_cluster.parent_group = self._name_generator.generate_parent_group(
                specific_groups=child_specific_groups,
                general_topics=child_general_topics,
            )

        output_records = self._label_assigner.assign(
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

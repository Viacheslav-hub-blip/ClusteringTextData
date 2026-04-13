"""Universal text clustering pipeline orchestration."""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ..label_consolidation import SemanticLabelConsolidator
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
        self._label_consolidator = SemanticLabelConsolidator(embeddings)
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

        if len(specific_clusters) > 1:
            specific_reconciliation_input: list[str] = []
            for cluster in specific_clusters:
                frame = prototypes_by_id[
                    cluster.representative_prototype_id
                ].representative_frame
                specific_reconciliation_input.append(
                    "\n".join(
                        [
                            f"specific_cluster_id: {cluster.specific_cluster_id}",
                            f"current_specific_group: {cluster.specific_group}",
                            f"parent_key: {frame.parent_key}",
                            f"general_topic: {frame.general_topic}",
                            f"core_case: {frame.core_case}",
                            f"exact_case: {frame.exact_case}",
                            f"key_qualifiers: {frame.key_qualifiers}",
                            f"context_details: {frame.context_details}",
                            f"entities: {frame.entities}",
                        ]
                    )
                )

            reconciled_specific_groups = self._name_generator.reconcile_specific_groups(
                specific_reconciliation_input
            )
            for cluster in specific_clusters:
                reconciled_specific_group = reconciled_specific_groups.get(
                    cluster.specific_cluster_id
                )
                if reconciled_specific_group:
                    cluster.specific_group = reconciled_specific_group

        consolidated_specific_groups = self._label_consolidator.consolidate(
            specific_clusters,
            item_id_getter=lambda cluster: cluster.specific_cluster_id,
            label_getter=lambda cluster: cluster.specific_group,
            semantic_text_getter=lambda cluster: " | ".join(
                [
                    cluster.specific_group,
                    prototypes_by_id[
                        cluster.representative_prototype_id
                    ].representative_frame.parent_key,
                    prototypes_by_id[
                        cluster.representative_prototype_id
                    ].representative_frame.general_topic,
                    prototypes_by_id[
                        cluster.representative_prototype_id
                    ].representative_frame.core_case,
                    prototypes_by_id[
                        cluster.representative_prototype_id
                    ].representative_frame.canonical_key,
                ]
            ),
            family_key_getter=lambda cluster: (
                prototypes_by_id[
                    cluster.representative_prototype_id
                ].representative_frame.parent_key
                or prototypes_by_id[
                    cluster.representative_prototype_id
                ].representative_frame.general_topic
            ),
            size_getter=lambda cluster: len(cluster.member_comment_ids),
            min_similarity=0.90,
            min_token_overlap=0.5,
        )
        for cluster in specific_clusters:
            consolidated_specific_group = consolidated_specific_groups.get(
                cluster.specific_cluster_id
            )
            if consolidated_specific_group:
                cluster.specific_group = consolidated_specific_group

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
            child_cluster_descriptions: list[str] = []
            child_general_topics: list[str] = []

            for specific_cluster_id in parent_cluster.child_specific_cluster_ids:
                specific_cluster = specific_clusters_by_id[specific_cluster_id]
                frame = prototypes_by_id[specific_cluster.representative_prototype_id].representative_frame
                child_general_topics.append(frame.parent_key or frame.general_topic)
                child_cluster_descriptions.append(
                    "\n".join(
                        [
                            f"- specific_group: {specific_cluster.specific_group}",
                            f"- parent_key: {frame.parent_key}",
                            f"- general_topic: {frame.general_topic}",
                            f"- core_case: {frame.core_case}",
                            f"- exact_case: {frame.exact_case}",
                            f"- key_qualifiers: {frame.key_qualifiers}",
                            f"- context_details: {frame.context_details}",
                            f"- entities: {frame.entities}",
                        ]
                    )
                )

            parent_cluster.parent_group = self._name_generator.generate_parent_group(
                child_cluster_descriptions=child_cluster_descriptions,
                general_topics=child_general_topics,
            )

        if len(parent_clusters) > 1:
            parent_reconciliation_input: list[str] = []
            for parent_cluster in parent_clusters:
                child_lines: list[str] = []
                for specific_cluster_id in parent_cluster.child_specific_cluster_ids:
                    specific_cluster = specific_clusters_by_id[specific_cluster_id]
                    frame = prototypes_by_id[
                        specific_cluster.representative_prototype_id
                    ].representative_frame
                    child_lines.append(
                        "\n".join(
                            [
                                f"  - specific_group: {specific_cluster.specific_group}",
                                f"    parent_key: {frame.parent_key}",
                                f"    general_topic: {frame.general_topic}",
                                f"    core_case: {frame.core_case}",
                                f"    key_qualifiers: {frame.key_qualifiers}",
                                f"    context_details: {frame.context_details}",
                                f"    entities: {frame.entities}",
                            ]
                        )
                    )
                parent_reconciliation_input.append(
                    "\n".join(
                        [
                            f"parent_cluster_id: {parent_cluster.parent_cluster_id}",
                            f"current_parent_group: {parent_cluster.parent_group}",
                            f"representative_topic: {parent_cluster.representative_topic}",
                            "children:",
                            *child_lines,
                        ]
                    )
                )

            reconciled_parent_groups = self._name_generator.reconcile_parent_groups(
                parent_reconciliation_input
            )
            for parent_cluster in parent_clusters:
                reconciled_parent_group = reconciled_parent_groups.get(
                    parent_cluster.parent_cluster_id
                )
                if reconciled_parent_group:
                    parent_cluster.parent_group = reconciled_parent_group

        consolidated_parent_groups = self._label_consolidator.consolidate(
            parent_clusters,
            item_id_getter=lambda cluster: cluster.parent_cluster_id,
            label_getter=lambda cluster: cluster.parent_group,
            semantic_text_getter=lambda cluster: " | ".join(
                [
                    cluster.parent_group,
                    cluster.representative_topic,
                    " | ".join(
                        specific_clusters_by_id[specific_cluster_id].specific_group
                        for specific_cluster_id in cluster.child_specific_cluster_ids
                    ),
                ]
            ),
            family_key_getter=lambda cluster: cluster.representative_topic,
            size_getter=lambda cluster: len(cluster.child_specific_cluster_ids),
            min_similarity=0.90,
            min_token_overlap=0.5,
        )
        for parent_cluster in parent_clusters:
            consolidated_parent_group = consolidated_parent_groups.get(
                parent_cluster.parent_cluster_id
            )
            if consolidated_parent_group:
                parent_cluster.parent_group = consolidated_parent_group

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

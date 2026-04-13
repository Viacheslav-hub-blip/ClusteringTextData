"""Candidate retrieval for structured signatures."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class CandidateRetriever:
    """Retrieve nearest structured candidates and cache vectors."""

    def __init__(self, embeddings: Embeddings, top_k: int = 8):
        self._embeddings = embeddings
        self._top_k = top_k

    def retrieve(
        self,
        texts_by_id: dict[str, str],
    ) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
        """Return candidate ids and cached vectors for the same texts."""
        if len(texts_by_id) <= 1:
            return {item_id: [] for item_id in texts_by_id}, {}

        ordered_ids = list(texts_by_id)
        ordered_texts = [texts_by_id[item_id] for item_id in ordered_ids]
        vectors = self._embeddings.embed_documents(ordered_texts)
        vectors_by_id = {
            item_id: vector for item_id, vector in zip(ordered_ids, vectors, strict=True)
        }

        vectorstore = FAISS.from_texts(
            ordered_texts,
            self._embeddings,
            metadatas=[{"item_id": item_id} for item_id in ordered_ids],
        )
        candidate_map: dict[str, list[str]] = {}

        for item_id in ordered_ids:
            query_vector = vectors_by_id[item_id]
            limit = min(len(ordered_ids), self._top_k + 1)
            if hasattr(vectorstore, "similarity_search_with_score_by_vector"):
                hits = vectorstore.similarity_search_with_score_by_vector(
                    query_vector,
                    k=limit,
                )
            else:
                hits = [
                    (document, 0.0)
                    for document in vectorstore.similarity_search_by_vector(
                        query_vector,
                        k=limit,
                    )
                ]
            candidate_ids: list[str] = []
            for document, _score in hits:
                candidate_id = str(document.metadata.get("item_id", "")).strip()
                if candidate_id and candidate_id != item_id and candidate_id not in candidate_ids:
                    candidate_ids.append(candidate_id)
            candidate_map[item_id] = candidate_ids

        return candidate_map, vectors_by_id

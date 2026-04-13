"""Dense retriever built on top of LangChain vector store retrievers."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from ..models import Prototype


class DensePrototypeRetriever:
    """Retrieve semantically close prototype candidates using LangChain retrievers."""

    def __init__(self, embeddings: Embeddings, top_k: int = 10):
        self._embeddings = embeddings
        self._top_k = top_k
        self._vectorstore: FAISS | None = None
        self._retriever: VectorStoreRetriever | None = None

    @staticmethod
    def _prototype_text(prototype: Prototype) -> str:
        frame = prototype.representative_frame
        return " | ".join(
            [
                frame.general_topic,
                frame.parent_key,
                frame.core_case,
                frame.exact_case,
                " ".join(frame.key_qualifiers),
                " ".join(frame.context_details),
                " ".join(frame.entities),
                frame.canonical_key,
            ]
        )

    def build_index(self, prototypes: list[Prototype]) -> None:
        """Build FAISS index and expose it through LangChain retriever."""
        documents = [
            Document(
                page_content=self._prototype_text(prototype),
                metadata={"prototype_id": prototype.prototype_id},
            )
            for prototype in prototypes
        ]
        self._vectorstore = FAISS.from_documents(documents, self._embeddings)
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._top_k + 1},
        )

    def retrieve_for_prototype(self, prototype: Prototype) -> list[str]:
        """Return candidate prototype ids for one prototype."""
        if self._retriever is None:
            raise RuntimeError("Retriever index has not been built yet.")

        query = self._prototype_text(prototype)
        documents = self._retriever.invoke(query)
        prototype_ids = [
            document.metadata["prototype_id"]
            for document in documents
            if document.metadata.get("prototype_id") != prototype.prototype_id
        ]
        return list(dict.fromkeys(prototype_ids))

    def retrieve_all(self, prototypes: list[Prototype]) -> dict[str, list[str]]:
        """Return candidate maps for all prototypes."""
        return {
            prototype.prototype_id: self.retrieve_for_prototype(prototype)
            for prototype in prototypes
        }

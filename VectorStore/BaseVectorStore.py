from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from typing import List, Any, Optional, Dict
from llama_index.core.schema import TextNode, BaseNode
import os


class BaseVectorStore(BasePydanticVectorStore):
    """Simple custom Vector Store.

    Stores documents in a simple in-memory dict.

    """

    stores_text: bool = True

    def get(self, text_id: str) -> List[float]:
        pass

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        pass

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        pass

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        pass

    def persist(self, persist_path, fs=None) -> None:
        pass
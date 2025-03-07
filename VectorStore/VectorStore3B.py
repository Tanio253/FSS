from VectorStore.VectorStore2 import VectorStore2
from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from typing import Any
from VectorStore.utils import filter_nodes, dense_search

class VectorStore3B(VectorStore2):


    def client(self):
        return None
    
    """Implements Metadata Filtering."""
    
    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        # 1. First filter by metadata
        nodes = self.node_dict.values()
        if query.filters is not None:
            nodes = filter_nodes(nodes, query.filters)
        if len(nodes) == 0:
            result_nodes = []
            similarities = []
            node_ids = []
        else:
            # 2. Then perform semantic search
            similarities, node_ids = dense_search(query, nodes)
            result_nodes = [self.node_dict[node_id] for node_id in node_ids]
        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )
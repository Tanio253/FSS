from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
import asyncio
from utils import run_queries
from typing import List
from utils import fuse_results, generate_queries
class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries = generate_queries(
            self._llm, query_bundle.query_str, num_queries=4
        )

        queries = [q for q in queries if q != ""]
        print(queries)
        # queries = [query_bundle.query_str]
        results = asyncio.run(run_queries(queries, self._retrievers))
        final_results = fuse_results(
            results, similarity_top_k=self._similarity_top_k
        )

        return final_results
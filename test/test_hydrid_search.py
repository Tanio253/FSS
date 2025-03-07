from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from VectorStore.VectorStore3B import VectorStore3B
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from utils import run_queries, fuse_results
import asyncio

from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)
import Stemmer

EMBED_MODEL_PATH = "intfloat/multilingual-e5-small"
queries = ["Tahukah Anda posisi Owen Bennett?"]
loader = PyMuPDFReader()
documents = loader.load(file_path="./FoodServingSystemIndo.pdf")

embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH)
node_parser = SentenceSplitter(chunk_size=256)
nodes = node_parser.get_nodes_from_documents(documents)

# Semantic Chunking (I realize it work not well)
# node_parser = SemanticSplitterNodeParser(embed_model = embed_model, include_metadata = True,include_prev_next_rel=True, buffer_size = 10, breakpoint_percentile_threshold = 70)
# nodes = node_parser.build_semantic_nodes_from_documents(documents)


for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

vector_store = VectorStore3B()
# load data into the vector stores
vector_store.add(nodes)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model = embed_model)
vector_retriever = index.as_retriever(similarity_top_k = 3)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes, 
    similarity_top_k=3,
    # stemmer=Stemmer.Stemmer("english"),
    # language="english"
)

results_dict = asyncio.run(run_queries(queries, [vector_retriever, bm25_retriever]))
final_results = fuse_results(results_dict)
for n in final_results:
    print(n.score, "\n", n.text, "\n********\n")
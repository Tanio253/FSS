from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from VectorStore.VectorStore3B import VectorStore3B
from llama_index.core.vector_stores import (
    VectorStoreQuery,
    VectorStoreQueryResult,
)

EMBED_MODEL_PATH = "intfloat/multilingual-e5-small"
loader = PyMuPDFReader()
documents = loader.load(file_path="./FoodServingSystemIndo.pdf")

node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH)

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

vector_store = VectorStore3B()
# load data into the vector stores
vector_store.add(nodes)


query_str = "Cara membuat kentang goreng"
query_embedding = embed_model.get_query_embedding(query_str)

query_obj = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5
)

query_result = vector_store.query(query_obj)

for similarity, node in zip(query_result.similarities, query_result.nodes):
    print(
        "\n----------------\n"
        f"[Node ID {node.node_id}] Similarity: {similarity}\n\n"
        f"{node.get_content(metadata_mode='all')}"
        "\n----------------\n\n"
    )
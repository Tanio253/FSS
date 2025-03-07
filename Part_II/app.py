import streamlit as st
from pathlib import Path  
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from VectorStore.VectorStore3B import VectorStore3B
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from Retriever.retriever import FusionRetriever
from llama_index.core.settings import Settings
from dotenv import load_dotenv
import os
from llama_index.llms.gemini import Gemini

st.set_page_config(page_title="Food Serving System", layout="wide")
st.title("Food Serving System")
st.markdown("Enter your query and get relevant information from the document.")

# Sample queries
sample_queries = [
    "Berapa pendapatan Restoran Ayam Spice Route?",
    "Apa cerita di balik Golden Crust Delights?",
    "Apa pekerjaan Olivia Chen?",
    "Siapa pemilik Seaside Bites?",
    "Ceritakan kisah tentang French Fry Kingdom?",
    "Berapa harga es krim?",
    "Bagaimana cara membuat nugget ayam?",
    "Beritahu saya bahan-bahan untuk membuat kentang goreng?",
    "Apa itu spageti?",
    "Berapa nomor telepon Zoe Bailey?"
]

with st.sidebar:
    st.header("Settings")
    default_file_path = "Part_I/Thanh.pdf"
    
    # Optional Chunk Size and Top-K returned
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size", min_value=128, max_value=512, value=256, step=32)
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=3, step=1)
    
    st.header("Sample Queries")
    st.write("Click on any sample query to use it:")

    for i, query in enumerate(sample_queries):
        if st.button(f"Sample {i+1}", key=f"sample_{i}"):
            st.session_state.query = query

@st.cache_resource
def initialize_query_engine(file_path, chunk_size=256, top_k=3):
    load_dotenv()
    EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH")
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
    
    loader = PyMuPDFReader()
    documents = loader.load(file_path=file_path)
    
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH)
    
    node_parser = SentenceSplitter(chunk_size=chunk_size)
    nodes = node_parser.get_nodes_from_documents(documents)
    
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    
    vector_store = VectorStore3B()
    vector_store.add(nodes)
    
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
    )
    
    llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key=GOOGLE_GEMINI_API_KEY
    )
    
    # Since Settings.llm = None, we must set it.
    Settings.llm = llm
    fusion_retriever = FusionRetriever(
        llm, [vector_retriever, bm25_retriever], similarity_top_k=top_k
    )
    
    query_engine = RetrieverQueryEngine(fusion_retriever)
    
    return query_engine

file_path = default_file_path
with st.spinner("Initializing query engine..."):
    query_engine = initialize_query_engine(
        file_path=file_path,
        chunk_size=chunk_size if 'chunk_size' in locals() else 256,
        top_k=top_k if 'top_k' in locals() else 3
    )
    st.success(f"Query engine initialized using file: {file_path.split('/')[-1]}")

if 'query' not in st.session_state:
    st.session_state.query = ""

st.subheader("Enter your query")

sample_selection = st.selectbox(
    "Or select a sample query:",
    [""] + sample_queries,
    key="sample_dropdown"
)

if sample_selection:
    st.session_state.query = sample_selection

query = st.text_area("Query", height=100, value=st.session_state.query, placeholder="Enter your question here...")

if st.button("Submit Query") and query:
    with st.spinner("Processing query..."):
        try:
            response = query_engine.query(query)
        
            st.subheader("Response")
            st.write(str(response))
            
            # Display the retrieved context
            with st.expander("View Source Nodes"):
                for idx, node in enumerate(response.source_nodes):
                    st.markdown(f"**Source {idx+1}**")
                    st.markdown(f"Score: {node.score:.4f}")
                    st.markdown(f"Content: {node.node.get_content()}")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

st.subheader("Sample Queries")
cols = st.columns(2)
for i, query in enumerate(sample_queries):
    col_idx = i % 2
    with cols[col_idx]:
        if st.button(query, key=f"main_sample_{i}"):
            st.session_state.query = query

if __name__ == "__main__":
    pass
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import asyncio
from langchain_community.document_loaders import PyPDFLoader
FILE_PATH = "./FoodServingSystemIndo.pdf"
EMBEDDING_MODEL_PATH = 'intfloat/multilingual-e5-small'
async def load_pdf_async(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

import urllib3, socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000), 
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
    ])

# Run the async function
pages = asyncio.run(load_pdf_async(FILE_PATH))

embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_PATH)
vector_store = InMemoryVectorStore.from_documents(pages, embeddings)

docs = vector_store.similarity_search("What is Hamburger?", k = 2)
for doc in docs:
    print(f"Page: {doc.metadata['page']}: {doc.page_content}\n")
# from llmsherpa.readers import LayoutPDFReader

# llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# pdf_url = "./FoodServingSystemIndo.pdf" # also allowed is a file path e.g. /home/downloads/xyz.pdf
# pdf_reader = LayoutPDFReader(llmsherpa_api_url)
# doc = pdf_reader.read_pdf(pdf_url)

# from llama_index.readers.schema.base import Document
# from llama_index import VectorStoreIndex

# index = VectorStoreIndex([])
# for chunk in doc.chunks():
#     index.insert(Document(text=chunk.to_context_text(), extra_info={}))
# query_engine = index.as_query_engine()

# # Let's run one query
# response = query_engine.query("How to make an ice cream")
# print(response)

from llama_index.readers.upstage.base import UpstageLayoutAnalysisReader

reader = UpstageLayoutAnalysisReader(api_key="<API_KEY>", use_ocr=True)
docs = reader.load_data( "./FoodServingSystemIndo.pdf")
for doc in docs:
    print(doc, "\n")
# from llama_index.core import VectorStoreIndex


# index = VectorStoreIndex.from_documents(doc)
# query_engine = index.as_query_engine()
# # Let's run one query
# response = query_engine.query("How to make an ice cream")
# print(response)

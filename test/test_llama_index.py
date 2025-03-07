import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse
API_KEY = ""
PDF_PATH = ""
parser = LlamaParse(api_key=API_KEY, result_type="markdown", verbose=True)

documents = parser.load_data(PDF_PATH)
print(documents)
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import magic

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

urls = [
    "https://docs.streamlit.io/library/api-reference/control-flow/st.form_submit_button",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",
]
loader = UnstructuredURLLoader(urls=urls)
loader.requests_kwargs = {"verify": False, "headers": headers}
loader.headers = headers
loader.continue_on_failure = True
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print(texts)

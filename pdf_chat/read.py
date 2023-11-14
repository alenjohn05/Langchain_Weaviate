import io
import os
from pathlib import Path
import time
from weaviate.util import get_valid_uuid
from uuid import uuid4
from langchain.vectorstores import Weaviate
from langchain.text_splitter import CharacterTextSplitter
from pypdf import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.chat_models import ChatOpenAI

import time
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.weaviate import Weaviate


os.environ["OPENAI_API_KEY"] = "sk-foo"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
pdf_loader = DirectoryLoader("pdflist/", glob="**/*.pdf", loader_cls=PyPDFLoader)
markdown_loader = DirectoryLoader(
        "pdflist/", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
text_loader = DirectoryLoader("pdflist/", glob="**/*.txt", loader_cls=TextLoader)

all_loaders = [pdf_loader, markdown_loader, text_loader]

# Load documents from all loaders
loaded_documents = []
for loader in all_loaders:
    loaded_documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
chunked_documents = text_splitter.split_documents(loaded_documents)

doc_upload_start = time.time()
embeddings = OpenAIEmbeddings(
    openai_api_key="sk-foo"
)
Weaviate.from_documents(
    chunked_documents, embeddings, weaviate_url="http://localhost:8080", by_text=False
)
print(chunked_documents)
print(
    f"Uploaded {len(chunked_documents)} documents in {time.time() - doc_upload_start} seconds."
)
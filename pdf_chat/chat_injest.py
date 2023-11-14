import os
import time
from weaviate.util import get_valid_uuid
from uuid import uuid4
import weaviate
from langchain.vectorstores import Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)

os.environ["OPENAI_API_KEY"] = "sk-foo"
llmClinet = OpenAI()


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return llmClinet.embeddings.create(input=[text], model=model)["data"][0][
        "embedding"
    ]


def create_class(
    client: weaviate.Client, drop: bool = False, class_name: str = "DocumentOP"
):
    if drop:
        client.schema.delete_class(class_name)
    schema = {
        "class": class_name,
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {"model": "ada", "modelVersion": "002", "type": "text"},
        },
        "vectorIndexType": "hnsw",
        "vectorizer": "text2vec-openai",
        "properties": [
            {
                "name": "source",
                "dataType": ["text"],
                "description": "The text content of the podcast clip",
                "moduleConfig": {
                    "text2vec-openai": {
                        "skip": False,
                        "vectorizePropertyName": False,
                        "vectorizeClassName": False,
                    }
                },
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The text content of the podcast clip",
                "moduleConfig": {
                    "text2vec-openai": {
                        "skip": False,
                        "vectorizePropertyName": False,
                        "vectorizeClassName": False,
                    }
                },
            },
        ],
    }
    if not client.schema.exists(class_name):
        client.schema.create_class(schema)


# chunks = text_splitter.split_text(text=text)
weaviate_client = weaviate.Client(f"http://localhost:8080")
create_class(
    weaviate_client,
    "True",
    "DocumentOP",
)
vectorstore = Weaviate(
    client=weaviate_client,
    index_name="DocumentOP",
    text_key="content",
)

print("doing Things.... Add Vector")
pdf_loader = DirectoryLoader("pdflist/", glob="**/*.pdf", loader_cls=PyPDFLoader)
all_loaders = [pdf_loader]
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
print(
    f"Uploaded {len(chunked_documents)} documents in {time.time() - doc_upload_start} seconds."
)


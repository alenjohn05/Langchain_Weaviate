import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain.document_loaders import PyMuPDFLoader

print("doing Things.... Add Vector")
# pdf_loader = DirectoryLoader("pdflist/", glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_loader = PyMuPDFLoader('https://innov-dev.beta.injomo.com/sys/core/files.pxy/1152-885-0/2137')
all_loaders = [pdf_loader]
loaded_documents = []
for loader in all_loaders:
    loaded_documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
chunked_documents = text_splitter.split_documents(loaded_documents)

doc_upload_start = time.time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
docs = text_splitter.split_documents(chunked_documents)

embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_KEY")

client = weaviate.Client("http://localhost:8080")

client.schema.delete_all()
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Chatbot",
            "description": "Documents for chatbot",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {"text2vec-openai": {"model": "ada", "type": "text"}},
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
            ],
        },
    ]
}
client.schema.create(schema)
vectorstore = Weaviate(client, "Chatbot", "content", attributes=["source, page"])
# load text into the vectorstore
text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]
texts, meta = list(zip(*text_meta_pair))
vectorstore.add_texts(texts, meta)
# data_to_add = []
# for text, meta in zip(texts, meta_list):
#     chatbot_instance = {
#         "pdfname": "PDF_Name",
#         "content": [{"texts": text, "source": meta["source"], "page": meta["page"]}],
#     }
#     data_to_add.append(chatbot_instance)
# vectorstore.add_texts(data_to_add)
print(
    f"Uploaded {len(chunked_documents)} documents in {time.time() - doc_upload_start} seconds."
)

schema = client.schema.get()
file_name = "schema_data.json"
with open(file_name, "w") as file:
    json.dump(schema, file, indent=4)

from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
import weaviate
import os
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
import json
import pandas as pd

os.environ["OPENAI_API_KEY"] = "sk-foo"
client = weaviate.Client("http://localhost:8080")


def contains(list, isInList):
    for x in list:
        if isInList(x):
            return True
    return False


def getOrCreateClass(className: str):
    try:
        schema = client.schema.get()
        if contains(schema["classes"], lambda x: x["class"] == className):
            print("Class already exists")
            return
        else:
            class_obj = {"class": className}
            client.schema.create_class(class_obj)
    except Exception as e:
        print(e)
        print("Error in getOrCreateClass")


className = "Smaple01"
objectName = "ObjectSample01"
pdf_loader = DirectoryLoader("pdflist/", glob="**/*.pdf", loader_cls=PyPDFLoader)
all_loaders = [pdf_loader]
loaded_documents = []
for loader in all_loaders:
    loaded_documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
chunked_documents = text_splitter.split_documents(loaded_documents)
composite_id = f"{className}-{objectName}"
getOrCreateClass(className)
from openai import OpenAI

clientllm = OpenAI()


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return clientllm.embeddings.create(input = [text], model=model)['data'][0]['embedding']


dataText = []
for doc in chunked_documents:
  response = clientllm.embeddings.create(input=[doc.page_content], model="text-embedding-ada-002")
  embedding = response.data[0].embedding
  content = {
    "text": doc.page_content,
    "metadata": doc.metadata,
    "embedding": embedding
  }
  
  dataText.append(content)
#  # batch create data objects
client.batch.configure(batch_size=100)
with client.batch as batch:
    for doc in dataText:
        data_object = {
            "text": doc["text"],
            "metadata": doc["metadata"]
        }
        batch.add_data_object(
            data_object=data_object, 
            class_name=className,
            vector=doc["embedding"]
        )
schema = client.schema.get()

# print the schema
print(json.dumps(schema, indent=4))
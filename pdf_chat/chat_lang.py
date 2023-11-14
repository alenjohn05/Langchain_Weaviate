from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
import weaviate
from langchain.vectorstores import Weaviate
import json
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

client = weaviate.Client("http://localhost:8080")
os.environ["OPENAI_API_KEY"] = "sk-foo"
vectorstore = Weaviate(
    client=client,
    index_name="Smaple01",
    text_key="text",
)

MyOpenAI = OpenAI(
    temperature=0.2,
    openai_api_key="sk-foo",
)


print("Welcome to the Weaviate ChatVectorDBChain Demo!")
print("Please enter a question or dialogue to get started!")



def queryContent(query):
    embedding_vector = OpenAIEmbeddings().embed_query(query)
    resultcontent = vectorstore.similarity_search_by_vector(embedding_vector)
    embedding = OpenAIEmbeddings()
    print(
        resultcontent,
        "\n\n========================================================= \n\n\n\n",
    )
    db = FAISS.from_documents(resultcontent, embedding)
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
    )
    result = conversation_chain({"question": query})
    return result


while True:
    query = input("")
    finalresult = queryContent(query)
    print(finalresult)
    schema = client.schema.get()
    file_name = "schema_data.json"
    with open(file_name, "w") as file:
        json.dump(schema, file, indent=4)
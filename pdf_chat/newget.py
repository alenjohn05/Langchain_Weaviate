from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.chains import (
    RetrievalQA,
)
import weaviate
from langchain.vectorstores import Weaviate
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.schema import Document


client = weaviate.Client("http://localhost:8080")

retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name="LangChain",
    text_key="text",
    attributes=[],
    create_schema_if_missing=True,
)

MyOpenAI = OpenAI(
    temperature=0.2,
    openai_api_key="sk-foo",
)
prompt_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer from the Retrieval tool.
The example of your response should be:

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


qa_prompt = set_custom_prompt()


def queryContent(query):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=MyOpenAI,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )
    result = retrieval_qa({"query": query})
    return result


chat_history = []
print("Welcome to the Weaviate ChatVectorDBChain Demo!")
query = "What is FIFA?"
result = queryContent(query)
print(" RetrievalQA----------------------------------- \n\n\n\n")
print(result)
print("----------------------------------- \n")

get_first_object_weaviate_query = """
{
  Get {
    DocumentOP {
      _additional {
        id
      }
    }
  }
}
"""

results = client.query.raw(get_first_object_weaviate_query)
print(results)

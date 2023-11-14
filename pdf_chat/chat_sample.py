from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.chains import (
    RetrievalQA,
    StuffDocumentsChain,
    LLMChain,
)
import weaviate
from langchain.vectorstores import Weaviate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.llms import OpenAI


client = weaviate.Client("http://localhost:8080")

vectorstore = Weaviate(client, "Document", "content")

MyOpenAI = OpenAI(
    temperature=0.2,
    openai_api_key="sk-foo",
)


prompt_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
The example of your response should be:

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt

def queryContent(query):
    messages = [
        SystemMessage(content="You are a world class algorithm to answer questions."),
        HumanMessage(
            content="Answer question using only information contained in the following context: "
        ),
        HumanMessagePromptTemplate.from_template("{context}"),
        HumanMessage(
            content="Tips: If you can't find a relevant answer in the context, then say you don't know. Be concise!"
        ),
        HumanMessagePromptTemplate.from_template("Question: {question}"),
    ]
    prompt = ChatPromptTemplate(messages=messages)
    qa_chain = LLMChain(llm=MyOpenAI, prompt=prompt)
    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name="context",
        document_prompt=set_custom_prompt(),
    )
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=MyOpenAI,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        # # combine_documents_chain=final_qa_chain,
        # return_source_documents=True,
    )

    result = retrieval_qa({"query":query})
    return result


chat_history = []
print("Welcome to the Weaviate ChatVectorDBChain Demo!")
query = "what is the story of cricket?"
result = queryContent(query)
print(" RetrievalQA----------------------------------- \n\n\n\n")
print(result)
print("----------------------------------- \n")


query_all_classes = """
{
  Get {
    Class {
      Document
    }
  }
}
"""

results = client.query.raw(query_all_classes)
print(results)

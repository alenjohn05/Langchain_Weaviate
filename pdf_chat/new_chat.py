import weaviate
from langchain.vectorstores import Weaviate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import json

client = weaviate.Client("http://localhost:8080")
query = "What is blue LED value in Clean water?"


vectorstore = Weaviate(client, "Chatbot", "content", attributes=["source","page"])
# jeopardy = client.collections.get("Chatbot")
# jeopardy = client.query.get("Chatbot", ["pdfname"]).do()
# query_builder = client.query.get("Chatbot", ["pdfname", "_additional {certainty}"])
# where_filter = {
#     "operator": "Like",
#     "operands": [{"path": "pdfname", "operator": "Equal", "valueText": "fdsf"}],
# }
# query_builder = query_builder.with_where(where_filter).do()
# print(json.dumps(query_builder, indent=2))
# response = (
#     client.query
#     .get('JeopardyQuestion', ['question', 'answer'])
#     .with_hybrid(
#       query='safety'
#     )
#     .with_additional('score')
#     .with_limit(3)
#     .do()
# )

docs = vectorstore.similarity_search(query, top_k=20)
print(docs)
chain = load_qa_chain(
    OpenAI(
        openai_api_key="sk-foo",
        temperature=0,
    ),
    chain_type="stuff",
)
# create answer
response = chain.run(input_documents=docs, question=query)

print(response)

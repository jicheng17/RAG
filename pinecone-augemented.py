import os
import openai
from dotenv import load_dotenv
import pinecone
import langchain
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# load configuration
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
env = os.getenv("env")
api_key = os.getenv("api-key")


pinecone.init(api_key=api_key, environment=env)
index_name = 'jayceecone'
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
index = Pinecone.from_existing_index(
    index_name=index_name, embedding=embeddings)


def get_similiar_docs(query, k=5, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    print("### similar_docs ###")
    for i in range(0, len(similar_docs)):
        print(similar_docs[i].page_content),

    return similar_docs


model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")


#### RAG flow ####
query = "What did LVMH do during COVID?"
similar_docs = get_similiar_docs("LVMH")

result = chain.run(input_documents=similar_docs, question=query)
print("##############################################")
print("################# result #####################")
print("##############################################")
print(result)
print("##############################################")

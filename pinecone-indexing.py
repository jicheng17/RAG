import os
import openai
from dotenv import load_dotenv
import pinecone
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI


# load configuration
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
env = os.getenv("env")
api_key = os.getenv("api-key")

directory_path = 'dataset/'


def load_docs(directory_path):
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
    return documents


documents = load_docs(directory_path)
print("Total number of documents :", len(documents))


def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


docs = split_docs(documents)
print(len(docs))
print(docs[0])

# Example
# query_result = embeddings.embed_query("large language model")
# print(query_result)
# print(len(query_result))

pinecone.init(api_key=api_key, environment=env)
index_name = 'jayceecone'
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

print("data indexed")

import os
import openai
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the document as a string
context = '''A phenotype refers to the observable physical
properties of an organism, including its appearance, development, and behavior.
It is determined by both the organism's genotype, which is the set of genes
it carries, and environmental influences upon these genes.'''

# Create the Prompt Template for base qa_chain
qa_template = """Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, 
    answer the question: {question}
    Answer:
"""
PROMPT = PromptTemplate(
    template=qa_template, input_variables=["context", "question"]
)
chain = LLMChain(llm=OpenAI(temperature=0), prompt=PROMPT)
query = "What's a phenotype?"
print(chain({"context": context, "question": query}, return_only_outputs=True))

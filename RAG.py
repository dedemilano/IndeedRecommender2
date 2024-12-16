import getpass
import os
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langgraph.graph import START , StateGraph ,MessagesState
from typing_extensions import List , TypedDict
import subprocess
from langchain_core.prompts import PromptTemplate
from typing import Literal

from typing_extensions import Annotated
from langgraph.checkpoint.memory import MemorySaver
from tqdm import tqdm
import time
import math
from langchain_mistralai import MistralAIEmbeddings
import concurrent.futures
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Utiliser les variables
MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

print("MISTRALAI_API_KEY :", MISTRALAI_API_KEY)
print("LANGCHAIN_API_KEY :", LANGCHAIN_API_KEY)
print("ANTHROPIC_API_KEY :", ANTHROPIC_API_KEY)

# 1. Read JSON with proper encoding
global loader
try:
    with open('./indeedJobData.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # Check data structure
        if isinstance(data, list):
            print("Data is a list of", len(data), "items")
            print("\nFirst item keys:", data[0].keys())
        else:
            print("Data structure:", type(data))
            print("\nKeys:", data.keys())
            
        # 2. Create appropriate loader
        loader = JSONLoader(
            file_path='./indeedJobData.json',
            jq_schema='.[]',  # Use .[] for array of objects
            text_content=False
        )
        
        # 3. Load documents
        docs = list(loader.lazy_load())
        
        # 4. Verify first document
        if docs:
            print("\nFirst document content:")
            print(docs[0].page_content[:-1])
            
except UnicodeDecodeError:
    # Fallback to different encoding if utf-8 fails
    with open('./indeedJobData.json', 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
        # Repeat steps above...

# 5. Load documents
docs = []
docs_lazy = loader.lazy_load()

for doc in docs_lazy:
    docs.append(doc)
print(docs[0].page_content[:100])
print(docs[0].metadata)

#SPLITTING PHASE
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
all_splits = text_splitter.split_documents(docs) 
print("Number of splits:", len(all_splits))

from langchain_mistralai import MistralAIEmbeddings

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRALAI_API_KEY,
)

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="indeedChromaCollection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # Where to save data locally, remove if not necessary
)

vector_store.add_documents(documents=all_splits)













template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

#Query analysis to boost queries with more context
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

all_splits[0].metadata

class Search(TypedDict):
    """"Search query."""
    
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

input_message = "List me all the skills to be a good data scientist in 2024?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()

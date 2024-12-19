# 1. Imports
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
from langgraph.graph import START, StateGraph, MessagesState
from typing_extensions import List, TypedDict
import subprocess
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from typing import Literal, Dict, Any
from typing_extensions import Annotated
from langgraph.checkpoint.memory import MemorySaver
from tqdm import tqdm
import time
import math
from langchain_mistralai import MistralAIEmbeddings
import concurrent.futures
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

import getpass
import os
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import json
import gc
import time
from tqdm import tqdm
import shutil

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


def reset_chroma_db():
    """Supprime la base de données Chroma existante"""
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("Base de données Chroma réinitialisée")

# 1. Chargement optimisé des données
def load_json_data(file_path='./indeedJobData.json', batch_size=1000):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Data is a list of {len(data)} items")
        print("\nFirst item keys:", data[0].keys())
        
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.[]',
            text_content=False
        )
        return loader, len(data)

# 2. Préparation des documents par lots
def prepare_documents_batch(loader, batch_size=1000):
    docs_lazy = loader.lazy_load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Augmenté pour réduire le nombre de splits
        chunk_overlap=100  # Réduit pour économiser la mémoire
    )
    
    batch = []
    splits = []
    
    for doc in docs_lazy:
        batch.append(doc)
        if len(batch) >= batch_size:
            batch_splits = text_splitter.split_documents(batch)
            splits.extend(batch_splits)
            batch = []
            # Force garbage collection
            gc.collect()
    
    # Traiter le dernier lot si nécessaire
    if batch:
        batch_splits = text_splitter.split_documents(batch)
        splits.extend(batch_splits)
    
    print(f"Total splits created: {len(splits)}")
    return splits

# 3. Configuration des embeddings
# 3. Configuration des embeddings
def setup_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


# 4. Configuration et ajout au vectorstore par lots
def setup_and_add_to_vectorstore(splits, embeddings, batch_size=100):
    vector_store = Chroma(
        collection_name="indeedChromaCollection",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    total_batches = len(splits) // batch_size + (1 if len(splits) % batch_size else 0)
    
    for i in tqdm(range(0, len(splits), batch_size), desc="Adding documents", total=total_batches):
        batch = splits[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
            vector_store.persist()  # Sauvegarder après chaque lot
            time.sleep(0.1)  # Petit délai pour éviter la surcharge
        except Exception as e:
            print(f"Error adding batch {i//batch_size + 1}: {e}")
            continue
        gc.collect()  # Force garbage collection après chaque lot
    
    return vector_store

# 5. Configuration du LLM
def setup_llm():
    return ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.7
    )

# 6. Création de la chaîne RAG
def create_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant spécialisé dans l'analyse d'offres d'emploi. 
        Utilise le contexte fourni pour répondre aux questions de manière précise.
        
        Contexte: {context}
        Question: {question}"""),
        ("human", "{question}")
    ])
    
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

# 7. Fonction principale d'initialisation
def initialize_rag_system(batch_size=1000):
    print("Loading data...")
    loader, total_docs = load_json_data(batch_size=batch_size)
    
    print("\nPreparing documents in batches...")
    splits = prepare_documents_batch(loader, batch_size=batch_size)
    
    print("\nSetting up embeddings...")
    embeddings = setup_embeddings()
    
    print("\nSetting up vector store and adding documents...")
    vector_store = setup_and_add_to_vectorstore(splits, embeddings)
    
    print("\nSetting up LLM...")
    llm = setup_llm()
    
    print("\nCreating RAG chain...")
    chain, retriever = create_rag_chain(vector_store, llm)
    
    return chain, retriever

# 8. Fonction de requête
def query_rag_system(chain, retriever, question: str):
    try:
        response = chain.invoke({
            "question": question
        })
        
        context_docs = retriever.get_relevant_documents(question)
        
        return {
            "answer": response,
            "sources": [doc.metadata for doc in context_docs]
        }
    except Exception as e:
        print(f"Error querying RAG system: {e}")
        raise

# Test du système
if __name__ == "__main__":
    try:
        # Initialiser avec un batch_size plus petit
        chain, retriever = initialize_rag_system(batch_size=500)
        
        # Questions de test
        questions = [
            "Quels sont les emplois les plus demandés ?",
            "Quelles sont les compétences les plus recherchées ?",
            "Quel est le salaire moyen proposé ?"
        ]
        
        print("\nTesting the system with sample questions...")
        for question in questions:
            print(f"\nQuestion: {question}")
            try:
                result = query_rag_system(chain, retriever, question)
                print(f"Réponse: {result['answer']}")
                print("\nSources utilisées:")
                for source in result['sources']:
                    print(f"- {source}")
            except Exception as e:
                print(f"Error processing question: {e}")
    
    except Exception as e:
        print(f"Fatal error during initialization: {e}")
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

# 2. Configuration des variables d'environnement
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# 3. Chargement des données
def load_json_data(file_path='./indeedJobData.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            if isinstance(data, list):
                print(f"Data is a list of {len(data)} items")
                print("\nFirst item keys:", data[0].keys())
            else:
                print("Data structure:", type(data))
                print("\nKeys:", data.keys())
            
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.[]',
                text_content=False
            )
            
            return loader
            
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.[]',
                text_content=False
            )
            return loader

# 4. Préparation des documents
def prepare_documents(loader):
    docs = []
    docs_lazy = loader.lazy_load()
    for doc in docs_lazy:
        docs.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print(f"Number of splits: {len(all_splits)}")
    
    return all_splits

# 5. Configuration des embeddings
def setup_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

# 6. Configuration du vector store
def setup_vector_store(embeddings):
    return Chroma(
        collection_name="indeedChromaCollection",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

# 7. Ajout des documents au vector store
def add_documents_to_vectorstore(vector_store, all_splits):
    try:
        vector_store.add_documents(all_splits)
        vector_store.persist()
        print(f"Added {len(all_splits)} documents to vector store")
    except Exception as e:
        print(f"Error adding documents: {e}")
        raise

# 8. Configuration du modèle de langage
def setup_llm():
    return ChatAnthropic(
        model="claude-3-opus-20240229",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.7
    )

# 9. Configuration du prompt
def setup_prompt():
    system_prompt = """Tu es un assistant spécialisé dans l'analyse d'offres d'emploi. Utilise le contexte fourni pour répondre aux questions de manière précise et pertinente.
    Contexte additionnel: {context}
    Historique de la conversation: {chat_history}
    Question: {question}

    Réponds de manière structurée en utilisant les informations du contexte. Si tu n'as pas assez d'informations, indique-le clairement.
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

# 10. Configuration de la chaîne RAG
def create_rag_chain(vector_store: Chroma, llm, memory: ConversationBufferMemory):
    prompt = setup_prompt()
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

# 11. Fonction d'interrogation
def query_rag_system(chain, question: str) -> Dict[Any, Any]:
    try:
        response = chain({"question": question})
        return {
            "answer": response["answer"],
            "sources": [doc.metadata for doc in response["source_documents"]],
            "chat_history": chain.memory.chat_memory.messages
        }
    except Exception as e:
        print(f"Error querying RAG system: {e}")
        raise

# 12. Initialisation du système complet
def initialize_complete_rag_system():
    # Charger les données
    loader = load_json_data()
    
    # Préparer les documents
    all_splits = prepare_documents(loader)
    
    # Configurer les embeddings
    embeddings = setup_embeddings()
    
    # Configurer le vector store
    vector_store = setup_vector_store(embeddings)
    
    # Ajouter les documents si nécessaire
    if vector_store.get()["ids"] == []:
        add_documents_to_vectorstore(vector_store, all_splits)
    
    # Configurer le LLM
    llm = setup_llm()
    
    # Configurer la mémoire
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Créer la chaîne RAG
    rag_chain = create_rag_chain(vector_store, llm, memory)
    
    return rag_chain

# 13. Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le système
    print("Initializing RAG system...")
    rag_chain = initialize_complete_rag_system()
    
    # Exemple de questions
    questions = [
        "Quels sont les emplois les plus demandés ?",
        "Quelles sont les compétences les plus recherchées ?",
        "Quel est le salaire moyen proposé ?"
    ]
    
    # Interroger le système
    print("\nTesting the system with sample questions...")
    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            result = query_rag_system(rag_chain, question)
            print(f"Réponse: {result['answer']}")
            print("\nSources utilisées:")
            for source in result['sources']:
                print(f"- {source}")
        except Exception as e:
            print(f"Error processing question: {e}")
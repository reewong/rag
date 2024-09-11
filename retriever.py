from typing import List, Dict, Optional
from langchain.retrievers import VectorStoreRetriever
class CodeRetriever:
    def __init__(self, vector_store):
        self.retriever = VectorStoreRetriever(vectorstore=vector_store)

    def get_relevant_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)
    
# vector_store/faiss_store.py
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

class GenVectorStore:
    def __init__(self, embedding):
        self.embeddings = embedding
        self.vector_store = None

    def create_vector_store(self, documents):
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def get_vector_store(self):
        return self.vector_store

    def search(self, query: str, k: int = 4):
        return self.vector_store.similarity_search(query, k=k)
import os
from typing import List, Dict, Optional
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from load_and_splitter import load_and_split_project
def _format_relevant_docs(relevant_docs: List[Dict]):
    prompt = ''        
    for i, doc in enumerate(relevant_docs, 1):
        prompt += f"\nSnippet {i}:\n"
        prompt += f"SourceFile: {doc['metadata']['source']}\n"
        prompt += f"Page content: {doc['page_content']}\n"
    return prompt
class GenVectorStore:
    def __init__(self, embedding):
        self.embeddings = embedding
        self.vector_store = None

    def create_vector_store(self, documents: List[Dict], store_path: str):
        """Create a new vector store and save it to disk."""
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            if i == 0:
                self.vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=store_path
                )
            else:
                self.vector_store.add_documents(batch)
            
            print(f"Processed batch {i//batch_size + 1}/{len(documents)//batch_size + 1}")
    
        print(f"Vector store created and saved to {store_path}")

    def load_vector_store(self, store_path: str) -> bool:
        """Load a vector store from disk if it exists and is valid."""
        if not os.path.exists(store_path):
            print(f"Vector store not found at {store_path}")
            return False
        try:
            self.vector_store = Chroma(
                persist_directory=store_path,
                embedding_function=self.embeddings
            )
            print(f"Successfully loaded vector store from {store_path}")
            return True
        except Exception as e:
            print(f"Error loading vector store from {store_path}: {str(e)}")
            return False

    def get_or_create_vector_store(self, code_path, store_path: str):
        """Load the vector store if it exists and is valid, otherwise create a new one."""
        if not self.load_vector_store(store_path):
            documents = load_and_split_project(code_path)
            print(f"Creating new vector store at {store_path}")
            os.makedirs(store_path)
            self.create_vector_store(documents, store_path)

    def _save_vector_store(self, store_path: str):
        """Save the vector store to disk."""
        if self.vector_store is None:
            raise ValueError("Vector store has not been created or loaded yet.")
        self.vector_store.persist()
        print(f"Vector store saved to {store_path}")

    def get_vector_store(self):
        return self.vector_store
    def get_retriever(self, k = 10):
        """Get a retriever for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store has not been created or loaded yet.")
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 50,
                "lambda_mult": 0.7
            }
        )
        return retriever
    def search(self, query: str, k: int = 10) -> List[Dict]:
        if self.vector_store is None:
            raise ValueError("Vector store has not been created or loaded yet.")
        # results = self.vector_store.similarity_search(query, k)
        results = self.get_retriever(k).invoke(query)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]

    def add_documents(self, documents: List[Dict], store_path: str):
        """Add new documents to the existing vector store and update the stored version."""
        if self.vector_store is None:
            raise ValueError("Vector store has not been created or loaded yet.")
        self.vector_store.add_documents(documents)
        self._save_vector_store(store_path)
        print(f"Added {len(documents)} documents to vector store")


from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings  # 或者您使用的其他嵌入模型

class GenVectorStore:
    def __init__(self, embedding):
        self.embeddings = embedding
        self.vector_store = None

    def create_vector_store(self, documents):
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def get_retriever(self):
        if self.vector_store:
            return self.vector_store.as_retriever()
        else:
            raise ValueError("Vector store has not been created yet.")

    def search(self, query: str, k: int = 4):
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        else:
            raise ValueError("Vector store has not been created yet.")
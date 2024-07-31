from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# loads BAAI/bge-small-en
# embed_model = HuggingFaceEmbedding()

# loads BAAI/bge-small-en-v1.5

Settings.llm = Ollama(model="mistral-nemo:latest", request_timeout=300.0)
Settings.embed_model= OllamaEmbedding("mistral-nemo:latest")
documents = SimpleDirectoryReader("D:\sql\openGauss-server\src").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("这是一个代码仓吗？主要是关于什么的")
print(response)

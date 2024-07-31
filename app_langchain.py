import sys
import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# load the code and split it into chunks
repo_path = r"D:\sql\openGauss-server\src\gausskernel\storage\access\redo"
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".cpp"],
    parser=LanguageParser(language=Language.CPP, parser_threshold=500),
)
data = loader.load()
print(len(data))


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
all_splits = text_splitter.split_documents(data)
print(len(all_splits))
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
def indexing_vec():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(model="mistral-nemo:latest"), persist_directory="./chroma_db")
    return vectorstore

def load_vec():
    vectorstore = Chroma(persist_directory="./chroma_db", embedding=OllamaEmbeddings(model="mistral-nemo:latest"))
    return vectorstore
def get_vectorstore(flag=False):
    persist_directory = "./chroma_db"
    
    # 检查持久化文件夹是否非空
    is_non_empty_directory = os.path.isdir(persist_directory) and bool(os.listdir(persist_directory))
    
    if flag or is_non_empty_directory:
        return load_vec()
    else:
        return indexing_vec()
vectorstore = get_vectorstore()
while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    # Prompt
    template = """使用以下上下文来回答最后的问题。
    如果你不知道答案，就直接说不知道，不要试图编造答案。
    请基于提供的代码片段回答问题，重点关注代码的功能、结构和关键逻辑。.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="gemma2:27b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})
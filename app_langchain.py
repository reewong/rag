import logging
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, List, Any

# 设置基本的日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGDebugHandler(BaseCallbackHandler):
    def __init__(self):
        self.retriever_inputs = []
        self.retriever_outputs = []
        self.llm_inputs = []
        self.llm_outputs = []

    def on_retriever_start(self, query: str, **kwargs: Any) -> None:
        logging.info(f"Retriever query: {query}")
        self.retriever_inputs.append(query)

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> None:
        logging.info(f"Retrieved {len(documents)} documents")
        for i, doc in enumerate(documents):
            logging.info(f"Document {i + 1}:\n{doc.page_content[:200]}...")
        self.retriever_outputs.extend(documents)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        for i, prompt in enumerate(prompts):
            logging.info(f"LLM Prompt {i + 1}:\n{prompt}")
        self.llm_inputs.extend(prompts)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        logging.info(f"LLM Response:\n{response}")
        self.llm_outputs.append(response)

# 修改您的RAG设置以包含调试处理器
debug_handler = RAGDebugHandler()
std_out_handler = StdOutCallbackHandler()


import sys
import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from pprint import pprint
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
    parser=LanguageParser(language=Language.CPP),
)
data = loader.load()
print(len(data))
contents = [document.page_content for document in data]

# 用分隔符 '\n\n--8<--\n\n' 连接所有文档内容
combined_contents = "\n\n--8<--\n\n".join(contents)

# 将连接后的内容写入文件
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(combined_contents)

from langchain.text_splitter import Language,RecursiveCharacterTextSplitter
cpp_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=3500, chunk_overlap=200)
all_splits = cpp_splitter.split_documents(data)
print(len(all_splits))

contents = [document.page_content for document in all_splits]

# 用分隔符 '\n\n--8<--\n\n' 连接所有文档内容
combined_contents = "\n\n--8<--\n\n".join(contents)

# 将连接后的内容写入文件
with open('output_split.txt', 'w', encoding='utf-8') as file:
    file.write(combined_contents)
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

embed_model_code = HuggingFaceEmbeddings(
    model_name="./jina-embeddings-v2-base-code",
    model_kwargs = {
                    "device": "cpu",
                    "trust_remote_code": True
                },
    encode_kwargs = {"normalize_embeddings": True}
)
# embedding_ollama = OllamaEmbeddings(model="mistral-nemo:latest")
def indexing_vec():
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model_code, persist_directory="./chroma_db")
    return vectorstore

def load_vec():
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embed_model_code)
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
    请基于提供的代码片段回答问题，重点关注代码的功能、结构和关键逻辑。
    使用中文回答
    {context}
    问题: {question}
    中文回答:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    llm = Ollama(model="gemma2:27b", callback_manager=CallbackManager([debug_handler]))
    retreval = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 50,  # 初始检索更多文档，然后从中选择最终的 10 个
            "lambda_mult": 0.7  # 在相关性和多样性之间取得平衡
        }
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = retreval,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,  # 这将返回源文档
        # callbacks=[debug_handler]
    )

    result = qa_chain({"query": query})
    print("\n--- 调试信息摘要 ---")
    if debug_handler.retriever_inputs:
        print(f"检索器查询: {debug_handler.retriever_inputs[-1]}")
    print(f"检索到的文档数量: {len(debug_handler.retriever_outputs)}")
    print("LLM输入提示:")
    if debug_handler.llm_inputs:
        print(debug_handler.llm_inputs[-1])
    print("LLM输出:")
    if debug_handler.llm_outputs:
        print(debug_handler.llm_outputs[-1])
    # 检查源文档
    if 'source_documents' in result:
        print("\n--- 源文档 ---")
        for i, doc in enumerate(result['source_documents']):
            print(f"文档 {i + 1}:")
            print(doc.page_content[:200] + "...")  # 打印每个文档的前200个字符
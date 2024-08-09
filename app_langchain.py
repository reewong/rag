import logging
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from typing import List, Dict, Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_chroma import Chroma
import sys
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import Language,RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
class RAGDebugHandler(BaseCallbackHandler):
    def __init__(self):
        self.retriever_inputs = []
        self.retriever_outputs = []
        self.llm_inputs = []
        self.llm_outputs = []
        self.current_response = ""
        # self.placeholder = st.empty()

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
        # logging.info(f"LLM Response:\n{response}")
        self.llm_outputs.append(response)

    # def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    #     """Run on new LLM token. Only available when streaming is enabled."""
    #     st.write(token, end="")
embed_model_code = HuggingFaceEmbeddings(
    model_name="./jina-embeddings-v2-base-code",
    model_kwargs = {
                    "device": "cpu",
                    "trust_remote_code": True
                },
    encode_kwargs = {"normalize_embeddings": True}
)

def indexing_vec():
    # load the code and split it into chunks
    # repo_path = r"D:\sql\openGauss-server\src\gausskernel\storage\access\redo"jnnjnjjjjn
    repo_path = r"/home/code/sql/openGauss-server/src/gausskernel/storage/access/redo"
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
    cpp_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=3500, chunk_overlap=200)
    all_splits = cpp_splitter.split_documents(data)
    print(len(all_splits))
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
debug_handler = RAGDebugHandler()

def main():
    st.set_page_config(page_title="本地大模型知识库RAG应用", page_icon="?", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("本地大模型知识库RAG应用")
    st.info("By ree", icon="👤")  # 使用用户图标

    if "messages" not in st.session_state.keys(): # 初始化聊天历史记录
       st.session_state.messages = [
            {"role": "assistant", "content": "关于文档里的内容，请随便问"}
    ]
    # Prompt
    template = """使用以下上下文来回答我最后提出的问题。
    如果你不知道答案，就直接说不知道，不要试图编造答案。
    请基于提供的代码片段回答问题，重点关注代码的功能、结构和关键逻辑。
    务必使用中文回答
    {context}
    问题: 基于以上给出的代码片段，{question}， 使用中文回答"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    # 初始化检索引擎)
    llm = Ollama(
        model="mistral-nemo:12b-instruct-2407-q8_0", 
        callback_manager=CallbackManager([debug_handler]), 
        base_url="http://host.docker.internal:11434"
    )

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
        callbacks=[debug_handler]
    )

    # 提示用户输入问题，并将问题添加到消息历史记录
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示此前的问答记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 生成回答
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain({"query": prompt})
                st.write(response["result"])
                message = {"role": "assistant", "content": response["result"]}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()

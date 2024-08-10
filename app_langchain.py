import logging
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from typing import List, Dict, Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_chroma import Chroma
import sys
import os
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from queue import Queue
from transformers import GPT2TokenizerFast

# import warnings

# Suppress the optimum warning
# warnings.filterwarnings("ignore", message="optimum is not installed.")

# Configure logging
logging.basicConfig(level=logging.INFO)

def cal_token(text):
    # 初始化GPT-2分词器
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # 使用分词器进行编码，并计算Token数量
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    return token_count

class ThreadSafeRAGDebugHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = Queue()

    def on_retriever_start(self, query: str, **kwargs: Any) -> None:
        self.queue.put(("retriever_start", query))

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> None:
        self.queue.put(("retriever_end", documents))

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.queue.put(("llm_start", prompts))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.queue.put(("llm_end", response))

    def get_debug_info(self):
        debug_info = {
            "retriever_inputs": [],
            "retriever_outputs": [],
            "llm_inputs": [],
            "llm_outputs": []
        }
        while not self.queue.empty():
            event_type, data = self.queue.get()
            if event_type == "retriever_start":
                debug_info["retriever_inputs"].append(data)
            elif event_type == "retriever_end":
                debug_info["retriever_outputs"].extend(data)
            elif event_type == "llm_start":
                debug_info["llm_inputs"].extend(data)
            elif event_type == "llm_end":
                debug_info["llm_outputs"].append(data)
        return debug_info

# ... (keep the existing embed_model_code, indexing_vec, load_vec, and get_vectorstore functions)
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
debug_handler = ThreadSafeRAGDebugHandler()
# 初始化LLM
llm = Ollama(
    model="mistral-nemo:12b-instruct-2407-q8_0", 
    callback_manager=CallbackManager([debug_handler]), 
    base_url="http://host.docker.internal:11434"
)
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
if "memory" not in st.session_state:
    st.session_state.memory = ConversationTokenBufferMemory(
        llm = llm,
        max_token_limit = 5000,
        return_massages = True
    )
def summary_conversation(messages):
    summary_prompt = f"""总结以下对话内容，关注关键点和上下文：
    {messages}
    总结:"""
    return llm.invoke(summary_prompt)
def main():
    st.set_page_config(page_title="本地大模型知识库RAG应用", page_icon="?", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("本地大模型知识库RAG应用")
    st.info("By ree", icon="👤")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "关于文档里的内容，请随便问"}
        ]

    # 更新后的Prompt，包含聊天历史
    template = """使用以下提供的上下文和聊天历史来回答最后的问题。如果你不知道答案，就直接说不知道，不要试图编造答案。
    请基于提供的代码片段和历史对话回答问题，重点关注代码的功能、结构和关键逻辑。
    务必使用中文回答。

    聊天历史：
    {chat_history}

    上下文信息：
    {context}

    问题: {input}
    请务必使用中文作答
    """
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "input"],
        template=template,
    )



    # 初始化检索器
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 50,
            "lambda_mult": 0.7
        }
    )

    # 创建文档链
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 创建检索链
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 聊天界面
    if user_input := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.memory.chat_memory.add_user_message(user_input)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = st.session_state.memory.load_memory_variables({})["history"]
                st.write(f"===================chat history===================")
                st.write(chat_history)
                if cal_token(chat_history) > 5000:
                    summary = summary_conversation(chat_history)
                    chat_history = [HumanMessage(content = "上次的对话总结为："),
                                    AIMessage(content = summary)]
                response = retrieval_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                # st.write("==================document info======================")
                # for i, doc in enumerate(response['context']):
                #     st.write(f"Document {i + 1} Metadata: {doc.metadata}")
                #     st.write(f"Document {i + 1} Content:\n{doc.page_content}\n")
                
                st.write("====================answer=============================")
                st.write(response["answer"])
                message = {"role": "assistant", "content": response["answer"]}
                st.session_state.messages.append(message)
                st.session_state.memory.chat_memory.add_ai_message(response["answer"])
                # 显示调试信息
                st.write("====================debug info=========================")
                debug_info = debug_handler.get_debug_info()
                # st.write("Retriever Inputs:", debug_info["retriever_inputs"])
                st.write("LLM Inputs:", debug_info["llm_inputs"])

if __name__ == "__main__":
    main()
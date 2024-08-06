
###################################
#
# 第1步：配置本地模型
#
###################################

from llama_index.core import Settings

# 配置ollama的LLM模型，这里我们用gemma:2b
from llama_index.llms.ollama import Ollama
llm_ollama = Ollama(model="mistral-nemo:12b-instruct-2407-q8_0", request_timeout=600.0)

from llama_index.embeddings.ollama import OllamaEmbedding
embed_model= OllamaEmbedding("mistral-nemo:latest")


# 配置Rerank模型，这里我们用BAAI/bge-reranker-base
# from llama_index.core.postprocessor import SentenceTransformerRerank
# rerank_model_bge_base = SentenceTransformerRerank(
# model="D:\ollama\embed_model\bge-reranker-base", 
# top_n=2
# )

# 配置使用SpacyTextSplitter
from llama_index.core.text_splitter import CodeSplitter
Settings.llm = llm_ollama
Settings.embed_model = embed_model
Settings.text_splitter = CodeSplitter(
        language="cpp",
        chunk_lines = 2000,
        chunk_lines_overlap = 1024,
        max_chars = 5120
    )

###################################
#
# 第2步：配置向量数据库
#
###################################

import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

STORAGE_DIR = "./storage"# 定义索引保存的目录

db = chromadb.PersistentClient(path=STORAGE_DIR)
chroma_collection = db.get_or_create_collection("think")
chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
chroma_storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)

###################################
#
# 第3步：初始化Streamlit Web应用
#
###################################

import streamlit as st

st.set_page_config(page_title="本地大模型知识库RAG应用", page_icon="?", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("本地大模型知识库RAG应用")
st.info("By ree", icon="👤")  # 使用用户图标

if "messages" not in st.session_state.keys(): # 初始化聊天历史记录
    st.session_state.messages = [
       {"role": "assistant", "content": "关于文档里的内容，请随便问"}
    ]

###################################
#
# 第4步：加载文档建立索引
#
###################################

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

DATA_DIR = r"D:\sql\openGauss-server\src\gausskernel\storage\access\redo"# 知识库文档所在目录

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="加载文档并建立索引，需要1-2分钟"):
# 将指定目录下的文档建立索引，保存到向量数据库
        documents = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True, exclude=["*.txt", "Makefile"]).load_data() 
        index = VectorStoreIndex.from_documents(
            documents, storage_context=chroma_storage_context)
    return index

def load_index():
# 直接从向量数据库读取索引
    index = VectorStoreIndex.from_vector_store(chroma_vector_store)
    return index

###################################
#
# 第5步：定制Prompt模板
#
###################################

from llama_index.core import PromptTemplate

text_qa_template_str = (
"以下为上下文信息\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"请根据上下文信息回答我的问题或回复我的指令。前面的上下文信息可能有用，也可能没用，你需要从我给出的上下文信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。\n"
"问题：{query_str}\n"
"你的回复： "
)

text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
"这是原本的问题： {query_str}\n"
"我们已经提供了回答: {existing_answer}\n"
"现在我们有机会改进这个回答 "
"使用以下更多上下文（仅当需要用时）\n"
"------------\n"
"{context_msg}\n"
"------------\n"
"根据新的上下文, 请改进原来的回答。"
"如果新的上下文没有用, 直接返回原本的回答。\n"
"改进的回答: "
)
refine_template = PromptTemplate(refine_template_str)

###################################
#
# 第6步：创建检索引擎，并提问
#
###################################

# 仅第一次运行时使用load_data建立索引，再次运行使用load_index读取索引
# index = load_data();
index = load_index();
st.write("Debug:  print session keys 1", st.session_state.keys())

# 初始化检索引擎
if "query_engine" not in st.session_state.keys():
    st.write("Debug:  print session keys 2", st.session_state.keys())
    query_engine = index.as_query_engine(
    text_qa_template=text_qa_template, 
    refine_template=refine_template,
    similarity_top_k=6, 
    response_mode="compact", 
    verbose=True)
    st.write("Debug: query engine is", query_engine)
    st.session_state.query_engine = query_engine

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
            response = st.session_state.query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
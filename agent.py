import os
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from retriever import GenVectorStore
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
# 初始化 LLM
# llm = Ollama(
#     # model="mistral-nemo:12b-instruct-2407-q8_0", 
#     model = "qwen2.5:32b",
#     # callback_manager=CallbackManager([debug_handler]), 
#     # docker专用
#     # base_url="http://host.docker.internal:11434"
# )
llm = ChatOpenAI(model="qwen2.5:32b", api_key="ollama",     base_url="http://localhost:11434/v1",
)
# 递归获取文件目录结构并存储到内存中
def get_codebase_structure(directory):
    file_structure = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.cpp', '.h', '.py', '.java')):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                file_structure.append(relative_path)
    return file_structure

# 模拟读取文件内容
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return str(e)

# 获取用户自定义反馈
def get_user_feedback():
    feedback = input("请提供进一步的提示或反馈（留空则跳过）：")
    return feedback if feedback.strip() != "" else None

# 向量数据库模拟检索
def search_code_in_vector_db(query):
    # 假设数据库检索返回的代码片段
    res = vdb_mgr.search(query, 10)
    return res

# 调用 LangChain 的 LLM 进行代码分析
def analyze_code_with_context(topic, file_content):
    # 创建分析提示
    analysis_prompt = PromptTemplate(
        template=f"请分析以下代码片段与主题 '{topic}' 的关系：\n\n{{code_content}}",
        input_variables=["code_content"]
    )
    
    # 创建 LLMChain 来进行分析
    analysis_res = llm.invoke(analysis_prompt)
    
    # 执行分析
    return analysis_res

# 向量数据库检索工具，先显示结果再允许用户反馈
def search_code_in_vector_db_with_feedback(question):
    code_snippets = search_code_in_vector_db(question)
    # 保存检索到的代码片段到内存
    memory.save_context({"question": question}, {"result": code_snippets})
    
    # 显示检索结果给用户
    print(f"检索结果：\n{code_snippets}")
    
    # 询问用户是否提供反馈
    user_feedback = get_user_feedback()
    
    # 如果有用户反馈，结合反馈进行下一轮推理
    if user_feedback:
        question = f"{question}\n用户提示：{user_feedback}"
    
    return code_snippets if code_snippets != "No relevant results found." else "No relevant results found."

# 文件读取工具，先显示结果再允许用户反馈
def read_files_and_save_with_feedback(question, codebase_directory, file_structure):
    # 使用 LLM 选择相关文件
    selected_files = "\n".join(file_structure[:2])  # 假设选择前两个文件
    
    combined_results = ""
    for file in selected_files.split("\n"):
        file_path = os.path.join(codebase_directory, file.strip())
        file_content = read_file(file_path)
        # 保存文件内容到内存
        memory.save_context({"question": question}, {"result": file_content})
        combined_results += f"File: {file_path}\n{file_content}\n\n"
    
    # 显示读取结果给用户
    print(f"文件读取结果：\n{combined_results}")
    
    # 询问用户是否提供反馈
    user_feedback = get_user_feedback()
    
    # 如果有用户反馈，结合反馈进行下一轮推理
    if user_feedback:
        question = f"{question}\n用户提示：{user_feedback}"
    
    return combined_results if combined_results else "No relevant code found."

# 综合分析工具，结合用户反馈进行分析
def analyze_combined_code_snippets_with_feedback(question):
    # 获取历史记录中的所有检索到的代码片段
    history = memory.load_memory_variables({})["history"]
    combined_snippets = "\n\n".join([item["result"] for item in history])
    
    # 调用 LLM 进行分析
    analysis = analyze_code_with_context(question, combined_snippets)
    
    # 显示分析结果给用户
    print(f"综合分析结果：\n{analysis}")
    
    # 询问用户是否提供进一步的分析提示
    user_feedback = get_user_feedback()
    
    # 如果有用户反馈，将其加入分析中
    if user_feedback:
        question = f"{question}\n用户提示：{user_feedback}"
        analysis = analyze_code_with_context(question, combined_snippets)
    
    return analysis if analysis else "No relevant analysis found."


# 一开始获取代码库的文件结构并存储
codebase_directory = r"D:\sql\openGauss-server"
file_structure = get_codebase_structure(codebase_directory)

embed_model= OllamaEmbeddings(model="unclemusclez/jina-embeddings-v2-base-code")
vdb_mgr = GenVectorStore(embed_model)


store_path = f"{codebase_directory}/vector_store"
vdb_mgr.get_or_create_vector_store(store_path, codebase_directory)
# 定义工具
# 将工具封装成 LangChain 的 Tool 对象
tools = [
    Tool.from_function(
        func=read_files_and_save_with_feedback,
        name="FileReadTool",
        description="Reads files from the codebase and returns the content."
    ),
    Tool.from_function(
        func=search_code_in_vector_db_with_feedback,
        name="VectorSearchTool",
        description="Searches the codebase using a vector search and returns the most relevant code snippets."
    ),
    Tool.from_function(
        func=analyze_combined_code_snippets_with_feedback,
        name="AnalyzeCodeTool",
        description="Analyzes code snippets in relation to a specified topic."
    )
]



system_message = "You are a helpful assistant."
# This could also be a SystemMessage object
# system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")

memory = MemorySaver()
langgraph_agent_executor = create_react_agent(
    llm, tools, state_modifier=system_message, checkpointer=memory
)

config = {"configurable": {"thread_id": "test-thread"}}
print(
    langgraph_agent_executor.invoke(
        {"messages": [("user", "已知该代码为某个数据库内核代码，详细讲讲该数据库代码中回放线程的工作原理")]}, config
    )["messages"][-1].content
)
print("---")
print(
    langgraph_agent_executor.invoke(
        {"messages": [("user", "what was that output again?")]}, config
    )["messages"][-1].content
)

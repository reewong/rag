from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from retriever import GenVectorStore
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
from typing import List
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import re

# 初始化 LLM
# llm = ChatOpenAI(model="mistral-nemo:12b-instruct-2407-q8_0", api_key="ollama", base_url="http://localhost:11434/v1")
llm = ChatOpenAI(model="deepseek-chat", api_key="sk-dfa01c38b5d345728d2517c04012b2c1", base_url="https://api.deepseek.com/v1")
# gemma2:27b

MAX_TOKENS = 131072
# 创建 prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# 创建 LLM 链
chain = prompt | llm




store = {}
session_id = "foo"  # 固定 session_id 用于演示
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
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
def count_tokens(text: str) -> int:
    """
    估算文本的 token 数量，每 4 个字符约为 1 个 token。
    可以根据实际的 tokenizer (如 GPT-3/4) 替换此估算函数。
    """
    return math.ceil(len(text) / 4)

def split_file_by_tokens(file_path: str, max_tokens: int):
    parts = []  # 存储分割的文件块
    current_part = []  # 当前部分的行
    current_tokens = 0  # 当前部分的 token 数
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_tokens = count_tokens(line)

                # 如果当前部分加上这行超过了 max_tokens，先保存当前部分，开始新部分
                if current_tokens + line_tokens > max_tokens:
                    parts.append(''.join(current_part))  # 将当前部分合并为字符串并存储
                    current_part = []  # 清空当前部分
                    current_tokens = 0  # 重置 token 计数

                # 将当前行添加到当前部分
                current_part.append(line)
                current_tokens += line_tokens

            # 如果最后还有剩余的部分，加入到结果中
            if current_part:
                parts.append(''.join(current_part))
            return parts
    except Exception as e:
        return str(e)

cache_file_dict ={}
# 模拟读取文件内容
def read_file(file_path):
    addr_message = ""
    if file_path in cache_file_dict:
        current_part = cache_file_dict[file_path][0]
        part_num = len(cache_file_dict[file_path][1])
        if current_part + 1 == part_num:
            cache_file_dict[file_path][0] = part_num - 1
        else:
            cache_file_dict[file_path][0]+=1 
        n = cache_file_dict[file_path][0]
        addr_message = """文件过大，所以拆分为{part_num}个部分，返回开始的第{n}部分"""
        return [file_parts[n], addr_message]
    file_parts = split_file_by_tokens(file_path, MAX_TOKENS)
    parts_num = len(file_parts)
    if parts_num > 1 :
        cache_file_dict[file_path] = [0,file_parts]
        addr_message = """文件过大，所以拆分为{parts_num}个部分，返回开始的第一部分"""
    return [file_parts[0], addr_message]

def extract_json(text):
    # 使用正则表达式匹配JSON格式的内容
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group()
        try:
            # 尝试将字符串解析为JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Error: The extracted string is not a valid JSON format.")
            return None
    else:
        print("Error: No JSON object found.")
        return None

# 获取用户自定义反馈
def get_user_feedback():
    feedback = input("请提供进一步的提示或反馈（留空则跳过）：")
    return feedback if feedback.strip() != "" else None

# 向量数据库模拟检索
def search_code_in_vector_db(input_paras, vdb_mgr):
    res = ''
    input_paras_split = input_paras.split(',')
    for input_para in input_paras_split:
        search_docs = vdb_mgr.search(input_para, 10)
        if len(search_docs) == 0:
            continue
        docstr = json.dumps(search_docs[0], ensure_ascii=False, indent=4)
        res+= docstr
        res+='\n'
    return res

# 调用 LangChain 的 LLM 进行代码分析
def analyze_code_with_context(question, final_response):
    analyze_prompt = PromptTemplate(
        template = """你选择的工具返回结果:
        {tool_response}
        根据工具结果，并结合历史聊天记录对背景问题进行分析，得出目前的结论，并指出下一步的分析方向""",
        input_variables = ["tool_response"]
    )
    return chain_with_history.invoke({"system_prompt":"背景问题是"+question, "question": analyze_prompt.format(tool_response = final_response)}, config={"configurable": {"session_id": session_id}})
tool_descriptions = {
    "FileReadTool": "输入：文件路径列表。输出：文件内容的文本以及额外信息（文件过大时会输出部分文件，额外信息中会注明是第几部分）。用于读取代码库中的文件内容，适合需要直接查看源代码的情况。返回格式：{\"file_path\": \"...\", \"additonal_message\": \"...\",\"content\": \"...\"}",
    "VectorSearchTool": "输入：查询字符串(你认为需要查询的关键问题，请务必用英文)。输出：相关代码片段的列表。用于在代码库中进行向量化检索，适合根据特定关键词或主题查找相关代码片段。返回格式：{\"results\": [...]}",
}
# 自定义的 agent 实现
def custom_agent(question, chain_with_history, vdb_mgr, file_structure, codebase_directory):
    iteration_count = 0
    max_iterations = 50
    final_response = None
    user_feedback = ""
    while iteration_count < max_iterations:
        # 使用 LLM 选择合适的工具

        tool_selection_prompt = PromptTemplate(
            template="""用户反馈：
{user_feedback}
为解决系统提示中的背景问题，请结合背景问题,上一轮的用户反馈,以及我让你记住的代码目录结构，可用工具列表，从我给你的可用工具列表中选择最合适的工具，输出工具的标准名称和你认为合适的输入参数以及原因:
请保证输出格式为 JSON，不要有其他多余字样，回答范例如下，严格按以下格式回答，不要有其他字符串：
{{"selected_tool": "工具名称", "input_paras": "参数的字符串(比如FileReadTool，输入你认为这次需要读取的文件路径列表,用逗号隔开;比如VectorSearchTool,输入你认为应该查询的关键问题,用逗号隔开)",“reason": "这样选择和给出这样参数的原因"}}""",
            input_variables=["user_feedback"]
        )

        format_tool_prompt = tool_selection_prompt.format(user_feedback=user_feedback)
        selected_tool_response = chain_with_history.invoke({"system_prompt": '总的背景是要解决:'+ question,"question": format_tool_prompt}, config={"configurable": {"session_id": session_id}})
        sec_data = extract_json(selected_tool_response.content)
        selected_tool = sec_data["selected_tool"]
        input_paras = sec_data["input_paras"]

        # 根据 LLM 选择的工具进行操作
        if selected_tool == "FileReadTool":
            selected_files = input_paras.split(',')
            combined_results = []
            for file in selected_files:
                file_path = os.path.join(codebase_directory, file.strip())
                file_res = read_file(file_path)
                file_content = file_res[0]
                additonal_message = file_res[1]
                combined_results.append({"file_path": file_path, "additonal_message": additonal_message,"content": file_content})
            final_response = {"results": combined_results} if combined_results else {"message": "未找到相关代码。"}
        elif selected_tool == "VectorSearchTool":
            code_snippets = search_code_in_vector_db(input_paras, vdb_mgr)
            final_response = {"results": code_snippets} if code_snippets != "No relevant results found." else {"message": "未找到相关代码片段。"}

        else:
            final_response = {"message": "未识别的问题类型，请尝试提供更明确的需求。"}
        analysis = analyze_code_with_context(question, final_response)
        # 使用 LLM 评估是否需要继续迭代，加入人类反馈
        evaluation_prompt = PromptTemplate(
            template="""这次你选择的工具返回的结果为：
{response}
以下是当前的分析结果：
{analysis}
请评估是否已经收集到足够的信息和得到足够翔实的结论来回答系统提示中的背景问题，或者是否需要进一步的操作。保证输出的格式为json,不要包含其他的多余字样，如：
{{"continue_iteration": true/false, "reason": "..."}}""",
            input_variables=["response", "analysis"]
        )
        evaluation_raw = chain_with_history.invoke({"system_prompt": '总的背景是要解决如下问题:'+ question,"question":evaluation_prompt.format(response=json.dumps(final_response), analysis = analysis)},
                                                        config={"configurable": {"session_id": session_id}})
        evaluation_response = extract_json(evaluation_raw.content)
        
        if evaluation_response["continue_iteration"] != True:
            break

        iteration_count += 1
        user_feedback = get_user_feedback()
    return [analysis, evaluation_response]

# 一开始获取代码库的文件结构并存储
codebase_directory = r"D:\sql\openGauss-server"
file_structure = get_codebase_structure(codebase_directory)

embed_model = OllamaEmbeddings(model="unclemusclez/jina-embeddings-v2-base-code")
vdb_mgr = GenVectorStore(embed_model)

store_path = f"{codebase_directory}/vector_store"
vdb_mgr.get_or_create_vector_store([],store_path)

# 使用自定义 agent
def main():
    question = " 已知这是一个数据库内核代码，请你应用我提供的工具，获取你需要的代码，分析数据库回放相关线程的工作原理"
    background_prompt = PromptTemplate(
        template="""请务必记住以下信息，后面分析会用到：
可用工具：
{tools}
代码仓目录结构：
{file_structure}
目前你只需记住以上信息，不需要回答太多，后面会用到
""",

        input_variables=["tools","file_structure"]
    )
    tools_description_text = "\n".join([f"{name}: {desc}" for name, desc in tool_descriptions.items()])
    format_back_prompt = background_prompt.format(tools = tools_description_text, file_structure = file_structure)
    back_response = chain_with_history.invoke({"system_prompt": '你是一个擅长搜索代码并分析代码的代码分析员，总的背景是要回答如下问题:'+ question,
        "question": format_back_prompt},
        config={"configurable": {"session_id": session_id}})
    print(back_response.content)

    while True:
        continue_flag = input("输入这轮迭代的操作指令\n")
        if continue_flag.lower() in ["exit", "quit"]:
            break
        response = custom_agent(question, chain_with_history, vdb_mgr, file_structure, codebase_directory)
        print(f"响应：\n{response}")

if __name__ == "__main__":
    main()
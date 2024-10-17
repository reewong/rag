from parser import parse_doxygen_output, generate_directory_structure, get_basic_structure_by_cmake
from call_graph_manager import CallGraphManager
from retriever import GenVectorStore
from neo4j_manager import Neo4jManager
from processor import Query_Processor
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from pathlib import Path
import config
def get_project_version(project_path):
    # 这个函数应该返回项目的当前版本
    # 可以从版本控制系统（如 git）获取，或者从项目的某个配置文件中读取
    # 这里只是一个示例实现
    return "1.0.0"  # 替换为实际的版本获取逻辑
def analyze_cpp_project(project_path: str, query: str) -> str:
    llm = Ollama(
        # model="mistral-nemo:12b-instruct-2407-q8_0", 
        model = "gemma2:27b",
        # callback_manager=CallbackManager([debug_handler]), 
        # docker专用
        # base_url="http://host.docker.internal:11434"
    )
#   basic file
    exclude_dirs = {'.git', '.vscode', 'build', 'doxygen_output', 'html'}  # 你可能想要排除的目录
    exclude_files = {'.gitignore', 'README.md'}  # 你可能想要排除的文件

    directory_structure = generate_directory_structure(project_path, exclude_dirs, exclude_files)
    cmake_module_structure = get_basic_structure_by_cmake(project_path)
    print(directory_structure)
    print(cmake_module_structure)
    
    
    doxygen_output_path = Path(f"{project_path}/doxygen_output")
    parsed_data, relationships = parse_doxygen_output(doxygen_output_path)
    call_graph = CallGraphManager(parsed_data, relationships)
    call_graph.build_call_graph()
    call_graph.get_related_functions('StartupXLOG', 1)
    embed_model= OllamaEmbeddings(model="unclemusclez/jina-embeddings-v2-base-code")
    source_code_vdb_mgr = GenVectorStore(embed_model)
    store_path = f"{project_path}/vector_store"
    source_code_vdb_mgr.get_or_create_vector_store(project_path, store_path)

    # Populate graph database
    current_version = get_project_version(project_path)
    graph_db_mgr = Neo4jManager(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD, llm)
    graph_db_mgr.populate_graph_database(parsed_data, relationships, current_version)
    
    # Process the query
    query_processor = Query_Processor(llm, graph_db_mgr, source_code_vdb_mgr, call_graph, directory_structure, cmake_module_structure)
    query_result = query_processor.process_query(query)    
    
    graph_db_mgr.close()

    return query_result

if __name__ == "__main__":
    project_path = r"D:\sql\openGauss-server"
    # project_path = r"/home/code/sql/openGauss-server"
    user_query = "已知本代码仓是一个数据库内核的代码，基于代码仓代码片段，讲讲数据库redo过程"
    
    answer = analyze_cpp_project(project_path, user_query)
    print(answer) 
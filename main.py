from static_parser.parser import parse_doxygen_output, generate_directory_structure, get_basic_structure_by_cmake
from code_splitter import load_and_splitter
from call_graph.call_graph_manager import CallGraphManager
from code_retrieval.retriever import GenVectorStore
# from vector_store.embeddings import create_vector_embeddings
from graph_database.neo4j_manager import Neo4jManager
from query_processing.processor import Query_Processor
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import config

def analyze_cpp_project(project_path: str, query: str) -> str:
    llm = Ollama(
        # model="mistral-nemo:12b-instruct-2407-q8_0", 
        model = "gemma2:27b",
        # callback_manager=CallbackManager([debug_handler]), 
        # docker专用
        # base_url="http://host.docker.internal:11434"
    )
#   basic file
    exclude_dirs = {'.git', '.vscode', 'build'}  # 你可能想要排除的目录
    exclude_files = {'.gitignore', 'README.md'}  # 你可能想要排除的文件

    directory_structure = generate_directory_structure(project_path, exclude_dirs, exclude_files)
    cmake_module_structure = get_basic_structure_by_cmake(project_path)
    
    split_code = load_and_splitter(project_path)

    parsed_static_docs = parse_doxygen_output(f"{project_path}/doxygen_output")
    call_graph = CallGraphManager()
    call_graph.build_call_graph(f"{project_path}/doxygen_output")
    embed_model= OllamaEmbeddings(model="unclemusclez/jina-embeddings-v2-base-code")
    source_code_vdb_mgr = GenVectorStore(embed_model)
    source_code_vdb_mgr.create_vector_store(split_code)


    # Populate graph database
    graph_db_mgr = Neo4jManager(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, llm)
    graph_db_mgr.populate_graph_database(parsed_static_docs)
    
    # Process the query
    query_processor = Query_Processor(llm, graph_db_mgr, source_code_vdb_mgr, call_graph, directory_structure, cmake_module_structure)
    query_result = query_processor.process_query(query)    
    
    graph_db_mgr.close()
    
    return query_result

if __name__ == "__main__":
    project_path = "/path/to/cpp/project"
    user_query = "Explain the main function and its key dependencies"
    
    answer = analyze_cpp_project(project_path, user_query)
    print(answer)
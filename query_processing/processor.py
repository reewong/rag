from typing import List
import numpy as np
from neo4j import GraphDatabase
from graph_database.neo4j_manager import Neo4jManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from call_graph.call_graph_manager import CallGraphManager
from typing import List, Dict, Set, Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from code_retrieval.retriever import CodeRetriever
decomposition_prompt = PromptTemplate(
    input_variables=["question"],
    template="Break down this complex question into simpler sub-questions:\n{question}\nSub-questions:"
)
class Query_Processor:
    def __init__(self, llm, gragh_db_mgr, source_code_store_mgr, call_gragh_mgr) -> None:
        self.llm = llm
        self.gragh_db_mgr = gragh_db_mgr
        self.source_mgr = source_code_manager(source_code_store_mgr)
        self.retriever = CodeRetriever(source_code_store_mgr.get_vector_store()) 
        self.call_graph_mgr = call_gragh_mgr
        self.parser = StrOutputParser()
    def decompose(self, question: str) -> list:
        chain = decomposition_prompt | self.llm | self.parser
        result = chain.invoke(question)
        return [q.strip() for q in result.split('\n') if q.strip()]
    def _synthesize_answers(self, answers: List[str]) -> str:
        # Combine answers from sub-questions into a coherent response
        # This could be improved with more sophisticated NLP techniques
        return "\n\n".join([f"Sub-answer {i+1}: {answer}" for i, answer in enumerate(answers)])
    def process_query(self, query: str) -> str:
        print(f"Processing query: {query}")
        # Step 1: Decompose the question if it's complex
        sub_questions = self.decompose(query)
        
        if len(sub_questions) > 1:
            print(f"Decomposed into {len(sub_questions)} sub-questions")
            answers = []
            for sub_q in sub_questions:
                answers.append(self._answer_question(sub_q))
            return self._synthesize_answers(answers)
        else:
            return self._answer_question(query)
        
    def _answer_question(self, question: str) -> str:
        relevant_docs = self.retriever.get_relevant_documents(question)
        graph_info = self.gragh_db_mgr.query_graph_database(question, relevant_docs)
        
        # Identify the main function in question
        main_function = self._identify_main_function(question, relevant_docs)
        if main_function:
            # Get related functions based on call graph
            related_functions = self._get_related_functions(main_function)
            
            # Fetch code snippets for main and related functions
            code_snippets = self._get_code_snippets([main_function] + list(related_functions))
        # Construct prompt with call graph information
        prompt = self._construct_prompt(question, relevant_docs, graph_info, code_snippets, main_function, related_functions)
        
        # Generate answer using the updated prompt
        answer = self.qa_system.answer_question(prompt)
        
        return answer

    def _identify_main_function(self, question: str, relevant_docs: List[Dict]) -> str:
        if not relevant_docs:
            return ""
        # Prepare the prompt for the LLM
        prompt = self._prepare_main_function_prompt(question, relevant_docs)

        chain = prompt | self.llm | self.parser
        result = chain.invoke(prompt)
        return result.strip()

    def _prepare_main_function_prompt(self, question: str, relevant_docs: List[Dict]) -> str:
        prompt = f"""Given the following user question and relevant code snippets, identify which function should be considered the main function for answering the user's question. Return only the name of the function.

User Question: {question}

Relevant Code Snippets:
"""
        for i, doc in enumerate(relevant_docs, 1):
            prompt += f"\nSnippet {i}:\n"
            prompt += f"Function Name: {doc['metadata']['name']}\n"
            prompt += f"Code:\n{doc['metadata'].get('snippet', 'No code available')}\n"

        prompt += "\nBased on these snippets, which function is most relevant to answering the user's question? Please return only the function name.Then you may exlain your choice"

        return prompt


    def _get_code_snippets(self, functions: List[str]) -> Dict[str, str]:
        snippets = {}
        for func in functions:
            snippet = self.source_mgr.search_fucntion_def(func)        
            snippets[func] = snippet
        return snippets    

    def _get_relevant_code_snippets(self, relevant_docs: List[Dict]) -> List[str]:
        snippets = []
        for i, doc in enumerate(relevant_docs, 1):
            prompt = ''
            prompt += f"\nSnippet {i}:\n"
            prompt += f"Function Name: {doc['metadata']['name']}\n"
            prompt += f"Code:\n{doc['metadata'].get('snippet', 'No code available')}\n"
            snippets.append(prompt)
        return snippets
    def _get_related_functions(self, main_function: str, max_depth: int = 2) -> Set[str]:
        called_functions = self.call_graph_mgr.get_called_functions(main_function, max_depth)
        calling_functions = self.call_graph_mgr.get_calling_functions(main_function, 1)  # Only immediate callers
        return called_functions.union(calling_functions)

        

    def _construct_prompt(self, question: str, relevant_docs: List[Dict], graph_info: str, 
                            code_snippets: Dict[str, str], main_function: str, related_functions: Set[str]) -> str:
        prompt = f"""
        User Query: {question}

        Main Function: {main_function}

        Related Functions:
        {', '.join(related_functions)}

        Call Graph Information:
        {self._format_call_graph_info(main_function, related_functions)}

        Code Snippets:
        """
        for func, snippet in code_snippets.items():
            prompt += f"\nFunction: {func}\n{snippet}\n"

        prompt += f"""
        Additional Context:
        {self._format_relevant_docs(relevant_docs)}
        {graph_info}

        Instructions:
        1. Analyze the main function and its related functions based on the call graph.
        2. Explain the purpose and functionality of the main function.
        3. Describe how the related functions contribute to or interact with the main function.
        4. If relevant, discuss the flow of data or control through these functions.
        5. Reference specific parts of the code snippets to support your explanation.
        6. If you need more information about any function or its context, say so explicitly.

        Please provide a comprehensive answer based on this information.
        """
        return prompt
    def _format_relevant_docs(self, relevant_docs: List[Dict]):
        prompt = ''        
        for i, doc in enumerate(relevant_docs, 1):
            prompt += f"\nSnippet {i}:\n"
            prompt += f"Function Name: {doc['metadata']['name']}\n"
            prompt += f"Code:\n{doc['metadata'].get('snippet', 'No code available')}\n"
        return prompt
    def _format_call_graph_info(self, main_function: str, related_functions: Set[str]) -> str:
        info = f"Main Function: {main_function}\n"
        info += "Calls:\n"
        for func in related_functions:
            if func in self.call_graph_mgr.get_called_functions(main_function, 1):
                info += f"  - {func}\n"
        info += "Called By:\n"
        for func in related_functions:
            if func in self.call_graph_mgr.get_calling_functions(main_function, 1):
                info += f"  - {func}\n"
        return info           
import tree_sitter    
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
class source_code_manager:
    def __init__(self, source_db_mgr):        
        self.cpp_parser = tree_sitter.Parser()
        self.cpp_parser.set_language(tree_sitter.Language('path/to/tree-sitter-cpp', 'cpp')) # 替换为你的 tree-sitter-cpp 路径
        self.source_db_mgr = source_db_mgr
    def search_fucntion_def(self, function_name_to_search): 
        coarse_results = self.coarse_search(function_name_to_search)
        definition = self.precise_search([result.page_content for result in coarse_results])
        return definition
    # 2. 解析 C++ 代码并提取函数定义
    def extract_function_definitions(self, code):
        tree = self.cpp_parser.parse(code.encode())
        definitions = []

        def traverse_tree(node):
            if node.type == 'function_definition':
                function_name = node.child_by_field_name('declarator').text.decode()
                function_body = code[node.start_byte:node.end_byte].decode() 
                definitions.append((function_name, function_body))
            for child in node.children:
                traverse_tree(child)
        traverse_tree(tree.root_node)
        return definitions

    # 2. 粗略搜索
    def coarse_search(self, function_name, k=3): # k 是返回的片段数量
        results = self.source_db_mgr.search(function_name, k=k)
        return results

    # 3. 精确解析
    def precise_search(self, code_snippets, function_name):
        for snippet in code_snippets:
            function_definitions = self.extract_function_definitions(snippet)
            for func_name, func_body in function_definitions:
                if func_name == function_name:
                    return func_body
        return "============no definitions============"
    
from typing import List
import numpy as np
from neo4j import GraphDatabase
from neo4j_manager import Neo4jManager
# from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from call_graph_manager import CallGraphManager
from typing import List, Dict, Set, Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
import json
from pydantic import BaseModel, Field

class SubQuestions(BaseModel):
    questions: List[str] = Field(description="A list of sub-questions")
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
class Query_Processor:
    def __init__(self, llm, gragh_db_mgr, source_code_store_mgr, call_gragh_mgr, directory_structure, cmake_module_structure) -> None:
        self.llm = llm
        self.gragh_db_mgr = gragh_db_mgr
        self.source_mgr = source_code_manager(source_code_store_mgr)
        self.retriever = source_code_store_mgr.get_retriever() 
        self.call_graph_mgr = call_gragh_mgr
        self.parser = StrOutputParser()
        self.directory_structure = directory_structure
        self.cmake_module_structure = cmake_module_structure
        self.decomposition_parser = PydanticOutputParser(pydantic_object=SubQuestions)

        self.decomposition_prompt = PromptTemplate(
            template="为了全面而透彻的理解这个问题，并结合一般人类读代码的习惯，可以将以下问题划分几个简单的子问题."
                     "提示一下，如果是询问的是一个简单的函数等局部代码片段就可以解答的问题，就不需要划分子问题，如果问的是比较复杂的问题，按照问题涉及的过程,"
                     "以你对背景的理解，把这个过程拆分为几步,每一步对应一个子问题" 
                     "请以JSON格式提供您的答案，使用'questions'键包含一个字符串列表。代表子问题的内容\n\n"
                     "复杂问题：{question}\n\n"
                     "在子问题中，请确保包含原问题的背景和上下文\n\n"
                     "JSON输出：",
            input_variables=["question"],
            partial_variables={"format_instructions": self.decomposition_parser.get_format_instructions()}
        )

    def decompose(self, question: str) -> List[str]:
        chain = self.decomposition_prompt | self.llm | self.parser
        result = chain.invoke(question)

        try:
            parsed_output = self.decomposition_parser.parse(result)
            if not parsed_output.questions:
                raise ValueError("No sub-questions generated")
            return parsed_output.questions
        except json.JSONDecodeError:
            print(f"Failed to parse LLM output as JSON: {result}")
            return [question]  # 如果解析失败，返回原问题
        except Exception as e:
            print(f"Error in decomposing question: {e}")
            return [question] 
    def _synthesize_answers(self, answers: List[str]) -> str:
        # Combine answers from sub-questions into a coherent response
        # This could be improved with more sophisticated NLP techniques
        return "\n\n".join([f"Sub-answer {i+1}: {answer}" for i, answer in enumerate(answers)])
    def process_query(self, query: str, ) -> str:
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
        relevant_docs = self.retriever.invoke(question)
        graph_info = self.gragh_db_mgr.query_graph_database(question, relevant_docs)
        
        # Identify the main function in question
        main_function = self._identify_main_function(question, relevant_docs)
        if main_function:
            # Get related functions based on call graph
            related_functions = self._get_related_functions(main_function)
            
            # Fetch code snippets for main and related functions
            code_snippets = self._get_code_snippets([main_function] + list(related_functions))
        else:
            related_functions = set()
            code_snippets = {}
        # Construct prompt with call graph information
        raw_prompt = self._construct_prompt(question, relevant_docs, graph_info, code_snippets, main_function, related_functions)
        # prompt = PromptTemplate.from_template(raw_prompt)
        # chain = prompt | self.llm | self.parser
        answer = self.llm.invoke(raw_prompt)
        
        return answer

    def _identify_main_function(self, question: str, relevant_docs: List[Dict]) -> str:
        if not relevant_docs:
            return ""
        # Prepare the prompt for the LLM
        raw_prompt = self._prepare_main_function_prompt(question, relevant_docs)
        result = self.llm.invoke(raw_prompt)
        lines = result.splitlines()
        function_name = lines[0].strip() 
        return function_name

    def _prepare_main_function_prompt(self, question: str, relevant_docs: List[Dict]) -> str:
        prompt = f"""Related Code Snippets:"""
        for i, doc in enumerate(relevant_docs, 1):
            prompt += f"\nSnippet {i}:\n"
            prompt += f"SourceFile: {doc.metadata}\n"
            prompt += f"Page content: {doc.page_content}\n"

            prompt += f"""Given the following user question and relevant code snippets above, identify which function should be considered the main function for answering the user's question. Return the complete function name, including the namespace or class name if applicable, according to C++ syntax.

User Question: {question}

In other words, from the code snippets above, which function is most relevant to answering the user's question? Please return only the complete function name. Then you may explain your choice."""

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
            prompt += f"\nSnippet {i}:\n"
            prompt += f"SourceFile: {doc.metadata}\n"
            prompt += f"Page content: {doc.page_content}\n"
            snippets.append(prompt)
        return snippets
    def _get_related_functions(self, main_function: str, max_depth: int = 2) -> Set[str]:
        return self.call_graph_mgr.get_related_functions(main_function, max_depth)

        

    def _construct_prompt(self, question: str, relevant_docs: List[Dict], graph_info: str, 
                            code_snippets: Dict[str, str], main_function: str, related_functions: Set[str]) -> str:
        prompt = f"""
        用户问题: {question}

        主函数: {main_function}

        相关函数:
        {', '.join(related_functions)}

        调用图信息:
        {self._format_call_graph_info(main_function, related_functions)}

        代码片段:
        """
        for func, snippet in code_snippets.items():
            prompt += f"\nFunction: {func}\n{snippet}\n"

        prompt += f"""
        项目基本结构:
        {self.directory_structure}
        {self.cmake_module_structure}
        其余信息:
        {self._format_relevant_docs(relevant_docs)}
        {graph_info}

        指令：
        根据调用图分析主函数及其相关函数。
        解释主函数的目的和功能。
        描述相关函数如何为主函数做出贡献或与主函数交互。
        如果相关，讨论数据或控制流通过这些函数的方式。
        参考代码片段的特定部分以支持您的解释。
        如果您需要有关任何功能或其上下文的更多信息，请明确说明。
        """
        print(prompt)
        return prompt
    def _format_relevant_docs(self, relevant_docs: List[Dict]):
        prompt = ''        
        for i, doc in enumerate(relevant_docs, 1):
            prompt += f"\nSnippet {i}:\n"
            prompt += f"SourceFile: {doc.metadata}\n"
            prompt += f"Page content: {doc.page_content}\n"
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
class source_code_manager:
    def __init__(self, source_db_mgr):        
        self.cpp_parser = tree_sitter.Parser()
        self.cpp_parser.set_language(tree_sitter.Language(r'E:\tree-sitter-cpp\tree_sitter_cpp.dll', 'cpp')) # 替换为你的 tree-sitter-cpp 路径
        self.source_db_mgr = source_db_mgr
    def search_fucntion_def(self, function_name_to_search): 
        coarse_results = self.coarse_search(function_name_to_search)
        definition = self.precise_search([result['page_content'] for result in coarse_results], function_name_to_search)
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
    def coarse_search(self, function_name, k=10): # k 是返回的片段数量
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
    
import networkx as nx
from typing import List, Dict, Set, Tuple
import tree_sitter  
class CallGraphManager:
    # def __init__(self, parsed_data: Dict, relationships: List[Tuple[str, str, str]]):
    #     self.call_graph = nx.DiGraph()
    #     self.cpp_parser = tree_sitter.Parser()
    #     # self.cpp_parser.set_language('path to so',tree_sitter.Language('cpp'))
    #     self.relationships = relationships
    #     self.parsed_data = parsed_data

    # def build_call_graph(self):
    #          # Build initial graph from relationships
    #     for from_entity, rel_type, to_entity in self.relationships:
    #         if rel_type == 'CALLS':
    #             self.call_graph.add_edge(from_entity, to_entity)
        
    #     # Process function bodies for more detailed analysis
    #     self._process_function_bodies(self.parsed_data['functions'])
    def __init__(self, parsed_data: Dict, relationships: List[Tuple[str, str, str]]):
        self.call_graph = nx.DiGraph()
        self.relationships = relationships
        self.parsed_data = parsed_data
        # self.build_call_graph(relationships)

    def build_call_graph(self):
        for from_entity, rel_type, to_entity in self.relationships:
            if rel_type == 'CALLS':
                self.call_graph.add_edge(from_entity, to_entity)
    # def print_call_graph(self):
    #     print(self.call_graph)
    # def _process_function_bodies(self, functions: Dict):
    #     for func_name, func_data in functions.items():
    #         func_body = func_data.get('details', '')  # Assuming 'details' contains the function body
    #         called_functions = self._analyze_function_body(func_body)
    #         for called_func in called_functions:
    #             self.call_graph.add_edge(func_name, called_func)

    # def _analyze_function_body(self, func_body: str) -> List[str]:
    #     called_functions = []
    #     tree = self.cpp_parser.parse(func_body.encode())
        
    #     def traverse(node):
    #         if node.type == 'call_expression':
    #             func_name = node.child_by_field_name('function').text.decode()
    #             called_functions.append(func_name)
    #         for child in node.children:
    #             traverse(child)
        
    #     traverse(tree.root_node)
        # return called_functions    
    def add_function_call(self, caller: str, callee: str):
        self.call_graph.add_edge(caller, callee)

    def get_called_functions(self, function_name: str, depth: int = 1) -> Set[str]:
        if depth == 0:
            return set()
        called_functions = set(self.call_graph.successors(function_name))
        for func in list(called_functions):
            called_functions.update(self.get_called_functions(func, depth - 1))
        print(called_functions)
        return called_functions

    def get_calling_functions(self, function_name: str, depth: int = 1) -> Set[str]:
        if depth == 0:
            return set()
        calling_functions = set(self.call_graph.predecessors(function_name))
        for func in list(calling_functions):
            calling_functions.update(self.get_calling_functions(func, depth - 1))
        print(calling_functions)
        return calling_functions
    def get_related_functions(self, main_function: str, max_depth: int = 2) -> Set[str]:
        called_functions = self.get_called_functions(main_function, max_depth)
        calling_functions = self.get_calling_functions(main_function, 1)  # Only immediate callers
        return called_functions.union(calling_functions)

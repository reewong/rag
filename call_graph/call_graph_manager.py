import networkx as nx
from typing import List, Dict, Set
class CallGraphManager:
    def __init__(self):
        self.call_graph = nx.DiGraph()
    def build_call_graph(self, static_path):
        pass
    def add_function_call(self, caller: str, callee: str):
        self.call_graph.add_edge(caller, callee)

    def get_called_functions(self, function_name: str, depth: int = 1) -> Set[str]:
        if depth == 0:
            return set()
        called_functions = set(self.call_graph.successors(function_name))
        for func in list(called_functions):
            called_functions.update(self.get_called_functions(func, depth - 1))
        return called_functions

    def get_calling_functions(self, function_name: str, depth: int = 1) -> Set[str]:
        if depth == 0:
            return set()
        calling_functions = set(self.call_graph.predecessors(function_name))
        for func in list(calling_functions):
            calling_functions.update(self.get_calling_functions(func, depth - 1))
        return calling_functions
    def _get_related_functions(self,main_function: str, max_depth: int = 2) -> Set[str]:
        called_functions = self.get_called_functions(main_function, max_depth)
        calling_functions = self.get_calling_functions(main_function, 1)  # Only immediate callers
        return called_functions.union(calling_functions)

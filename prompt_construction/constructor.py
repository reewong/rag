from data_structures import PromptComponent, QueryResult

def construct_prompt(query: str, query_result: QueryResult, code_snippets: Dict[str, str]) -> PromptComponent:
    main_entity = query_result.relevant_entities[0] if query_result.relevant_entities else None
    
    return PromptComponent(
        question=query,
        main_entity=main_entity,
        related_entities=query_result.relevant_entities[1:],
        relationships=query_result.relevant_relationships,
        code_snippets=code_snippets
    )

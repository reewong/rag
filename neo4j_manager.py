from neo4j import GraphDatabase
from typing import List, Dict, Tuple
from langchain_core.output_parsers import StrOutputParser

class Neo4jManager:
    def __init__(self, uri, user, password, llm):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.llm = llm
        self.parser = StrOutputParser()
    def close(self):
        self.driver.close()

    def add_relationship(self, from_entity, to_entity, relationship_type):
        with self.driver.session() as session:
            session.run(
                "MERGE (a:CodeEntity {name: $from_name}) "
                "MERGE (b:CodeEntity {name: $to_name}) "
                "MERGE (a)-[r:$rel_type]->(b)",
                from_name=from_entity, to_name=to_entity, rel_type=relationship_type
            )

    def add_entity_details(self, entity_name, entity_type, details, file, line):
        with self.driver.session() as session:
            session.run(
                "MERGE (e:CodeEntity {name: $name}) "
                "SET e.type = $type, e.details = $details, e.file = $file, e.line = $line",
                name=entity_name, type=entity_type, details=details, file=file, line=line
            )

    def query_relationships(self, entity_name):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (a:CodeEntity {name: $name})-[r]->(b) "
                "RETURN type(r) as relationship, b.name as related_entity, b.type as related_type",
                name=entity_name
            )
            return [record for record in result]
    def populate_graph_database(self, parsed_data: Dict, relationships: List[Tuple[str, str, str]]):
        for entity_type, entities in parsed_data.items():
            for entity_name, entity_data in entities.items():
                self.graph_db.add_entity_details(
                    entity_name,
                    entity_type,
                    entity_data['details'],
                    entity_data['file'],
                    entity_data['line']
                )
        
        for from_entity, relationship_type, to_entity in relationships:
            self.graph_db.add_relationship(from_entity, to_entity, relationship_type)

    def query_graph_database(self, question: str) -> str:
        # Extract entity names from the question and query the graph database
        # This is a simplified version and might need more sophisticated NLP techniques
        words = question.split()
        for word in words:
            if len(word) > 3:  # Assuming entity names are longer than 3 characters
                relationships = self.graph_db.query_relationships(word)
                if relationships:
                    return f"Relationships for {word}: " + ", ".join([f"{r['related_entity']} ({r['relationship']})" for r in relationships])
        return ""
    def query_graph_database(self, question: str, relevant_docs: List[Dict]) -> str:
        # Step 1: Identify key entities
        key_entities = self._identify_key_entities(question, relevant_docs)
        
        # Step 2: Query relationships for key entities
        relationships = self._query_relationships_for_entities(key_entities)
        
        # Step 3: Format graph info
        graph_info = self._format_graph_info(relationships)
        
        return graph_info

    def _identify_key_entities(self, question: str, relevant_docs: List[Dict]) -> List[str]:
        prompt = self._prepare_key_entities_prompt(question, relevant_docs)
        chain = self.llm | self.parser
        result = chain.invoke(prompt)
        return self._parse_key_entities_response(result)

    def _prepare_key_entities_prompt(self, question: str, relevant_docs: List[Dict]) -> str:
        prompt = f"""Given the following user question and relevant code snippets, identify the key entities (classes, functions, or variables) that are most relevant to answering the question. Return only the names of these entities, separated by commas.

User Question: {question}

Relevant Code Snippets:
"""
        for i, doc in enumerate(relevant_docs, 1):
            prompt += f"\nSnippet {i}:\n"
            prompt += f"Name: {doc['metadata']['name']}\n"
            prompt += f"Type: {doc['metadata']['type']}\n"
            prompt += f"Description: {doc['metadata'].get('description', 'No description available')}\n"

        prompt += "\nBased on this information, what are the key entities most relevant to the user's question? Please list only the entity names, separated by commas."

        return prompt

    def _parse_key_entities_response(self, llm_response: str) -> List[str]:
        return [entity.strip() for entity in llm_response.split(',')]

    def _query_relationships_for_entities(self, entities: List[str]) -> List[Dict]:
        all_relationships = []
        for entity in entities:
            relationships = self.query_relationships(entity)
            all_relationships.extend(relationships)
        return all_relationships

    def _format_graph_info(self, relationships: List[Dict]) -> str:
        formatted_info = "Graph Relationships:\n"
        for rel in relationships:
            formatted_info += f"- {rel.source} {rel.type} {rel.target}\n"
        return formatted_info
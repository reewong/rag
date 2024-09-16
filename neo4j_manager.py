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

    def populate_graph_database(self, parsed_data, relationships, current_version):
        with self.driver.session() as session:
            if not self._is_update_needed(session, current_version):
                print("Database is up to date")
                return

            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            # Add new data
            self._add_entities(session, parsed_data)
            self._add_relationships(session, relationships)

            # Update version
            self._update_version(session, current_version)

            print(f"Database updated to version {current_version}")
    def _is_update_needed(self, session, current_version):
        result = session.run("MATCH (v:Version) RETURN v.number AS version")
        db_version = result.single()
        return db_version is None or db_version['version'] != current_version
    def _update_version(self, session, version):
        session.run(
            "MERGE (v:Version) SET v.number = $version",
            version=version
        )
    def _add_entities(self, session, parsed_data: Dict):
        for entity_type, entities in parsed_data.items():
            for entity_name, entity_data in entities.items():
                session.run(
                    "MERGE (e:CodeEntity {name: $name, type: $type}) "
                    "SET e.details = $details",
                    name=entity_name, type=entity_type, details=entity_data.get('details', '')
                )

    def _add_relationships(self, session, relationships: List[Tuple[str, str, str]]):
        for from_entity, relationship_type, to_entity in relationships:
            session.run(
                "MATCH (a:CodeEntity {name: $from_name}), (b:CodeEntity {name: $to_name}) "
                "MERGE (a)-[r:$rel_type]->(b)",
                from_name=from_entity, to_name=to_entity, rel_type=relationship_type
            )

    def query_entity_definition(self, entity_name: str) -> Dict:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:CodeEntity {name: $name}) "
                "RETURN e.type as type, e.details as details",
                name=entity_name
            )
            record = result.single()
            if record:
                return {"type": record["type"], "details": record["details"]}
            return None

    def query_graph_database(self, question: str, relevant_docs: List[Dict]) -> str:
        key_entities = self._identify_key_entities(question, relevant_docs)
        entity_definitions = self._query_entity_definitions(key_entities)
        relationships = self._query_relationships_for_entities(key_entities)
        graph_info = self._format_graph_info(entity_definitions, relationships)
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

    def _query_entity_definitions(self, entities: List[str]) -> Dict[str, Dict]:
        definitions = {}
        for entity in entities:
            definition = self.query_entity_definition(entity)
            if definition:
                definitions[entity] = definition
        return definitions

    def _query_relationships_for_entities(self, entities: List[str]) -> List[Dict]:
        relationships = []
        with self.driver.session() as session:
            for entity in entities:
                result = session.run(
                    "MATCH (a:CodeEntity {name: $name})-[r]->(b:CodeEntity) "
                    "RETURN type(r) as relationship, a.name as source, b.name as target",
                    name=entity
                )
                relationships.extend([dict(record) for record in result])
        return relationships

    def _format_graph_info(self, entity_definitions: Dict[str, Dict], relationships: List[Dict]) -> str:
        graph_info = "Entity Definitions:\n"
        for entity, definition in entity_definitions.items():
            graph_info += f"- {entity} ({definition['type']}): {definition['details']}\n"

        graph_info += "\nRelationships:\n"
        for rel in relationships:
            graph_info += f"- {rel['source']} {rel['relationship']} {rel['target']}\n"

        return graph_info
from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel


class CypherIngestRecord(BaseModel):
    example_id: str
    cypher_statement: str
    question: str
    question_embedding: List[float]
    embedding_model: Optional[str]
    domain: str
    query_type: str
    difficulty: str
    schema_version: str
    required_labels: List[str]
    required_relationships: List[str]
    required_properties: List[str]


class EmbedderResult(TypedDict):
    nodes: List[CypherIngestRecord]
    failed: List[Dict[str, str]]

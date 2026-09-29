"""Create and initialize the Neo4j-backed Cypher example vector store."""

import re
import threading
from typing import Any, Dict, List, Set

from langchain_neo4j import Neo4jGraph
from neo4j import Driver

from app.core.config import settings
from app.lg_agent.kg_sub_graph.schema_normalization import (
    ensure_canonical_ecommerce_schema,
)

from ....embeddings import SentenceTransformerEmbedder
from ....ingest.cypher_examples import embed_cypher_query_nodes, load_cypher_query_nodes
from ..curated_examples import get_curated_examples
from .neo4j_vector_example_retriever import Neo4jVectorSearchCypherExampleRetriever


_INITIALIZED_STORES: Set[str] = set()
_INITIALIZATION_LOCK = threading.Lock()


def _schema_capabilities(graph: Neo4jGraph) -> Dict[str, Set[str]]:
    """Extract the labels, relationship types and qualified properties in use."""

    schema = graph.structured_schema or {}
    node_props = schema.get("node_props", {})
    rel_props = schema.get("rel_props", {})
    labels = set(node_props)
    relationships = {
        str(rel.get("type"))
        for rel in schema.get("relationships", [])
        if rel.get("type")
    }
    properties = {
        f"{label}.{prop['property']}"
        for label, props in node_props.items()
        for prop in props
        if prop.get("property")
    }
    properties.update(
        f"{rel_type}.{prop['property']}"
        for rel_type, props in rel_props.items()
        for prop in props
        if prop.get("property")
    )
    return {
        "labels": labels,
        "relationships": relationships,
        "properties": properties,
    }


def _seed_examples() -> List[Dict[str, Any]]:
    """Return the complete versioned catalogue with retrieval metadata."""

    examples = get_curated_examples()
    if not examples:
        raise RuntimeError("No curated Cypher examples were found for vector indexing")
    return examples


def _ensure_safe_index_name(index_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", index_name):
        raise ValueError(f"Invalid Neo4j vector index name: {index_name!r}")
    return index_name


def _initialize_vector_store(
    driver: Driver,
    database: str,
    index_name: str,
    embedder: SentenceTransformerEmbedder,
) -> None:
    """Create the vector index and idempotently load all Few-shot examples."""

    safe_index_name = _ensure_safe_index_name(index_name)
    dimension = embedder.dimension
    store_key = f"{database}:{safe_index_name}:{embedder.model_name}:{dimension}"

    with _INITIALIZATION_LOCK:
        if store_key in _INITIALIZED_STORES:
            return

        create_index_query = f"""
CREATE VECTOR INDEX `{safe_index_name}` IF NOT EXISTS
FOR (example:CypherQuery) ON (example.questionEmbedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {dimension},
  `vector.similarity_function`: 'cosine'
}}}}
"""
        with driver.session(database=database) as session:
            session.run(create_index_query).consume()

        examples = _seed_examples()
        embedded = embed_cypher_query_nodes(
            embedder=embedder,
            nodes_to_embed=examples,
            embedding_model_name=embedder.model_name,
        )
        if embedded["failed"]:
            raise RuntimeError(
                f"Failed to embed {len(embedded['failed'])} Cypher examples"
            )
        load_cypher_query_nodes(
            driver=driver,
            nodes=embedded["nodes"],
            database=database,
        )

        with driver.session(database=database) as session:
            session.run(
                "CALL db.awaitIndex($index_name, $timeout_seconds)",
                index_name=safe_index_name,
                timeout_seconds=60,
            ).consume()

        _INITIALIZED_STORES.add(store_key)


def create_neo4j_vector_example_retriever(
    graph: Neo4jGraph,
) -> Neo4jVectorSearchCypherExampleRetriever:
    """Build the retriever used by the live Text2Cypher workflow."""

    ensure_canonical_ecommerce_schema(graph)

    driver = getattr(graph, "_driver", None)
    if driver is None:
        raise RuntimeError("Neo4jGraph does not expose a driver for vector retrieval")

    embedder = SentenceTransformerEmbedder(
        model_name=settings.CYPHER_EXAMPLE_EMBEDDING_MODEL
    )
    _initialize_vector_store(
        driver=driver,
        database=settings.NEO4J_DATABASE,
        index_name=settings.CYPHER_EXAMPLE_VECTOR_INDEX,
        embedder=embedder,
    )
    capabilities = _schema_capabilities(graph)
    return Neo4jVectorSearchCypherExampleRetriever(
        neo4j_driver=driver,
        neo4j_database=settings.NEO4J_DATABASE,
        vector_index_name=settings.CYPHER_EXAMPLE_VECTOR_INDEX,
        embedder=embedder,
        schema_version=settings.CYPHER_SCHEMA_VERSION,
        min_score=settings.CYPHER_EXAMPLE_MIN_SCORE,
        schema_labels=capabilities["labels"],
        schema_relationships=capabilities["relationships"],
        schema_properties=capabilities["properties"],
    )

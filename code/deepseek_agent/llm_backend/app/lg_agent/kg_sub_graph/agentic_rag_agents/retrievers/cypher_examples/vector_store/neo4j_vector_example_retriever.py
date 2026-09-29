"""
LangChain Embedder Base: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/embeddings/embeddings.py
Neo4j GraphRAG Embedder Base: https://github.com/neo4j/neo4j-graphrag-python/blob/main/src/neo4j_graphrag/embeddings/base.py
"""

from typing import Any, Dict, List, Set

from neo4j import Driver
from pydantic import Field

from ....embeddings import EmbedderProtocol
from ....exceptions import CypherExampleRetrieverError
from ..base import BaseCypherExampleRetriever


class Neo4jVectorSearchCypherExampleRetriever(BaseCypherExampleRetriever):
    neo4j_driver: Driver = Field(
        description="The Neo4j Python Driver to perform database operations with.",
    )
    neo4j_database: str = Field(
        default="neo4j", description="The Neo4j database name to connect to."
    )
    vector_index_name: str = Field(description="The name of the vector index to use.")
    embedder: EmbedderProtocol = Field(
        description="The embedder to generate an embedding from an input query."
    )
    schema_version: str = Field(description="Required Few-shot schema version.")
    min_score: float = Field(default=0.20, ge=0.0, le=1.0)
    schema_labels: Set[str] = Field(default_factory=set)
    schema_relationships: Set[str] = Field(default_factory=set)
    schema_properties: Set[str] = Field(default_factory=set)

    def get_examples(self, query: str, k: int = 5, *args: Any, **kwargs: Any) -> str:
        """
        Perform vector similarity search between the provided query and queries that exist in the Neo4j database.
        Returns Cypher queries associated with the top K most similar query results.

        Parameters
        ----------
        query : str
            The query to match against.
        k: int, optional
            The number of Cypher statements to return.
        Returns
        -------
        str
            A list of examples as a string.
        """

        domain = str(kwargs.get("domain", ""))
        query_type = str(kwargs.get("query_type", ""))
        schema_version = str(kwargs.get("schema_version", self.schema_version))
        allowed_labels = set(kwargs.get("allowed_labels") or self.schema_labels)
        examples = self._retrieve_examples(
            query,
            k,
            domain=domain,
            query_type=query_type,
            schema_version=schema_version,
            allowed_labels=allowed_labels,
        )
        if len(examples) > 0:
            return self._format_examples_list(examples)
        else:
            return ""

    def _retrieve_examples(
        self,
        query: str,
        k: int,
        *,
        domain: str,
        query_type: str,
        schema_version: str,
        allowed_labels: Set[str],
    ) -> List[Dict[str, Any]]:
        try:
            embedding = self._embed_query(query)
            vector_query = """
CALL db.index.vector.queryNodes($index_name, $candidate_k, $embedding)
YIELD node, score
WHERE node.active = true
  AND node.schemaVersion = $schema_version
  AND ($domain = '' OR node.domain = $domain)
RETURN node.question AS question,
       node.cypherStatement AS cypherStatement,
       node.exampleId AS exampleId,
       node.domain AS domain,
       node.queryType AS queryType,
       node.difficulty AS difficulty,
       node.schemaVersion AS schemaVersion,
       node.requiredLabels AS requiredLabels,
       node.requiredRelationships AS requiredRelationships,
       node.requiredProperties AS requiredProperties,
       score,
       CASE
         WHEN node.queryType = $query_type THEN 3
         WHEN node.queryType CONTAINS $query_type OR $query_type CONTAINS node.queryType THEN 2
         WHEN node.queryType CONTAINS 'aggregation' AND $query_type CONTAINS 'aggregation' THEN 1
         ELSE 0
       END AS queryTypePriority,
       score + CASE
         WHEN node.queryType = $query_type THEN 0.12
         WHEN node.queryType CONTAINS $query_type OR $query_type CONTAINS node.queryType THEN 0.08
         WHEN node.queryType CONTAINS 'aggregation' AND $query_type CONTAINS 'aggregation' THEN 0.07
         ELSE 0.0
       END AS retrievalScore
ORDER BY retrievalScore DESC
"""
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run(
                    vector_query,
                    index_name=self.vector_index_name,
                    candidate_k=max(k * 20, 200),
                    embedding=embedding,
                    domain=domain,
                    query_type=query_type,
                    schema_version=schema_version,
                )
                candidates = [record.data() for record in result]
                return [
                    example
                    for example in candidates
                    if float(example.get("score", 0.0)) >= self.min_score
                    and self._schema_compatible(example, allowed_labels)
                ][:k]
        except Exception as e:
            raise CypherExampleRetrieverError(
                f"Error occurred while retrieving Cypher examples: {e}"
            )

    def _schema_compatible(
        self, example: Dict[str, Any], allowed_labels: Set[str]
    ) -> bool:
        """Reject examples that require capabilities absent from this graph."""

        required_labels = set(example.get("requiredLabels") or [])
        required_relationships = set(example.get("requiredRelationships") or [])
        required_properties = set(example.get("requiredProperties") or [])
        return (
            required_labels.issubset(self.schema_labels)
            and required_labels.issubset(allowed_labels)
            and required_relationships.issubset(self.schema_relationships)
            and required_properties.issubset(self.schema_properties)
        )

    def _embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)

    def _format_examples_list(self, unformatted_examples: List[Dict[str, str]]) -> str:
        if len(unformatted_examples) > 0:
            return ("\n" * 2).join(
                [
                    f"Question: {el['question']}\nCypher:\n{el['cypherStatement']}"
                    for el in unformatted_examples
                ]
            )
        else:
            return ""

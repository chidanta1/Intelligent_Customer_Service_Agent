"""In-memory fallback over the versioned curated Cypher examples."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseCypherExampleRetriever
from .curated_examples import get_curated_examples


class NorthwindCypherRetriever(BaseCypherExampleRetriever):
    """Keyword-ranked fallback retained for compatibility and local tests."""

    def get_examples(
        self, query: str, k: int = 5, *args: Any, **kwargs: Any
    ) -> str:
        domain = kwargs.get("domain")
        query_type = kwargs.get("query_type")
        schema_version = kwargs.get("schema_version")
        examples = get_curated_examples()
        if domain:
            examples = [item for item in examples if item["domain"] == domain]
        if schema_version:
            examples = [
                item for item in examples if item["schema_version"] == schema_version
            ]

        query_words = set(re.findall(r"[\w]+", query.lower()))

        def score(item: dict[str, Any]) -> tuple[int, str]:
            example_words = set(re.findall(r"[\w]+", item["question"].lower()))
            type_bonus = 10 if query_type and item["query_type"] == query_type else 0
            return type_bonus + len(query_words & example_words), item["id"]

        selected = sorted(examples, key=score, reverse=True)[: max(k, 0)]
        return "\n\n".join(
            f"Question: {item['question']}\nCypher:\n{item['cql']}"
            for item in selected
        )

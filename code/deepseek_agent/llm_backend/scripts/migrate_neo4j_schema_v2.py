"""Apply and verify the additive e-commerce Schema v2 migration."""

import json

from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from app.lg_agent.kg_sub_graph.schema_normalization import (
    ensure_canonical_ecommerce_schema,
)


def main() -> int:
    graph = get_neo4j_graph()
    try:
        audit = ensure_canonical_ecommerce_schema(graph, force=True)
        expected_equal = [
            ("containsRelationships", "typedUnitPrice"),
            ("containsRelationships", "typedQuantity"),
            ("containsRelationships", "typedDiscount"),
            ("canonicalBusinessNodes", "versionedBusinessNodes"),
            ("canonicalBusinessNodes", "correctlyDomainedBusinessNodes"),
            ("businessRelationships", "versionedBusinessRelationships"),
            ("businessRelationships", "correctlyDomainedBusinessRelationships"),
        ]
        errors = [
            f"{right} does not cover {left}"
            for left, right in expected_equal
            if audit.get(left) != audit.get(right)
        ]
        audit["status"] = "PASS" if not errors else "FAIL"
        audit["errors"] = errors
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if not errors else 1
    finally:
        graph.close()


if __name__ == "__main__":
    raise SystemExit(main())

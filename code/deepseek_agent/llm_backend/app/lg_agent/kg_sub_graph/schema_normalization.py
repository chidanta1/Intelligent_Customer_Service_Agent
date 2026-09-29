"""Idempotent, additive normalization for the e-commerce Neo4j schema.

The imported dataset contains two Product/Review property families.  The
normalization keeps every legacy label and property for compatibility, adds
domain-specific labels, and exposes one typed canonical property family for
Text2Cypher.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from langchain_neo4j import Neo4jGraph

from app.core.config import settings


CATALOG_DOMAIN = "app_catalog"
GRAPH_DOMAIN = "legacy_graph"

_NORMALIZED_DATABASES: set[str] = set()
_NORMALIZATION_LOCK = threading.Lock()


NORMALIZATION_QUERIES = [
    # Preserve :Product for backwards compatibility, while giving each family
    # an unambiguous label and the same canonical name/price/stock types.
    """
MATCH (p:Product)
WHERE p.name IS NOT NULL AND p.ProductName IS NULL
SET p:CatalogProduct,
    p.dataDomain = $catalog_domain,
    p.schemaVersion = $schema_version,
    p.price = toFloat(p.price),
    p.stock = toInteger(p.stock)
""",
    """
MATCH (p:Product)
WHERE p.ProductName IS NOT NULL
SET p:GraphProduct,
    p.dataDomain = $graph_domain,
    p.schemaVersion = $schema_version,
    p.productId = coalesce(p.productId, p.ProductID),
    p.name = p.ProductName,
    p.price = toFloat(p.UnitPrice),
    p.stock = toInteger(p.UnitsInStock),
    p.category = p.CategoryName,
    p.quantityPerUnit = p.QuantityPerUnit
""",
    """
MATCH (r:Review)
WHERE r.content IS NOT NULL AND r.ReviewText IS NULL
SET r:CatalogReview,
    r.dataDomain = $catalog_domain,
    r.schemaVersion = $schema_version,
    r.rating = toFloat(r.rating),
    r.helpfulCount = toInteger(r.helpful_count),
    r.createdAt = r.created_at
""",
    """
MATCH (r:Review)
WHERE r.ReviewText IS NOT NULL
SET r:GraphReview,
    r.dataDomain = $graph_domain,
    r.schemaVersion = $schema_version,
    r.reviewId = coalesce(r.reviewId, r.ReviewID),
    r.rating = toFloat(r.Rating),
    r.content = r.ReviewText,
    r.createdAt = datetime(r.ReviewDate + 'T00:00:00Z')
""",
    # Canonical aliases for entities used by the relationship graph.
    """
MATCH (c:Category)
SET c.dataDomain = $graph_domain,
    c.schemaVersion = $schema_version,
    c.categoryId = coalesce(c.categoryId, c.CategoryID),
    c.name = c.CategoryName,
    c.description = c.Description
""",
    """
MATCH (s:Supplier)
SET s.dataDomain = $graph_domain,
    s.schemaVersion = $schema_version,
    s.supplierId = coalesce(s.supplierId, s.SupplierID),
    s.name = s.CompanyName,
    s.contactName = s.ContactName,
    s.contactTitle = s.ContactTitle,
    s.phone = s.Phone,
    s.country = s.Country,
    s.city = s.City,
    s.region = s.Region
""",
    """
MATCH (c:Customer)
SET c.dataDomain = $graph_domain,
    c.schemaVersion = $schema_version,
    c.customerId = coalesce(c.customerId, c.CustomerID),
    c.name = c.CompanyName,
    c.contactName = c.ContactName,
    c.phone = c.Phone,
    c.country = c.Country,
    c.city = c.City,
    c.region = c.Region
""",
    """
MATCH (s:Shipper)
SET s.dataDomain = $graph_domain,
    s.schemaVersion = $schema_version,
    s.shipperId = coalesce(s.shipperId, s.ShipperID),
    s.name = s.CompanyName,
    s.phone = s.Phone
""",
    """
MATCH (e:Employee)
SET e.dataDomain = $graph_domain,
    e.schemaVersion = $schema_version,
    e.employeeId = coalesce(e.employeeId, e.EmployeeID),
    e.firstName = e.FirstName,
    e.lastName = e.LastName,
    e.title = e.Title,
    e.city = e.City,
    e.reportsTo = CASE
        WHEN e.ReportsTo IS NULL THEN NULL
        ELSE toInteger(toFloat(e.ReportsTo))
    END,
    e.hireDate = CASE
        WHEN e.HireDate IS NULL THEN NULL
        ELSE date(e.HireDate)
    END
""",
    """
MATCH (o:Order)
SET o.dataDomain = $graph_domain,
    o.schemaVersion = $schema_version,
    o.orderId = coalesce(o.orderId, o.OrderID),
    o.customerName = o.CustomerName,
    o.freight = toFloat(o.Freight),
    o.orderDate = localdatetime(replace(o.OrderDate, ' ', 'T')),
    o.requiredDate = localdatetime(replace(o.RequiredDate, ' ', 'T')),
    o.shippedDate = CASE
        WHEN o.ShippedDate IS NULL THEN NULL
        ELSE localdatetime(replace(o.ShippedDate, ' ', 'T'))
    END,
    o.shipCity = o.ShipCity,
    o.shipCountry = o.ShipCountry,
    o.shipName = o.ShipName,
    o.shipAddress = o.ShipAddress
""",
    # Imported order-line numbers are strings.  Keep them, but add typed
    # canonical properties so generated aggregation queries need no casts.
    """
MATCH ()-[line:CONTAINS]->()
SET line.dataDomain = $graph_domain,
    line.schemaVersion = $schema_version,
    line.unitPrice = toFloat(line.UnitPrice),
    line.quantity = toInteger(line.Quantity),
    line.discount = toFloat(line.Discount)
""",
    """
MATCH (start)-[rel]->(end)
WHERE type(rel) <> 'CONTAINS'
SET rel.dataDomain = CASE
        WHEN start:CatalogReview AND end:CatalogProduct THEN $catalog_domain
        ELSE $graph_domain
    END,
    rel.schemaVersion = $schema_version
""",
]


def ensure_canonical_ecommerce_schema(
    graph: Neo4jGraph, *, force: bool = False
) -> Dict[str, Any]:
    """Apply the additive migration once per process and return audit counts."""

    database = settings.NEO4J_DATABASE
    process_key = f"{database}:{settings.CYPHER_SCHEMA_VERSION}"
    with _NORMALIZATION_LOCK:
        if force or process_key not in _NORMALIZED_DATABASES:
            params = {
                "catalog_domain": CATALOG_DOMAIN,
                "graph_domain": GRAPH_DOMAIN,
                "schema_version": settings.CYPHER_SCHEMA_VERSION,
            }
            for query in NORMALIZATION_QUERIES:
                graph.query(query, params=params)
            graph.refresh_schema()
            _NORMALIZED_DATABASES.add(process_key)

    rows: List[Dict[str, Any]] = graph.query(
        """
MATCH (n)
WHERE n:CatalogProduct OR n:GraphProduct OR n:CatalogReview OR n:GraphReview
RETURN count(CASE WHEN n:CatalogProduct THEN 1 END) AS catalogProducts,
       count(CASE WHEN n:GraphProduct THEN 1 END) AS graphProducts,
       count(CASE WHEN n:CatalogReview THEN 1 END) AS catalogReviews,
       count(CASE WHEN n:GraphReview THEN 1 END) AS graphReviews,
       count(CASE WHEN n.schemaVersion = $schema_version THEN 1 END) AS versionedNodes
""",
        params={"schema_version": settings.CYPHER_SCHEMA_VERSION},
    )
    relationship_rows = graph.query(
        """
MATCH ()-[line:CONTAINS]->()
RETURN count(line) AS containsRelationships,
       count(line.unitPrice) AS typedUnitPrice,
       count(line.quantity) AS typedQuantity,
       count(line.discount) AS typedDiscount
"""
    )
    canonical_rows = graph.query(
        """
MATCH (n)
WHERE n:CatalogProduct OR n:GraphProduct OR n:CatalogReview OR n:GraphReview
   OR n:Category OR n:Supplier OR n:Customer OR n:Employee OR n:Order OR n:Shipper
WITH n,
     CASE WHEN n:CatalogProduct OR n:CatalogReview
          THEN $catalog_domain ELSE $graph_domain END AS expectedDomain
RETURN count(n) AS canonicalBusinessNodes,
       sum(CASE WHEN n.schemaVersion = $schema_version THEN 1 ELSE 0 END)
           AS versionedBusinessNodes,
       sum(CASE WHEN n.dataDomain = expectedDomain THEN 1 ELSE 0 END)
           AS correctlyDomainedBusinessNodes
""",
        params={
            "catalog_domain": CATALOG_DOMAIN,
            "graph_domain": GRAPH_DOMAIN,
            "schema_version": settings.CYPHER_SCHEMA_VERSION,
        },
    )
    canonical_relationship_rows = graph.query(
        """
MATCH (start)-[rel]->(end)
WITH rel,
     CASE WHEN start:CatalogReview AND end:CatalogProduct
          THEN $catalog_domain ELSE $graph_domain END AS expectedDomain
RETURN count(rel) AS businessRelationships,
       sum(CASE WHEN rel.schemaVersion = $schema_version THEN 1 ELSE 0 END)
           AS versionedBusinessRelationships,
       sum(CASE WHEN rel.dataDomain = expectedDomain THEN 1 ELSE 0 END)
           AS correctlyDomainedBusinessRelationships
""",
        params={
            "catalog_domain": CATALOG_DOMAIN,
            "graph_domain": GRAPH_DOMAIN,
            "schema_version": settings.CYPHER_SCHEMA_VERSION,
        },
    )
    return {
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        **(rows[0] if rows else {}),
        **(relationship_rows[0] if relationship_rows else {}),
        **(canonical_rows[0] if canonical_rows else {}),
        **(canonical_relationship_rows[0] if canonical_relationship_rows else {}),
    }

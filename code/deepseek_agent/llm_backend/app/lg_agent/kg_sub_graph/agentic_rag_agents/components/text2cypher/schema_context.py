"""Question-aware canonical Schema context for Text2Cypher generation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from langchain_neo4j import Neo4jGraph


NODE_PROPERTY_WHITELIST: Dict[str, List[str]] = {
    "CatalogProduct": ["name", "price", "stock", "brand", "category", "description"],
    "CatalogReview": ["rating", "content", "helpfulCount", "createdAt"],
    "GraphProduct": ["productId", "name", "price", "stock", "category", "quantityPerUnit"],
    "GraphReview": ["reviewId", "rating", "content", "createdAt"],
    "Category": ["categoryId", "name", "description"],
    "Supplier": ["supplierId", "name", "contactName", "contactTitle", "phone", "country", "city", "region"],
    "Customer": ["customerId", "name", "contactName", "phone", "country", "city", "region"],
    "Employee": ["employeeId", "firstName", "lastName", "title", "city", "reportsTo", "hireDate"],
    "Order": ["orderId", "customerName", "freight", "orderDate", "requiredDate", "shippedDate", "shipName", "shipAddress", "shipCity", "shipCountry"],
    "Shipper": ["shipperId", "name", "phone"],
}

RELATIONSHIPS: List[Tuple[str, str, str]] = [
    ("CatalogReview", "ABOUT", "CatalogProduct"),
    ("GraphReview", "ABOUT", "GraphProduct"),
    ("Customer", "WROTE", "GraphReview"),
    ("GraphProduct", "BELONGS_TO", "Category"),
    ("GraphProduct", "SUPPLIED_BY", "Supplier"),
    ("Customer", "PLACED", "Order"),
    ("Employee", "PROCESSED", "Order"),
    ("Order", "CONTAINS", "GraphProduct"),
    ("Order", "SHIPPED_VIA", "Shipper"),
]

RELATIONSHIP_PROPERTY_WHITELIST: Dict[str, List[str]] = {
    "CONTAINS": ["unitPrice", "quantity", "discount"],
}


def _property_types(graph: Neo4jGraph) -> Dict[str, str]:
    schema = graph.structured_schema or {}
    result: Dict[str, str] = {}
    for label, properties in schema.get("node_props", {}).items():
        for prop in properties:
            result[f"{label}.{prop.get('property')}"] = str(prop.get("type", "ANY"))
    for rel_type, properties in schema.get("rel_props", {}).items():
        for prop in properties:
            result[f"{rel_type}.{prop.get('property')}"] = str(prop.get("type", "ANY"))
    return result


def _available_labels(graph: Neo4jGraph) -> Set[str]:
    return set((graph.structured_schema or {}).get("node_props", {}))


def relevant_schema_capabilities(
    graph: Neo4jGraph, question: str, domain: str
) -> Dict[str, Set[str]]:
    """Return the qualified allow-list represented by a question's Schema slice."""

    labels = select_relevant_labels(question, domain) & _available_labels(graph)
    relationships = {
        rel_type
        for start, rel_type, end in RELATIONSHIPS
        if start in labels and end in labels
    }
    property_types = _property_types(graph)
    properties = {
        f"{label}.{name}"
        for label in labels
        for name in NODE_PROPERTY_WHITELIST.get(label, [])
        if f"{label}.{name}" in property_types
    }
    properties.update(
        f"{rel_type}.{name}"
        for rel_type in relationships
        for name in RELATIONSHIP_PROPERTY_WHITELIST.get(rel_type, [])
        if f"{rel_type}.{name}" in property_types
    )
    return {
        "labels": labels,
        "relationships": relationships,
        "properties": properties,
    }


def select_relevant_labels(question: str, domain: str) -> Set[str]:
    if domain == "app_catalog":
        labels = {"CatalogProduct"}
        if any(word in question for word in ("评价", "评论", "星级", "评分", "有帮助")):
            labels.add("CatalogReview")
        return labels

    labels: Set[str] = set()
    if any(word in question for word in ("商品", "产品", "购买", "消费", "销售", "收入", "评价", "评论")):
        labels.add("GraphProduct")
    if "类别" in question:
        labels.update(("Category", "GraphProduct"))
    if any(word in question for word in ("供应商", "供应", "供货", "所供")):
        labels.update(("Supplier", "GraphProduct"))
    if any(word in question for word in ("评价", "评论", "星级", "评分")):
        labels.update(("GraphReview", "GraphProduct"))
    if "客户" in question:
        labels.add("Customer")
    if any(word in question for word in ("订单", "下单", "购买", "消费", "销售", "收入", "运费", "日期", "发货")):
        labels.add("Order")
    if "员工" in question or "处理" in question or "汇报" in question:
        labels.update(("Employee", "Order"))
    if any(word in question for word in ("物流", "配送", "承运")):
        labels.update(("Shipper", "Order"))

    # Add the intermediate nodes required by common multi-hop questions.
    if (
        "Customer" in labels
        and "GraphProduct" in labels
        and "GraphReview" not in labels
    ):
        labels.add("Order")
    if "Order" in labels and any(word in question for word in ("包含", "商品", "产品", "购买", "消费", "销售", "收入")):
        labels.add("GraphProduct")
    if not labels:
        labels.add("GraphProduct")
    return labels


def _format_properties(
    owner: str,
    names: Iterable[str],
    property_types: Dict[str, str],
) -> str:
    present = [
        f"{name}: {property_types[f'{owner}.{name}']}"
        for name in names
        if f"{owner}.{name}" in property_types
    ]
    return ", ".join(present)


def retrieve_relevant_schema_for_prompt(
    graph: Neo4jGraph,
    question: str,
    domain: str,
    schema_version: str,
) -> str:
    """Return only the canonical schema slice relevant to the current question."""

    selected = relevant_schema_capabilities(graph, question, domain)["labels"]
    property_types = _property_types(graph)
    lines = [
        f"Schema version: {schema_version}",
        f"Data domain: {domain}",
        "Only use the labels, properties and relationship directions listed below.",
        "Node properties:",
    ]
    for label in NODE_PROPERTY_WHITELIST:
        if label not in selected:
            continue
        properties = _format_properties(
            label, NODE_PROPERTY_WHITELIST[label], property_types
        )
        lines.append(f"- {label} [{properties}]")

    relationship_lines = [
        f"- (:{start})-[:{rel_type}]->(:{end})"
        for start, rel_type, end in RELATIONSHIPS
        if start in selected and end in selected
    ]
    lines.append("Relationships:")
    lines.extend(relationship_lines or ["- none required for this question"])

    selected_relationships = {
        rel_type
        for start, rel_type, end in RELATIONSHIPS
        if start in selected and end in selected
    }
    rel_property_lines = []
    for rel_type, names in RELATIONSHIP_PROPERTY_WHITELIST.items():
        if rel_type not in selected_relationships:
            continue
        properties = _format_properties(rel_type, names, property_types)
        if properties:
            rel_property_lines.append(f"- {rel_type} [{properties}]")
    if rel_property_lines:
        lines.append("Relationship properties:")
        lines.extend(rel_property_lines)

    return "\n".join(lines).replace("{", "[").replace("}", "]")

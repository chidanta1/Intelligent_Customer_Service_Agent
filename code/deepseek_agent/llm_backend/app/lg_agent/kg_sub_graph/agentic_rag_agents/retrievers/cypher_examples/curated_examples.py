"""Versioned, executable Few-shot examples for the canonical e-commerce Schema."""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.config import settings


def _example(
    example_id: str,
    question: str,
    cql: str,
    *,
    domain: str,
    query_type: str,
    difficulty: str = "medium",
    labels: List[str],
    relationships: List[str] | None = None,
    properties: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "id": example_id,
        "question": question,
        "cql": cql.strip(),
        "domain": domain,
        "query_type": query_type,
        "difficulty": difficulty,
        "schema_version": settings.CYPHER_SCHEMA_VERSION,
        "required_labels": labels,
        "required_relationships": relationships or [],
        "required_properties": properties or [],
    }


# Repaired counterparts of the original 25 examples. Every statement uses
# Schema v2 labels/properties and is checked against the real Neo4j database.
BASE_EXAMPLES: List[Dict[str, Any]] = [
    _example("B001", "查询所有智能音箱类应用商品", """
MATCH (p:CatalogProduct)
WHERE p.category = '智能音箱'
RETURN p.name AS product, p.price AS price, p.stock AS stock
ORDER BY product
""", domain="app_catalog", query_type="filter", difficulty="easy",
        labels=["CatalogProduct"], properties=["CatalogProduct.category", "CatalogProduct.name", "CatalogProduct.price", "CatalogProduct.stock"]),
    _example("B002", "查找库存少于50件的应用商品", """
MATCH (p:CatalogProduct)
WHERE p.stock < 50
RETURN p.name AS product, p.stock AS stock
ORDER BY stock, product
""", domain="app_catalog", query_type="range_filter", difficulty="easy",
        labels=["CatalogProduct"], properties=["CatalogProduct.name", "CatalogProduct.stock"]),
    _example("B003", "哪些应用商品的价格高于1000元", """
MATCH (p:CatalogProduct)
WHERE p.price > 1000
RETURN p.name AS product, p.price AS price
ORDER BY price DESC
""", domain="app_catalog", query_type="range_filter", difficulty="easy",
        labels=["CatalogProduct"], properties=["CatalogProduct.name", "CatalogProduct.price"]),
    _example("B004", "图谱中有哪些商品类别",
        "MATCH (c:Category) RETURN c.name AS category, c.description AS description ORDER BY category",
        domain="legacy_graph", query_type="list", difficulty="easy",
        labels=["Category"], properties=["Category.name", "Category.description"]),
    _example("B005", "智能开关类别下有哪些图谱商品", """
MATCH (p:GraphProduct)-[:BELONGS_TO]->(c:Category)
WHERE c.name = '智能开关'
RETURN p.name AS product, p.price AS price
ORDER BY product
""", domain="legacy_graph", query_type="one_hop",
        labels=["GraphProduct", "Category"], relationships=["BELONGS_TO"],
        properties=["GraphProduct.name", "GraphProduct.price", "Category.name"]),
    _example("B006", "供应商苹果智能家庭提供了哪些图谱商品", """
MATCH (p:GraphProduct)-[:SUPPLIED_BY]->(s:Supplier)
WHERE s.name = '苹果智能家庭'
RETURN p.name AS product, p.quantityPerUnit AS quantity_per_unit, p.price AS price
ORDER BY product
""", domain="legacy_graph", query_type="one_hop",
        labels=["GraphProduct", "Supplier"], relationships=["SUPPLIED_BY"],
        properties=["GraphProduct.name", "GraphProduct.quantityPerUnit", "GraphProduct.price", "Supplier.name"]),
    _example("B007", "中国供应商提供了哪些图谱商品", """
MATCH (p:GraphProduct)-[:SUPPLIED_BY]->(s:Supplier)
WHERE s.country = '中国'
RETURN s.name AS supplier, p.name AS product, p.price AS price
ORDER BY supplier, product
""", domain="legacy_graph", query_type="one_hop_filter",
        labels=["GraphProduct", "Supplier"], relationships=["SUPPLIED_BY"],
        properties=["Supplier.country", "Supplier.name", "GraphProduct.name", "GraphProduct.price"]),
    _example("B008", "79号订单包含哪些图谱商品", """
MATCH (o:Order)-[:CONTAINS]->(p:GraphProduct)
WHERE o.orderId = '79'
RETURN p.name AS product, p.price AS price, o.orderDate AS order_date
ORDER BY product
""", domain="legacy_graph", query_type="one_hop_filter",
        labels=["Order", "GraphProduct"], relationships=["CONTAINS"],
        properties=["Order.orderId", "Order.orderDate", "GraphProduct.name", "GraphProduct.price"]),
    _example("B009", "谁处理了79号订单", """
MATCH (e:Employee)-[:PROCESSED]->(o:Order)
WHERE o.orderId = '79'
RETURN e.firstName AS first_name, e.lastName AS last_name, e.title AS title
""", domain="legacy_graph", query_type="one_hop_filter",
        labels=["Employee", "Order"], relationships=["PROCESSED"],
        properties=["Order.orderId", "Employee.firstName", "Employee.lastName", "Employee.title"]),
    _example("B010", "客户LS391下了哪些订单", """
MATCH (c:Customer)-[:PLACED]->(o:Order)
WHERE c.customerId = 'LS391'
RETURN o.orderId AS order_id, o.orderDate AS order_date, o.shippedDate AS shipped_date
ORDER BY order_date DESC
""", domain="legacy_graph", query_type="one_hop_filter",
        labels=["Customer", "Order"], relationships=["PLACED"],
        properties=["Customer.customerId", "Order.orderId", "Order.orderDate", "Order.shippedDate"]),
    _example("B011", "孙凤英处理了哪些订单", """
MATCH (e:Employee)-[:PROCESSED]->(o:Order)
WHERE e.firstName = '凤英' AND e.lastName = '孙'
RETURN o.orderId AS order_id, o.orderDate AS order_date, o.shippedDate AS shipped_date
ORDER BY order_date DESC
""", domain="legacy_graph", query_type="one_hop_filter",
        labels=["Employee", "Order"], relationships=["PROCESSED"],
        properties=["Employee.firstName", "Employee.lastName", "Order.orderId", "Order.orderDate", "Order.shippedDate"]),
    _example("B012", "谁向孙凤英汇报", """
MATCH (manager:Employee), (employee:Employee)
WHERE manager.firstName = '凤英' AND manager.lastName = '孙'
  AND employee.reportsTo = toInteger(manager.employeeId)
RETURN employee.firstName AS first_name, employee.lastName AS last_name, employee.title AS title
""", domain="legacy_graph", query_type="self_join_filter",
        labels=["Employee"],
        properties=["Employee.firstName", "Employee.lastName", "Employee.reportsTo", "Employee.employeeId", "Employee.title"]),
    _example("B013", "79号订单通过哪个物流公司配送", """
MATCH (o:Order)-[:SHIPPED_VIA]->(s:Shipper)
WHERE o.orderId = '79'
RETURN s.name AS shipper, s.phone AS phone, o.shippedDate AS shipped_date
""", domain="legacy_graph", query_type="one_hop_filter",
        labels=["Order", "Shipper"], relationships=["SHIPPED_VIA"],
        properties=["Order.orderId", "Order.shippedDate", "Shipper.name", "Shipper.phone"]),
    _example("B014", "顺丰速运配送了哪些订单", """
MATCH (o:Order)-[:SHIPPED_VIA]->(s:Shipper)
WHERE s.name = '顺丰速运'
RETURN o.orderId AS order_id, o.shipName AS recipient, o.shipCity AS city, o.shippedDate AS shipped_date
ORDER BY order_id
LIMIT 10
""", domain="legacy_graph", query_type="one_hop_filter",
        labels=["Order", "Shipper"], relationships=["SHIPPED_VIA"],
        properties=["Shipper.name", "Order.orderId", "Order.shipName", "Order.shipCity", "Order.shippedDate"]),
    _example("B015", "哪些客户来自新疆维吾尔自治区", """
MATCH (c:Customer)
WHERE c.region = '新疆维吾尔自治区'
RETURN c.name AS customer, c.contactName AS contact, c.phone AS phone
ORDER BY customer
""", domain="legacy_graph", query_type="filter", difficulty="easy",
        labels=["Customer"], properties=["Customer.region", "Customer.name", "Customer.contactName", "Customer.phone"]),
    _example("B016", "客户LS391的订单配送到哪些城市", """
MATCH (c:Customer)-[:PLACED]->(o:Order)
WHERE c.customerId = 'LS391'
RETURN o.orderId AS order_id, o.shipAddress AS address, o.shipCity AS city, o.shipCountry AS country
ORDER BY order_id
""", domain="legacy_graph", query_type="one_hop_filter",
        labels=["Customer", "Order"], relationships=["PLACED"],
        properties=["Customer.customerId", "Order.orderId", "Order.shipAddress", "Order.shipCity", "Order.shipCountry"]),
    _example("B017", "销售数量最多的图谱商品是什么", """
MATCH (:Order)-[line:CONTAINS]->(p:GraphProduct)
RETURN p.name AS product, sum(line.quantity) AS total_quantity
ORDER BY total_quantity DESC
LIMIT 5
""", domain="legacy_graph", query_type="aggregation_ranking", difficulty="hard",
        labels=["Order", "GraphProduct"], relationships=["CONTAINS"],
        properties=["GraphProduct.name", "CONTAINS.quantity"]),
    _example("B018", "79号订单中的图谱商品分别由哪些供应商提供", """
MATCH (o:Order)-[:CONTAINS]->(p:GraphProduct)-[:SUPPLIED_BY]->(s:Supplier)
WHERE o.orderId = '79'
RETURN p.name AS product, s.name AS supplier, s.contactName AS contact, s.phone AS phone
ORDER BY product
""", domain="legacy_graph", query_type="multi_hop_filter",
        labels=["Order", "GraphProduct", "Supplier"], relationships=["CONTAINS", "SUPPLIED_BY"],
        properties=["Order.orderId", "GraphProduct.name", "Supplier.name", "Supplier.contactName", "Supplier.phone"]),
    _example("B019", "孙凤英处理的订单包含哪些智能门铃图谱商品", """
MATCH (e:Employee)-[:PROCESSED]->(o:Order)-[:CONTAINS]->(p:GraphProduct)-[:BELONGS_TO]->(c:Category)
WHERE e.firstName = '凤英' AND e.lastName = '孙' AND c.name = '智能门铃'
RETURN DISTINCT p.name AS product, p.price AS price, o.orderId AS order_id
ORDER BY product
""", domain="legacy_graph", query_type="four_hop_filter", difficulty="hard",
        labels=["Employee", "Order", "GraphProduct", "Category"],
        relationships=["PROCESSED", "CONTAINS", "BELONGS_TO"],
        properties=["Employee.firstName", "Employee.lastName", "Category.name", "GraphProduct.name", "GraphProduct.price", "Order.orderId"]),
    _example("B020", "查询华为Sound X的应用评价", """
MATCH (r:CatalogReview)-[:ABOUT]->(p:CatalogProduct)
WHERE p.name = '华为Sound X'
RETURN r.content AS review, r.rating AS rating, r.createdAt AS created_at
ORDER BY created_at DESC
""", domain="app_catalog", query_type="one_hop_filter",
        labels=["CatalogReview", "CatalogProduct"], relationships=["ABOUT"],
        properties=["CatalogProduct.name", "CatalogReview.content", "CatalogReview.rating", "CatalogReview.createdAt"]),
    _example("B021", "智能开关图谱商品中平均评价超过4分的有哪些", """
MATCH (r:GraphReview)-[:ABOUT]->(p:GraphProduct)-[:BELONGS_TO]->(c:Category)
WHERE c.name = '智能开关'
WITH p, avg(r.rating) AS avg_rating, count(r) AS review_count
WHERE avg_rating > 4
RETURN p.name AS product, round(avg_rating, 2) AS avg_rating, review_count
ORDER BY avg_rating DESC
""", domain="legacy_graph", query_type="multi_hop_aggregation_filter", difficulty="hard",
        labels=["GraphReview", "GraphProduct", "Category"], relationships=["ABOUT", "BELONGS_TO"],
        properties=["Category.name", "GraphProduct.name", "GraphReview.rating"]),
    _example("B022", "每个月的订单数量是多少", """
MATCH (o:Order)
RETURN date.truncate('month', o.orderDate) AS month, count(o) AS order_count
ORDER BY month
""", domain="legacy_graph", query_type="temporal_aggregation", difficulty="hard",
        labels=["Order"], properties=["Order.orderDate"]),
    _example("B023", "每个图谱商品类别的销售金额是多少", """
MATCH (:Order)-[line:CONTAINS]->(p:GraphProduct)-[:BELONGS_TO]->(c:Category)
RETURN c.name AS category,
       round(sum(line.unitPrice * line.quantity * (1 - line.discount)), 2) AS revenue
ORDER BY revenue DESC
""", domain="legacy_graph", query_type="multi_hop_aggregation", difficulty="hard",
        labels=["Order", "GraphProduct", "Category"], relationships=["CONTAINS", "BELONGS_TO"],
        properties=["Category.name", "CONTAINS.unitPrice", "CONTAINS.quantity", "CONTAINS.discount"]),
    _example("B024", "各城市的客户数量是多少", """
MATCH (c:Customer)
RETURN c.city AS city, count(c) AS customer_count
ORDER BY customer_count DESC
LIMIT 10
""", domain="legacy_graph", query_type="aggregation_ranking",
        labels=["Customer"], properties=["Customer.city"]),
    _example("B025", "统计各省份的订单数和销售额", """
MATCH (c:Customer)-[:PLACED]->(o:Order)-[line:CONTAINS]->(:GraphProduct)
RETURN c.region AS province,
       count(DISTINCT o) AS order_count,
       round(sum(line.unitPrice * line.quantity * (1 - line.discount)), 2) AS revenue
ORDER BY revenue DESC
""", domain="legacy_graph", query_type="multi_hop_aggregation", difficulty="hard",
        labels=["Customer", "Order", "GraphProduct"], relationships=["PLACED", "CONTAINS"],
        properties=["Customer.region", "CONTAINS.unitPrice", "CONTAINS.quantity", "CONTAINS.discount"]),
]


# Supplemental templates cover the failure-heavy query shapes in the benchmark.
# Their wording and statements are deliberately distinct from every held-out case.
HARD_EXAMPLES: List[Dict[str, Any]] = [
    _example("H001", "挑出售价最低的三款商城商品", """
MATCH (p:CatalogProduct)
RETURN p.name AS product, p.price AS price
ORDER BY price ASC, product
LIMIT 3
""", domain="app_catalog", query_type="ranking", difficulty="hard",
        labels=["CatalogProduct"], properties=["CatalogProduct.name", "CatalogProduct.price"]),
    _example("H002", "比较商城各品牌的最高售价与在售款数", """
MATCH (p:CatalogProduct)
WHERE p.brand IS NOT NULL
RETURN p.brand AS brand, max(p.price) AS max_price, count(p) AS product_count
ORDER BY max_price DESC, brand
""", domain="app_catalog", query_type="aggregation_ranking", difficulty="hard",
        labels=["CatalogProduct"], properties=["CatalogProduct.brand", "CatalogProduct.price"]),
    _example("H003", "汇总商城每件商品的评分均值及评论量", """
MATCH (r:CatalogReview)-[:ABOUT]->(p:CatalogProduct)
WITH p, avg(r.rating) AS avg_rating, count(r) AS review_count
RETURN p.name AS product, round(avg_rating, 2) AS avg_rating, review_count
ORDER BY avg_rating DESC, product
""", domain="app_catalog", query_type="aggregation", difficulty="hard",
        labels=["CatalogReview", "CatalogProduct"], relationships=["ABOUT"],
        properties=["CatalogReview.rating", "CatalogProduct.name"]),
    _example("H004", "选出获赞数最多的三条商城评论并显示商品", """
MATCH (r:CatalogReview)-[:ABOUT]->(p:CatalogProduct)
RETURN p.name AS product, r.content AS review, r.helpfulCount AS helpful_count
ORDER BY helpful_count DESC, product
LIMIT 3
""", domain="app_catalog", query_type="ranking", difficulty="hard",
        labels=["CatalogReview", "CatalogProduct"], relationships=["ABOUT"],
        properties=["CatalogReview.helpfulCount", "CatalogReview.content", "CatalogProduct.name"]),
    _example("H005", "找出从未提交过订单的客户公司", """
MATCH (c:Customer)
WHERE NOT (c)-[:PLACED]->(:Order)
RETURN c.name AS customer
ORDER BY customer
""", domain="legacy_graph", query_type="negative_pattern", difficulty="hard",
        labels=["Customer", "Order"], relationships=["PLACED"],
        properties=["Customer.name"]),
    _example("H006", "列出仍缺少发货时间的订单及收件人", """
MATCH (o:Order)
WHERE o.shippedDate IS NULL
RETURN o.orderId AS order_id, o.shipName AS recipient
ORDER BY toInteger(order_id)
""", domain="legacy_graph", query_type="null_filter", difficulty="hard",
        labels=["Order"], properties=["Order.orderId", "Order.shippedDate", "Order.shipName"]),
    _example("H007", "筛选实际发货早于要求时间的订单并计算提前天数", """
MATCH (o:Order)
WHERE o.shippedDate IS NOT NULL AND o.shippedDate < o.requiredDate
RETURN o.orderId AS order_id,
       duration.between(o.shippedDate, o.requiredDate).days AS days_early
ORDER BY days_early DESC
LIMIT 20
""", domain="legacy_graph", query_type="temporal_filter", difficulty="hard",
        labels=["Order"], properties=["Order.orderId", "Order.shippedDate", "Order.requiredDate"]),
    _example("H008", "筛出订单量达到四单以上的客户", """
MATCH (c:Customer)-[:PLACED]->(o:Order)
WITH c, count(o) AS order_count
WHERE order_count >= 4
RETURN c.name AS customer, order_count
ORDER BY order_count DESC, customer
""", domain="legacy_graph", query_type="aggregation_filter", difficulty="hard",
        labels=["Customer", "Order"], relationships=["PLACED"],
        properties=["Customer.name"]),
    _example("H009", "依据成交明细给图谱产品计算净销售额并取前八名", """
MATCH (:Order)-[line:CONTAINS]->(p:GraphProduct)
WITH p, sum(line.unitPrice * line.quantity * (1 - line.discount)) AS revenue
RETURN p.name AS product, round(revenue, 2) AS revenue
ORDER BY revenue DESC, product
LIMIT 8
""", domain="legacy_graph", query_type="multi_hop_aggregation", difficulty="hard",
        labels=["Order", "GraphProduct"], relationships=["CONTAINS"],
        properties=["GraphProduct.name", "CONTAINS.unitPrice", "CONTAINS.quantity", "CONTAINS.discount"]),
    _example("H010", "汇总每家供货商对应产品的成交净额", """
MATCH (:Order)-[line:CONTAINS]->(p:GraphProduct)-[:SUPPLIED_BY]->(s:Supplier)
RETURN s.name AS supplier,
       round(sum(line.unitPrice * line.quantity * (1 - line.discount)), 2) AS revenue
ORDER BY revenue DESC, supplier
""", domain="legacy_graph", query_type="multi_hop_aggregation", difficulty="hard",
        labels=["Order", "GraphProduct", "Supplier"], relationships=["CONTAINS", "SUPPLIED_BY"],
        properties=["Supplier.name", "CONTAINS.unitPrice", "CONTAINS.quantity", "CONTAINS.discount"]),
    _example("H011", "统计各经办员工的成交净额和不同订单数", """
MATCH (e:Employee)-[:PROCESSED]->(o:Order)-[line:CONTAINS]->(:GraphProduct)
RETURN e.lastName + e.firstName AS employee,
       count(DISTINCT o) AS order_count,
       round(sum(line.unitPrice * line.quantity * (1 - line.discount)), 2) AS revenue
ORDER BY revenue DESC, employee
""", domain="legacy_graph", query_type="multi_hop_aggregation", difficulty="hard",
        labels=["Employee", "Order", "GraphProduct"], relationships=["PROCESSED", "CONTAINS"],
        properties=["Employee.lastName", "Employee.firstName", "CONTAINS.unitPrice", "CONTAINS.quantity", "CONTAINS.discount"]),
    _example("H012", "统计在相同订单明细里搭配次数最多的八组产品", """
MATCH (o:Order)-[:CONTAINS]->(left:GraphProduct),
      (o)-[:CONTAINS]->(right:GraphProduct)
WHERE toInteger(left.productId) < toInteger(right.productId)
RETURN left.name AS product_1, right.name AS product_2,
       count(DISTINCT o) AS together_count
ORDER BY together_count DESC, product_1, product_2
LIMIT 8
""", domain="legacy_graph", query_type="self_join_aggregation", difficulty="hard",
        labels=["Order", "GraphProduct"], relationships=["CONTAINS"],
        properties=["GraphProduct.productId", "GraphProduct.name"]),
    _example("H013", "列出每个客户实际买到过的供货商组合", """
MATCH (c:Customer)-[:PLACED]->(:Order)-[:CONTAINS]->(p:GraphProduct)-[:SUPPLIED_BY]->(s:Supplier)
RETURN DISTINCT c.name AS customer, s.name AS supplier
ORDER BY customer, supplier
""", domain="legacy_graph", query_type="four_hop_filter", difficulty="hard",
        labels=["Customer", "Order", "GraphProduct", "Supplier"],
        relationships=["PLACED", "CONTAINS", "SUPPLIED_BY"],
        properties=["Customer.name", "Supplier.name"]),
    _example("H014", "找出客户与品类组合中的最高消费项", """
MATCH (c:Customer)-[:PLACED]->(:Order)-[line:CONTAINS]->(p:GraphProduct)-[:BELONGS_TO]->(category:Category)
RETURN c.name AS customer, category.name AS category,
       round(sum(line.unitPrice * line.quantity * (1 - line.discount)), 2) AS spend
ORDER BY spend DESC, customer, category
LIMIT 8
""", domain="legacy_graph", query_type="four_hop_aggregation", difficulty="hard",
        labels=["Customer", "Order", "GraphProduct", "Category"],
        relationships=["PLACED", "CONTAINS", "BELONGS_TO"],
        properties=["Customer.name", "Category.name", "CONTAINS.unitPrice", "CONTAINS.quantity", "CONTAINS.discount"]),
    _example("H015", "以月份为单位计算订单明细净收入", """
MATCH (o:Order)-[line:CONTAINS]->(:GraphProduct)
WITH date.truncate('month', o.orderDate) AS month,
     sum(line.unitPrice * line.quantity * (1 - line.discount)) AS revenue
RETURN month, round(revenue, 2) AS revenue
ORDER BY month
""", domain="legacy_graph", query_type="temporal_multi_hop_aggregation", difficulty="hard",
        labels=["Order", "GraphProduct"], relationships=["CONTAINS"],
        properties=["Order.orderDate", "CONTAINS.unitPrice", "CONTAINS.quantity", "CONTAINS.discount"]),
    _example("H016", "在客户撰写的图谱评论中汇总各产品平均分，保留三条以上", """
MATCH (:Customer)-[:WROTE]->(r:GraphReview)-[:ABOUT]->(p:GraphProduct)
WITH p, avg(r.rating) AS avg_rating, count(r) AS review_count
WHERE review_count >= 3
RETURN p.name AS product, round(avg_rating, 2) AS avg_rating, review_count
ORDER BY avg_rating DESC, product
""", domain="legacy_graph", query_type="multi_hop_aggregation_filter", difficulty="hard",
        labels=["Customer", "GraphReview", "GraphProduct"],
        relationships=["WROTE", "ABOUT"],
        properties=["GraphReview.rating", "GraphProduct.name"]),
    _example("H017", "查看图谱产品中库存超过五百件的条目", """
MATCH (p:GraphProduct)
WHERE p.stock > 500
RETURN p.name AS product, p.stock AS stock
ORDER BY stock DESC, product
""", domain="legacy_graph", query_type="range_filter", difficulty="medium",
        labels=["GraphProduct"],
        properties=["GraphProduct.name", "GraphProduct.stock"]),
    _example("H018", "按售价从低到高展示四款图谱产品", """
MATCH (p:GraphProduct)
RETURN p.name AS product, p.price AS price
ORDER BY price ASC, product
LIMIT 4
""", domain="legacy_graph", query_type="ranking", difficulty="medium",
        labels=["GraphProduct"],
        properties=["GraphProduct.name", "GraphProduct.price"]),
    _example("H019", "计算所有图谱产品的平均单价", """
MATCH (p:GraphProduct)
RETURN round(avg(p.price), 2) AS average_price
""", domain="legacy_graph", query_type="aggregation", difficulty="medium",
        labels=["GraphProduct"], properties=["GraphProduct.price"]),
    _example("H020", "按国家汇总客户公司数量", """
MATCH (c:Customer)
RETURN c.country AS country, count(c) AS customer_count
ORDER BY customer_count DESC, country
""", domain="legacy_graph", query_type="aggregation", difficulty="medium",
        labels=["Customer"], properties=["Customer.country"]),
]


def get_curated_examples() -> List[Dict[str, Any]]:
    """Return copies so callers cannot mutate the process-wide catalogue."""

    return [dict(example) for example in BASE_EXAMPLES + HARD_EXAMPLES]

"""Deterministic domain and query-shape classification for Text2Cypher."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Text2CypherQueryContext:
    domain: str
    query_type: str


LEGACY_MARKERS = re.compile(
    r"图谱|图关系|商品类别|订单|客户|供应商|员工|物流|配送|运费|下单|发货|收货|销售额|销售收入"
)
APP_MARKERS = re.compile(r"应用商品|应用评价|商城商品|目录|品牌")


def classify_text2cypher_question(question: str) -> Text2CypherQueryContext:
    """Classify without another LLM call, keeping the generation A/B boundary clean."""

    normalized = question.strip()
    if APP_MARKERS.search(normalized):
        domain = "app_catalog"
    elif LEGACY_MARKERS.search(normalized):
        domain = "legacy_graph"
    elif re.search(r"评价|评论|星级|有帮助", normalized):
        domain = "app_catalog"
    elif re.search(r"商品|产品|价格|售价|多少钱|库存|还剩|几款", normalized):
        domain = "app_catalog"
    else:
        domain = "legacy_graph"

    if re.search(r"\d+号订单", normalized) or (
        "类别" not in normalized and re.search(
        r"的.*(?:单价|售价|库存|联系电话|卖多少钱).*(?:多少|是什么)", normalized
        )
    ) or re.search(r"卖多少钱.*库存", normalized):
        query_type = "entity_lookup"
    elif "全部商品类别" in normalized:
        query_type = "list"
    elif "客户" in normalized and "类别" in normalized and re.search(
        r"最丰富|种类最多", normalized
    ):
        query_type = "four_hop_distinct_aggregation"
    elif "客户" in normalized and "类别" in normalized and re.search(
        r"消费|金额|销售", normalized
    ):
        query_type = "four_hop_aggregation"
    elif "客户" in normalized and (
        "供应商" in normalized or re.search(r"所供商品", normalized)
    ) and re.search(
        r"购买|买到", normalized
    ):
        query_type = "four_hop_filter"
    elif re.search(r"图谱.*(?:评价|评论)|(?:评价|评论).*图谱", normalized) and re.search(
        r"平均", normalized
    ) and re.search(r"至少|达到|超过", normalized):
        query_type = "multi_hop_aggregation_filter"
    elif re.search(r"尚未发货|未发货|缺少发货", normalized):
        query_type = "null_filter"
    elif re.search(r"没有|一笔.*都没有|从未", normalized):
        query_type = "negative_pattern"
    elif re.search(r"共同出现|同时出现|商品对", normalized):
        query_type = "self_join_aggregation"
    elif re.search(r"月份|按月", normalized) and re.search(
        r"销售|收入|金额|净额", normalized
    ):
        query_type = "temporal_multi_hop_aggregation"
    elif re.search(r"销售额|销售收入|净收入|成交净额|消费金额|累计消费", normalized):
        query_type = "multi_hop_aggregation"
    elif re.search(r"(?:属于.*类别|类别.*商品|供应了哪些|提供了哪些)", normalized) and not re.search(
        r"统计|汇总|平均|销售|收入", normalized
    ):
        query_type = "one_hop"
    elif re.search(r"月份|按月|日期|年以后|发货.*日期", normalized) and re.search(
        r"统计|汇总|数量", normalized
    ):
        query_type = "temporal_aggregation"
    elif re.search(r"月份|日期|年以后|之前|之后|发货", normalized):
        query_type = "temporal_filter"
    elif re.search(r"最高|最低|前\s*\d+|最多|最少|排名|排行", normalized) and re.search(
        r"平均|统计|数量|销售|金额|类别", normalized
    ):
        query_type = "aggregation_ranking"
    elif re.search(r"至少|超过.*分", normalized):
        query_type = "aggregation_filter"
    elif re.search(r"平均|统计|汇总|多少|累计|总额|订单数|评价数", normalized):
        query_type = "aggregation"
    elif re.search(r"最高|最低|前\s*\d+|最多|最少|排名|排行|最有帮助", normalized):
        query_type = "ranking"
    elif re.search(r"到|之间|少于|低于|高于|不超过|不少于", normalized) and re.search(
        r"价格|售价|库存|评分", normalized
    ):
        query_type = "range_filter"
    elif re.search(r"包含|属于|提供|处理|配送|下了|购买过|写下", normalized):
        query_type = "relation_filter"
    elif re.search(r"哪些|列出|查询|查找|来自", normalized):
        query_type = "filter"
    else:
        query_type = "entity_lookup"
    return Text2CypherQueryContext(domain=domain, query_type=query_type)

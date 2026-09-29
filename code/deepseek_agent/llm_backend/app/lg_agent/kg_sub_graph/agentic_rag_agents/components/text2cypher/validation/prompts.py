"""
This code is based on content found in the LangGraph documentation: https://python.langchain.com/docs/tutorials/graph/#advanced-implementation-with-langgraph
"""

from langchain_core.prompts import ChatPromptTemplate


def create_text2cypher_validation_prompt_template() -> ChatPromptTemplate:
    """
    Create a Text2Cypher validation prompt template.

    Returns
    -------
    ChatPromptTemplate
        The prompt template.
    """

    validate_cypher_system = """
    You are the semantic-review layer in a five-layer Cypher validation pipeline.
    Judge only whether the Cypher answers the user's question and whether variables
    needed for that answer are defined. Syntax, write safety, relationship direction,
    Neo4j function compatibility, and Schema existence are checked by other layers.
    Do not report errors owned by those layers and do not rewrite the query.
    """

    validate_cypher_user = """You must check the following:
    * Are variables required to answer the question defined?
    * Does the Cypher statement contain the filters, grouping, sorting, limits,
      aggregations and return fields required by the question?
    * Do not flag syntax, labels, relationship directions, properties, function
      signatures, read/write safety, or database value existence; dedicated
      deterministic layers own those checks.

    Examples of good errors:
    * The question asks for the top 5 products, but the query has no ordering or limit.
    * The question asks for revenue, but the query returns only order count.
    * The question asks for unshipped orders, but the query does not filter shippedDate.

    Output contract:
    * errors must be a JSON list of concise semantic discrepancies, or [].
    * filters must be a JSON list. Only include literal string equality predicates
      that occur in the Cypher, using exactly node_label, property_key and
      property_value. Do not include IS NULL, ranges, variables, relationship
      patterns or property-to-property comparisons. Use [] when there are none.

    Schema:
    {schema}

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    Make sure you don't make any mistakes!"""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                validate_cypher_system,
            ),
            (
                "human",
                (validate_cypher_user),
            ),
        ]
    )

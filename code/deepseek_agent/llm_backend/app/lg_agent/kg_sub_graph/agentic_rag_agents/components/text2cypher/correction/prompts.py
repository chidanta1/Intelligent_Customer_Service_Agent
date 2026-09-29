"""
This code is based on content found in the LangGraph documentation: https://python.langchain.com/docs/tutorials/graph/#advanced-implementation-with-langgraph
"""

from langchain.prompts import ChatPromptTemplate


def create_text2cypher_correction_prompt_template() -> ChatPromptTemplate:
    """
    Create a Text2Cypher query correction prompt template.

    Returns
    -------
    ChatPromptTemplate
        The prompt template.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Cypher expert reviewing a statement written by a junior developer. "
                    "Correct only the defects explicitly listed in the provided errors. "
                    "Preserve every clause, alias, filter and return field that is unrelated "
                    "to those errors. Make the smallest valid change. "
                    "Target Neo4j 5: use `property IS NULL` for null checks; "
                    "`round(value, precision)` is supported. No pre-amble. "
                    "No parameter map can be returned by this correction step, so do not "
                    "introduce `$parameter` placeholders; preserve existing supplied "
                    "placeholders or inline the literal required by the user question. "
                    "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
                ),
            ),
            (
                "human",
                (
                    """Check for invalid syntax or semantics and return a corrected Cypher statement.

    Schema:
    {schema}

    Note: Do not include any explanations or apologies in your responses.
    Do not wrap the response in any backticks or anything else.
    Respond with a Cypher statement only!

    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    The errors are:
    {errors}

    Corrected Cypher statement: """
                ),
            ),
        ]
    )

from typing import Any, Callable, Coroutine, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph

from ....components.text2cypher.generation.prompts import (
    create_text2cypher_generation_prompt_template,
)
from ....retrievers.cypher_examples.base import BaseCypherExampleRetriever
from app.core.config import settings
from ..query_context import classify_text2cypher_question
from ..schema_context import (
    retrieve_relevant_schema_for_prompt,
    select_relevant_labels,
)
from ..state import CypherInputState

# 定义text2cypher generation prompt
generation_prompt = create_text2cypher_generation_prompt_template()


def create_text2cypher_generation_node(
    llm: BaseChatModel,
    graph: Neo4jGraph,
    cypher_example_retriever: BaseCypherExampleRetriever,
    fewshot_k: int = 3,
) -> Callable[[CypherInputState], Coroutine[Any, Any, dict[str, Any]]]:
    text2cypher_chain = generation_prompt | llm | StrOutputParser()

    async def generate_cypher(state: CypherInputState) -> Dict[str, Any]:
        """
        Generates a cypher statement based on the provided schema and user input
        """
        task = state.get("task", "")
        question = task[0] if isinstance(task, list) else task
        query_context = classify_text2cypher_question(question)
        # k=0 is an explicit no-Few-shot control used by production-path A/B
        # evaluation. Avoid querying Neo4j's vector procedure because it only
        # accepts a positive neighbour count.
        examples: str = ""
        if fewshot_k > 0:
            examples = cypher_example_retriever.get_examples(
                query=question,
                k=fewshot_k,
                domain=query_context.domain,
                query_type=query_context.query_type,
                schema_version=settings.CYPHER_SCHEMA_VERSION,
                allowed_labels=select_relevant_labels(
                    question, query_context.domain
                ),
            )
        generated_cypher = await text2cypher_chain.ainvoke(
            {
                "question": question,
                "fewshot_examples": examples,
                "schema": retrieve_relevant_schema_for_prompt(
                    graph=graph,
                    question=question,
                    domain=query_context.domain,
                    schema_version=settings.CYPHER_SCHEMA_VERSION,
                ),
            }
        )
        generated_cypher = generated_cypher.strip().removeprefix("```cypher")
        generated_cypher = generated_cypher.removeprefix("```").removesuffix("```").strip()

        return {
            "statement": generated_cypher,
            "attempts": 0,
            "retries": 0,
            "errors": [],
            "mapping_errors": [],
            "validation_layers": [],
            "validation_trace": [],
            "correction_history": [],
            "fewshot_examples": examples,
            "fewshot_k": fewshot_k,
            "retrieval_domain": query_context.domain,
            "retrieval_query_type": query_context.query_type,
            "schema_version": settings.CYPHER_SCHEMA_VERSION,
            "steps": ["generate_cypher"],
        }

    return generate_cypher

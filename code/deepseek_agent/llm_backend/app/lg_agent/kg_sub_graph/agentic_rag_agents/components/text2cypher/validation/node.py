"""
This code is based on content found in the LangGraph documentation: https://python.langchain.com/docs/tutorials/graph/#advanced-implementation-with-langgraph
"""

import re
from typing import Any, Callable, Coroutine, Dict, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langgraph.types import Command

from app.core.config import settings

from ....constants import MAX_TEXT2CYPHER_CORRECTION_RETRIES
from ....components.text2cypher.validation.models import ValidateCypherOutput
from ....components.text2cypher.validation.prompts import (
    create_text2cypher_validation_prompt_template,
)
from ..state import CypherState
from ..query_context import classify_text2cypher_question
from ..schema_context import (
    relevant_schema_capabilities,
    retrieve_relevant_schema_for_prompt,
)
from .validators import (
    correct_cypher_query_relationship_direction,
    validate_cypher_query_parameters,
    validate_cypher_query_syntax,
    validate_cypher_query_with_llm,
    validate_cypher_query_with_schema,
    validate_no_writes_in_cypher_query,
)

validation_prompt_template = create_text2cypher_validation_prompt_template()


def create_text2cypher_validation_node(
    graph: Neo4jGraph,
    llm: Optional[BaseChatModel] = None,
    llm_validation: bool = True,
    max_retries: int = 3,
    attempt_cypher_execution_on_final_attempt: bool = False,
) -> Callable[
    [CypherState],
    Coroutine[
        Any,
        Any,
        Command[Literal["correct_cypher", "execute_cypher", "finalize_failure"]],
    ],
]:
    """
    Create a Text2Cypher query validation node for a LangGraph workflow.
    This is the last node in the workflow before Cypher execution may be attempted.
    If errors are detected and the retry limit has not been reached, then the Cypher Statement must be corrected by the Correction node.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    llm : Optional[BaseChatModel], optional
        The LLM to use for processing if LLM validation is desired. By default None
    llm_validation : bool, optional
        Whether to perform LLM validation with the provided LLM, by default True
    max_retries: int, optional
        The maximum number of correction retries after initial generation, by default 3
    attempt_cypher_execution_on_final_attempt, bool, optional
        THIS MAY BE DANGEROUS.
        Whether to attempt Cypher execution on the last attempt, regardless of if the Cypher contains errors, by default False

    Returns
    -------
    Callable[[CypherState], CypherState]
        The LangGraph node.
    """

    if not isinstance(max_retries, int) or isinstance(max_retries, bool):
        raise TypeError("max_retries must be an integer")
    if not 0 <= max_retries <= MAX_TEXT2CYPHER_CORRECTION_RETRIES:
        raise ValueError(
            "max_retries must be between 0 and "
            f"{MAX_TEXT2CYPHER_CORRECTION_RETRIES}"
        )

    if llm is not None and llm_validation:
        model_name = str(
            getattr(llm, "model_name", getattr(llm, "model", ""))
        ).lower()
        if "deepseek" in model_name:
            json_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        validation_prompt_template.messages[0].prompt.template
                        + "\nReturn only valid JSON in exactly this shape: "
                        + '{{"errors": [], "filters": []}}. '
                        + "Each filters item, if any, must contain exactly "
                        + "node_label, property_key and property_value.",
                    ),
                    (
                        "human",
                        validation_prompt_template.messages[1].prompt.template,
                    ),
                ]
            )
            validate_cypher_chain = json_prompt | llm | JsonOutputParser()
        else:
            validate_cypher_chain = (
                validation_prompt_template
                | llm.with_structured_output(ValidateCypherOutput)
            )

    async def validate_cypher(
        state: CypherState,
    ) -> Command[Literal["correct_cypher", "execute_cypher", "finalize_failure"]]:
        """
        Validates the Cypher statements and maps any property values to the database.
        """

        generation_attempt = state.get("attempts", 0) + 1
        retries = max(0, generation_attempt - 1)
        errors: list[str] = []
        mapping_errors: list[str] = []
        layer_results: list[dict[str, Any]] = []
        statement = state.get("statement", "").strip()
        task = state.get("task", "")
        question = task[0] if isinstance(task, list) else task
        domain = state.get("retrieval_domain") or classify_text2cypher_question(
            question
        ).domain
        schema_version = state.get(
            "schema_version", settings.CYPHER_SCHEMA_VERSION
        )
        prompt_schema = retrieve_relevant_schema_for_prompt(
            graph=graph,
            question=question,
            domain=domain,
            schema_version=schema_version,
        )
        capabilities = relevant_schema_capabilities(graph, question, domain)
        # Test doubles and pre-migration databases have no canonical labels;
        # retain the legacy full-Schema behavior in that case.
        enforce_slice = bool(capabilities["labels"])
        validation_layers = [
            "syntax",
            "write_operation",
            "relationship_direction",
            "llm",
            "schema",
        ]

        # Layer 1: ask Neo4j to parse and plan the statement without executing it.
        syntax_errors: list[str] = []
        try:
            syntax_errors = validate_cypher_query_syntax(
                graph=graph,
                cypher_statement=statement,
            )
        except Exception as exc:
            syntax_errors = [f"Syntax validation failed: {exc}"]
        syntax_errors.extend(
            validate_cypher_query_parameters(
                cypher_statement=statement,
                parameters=state.get("parameters"),
            )
        )
        errors.extend(syntax_errors)
        layer_results.append(
            {
                "layer": "syntax",
                "status": "failed" if syntax_errors else "passed",
                "errors": syntax_errors,
            }
        )

        # Layer 2: reject every write clause before execution.
        write_errors = validate_no_writes_in_cypher_query(statement)
        errors.extend(write_errors)
        layer_results.append(
            {
                "layer": "write_operation",
                "status": "failed" if write_errors else "passed",
                "errors": write_errors,
            }
        )

        # Layer 3: validate and automatically normalize relationship directions.
        corrected_cypher = statement
        direction_errors: list[str] = []
        direction_status = "passed"
        try:
            direction_corrected = correct_cypher_query_relationship_direction(
                graph=graph,
                cypher_statement=statement,
            )
            if direction_corrected:
                corrected_cypher = direction_corrected
                if direction_corrected.strip() != statement:
                    direction_status = "corrected"
            elif re.search(r"\[[^\]]*:[^\]]+\]", statement):
                # An empty result means either a known relationship has an
                # impossible direction or the relationship type is unknown.
                # Keep those concerns separate: unknown types belong solely to
                # the later Schema layer.
                declared_relationships = set(
                    re.findall(
                        r"\[[^\]]*:\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
                        statement,
                    )
                )
                known_relationships = {
                    str(item.get("type"))
                    for item in graph.structured_schema.get("relationships", [])
                    if item.get("type")
                }
                known_relationships.update(
                    graph.structured_schema.get("rel_props", {}).keys()
                )
                if declared_relationships and declared_relationships.issubset(
                    known_relationships
                ):
                    direction_errors = [
                        "Relationship direction does not match the provided Schema"
                    ]
                    direction_status = "failed"
        except Exception as exc:
            direction_errors = [
                f"Relationship direction validation failed: {exc}"
            ]
            direction_status = "failed"
        errors.extend(direction_errors)
        layer_results.append(
            {
                "layer": "relationship_direction",
                "status": direction_status,
                "errors": direction_errors,
            }
        )

        # Layer 4: independently validate whether the statement answers the question.
        llm_operational_failure = False
        if llm is not None and llm_validation:
            llm_errors: list[str] = []
            llm_mapping_errors: list[str] = []
            llm_warnings: list[str] = []
            try:
                llm_result = await validate_cypher_query_with_llm(
                    validate_cypher_chain=validate_cypher_chain,
                    question=question,
                    graph=graph,
                    cypher_statement=corrected_cypher,
                    schema=prompt_schema,
                )
                llm_errors = list(llm_result.get("errors", []))
                llm_mapping_errors = list(llm_result.get("mapping_errors", []))
                llm_warnings = list(llm_result.get("warnings", []))
            except Exception as exc:
                llm_operational_failure = True
                llm_errors = [f"LLM validation failed: {exc}"]
            errors.extend(llm_errors)
            mapping_errors.extend(llm_mapping_errors)
            layer_results.append(
                {
                    "layer": "llm",
                    "status": (
                        "failed"
                        if llm_errors or llm_mapping_errors
                        else "passed"
                    ),
                    "errors": llm_errors + llm_mapping_errors,
                    "warnings": llm_warnings,
                    "error_kind": (
                        "operational" if llm_operational_failure else None
                    ),
                }
            )
        else:
            layer_results.append(
                {
                    "layer": "llm",
                    "status": "skipped",
                    "errors": [],
                    "warnings": [],
                }
            )

        # Layer 5: always validate labels, relationships and properties against Schema.
        schema_errors: list[str] = []
        try:
            schema_errors = validate_cypher_query_with_schema(
                graph=graph,
                cypher_statement=corrected_cypher,
                allowed_labels=(capabilities["labels"] if enforce_slice else None),
                allowed_relationships=(
                    capabilities["relationships"] if enforce_slice else None
                ),
                allowed_properties=(
                    capabilities["properties"] if enforce_slice else None
                ),
            )
        except Exception as exc:
            schema_errors = [f"Schema validation failed: {exc}"]
        errors.extend(schema_errors)
        layer_results.append(
            {
                "layer": "schema",
                "status": "failed" if schema_errors else "passed",
                "errors": schema_errors,
            }
        )

        errors = list(dict.fromkeys(str(error) for error in errors if error))
        mapping_errors = list(
            dict.fromkeys(str(error) for error in mapping_errors if error)
        )

        has_validation_failure = bool(errors or mapping_errors)
        if llm_operational_failure:
            # A provider/transport/parser failure says nothing about the Cypher.
            # Fail closed without spending up to three correction calls on an
            # infrastructure problem or mutating a potentially correct query.
            next_action = "finalize_failure"
        elif has_validation_failure and retries < max_retries:
            next_action = "correct_cypher"
        elif has_validation_failure:
            # Never execute a query that still fails validation after the retry limit.
            next_action = "finalize_failure"
        else:
            next_action = "execute_cypher"

        validation_attempt = {
            "attempt": generation_attempt,
            "retry": retries,
            "statement_before": statement,
            "statement_after": corrected_cypher,
            "layers": layer_results,
            "errors": errors,
            "mapping_errors": mapping_errors,
            "operational_failure": llm_operational_failure,
            "next_action": next_action,
        }

        return Command(
            goto=next_action,
            update={
                "next_action_cypher": next_action,
                "statement": corrected_cypher,
                "errors": errors,
                "mapping_errors": mapping_errors,
                "attempts": generation_attempt,
                "retries": retries,
                "validation_layers": validation_layers,
                "validation_trace": [validation_attempt],
                "steps": [
                    "validate_syntax",
                    "validate_write_operation",
                    "validate_relationship_direction",
                    "validate_llm",
                    "validate_schema",
                ],
            },
        )

    return validate_cypher

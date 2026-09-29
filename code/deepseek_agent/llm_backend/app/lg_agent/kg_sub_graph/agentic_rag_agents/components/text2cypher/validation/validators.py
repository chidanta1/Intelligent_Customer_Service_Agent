"""
This file contains Cypher validators that may be used in the Text2Cypher validation node.
"""

import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from langchain_core.runnables.base import Runnable
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from neo4j.exceptions import CypherSyntaxError

from ....components.text2cypher.validation.models import Property, ValidateCypherOutput
from ....constants import WRITE_CLAUSES
from ...utils.utils import retrieve_and_parse_schema_from_graph_for_prompts
from .models import (
    CypherValidationTask,
    Neo4jStructuredSchema,
    Neo4jStructuredSchemaPropertyNumber,
)
from .utils.cypher_extractors import (
    extract_entities_for_validation,
)

# parse_labels_or_types,
from .utils.utils import update_task_list_with_property_type


def validate_cypher_query_syntax(graph: Neo4jGraph, cypher_statement: str) -> List[str]:
    """
    Validate the Cypher statement syntax by running an EXPLAIN query.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    cypher_statement : str
        The Cypher statement to validate.

    Returns
    -------
    List[str]
        If the statement contains invalid syntax, return an error message in a list
    """
    errors = list()
    try:
        graph.query(f"EXPLAIN {cypher_statement}")
    except CypherSyntaxError as e:
        errors.append(str(e.message))
    return errors


def validate_cypher_query_parameters(
    cypher_statement: str,
    parameters: Optional[Dict[str, Any]],
) -> List[str]:
    """Reject unresolved ``$parameter`` placeholders before execution.

    Text2Cypher correction currently returns only a statement, so silently
    introducing a placeholder produces a syntactically valid query that fails
    at execution.  Parameter completeness is therefore part of the syntax
    layer rather than a sixth validation layer.
    """

    # Remove quoted strings and comments so literal "$name" text is ignored.
    searchable = re.sub(r"(['\"])(?:\\.|(?!\1).)*\1", "", cypher_statement)
    searchable = re.sub(r"//[^\n]*|/\*.*?\*/", "", searchable, flags=re.DOTALL)
    required = set(re.findall(r"(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)", searchable))
    supplied = set((parameters or {}).keys())
    missing = sorted(required - supplied)
    if not missing:
        return []
    return [
        "Missing Cypher parameter values: " + ", ".join(f"${name}" for name in missing)
    ]


def correct_cypher_query_relationship_direction(
    graph: Neo4jGraph, cypher_statement: str
) -> str:
    """
    Correct Relationship directions in the Cypher statement with LangChain's `CypherQueryCorrector`.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    cypher_statement : str
        The Cypher statement to validate.

    Returns
    -------
    str
        The Cypher statement with corrected Relationship directions.
    """
    # Cypher query corrector is experimental
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in graph.structured_schema.get("relationships", list())
    ]
    cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    corrected_cypher: str = cypher_query_corrector(cypher_statement)

    return corrected_cypher


async def validate_cypher_query_with_llm(
    validate_cypher_chain: Runnable[Dict[str, Any], Any],
    question: str,
    graph: Neo4jGraph,
    cypher_statement: str,
    schema: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Validate the Cypher statement with an LLM.
    Use declared LLM to find Node and Property pairs to validate.
    Validate Node and Property pairs against the Neo4j graph.

    Parameters
    ----------
    validate_cypher_chain : RunnableSerializable
        The LangChain LLM to perform processing.
    question : str
        The question associated with the Cypher statement.
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    cypher_statement : str
        The Cypher statement to validate.

    Returns
    -------
    Dict[str, List[str]]
        A Python dictionary with keys `errors`, `mapping_errors` and `warnings`.
        Mapping candidates are extracted deterministically from the Cypher statement;
        the LLM's optional ``filters`` field is retained only as format telemetry.
    """

    errors: List[str] = []
    mapping_errors: List[str] = []
    warnings: List[str] = []

    llm_output = await validate_cypher_chain.ainvoke(
        {
            "question": question,
            "schema": schema or retrieve_and_parse_schema_from_graph_for_prompts(graph),
            "cypher": cypher_statement,
        }
    )
    if isinstance(llm_output, dict):
        raw_errors = llm_output.get("errors")
        raw_filters = llm_output.get("filters")
    else:
        raw_errors = getattr(llm_output, "errors", None)
        raw_filters = getattr(llm_output, "filters", None)

    if isinstance(raw_errors, str):
        errors.append(raw_errors)
    elif isinstance(raw_errors, (list, tuple, set)):
        errors.extend(str(item) for item in raw_errors if item)
    elif raw_errors:
        errors.append(str(raw_errors))

    # DeepSeek occasionally returns filters as strings or with invented field
    # names. Those format variations must not turn an otherwise valid Cypher
    # statement into a blocking validation error. Mapping validation is based on
    # the statement itself, so the LLM payload is useful only for observability.
    if raw_filters and not isinstance(raw_filters, list):
        warnings.append(
            "LLM filters was not a list; deterministic Cypher filter extraction was used"
        )
    elif isinstance(raw_filters, list):
        malformed_count = 0
        for item in raw_filters:
            if isinstance(item, Property):
                continue
            if isinstance(item, dict):
                try:
                    Property.model_validate(item)
                    continue
                except Exception:
                    pass
            malformed_count += 1
        if malformed_count:
            warnings.append(
                f"Ignored {malformed_count} non-canonical LLM filter item(s); "
                "deterministic Cypher filter extraction was used"
            )

    for cypher_filter in extract_string_equality_filters(
        graph=graph,
        cypher_statement=cypher_statement,
    ):
        mapping = graph.query(
            f"MATCH (n:`{cypher_filter.node_label}`) "
            f"WHERE toLower(n.`{cypher_filter.property_key}`) = toLower($value) "
            "RETURN 'yes' LIMIT 1",
            {"value": cypher_filter.property_value},
        )
        if not mapping:
            mapping_errors.append(
                "Missing value mapping for "
                f"{cypher_filter.node_label} on property "
                f"{cypher_filter.property_key} with value "
                f"{cypher_filter.property_value}"
            )
    return {
        "errors": errors,
        "mapping_errors": mapping_errors,
        "warnings": warnings,
    }


def extract_string_equality_filters(
    graph: Neo4jGraph,
    cypher_statement: str,
) -> List[Property]:
    """Extract known STRING literal equality filters without using an LLM.

    Unknown labels/properties are intentionally left to the independent Schema
    validator. Non-equality predicates such as ``IS NULL`` and ranges are not
    database value mappings and are ignored.
    """

    node_props = graph.structured_schema.get("node_props", {})
    variable_labels: Dict[str, str] = {}
    filters: List[Property] = []

    def is_known_string_property(label: str, property_key: str) -> bool:
        return any(
            item.get("property") == property_key and item.get("type") == "STRING"
            for item in node_props.get(label, [])
        )

    def add_filter(label: str, property_key: str, property_value: str) -> None:
        if not is_known_string_property(label, property_key):
            return
        candidate = Property(
            node_label=label,
            property_key=property_key,
            property_value=property_value,
        )
        if candidate not in filters:
            filters.append(candidate)

    for node_match in re.finditer(r"\(([^()]*)\)", cypher_statement):
        content = node_match.group(1)
        declaration = content.split("{", 1)[0]
        variable_match = re.match(
            r"\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*:\s*"
            r"`?([A-Za-z_][A-Za-z0-9_]*)`?",
            declaration,
        )
        if not variable_match:
            continue
        variable, label = variable_match.groups()
        variable_labels[variable] = label

        # Also support literal maps inside a node pattern:
        # MATCH (p:Product {brand: '小米'}).
        map_match = re.search(r"\{([^{}]*)\}", content)
        if map_match:
            for property_match in re.finditer(
                r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*:\s*"
                r"(['\"])([^'\"]*)\2",
                map_match.group(1),
            ):
                property_key, _, property_value = property_match.groups()
                add_filter(label, property_key, property_value)

    equality_pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
        r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*=\s*(?!~)"
        r"(['\"])([^'\"]*)\3"
    )
    for match in equality_pattern.finditer(cypher_statement):
        variable, property_key, _, property_value = match.groups()
        label = variable_labels.get(variable)
        if label:
            add_filter(label, property_key, property_value)

    return filters


def validate_cypher_query_with_schema(
    graph: Neo4jGraph,
    cypher_statement: str,
    *,
    allowed_labels: Optional[Set[str]] = None,
    allowed_relationships: Optional[Set[str]] = None,
    allowed_properties: Optional[Set[str]] = None,
) -> List[str]:
    """
    Validate the provided Cypher statement using the schema retrieved from the graph.
    This will ensure the existance of names nodes, relationships and properties.
    This will validate property values with enums and number ranges, if available.
    This method does not use an LLM.

    Parameters
    ----------
    graph : Neo4jGraph
        The Neo4j graph wrapper.
    cypher_statement : str
        The Cypher to be validated.

    Returns
    -------
    List[str]
        A list of any found errors.
    """

    structured_schema = graph.structured_schema
    node_props = structured_schema.get("node_props", {})
    rel_props = structured_schema.get("rel_props", {})
    known_labels = set(node_props)
    known_relationships = {
        relationship.get("type")
        for relationship in structured_schema.get("relationships", [])
    }
    known_relationships.update(rel_props)
    if allowed_labels is not None:
        known_labels.intersection_update(allowed_labels)
    if allowed_relationships is not None:
        known_relationships.intersection_update(allowed_relationships)

    node_variables: Dict[str, List[str]] = {}
    relationship_variables: Dict[str, List[str]] = {}
    errors: List[str] = []

    def property_names(items: List[Dict[str, Any]]) -> Set[str]:
        return {str(item.get("property")) for item in items}

    # Validate labels and properties declared directly in node patterns.
    for node_match in re.finditer(r"\(([^()]*)\)", cypher_statement):
        content = node_match.group(1)
        declaration = content.split("{", 1)[0]
        labels = re.findall(r":\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", declaration)
        variable_match = re.match(r"\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", declaration)
        if variable_match and labels:
            node_variables[variable_match.group(1)] = labels
        for label in labels:
            if label not in known_labels:
                errors.append(f"Unknown node label in Schema: {label}")
        map_match = re.search(r"\{([^{}]*)\}", content)
        if map_match:
            map_properties = re.findall(
                r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*:", map_match.group(1)
            )
            for prop in map_properties:
                if labels and not any(
                    (
                        f"{label}.{prop}" in allowed_properties
                        if allowed_properties is not None
                        else prop in property_names(node_props.get(label, []))
                    )
                    for label in labels
                ):
                    errors.append(
                        f"Unknown property in Schema for {labels}: {prop}"
                    )

    # Validate relationship types and properties declared in relationship patterns.
    for rel_match in re.finditer(r"\[([^\[\]]*)\]", cypher_statement):
        content = rel_match.group(1)
        declaration = content.split("{", 1)[0]
        rel_types = re.findall(r":\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", declaration)
        variable_match = re.match(r"\s*`?([A-Za-z_][A-Za-z0-9_]*)`?", declaration)
        if variable_match and rel_types:
            relationship_variables[variable_match.group(1)] = rel_types
        for rel_type in rel_types:
            if rel_type not in known_relationships:
                errors.append(f"Unknown relationship type in Schema: {rel_type}")
        map_match = re.search(r"\{([^{}]*)\}", content)
        if map_match:
            map_properties = re.findall(
                r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*:", map_match.group(1)
            )
            for prop in map_properties:
                if rel_types and not any(
                    (
                        f"{rel_type}.{prop}" in allowed_properties
                        if allowed_properties is not None
                        else prop in property_names(rel_props.get(rel_type, []))
                    )
                    for rel_type in rel_types
                ):
                    errors.append(
                        f"Unknown relationship property in Schema for {rel_types}: {prop}"
                    )

    # Validate properties referenced as variable.property anywhere in the query.
    for variable, prop in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*`?([A-Za-z_][A-Za-z0-9_]*)`?",
        cypher_statement,
    ):
        if variable in node_variables:
            labels = node_variables[variable]
            if not any(
                (
                    f"{label}.{prop}" in allowed_properties
                    if allowed_properties is not None
                    else prop in property_names(node_props.get(label, []))
                )
                for label in labels
            ):
                errors.append(f"Unknown property in Schema for {labels}: {prop}")
        elif variable in relationship_variables:
            rel_types = relationship_variables[variable]
            if not any(
                (
                    f"{rel_type}.{prop}" in allowed_properties
                    if allowed_properties is not None
                    else prop in property_names(rel_props.get(rel_type, []))
                )
                for rel_type in rel_types
            ):
                errors.append(
                    f"Unknown relationship property in Schema for {rel_types}: {prop}"
                )

    # Preserve order while removing duplicate errors.
    return list(dict.fromkeys(errors))


def _validate_node_property_values_with_enum(
    structure_graph_schema: Neo4jStructuredSchema, tasks: List[CypherValidationTask]
) -> List[str]:
    prop_values_enum = structure_graph_schema.get_node_property_values_enum()

    errors = list()

    for t in tasks:
        labels = t.parsed_labels_or_types

        prop_val_validation_error = _validate_property_value_with_enum(
            enum_dict=prop_values_enum,
            labels_or_types=labels,
            node_or_rel="Node",
            property_name=t.property_name,
            property_value=t.property_value,
        )
        if prop_val_validation_error:
            errors.append(prop_val_validation_error)

    return errors


def _validate_node_property_names_with_enum(
    structure_graph_schema: Neo4jStructuredSchema, tasks: List[CypherValidationTask]
) -> List[str]:
    prop_enum = structure_graph_schema.get_node_properties_enum()

    errors = list()

    for t in tasks:
        labels = t.parsed_labels_or_types

        prop_validation_error = _validate_property_with_enum(
            enum_dict=prop_enum,
            labels_or_types=labels,
            node_or_rel="Node",
            property_name=t.property_name,
        )

        if prop_validation_error:
            errors.append(prop_validation_error)
    return errors


def _validate_relationship_property_names_with_enum(
    structure_graph_schema: Neo4jStructuredSchema, tasks: List[CypherValidationTask]
) -> List[str]:
    prop_enum = structure_graph_schema.get_relationship_properties_enum()

    errors = list()

    for t in tasks:
        rel_types = t.parsed_labels_or_types

        prop_validation_error = _validate_property_with_enum(
            enum_dict=prop_enum,
            labels_or_types=rel_types,
            node_or_rel="Relationship",
            property_name=t.property_name,
        )

        if prop_validation_error:
            errors.append(prop_validation_error)
    return errors


def _validate_relationship_property_values_with_enum(
    structure_graph_schema: Neo4jStructuredSchema, tasks: List[CypherValidationTask]
) -> List[str]:
    prop_values_enum = structure_graph_schema.get_relationship_property_values_enum()

    errors = list()

    for t in tasks:
        rel_types = t.parsed_labels_or_types
        prop_val_validation_error = _validate_property_value_with_enum(
            enum_dict=prop_values_enum,
            labels_or_types=rel_types,
            node_or_rel="Relationship",
            property_name=t.property_name,
            property_value=t.property_value,
        )
        if prop_val_validation_error:
            errors.append(prop_val_validation_error)

    return errors


def _validate_node_property_values_with_range(
    structure_graph_schema: Neo4jStructuredSchema,
    tasks: List[CypherValidationTask],
) -> List[str]:
    prop_values_range = structure_graph_schema.get_node_property_values_range()

    errors = list()

    for t in tasks:
        rel_types = t.parsed_labels_or_types
        prop_val_validation_error = _validate_property_value_with_range(
            enum_dict=prop_values_range,
            labels_or_types=rel_types,
            node_or_rel="Node",
            property_name=t.property_name,
            property_value=t.property_value,
        )
        if prop_val_validation_error:
            errors.append(prop_val_validation_error)

    return errors


def _validate_relationship_property_values_with_range(
    structure_graph_schema: Neo4jStructuredSchema,
    tasks: List[CypherValidationTask],
) -> List[str]:
    prop_values_range = structure_graph_schema.get_relationship_property_values_range()

    errors = list()

    for t in tasks:
        rel_types = t.parsed_labels_or_types
        prop_val_validation_error = _validate_property_value_with_range(
            enum_dict=prop_values_range,
            labels_or_types=rel_types,
            node_or_rel="Relationship",
            property_name=t.property_name,
            property_value=t.property_value,
        )
        if prop_val_validation_error:
            errors.append(prop_val_validation_error)

    return errors


def _validate_property_value_with_enum(
    enum_dict: Dict[str, Dict[str, Set[str]]],
    labels_or_types: List[str],
    property_name: str,
    node_or_rel: str,
    property_value: str,
    and_or: Optional[Literal["and", "or"]] = None,
) -> Optional[str]:
    """Validate that a property value is found in the enum generated from the graph schema."""

    assert node_or_rel in {
        "Node",
        "Relationship",
    }, f"Invalid `node_or_rel`: {node_or_rel}"

    if and_or is None and len(labels_or_types) > 1:
        raise ValueError(
            f"Invalid combination of `labels_or_types` and `and_or`: {labels_or_types} | {and_or}"
        )

    # track labels or types that are invalid
    # compare the number of invalid to the number tested to determine if valid
    invalid_labels_or_types = list()

    for lt in labels_or_types:
        props = enum_dict.get(lt)
        if props is None:
            # return None
            continue
        enum = props.get(property_name)
        if enum is None:
            # return None
            continue
        if property_value not in enum:
            invalid_labels_or_types.append(lt)

    if and_or is None and len(invalid_labels_or_types) > 0:  # single label or type
        return f"{node_or_rel} {labels_or_types} with property {property_name} = {property_value} not found in graph database."
    elif (
        and_or == "and"
        and 0 < len(invalid_labels_or_types)
        and len(invalid_labels_or_types) <= len(labels_or_types)
    ):
        return f"{node_or_rel}(s) {invalid_labels_or_types} with property {property_name} = {property_value} not found in graph database."
    elif and_or == "or" and len(invalid_labels_or_types) == len(labels_or_types):
        return f"None of {node_or_rel}s {labels_or_types} have property {property_name} = {property_value} in graph database."

    else:
        return None


def _validate_property_value_with_range(
    enum_dict: Dict[str, Dict[str, Neo4jStructuredSchemaPropertyNumber]],
    labels_or_types: List[str],
    property_name: str,
    node_or_rel: Literal["Node", "Relationship"],
    property_value: Union[int, float],
    and_or: Optional[Literal["and", "or"]] = None,
) -> Optional[str]:
    """Validate that a property value is found within the range generated from the graph schema."""

    assert node_or_rel in {
        "Node",
        "Relationship",
    }, f"Invalid `node_or_rel`: {node_or_rel}"

    # track labels or types that are invalid
    # compare the number of invalid to the number tested to determine to determine if valid
    invalid_labels_or_types: List[Tuple[str, Neo4jStructuredSchemaPropertyNumber]] = (
        list()
    )
    error_message_invalid_labels_or_types: List[str] = list()

    for lt in labels_or_types:
        props = enum_dict.get(lt)
        if props is None:
            # return None
            continue
        r = props.get(property_name)

        if r is None:
            # return None
            continue
        if float(property_value) < r.min or float(property_value) > r.max:
            invalid_labels_or_types.append((lt, r))

    for e in invalid_labels_or_types:
        if len(e) == 2:
            e_lt: str = e[0]  # the label or type
            e_prop: Neo4jStructuredSchemaPropertyNumber = e[1]
            error_message_invalid_labels_or_types.append(
                f"{e_lt} with property {e_prop.property} range {e_prop.min} to {e_prop.max}"
            )

    if and_or is None and len(invalid_labels_or_types) > 0:  # single label or type
        example_lt, example_prop = invalid_labels_or_types[0]
        return f"{node_or_rel} {example_lt} has property {property_name} = {property_value} which is out of range {example_prop.min} to {example_prop.max} in graph database."
    elif (
        and_or == "and"
        and 0 < len(invalid_labels_or_types)
        and len(invalid_labels_or_types) <= len(labels_or_types)
    ):
        return f"{node_or_rel}(s) {', '.join(error_message_invalid_labels_or_types)} have property {property_name} = {property_value} which is out of range in graph database."
    elif and_or == "or" and len(invalid_labels_or_types) == len(labels_or_types):
        return f"All of {node_or_rel}s {', '.join(error_message_invalid_labels_or_types)} have property {property_name} = {property_value} which is out of range in graph database."

    else:
        return None


def _validate_property_with_enum(
    enum_dict: Dict[str, Set[str]],
    labels_or_types: List[str],
    property_name: str,
    node_or_rel: Literal["Node", "Relationship"],
    and_or: Optional[Literal["and", "or"]] = None,
) -> Optional[str]:
    """Validate that a property name is found in the enum generated from the graph schema."""
    assert node_or_rel in {
        "Node",
        "Relationship",
    }, f"Invalid `node_or_rel`: {node_or_rel}"

    if and_or is None and len(labels_or_types) > 1:
        raise ValueError(
            f"Invalid combination of `labels_or_types` and `and_or`: {labels_or_types} | {and_or}"
        )

    # track labels or types that are invalid
    # compare the number of invalid to the number tested to determine to determine if valid
    invalid_labels_or_types = list()

    for lt in labels_or_types:
        enum = enum_dict.get(lt)

        if enum is None:
            # return None
            continue
        if property_name not in enum:
            invalid_labels_or_types.append(lt)

    if and_or is None and len(invalid_labels_or_types) > 0:  # single label or type
        return f"{node_or_rel} {labels_or_types} does not have the property {property_name} in the graph database."
    elif (
        and_or == "and"
        and 0 < len(invalid_labels_or_types)
        and len(invalid_labels_or_types) <= len(labels_or_types)
    ):
        return f"{node_or_rel}(s) {invalid_labels_or_types} do(es) not have the property {property_name} in the graph database."
    elif and_or == "or" and len(invalid_labels_or_types) == len(labels_or_types):
        return f"None of {node_or_rel}s {labels_or_types} have the property {property_name} in the graph database."

    else:
        return None


def validate_no_writes_in_cypher_query(cypher_statement: str) -> List[str]:
    """
    Validate whether the provided Cypher contains any write clauses.

    Parameters
    ----------
    cypher_statement : str
        The Cypher statement to validate.

    Returns
    -------
    List[str]
        A list of any found errors.
    """
    errors: List[str] = list()

    # Keywords inside values, quoted identifiers and comments are data rather
    # than executable clauses. Mask them before applying the deny-list so, for
    # example, `RETURN 'DELETE'` is not rejected as a write query.
    code_only = re.sub(
        r"'(?:''|\\.|[^'])*'|\"(?:\"\"|\\.|[^\"])*\"|`(?:``|[^`])*`|//[^\r\n]*|/\*.*?\*/",
        " ",
        cypher_statement,
        flags=re.DOTALL,
    )

    for wc in sorted(WRITE_CLAUSES, key=lambda clause: (-len(clause), clause)):
        pattern = r"\b" + r"\s+".join(map(re.escape, wc.split())) + r"\b"
        if re.search(pattern, code_only, flags=re.IGNORECASE):
            errors.append(f"Cypher contains write clause: {wc}")

    return errors

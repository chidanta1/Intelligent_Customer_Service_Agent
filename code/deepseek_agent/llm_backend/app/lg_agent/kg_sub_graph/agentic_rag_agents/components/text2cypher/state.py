"""
This file contains classes that manage the state of a Text2Cypher Agent or subgraph.
"""

from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional

from typing_extensions import NotRequired, TypedDict


class CypherInputState(TypedDict):
    task: str


class ValidationLayerResult(TypedDict):
    """One validation layer's immutable result for a single attempt."""

    layer: Literal[
        "syntax",
        "write_operation",
        "relationship_direction",
        "llm",
        "schema",
    ]
    status: Literal["passed", "failed", "corrected", "skipped"]
    errors: List[str]
    warnings: NotRequired[List[str]]
    error_kind: NotRequired[Optional[str]]


class ValidationAttempt(TypedDict):
    """Complete evidence produced by one pass through all five layers."""

    attempt: int
    retry: int
    statement_before: str
    statement_after: str
    layers: List[ValidationLayerResult]
    errors: List[str]
    mapping_errors: List[str]
    operational_failure: NotRequired[bool]
    next_action: str


class CorrectionAttempt(TypedDict):
    """Input and output of one LLM correction retry."""

    retry: int
    statement_before: str
    statement_after: str
    errors: List[str]


class CypherState(TypedDict):
    task: str
    statement: str
    parameters: Optional[Dict[str, Any]]
    errors: List[str]
    mapping_errors: List[str]
    records: List[Dict[str, Any]]
    next_action_cypher: str
    attempts: int
    retries: int
    validation_layers: List[str]
    validation_trace: Annotated[List[ValidationAttempt], add]
    correction_history: Annotated[List[CorrectionAttempt], add]
    fewshot_examples: str
    fewshot_k: int
    retrieval_domain: str
    retrieval_query_type: str
    schema_version: str
    steps: Annotated[List[str], add]


class CypherOutputState(TypedDict):
    task: str
    statement: str
    parameters: Optional[Dict[str, Any]]
    errors: List[str]
    records: List[Dict[str, Any]]
    attempts: int
    retries: int
    validation_layers: List[str]
    validation_trace: List[ValidationAttempt]
    correction_history: List[CorrectionAttempt]
    steps: List[str]


class CypherAgentOutputState(TypedDict):
    cyphers: List[CypherOutputState]
    steps: List[str]

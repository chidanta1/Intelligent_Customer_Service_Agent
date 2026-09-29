WRITE_CLAUSES = {
    "ALTER",
    "CREATE",
    "DELETE",
    "DETACH DELETE",
    "DENY",
    "DROP",
    "SET",
    "REMOVE",
    "FOREACH",
    "GRANT",
    "INSERT",
    "MERGE",
    "RENAME",
    "REVOKE",
    "START DATABASE",
    "STOP DATABASE",
    "TERMINATE TRANSACTIONS",
}

MAX_TEXT2CYPHER_CORRECTION_RETRIES = 3


NO_CYPHER_RESULTS = [
    {"error": "I couldn't find any relevant information in the database."}
]

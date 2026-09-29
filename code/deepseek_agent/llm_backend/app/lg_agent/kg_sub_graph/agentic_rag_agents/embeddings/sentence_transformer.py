"""Sentence-Transformer adapter used by the Cypher example vector store."""

from functools import lru_cache
from typing import Any, List


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> Any:
    """Load each embedding model once per backend process."""

    # Keep torch/transformers lazy so ordinary LangGraph imports stay lightweight.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    """Synchronous embedder compatible with ``EmbedderProtocol``."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = model_name

    @property
    def dimension(self) -> int:
        model = _load_model(self.model_name)
        get_dimension = getattr(
            model,
            "get_embedding_dimension",
            model.get_sentence_embedding_dimension,
        )
        return int(get_dimension())

    def embed_query(self, text: str) -> List[float]:
        model = _load_model(self.model_name)
        vector = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.astype(float).tolist()

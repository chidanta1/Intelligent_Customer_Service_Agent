"""Deterministic local character n-gram embeddings for Chinese GraphRAG data."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import HashingVectorizer


class LocalHashEmbedding:
    """Stateless local embeddings with a fixed indexing/query vector space."""

    def __init__(self, *, name: str, config: Any, **_: Any) -> None:
        self.name = name
        self.config = config
        self.vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=1024,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )

    def embed_batch(self, text_list: list[str], **_: Any) -> list[list[float]]:
        matrix = self.vectorizer.transform(text_list)
        return matrix.toarray().astype(float).tolist()

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_batch([text], **kwargs)[0]

    async def aembed_batch(
        self, text_list: list[str], **kwargs: Any
    ) -> list[list[float]]:
        return self.embed_batch(text_list, **kwargs)

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed(text, **kwargs)

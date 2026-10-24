"""Dependency-light text similarity used for de-duplication and cross-meeting tracking.

`HashingEmbedder` maps text to L2-normalised hashed bag-of-(stemmed)-words + bigram vectors with
numpy. It is deterministic, needs no model download and is good enough for matching short task
descriptions ("Fix the promo code reset bug" vs "the promo code reset bug is fixed").
The `Embedder` protocol lets a sentence-transformers or API embedder be swapped in.
"""

from __future__ import annotations

import hashlib
import re
from itertools import pairwise
from typing import Protocol

import numpy as np

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "so",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "up",
        "down",
        "out",
        "over",
        "into",
        "onto",
        "about",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "its",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "then",
        "than",
        "just",
        "also",
        "too",
        "very",
        "really",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "let",
        "lets",
        "let's",
        "i'll",
        "we'll",
        "you'll",
        "it's",
        "that's",
        "i'm",
        "i've",
        "we've",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "how",
        "why",
        "all",
        "any",
        "some",
        "one",
        "two",
        "get",
        "got",
        "go",
        "going",
        "okay",
        "ok",
        "yeah",
        "yes",
        "sure",
        "great",
        "thanks",
        "quick",
    ]
)

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


IRREGULAR = {
    "ran": "run",
    "sent": "send",
    "wrote": "write",
    "written": "write",
    "built": "build",
    "made": "make",
    "did": "do",
    "done": "do",
    "went": "go",
    "taken": "take",
    "took": "take",
    "found": "find",
    "brought": "bring",
    "led": "lead",
    "spoke": "speak",
    "set": "set",
}


def stem(token: str) -> str:
    """Tiny suffix stripper; both 'file' and 'filed' map to 'fil', 'ran' maps to 'run'."""
    if token in IRREGULAR:
        return IRREGULAR[token]
    return _strip_e(_stem(token))


def _strip_e(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("e") else token


def _stem(token: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "edly", "ied", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            if suffix in ("ied", "ies"):
                base += "y"
            if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "ls":
                base = base[:-1]
            return base
    return token


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 4096, bigram_weight: float = 0.5):
        self.dim = dim
        self.bigram_weight = bigram_weight

    def _idx(self, feature: str) -> int:
        return int.from_bytes(hashlib.md5(feature.encode()).digest()[:4], "little") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = content_tokens(text)
            for t in toks:
                out[row, self._idx(t)] += 1.0
            for a, b in pairwise(toks):
                out[row, self._idx(f"{a}_{b}")] += self.bigram_weight
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


_default = HashingEmbedder()


def cosine_matrix(a: list[str], b: list[str], embedder: Embedder | None = None) -> np.ndarray:
    emb = embedder or _default
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return emb.embed(a) @ emb.embed(b).T


def similarity(a: str, b: str, embedder: Embedder | None = None) -> float:
    return float(cosine_matrix([a], [b], embedder)[0, 0])

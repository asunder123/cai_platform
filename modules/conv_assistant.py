# case_engine.py
# Deterministic case parsing + retrieval + sentence-level answer extraction

import re
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Text Utilities
# ============================================================

def clean_text(text: str) -> str:
    """Normalize whitespace and remove noisy formatting."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 120,
    overlap: int = 30
) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)

    return chunks


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no NLP dependency)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ============================================================
# Core Engine
# ============================================================

class CaseIndex:
    """
    Case text semantic index with sentence-level answer extraction.
    """

    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.chunks: List[str] = []

    # --------------------------------------------------------

    def build(self, chunks: List[str]) -> None:
        """Build TF-IDF index over case chunks."""
        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1
        )
        self.matrix = self.vectorizer.fit_transform(chunks)

    # --------------------------------------------------------

    def _rank_sentences(
        self,
        sentences: List[str],
        query_text: str,
        top_n: int
    ) -> List[Dict]:
        """
        Rank sentences inside a chunk for answer precision.
        """
        sent_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1
        )

        sent_matrix = sent_vectorizer.fit_transform(sentences)
        q_vec = sent_vectorizer.transform([query_text])

        scores = cosine_similarity(q_vec, sent_matrix)[0]
        ranked_idx = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_idx[:top_n]:
            score = float(scores[idx])
            if score > 0:
                results.append({
                    "sentence": sentences[idx],
                    "sentence_score": score
                })

        return results

    # --------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k_chunks: int = 3,
        sentences_per_chunk: int = 1,
        min_chunk_score: float = 0.05
    ) -> List[Dict]:
        """
        Query the case and return precise, sentence-level answers.
        """
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Case index has not been built")

        q_vec = self.vectorizer.transform([query_text])
        chunk_scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked_chunks = np.argsort(chunk_scores)[::-1]

        answers: List[Dict] = []

        for idx in ranked_chunks[:top_k_chunks]:
            chunk_score = float(chunk_scores[idx])
            if chunk_score < min_chunk_score:
                continue

            chunk_text = self.chunks[idx]
            sentences = split_sentences(chunk_text)

            if not sentences:
                continue

            ranked_sentences = self._rank_sentences(
                sentences,
                query_text,
                sentences_per_chunk
            )

            for sent in ranked_sentences:
                answers.append({
                    "answer": sent["sentence"],
                    "sentence_score": sent["sentence_score"],
                    "chunk_score": chunk_score,
                    "source_chunk": chunk_text
                })

        return answers


# ============================================================
# High-Level Builder
# ============================================================

def build_case_index(
    case_text: str,
    chunk_size: int,
    overlap: int
) -> CaseIndex:
    """End-to-end case parsing + indexing."""
    cleaned = clean_text(case_text)
    chunks = chunk_text(cleaned, chunk_size, overlap)

    index = CaseIndex()
    index.build(chunks)

    return index

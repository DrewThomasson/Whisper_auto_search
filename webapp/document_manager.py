"""
document_manager.py — Core document management and search logic.

Extracted from interview_assistant.py; no GUI dependencies.
Supports PDF, DOCX, and TXT; searches via the best available backend:
  1. Semantic  — sentence-transformers all-MiniLM-L6-v2
  2. Hybrid    — BM25 + TF-IDF combined
  3. TF-IDF    — cosine similarity fallback
  4. Keyword   — simple word-overlap bare minimum
"""

from __future__ import annotations

import re
import string
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── NLTK ─────────────────────────────────────────────────────────────────────
import nltk

for _ds in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(_ds, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# ── Optional backends ─────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _BM25 = True
except ImportError:
    _BM25 = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _SBERT = True
except ImportError:
    _SBERT = False


def _best_device() -> str:
    """Return the best available compute device (cuda → mps → cpu)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DocumentChunk:
    """One searchable chunk from a loaded document."""

    source_file: str
    text: str
    chunk_index: int = 0
    page_num: Optional[int] = None

    @property
    def display_source(self) -> str:
        name = Path(self.source_file).name
        if self.page_num is not None:
            return f"{name} · p.{self.page_num}"
        return name

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "display_source": self.display_source,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "page_num": self.page_num,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Document Manager
# ─────────────────────────────────────────────────────────────────────────────


class DocumentManager:
    """
    Loads documents (PDF / DOCX / TXT) and searches them using the best
    available backend.

    Thread-safety: ``load_file`` and ``clear`` must not be called concurrently;
    ``search`` is read-only and safe to call from any thread after indexing.
    """

    CHUNK_SENTENCES = 4
    OVERLAP_SENTENCES = 1
    MIN_CHUNK_LENGTH = 20

    SEMANTIC_MIN_SCORE: float = 0.20
    HYBRID_MIN_SCORE: float = 0.04
    KEYWORD_MIN_SCORE: float = 0.10

    _TFIDF_WEIGHT: float = 0.5
    _BM25_WEIGHT: float = 0.5

    EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self.chunks: List[DocumentChunk] = []
        self.backend_name: str = "keyword"
        self._stop_words = set(nltk_stopwords.words("english"))

        self._embed_model = None
        self._embeddings: Optional[np.ndarray] = None

        self._bm25 = None
        self._tfidf_vec = None
        self._tfidf_mat = None

        self._lock = threading.Lock()
        self._model_device: str = "cpu"

        # Automatically load the embedding model in a background thread.
        self._model_ready = threading.Event()
        self._load_model_thread = threading.Thread(
            target=self._load_embed_model_bg, daemon=True
        )
        self._load_model_thread.start()

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        return {
            "backend": self.backend_name,
            "chunk_count": len(self.chunks),
            "model_device": self._model_device,
            "model_ready": self._embed_model is not None,
        }

    # ── Embed model loading ───────────────────────────────────────────────────

    def _load_embed_model_bg(self) -> None:
        if not _SBERT:
            self._model_ready.set()
            return
        device = _best_device()
        try:
            try:
                from transformers import logging as hf_logging  # type: ignore

                hf_logging.set_verbosity_error()
            except ImportError:
                pass
            model = SentenceTransformer(self.EMBED_MODEL, device=device)
            with self._lock:
                self._embed_model = model
                self._model_device = device
                if self.backend_name in ("keyword", "tfidf", "hybrid"):
                    self.backend_name = "semantic (indexing…)"
        except Exception as exc:
            print(f"[DocumentManager] Embed model failed: {exc}")
        finally:
            self._model_ready.set()

    def _maybe_build_semantic_index(self) -> None:
        """If model is ready and we have chunks, build the embedding index."""
        if self._embed_model is None or not self.chunks:
            return
        if self._embeddings is not None and len(self._embeddings) == len(self.chunks):
            return
        texts = [c.text for c in self.chunks]
        try:
            embeddings = self._embed_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            with self._lock:
                self._embeddings = embeddings
                self.backend_name = "semantic"
        except Exception as exc:
            print(f"[DocumentManager] Embedding failed: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        with self._lock:
            self.chunks.clear()
            self._embeddings = None
            self._bm25 = None
            self._tfidf_vec = None
            self._tfidf_mat = None
            self.backend_name = "keyword"

    def load_file(self, path: str, display_name: str | None = None) -> int:
        """Load a file, rebuild keyword indices, then start semantic index.

        Args:
            path: Filesystem path to the file.
            display_name: Optional human-readable name shown in the UI.
                          Defaults to the filename part of *path*.
        """
        ext = Path(path).suffix.lower()
        source = display_name or path
        try:
            if ext == ".pdf":
                new_chunks = self._load_pdf(path, source=source)
            elif ext == ".docx":
                new_chunks = self._load_docx(path, source=source)
            else:
                new_chunks = self._load_txt(path, source=source)
        except Exception as exc:
            print(f"[DocumentManager] Error loading {path}: {exc}")
            return 0

        if new_chunks:
            with self._lock:
                self.chunks.extend(new_chunks)
            self._rebuild_keyword_index()
            # Build / refresh semantic index in a background thread so the
            # API response returns immediately.
            t = threading.Thread(target=self._maybe_build_semantic_index, daemon=True)
            t.start()
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return (chunk, score) pairs ranked by relevance."""
        if not self.chunks or not query.strip():
            return []

        if self._embeddings is not None and self._embed_model is not None:
            return self._search_semantic(
                query,
                top_k,
                min_score if min_score is not None else self.SEMANTIC_MIN_SCORE,
            )

        if self._bm25 is not None and self._tfidf_vec is not None:
            return self._search_hybrid(
                query,
                top_k,
                min_score if min_score is not None else self.HYBRID_MIN_SCORE,
            )

        if self._tfidf_vec is not None:
            return self._search_tfidf(
                query,
                top_k,
                min_score if min_score is not None else self.HYBRID_MIN_SCORE,
            )

        return self._search_keyword(
            query,
            top_k,
            min_score if min_score is not None else self.KEYWORD_MIN_SCORE,
        )

    # ── File loading ──────────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        source: str,
        page_num: Optional[int] = None,
    ) -> List[DocumentChunk]:
        sentences = sent_tokenize(text)
        step = max(1, self.CHUNK_SENTENCES - self.OVERLAP_SENTENCES)
        chunks: List[DocumentChunk] = []
        for i in range(0, len(sentences), step):
            chunk_text = " ".join(sentences[i : i + self.CHUNK_SENTENCES]).strip()
            if len(chunk_text) >= self.MIN_CHUNK_LENGTH:
                chunks.append(
                    DocumentChunk(
                        source_file=source,
                        text=chunk_text,
                        chunk_index=len(self.chunks) + len(chunks),
                        page_num=page_num,
                    )
                )
        return chunks

    def _load_txt(self, path: str, source: str | None = None) -> List[DocumentChunk]:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return self._chunk_text(fh.read(), source or path)

    def _load_pdf(self, path: str, source: str | None = None) -> List[DocumentChunk]:
        src = source or path
        chunks: List[DocumentChunk] = []
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(path) as pdf:
                for pnum, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        chunks.extend(self._chunk_text(text, src, pnum))
            return chunks
        except Exception:
            pass
        try:
            import PyPDF2  # type: ignore

            with open(path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for pnum, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        chunks.extend(self._chunk_text(text, src, pnum))
            return chunks
        except Exception:
            pass
        return self._load_txt(path, source=src)

    def _load_docx(self, path: str, source: str | None = None) -> List[DocumentChunk]:
        src = source or path
        try:
            from docx import Document  # type: ignore

            doc = Document(path)
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return self._chunk_text(full_text, src)
        except Exception:
            return self._load_txt(path, source=src)

    # ── Keyword index ─────────────────────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(
            w
            for w in word_tokenize(text)
            if w.isalpha() and w not in self._stop_words
        )

    def _tokenize(self, text: str) -> List[str]:
        return self._preprocess(text).split()

    def _rebuild_keyword_index(self) -> None:
        with self._lock:
            texts = [c.text for c in self.chunks]
        best = "keyword"

        if _SKLEARN:
            try:
                proc = [self._preprocess(t) for t in texts]
                vec = TfidfVectorizer(
                    ngram_range=(1, 2), max_features=8_000, sublinear_tf=True
                )
                mat = vec.fit_transform(proc)
                with self._lock:
                    self._tfidf_vec = vec
                    self._tfidf_mat = mat
                best = "tfidf"
            except Exception as exc:
                print(f"[TF-IDF] {exc}")

        if _BM25:
            try:
                corpus = [self._tokenize(t) for t in texts]
                bm25 = BM25Okapi(corpus)
                with self._lock:
                    self._bm25 = bm25
                if best == "tfidf":
                    best = "hybrid"
            except Exception as exc:
                print(f"[BM25] {exc}")

        with self._lock:
            if self.backend_name in ("keyword", "tfidf", "hybrid"):
                self.backend_name = best

    # ── Search backends ───────────────────────────────────────────────────────

    def _search_semantic(
        self, query: str, top_k: int, min_score: float
    ) -> List[Tuple[DocumentChunk, float]]:
        q_emb = self._embed_model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(q_emb, self._embeddings)[0]
        idx = np.argsort(scores)[::-1]
        return [
            (self.chunks[i], float(scores[i]))
            for i in idx
            if scores[i] >= min_score
        ][:top_k]

    def _search_hybrid(
        self, query: str, top_k: int, min_score: float
    ) -> List[Tuple[DocumentChunk, float]]:
        pq = self._preprocess(query)
        if pq.strip():
            q_vec = self._tfidf_vec.transform([pq])
            tfidf_raw = cosine_similarity(q_vec, self._tfidf_mat)[0]
        else:
            tfidf_raw = np.zeros(len(self.chunks))
        tfidf_norm = tfidf_raw / (tfidf_raw.max() or 1.0)

        tokens = self._tokenize(query)
        if tokens:
            bm25_raw = np.array(self._bm25.get_scores(tokens))
        else:
            bm25_raw = np.zeros(len(self.chunks))
        bm25_norm = bm25_raw / (bm25_raw.max() or 1.0)

        final = self._TFIDF_WEIGHT * tfidf_norm + self._BM25_WEIGHT * bm25_norm
        idx = np.argsort(final)[::-1]
        return [
            (self.chunks[i], float(final[i]))
            for i in idx
            if final[i] >= min_score
        ][:top_k]

    def _search_tfidf(
        self, query: str, top_k: int, min_score: float
    ) -> List[Tuple[DocumentChunk, float]]:
        pq = self._preprocess(query)
        if not pq.strip():
            return []
        q_vec = self._tfidf_vec.transform([pq])
        scores = cosine_similarity(q_vec, self._tfidf_mat)[0]
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            (self.chunks[i], float(scores[i]))
            for i in idx
            if scores[i] >= min_score
        ]

    def _search_keyword(
        self, query: str, top_k: int, min_score: float
    ) -> List[Tuple[DocumentChunk, float]]:
        qwords = set(self._tokenize(query))
        if not qwords:
            return []
        results: List[Tuple[DocumentChunk, float]] = []
        for chunk in self.chunks:
            cwords = set(self._tokenize(chunk.text))
            if cwords:
                score = len(qwords & cwords) / len(qwords)
                if score >= min_score:
                    results.append((chunk, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

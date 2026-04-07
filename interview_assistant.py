#!/usr/bin/env python3
"""
Interview Assistant — Live Speech-to-Document Reference Finder
==============================================================
As an interviewee speaks, this tool automatically surfaces relevant sections
from their resume and supplemental documents in real time, so the interviewer
always has context without needing to pre-read everything.

Search backend hierarchy (best available is used automatically):
  1. Semantic  — sentence-transformers all-MiniLM-L6-v2 (meaning-based, most accurate)
  2. Hybrid    — BM25 + TF-IDF combined  (good keyword precision)
  3. TF-IDF    — cosine similarity fallback
  4. Keyword   — simple word-overlap bare-minimum fallback

Usage:
    python interview_assistant.py

Requirements:
    pip install -r requirements.txt
"""

import os
import sys
import re
import queue
import string
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ── Qt ──────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QFrame, QScrollArea,
    QSplitter, QStatusBar, QGroupBox, QComboBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextCursor

# ── NLP ──────────────────────────────────────────────────────────────────────
import nltk

for _ds in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(_ds, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# ── Optional backends ────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Background threads for non-blocking model loading and indexing
# ─────────────────────────────────────────────────────────────────────────────

class ModelLoaderThread(QThread):
    """Loads the sentence-transformer embedding model in the background."""

    EMBED_MODEL = "all-MiniLM-L6-v2"  # 22 M params, ~80 MB, fast on CPU

    model_ready = pyqtSignal(object, str)   # (model_or_None, device_name)
    status_update = pyqtSignal(str)

    def run(self) -> None:
        if not _SBERT:
            self.status_update.emit(
                "sentence-transformers not installed → using TF-IDF+BM25  "
                "(pip install sentence-transformers for semantic search)"
            )
            self.model_ready.emit(None, "cpu")
            return

        device = _best_device()
        self.status_update.emit(
            f"Loading semantic model '{self.EMBED_MODEL}' on {device} "
            "(first run downloads ~80 MB)…"
        )
        try:
            # Suppress the harmless "embeddings.position_ids UNEXPECTED" warning
            # that transformers emits when loading all-MiniLM-L6-v2.
            try:
                from transformers import logging as hf_logging  # type: ignore
                hf_logging.set_verbosity_error()
            except ImportError:
                pass
            model = SentenceTransformer(self.EMBED_MODEL, device=device)
            self.status_update.emit(f"Semantic model ready  [{device}]  ✔")
            self.model_ready.emit(model, device)
        except Exception as exc:
            self.status_update.emit(
                f"Semantic model failed ({exc}) → falling back to TF-IDF+BM25"
            )
            self.model_ready.emit(None, "cpu")


class EmbeddingIndexThread(QThread):
    """Computes sentence embeddings for all chunks in the background."""

    embeddings_ready = pyqtSignal(object)   # numpy ndarray or None
    status_update = pyqtSignal(str)

    def __init__(
        self,
        model,
        chunks: List[DocumentChunk],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._chunks = list(chunks)  # snapshot

    def run(self) -> None:
        n = len(self._chunks)
        self.status_update.emit(f"Computing semantic embeddings for {n} chunks…")
        try:
            texts = [c.text for c in self._chunks]
            embeddings = self._model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self.status_update.emit(f"Semantic index ready  ({n} chunks)  ✔")
            self.embeddings_ready.emit(embeddings)
        except Exception as exc:
            self.status_update.emit(f"Embedding index failed: {exc}")
            self.embeddings_ready.emit(None)


# ─────────────────────────────────────────────────────────────────────────────
# Document Manager
# ─────────────────────────────────────────────────────────────────────────────

class DocumentManager:
    """
    Loads documents (PDF / DOCX / TXT) and searches them using the best
    available backend.

    Backend priority (set automatically as capabilities become available):
      1. Semantic  — sentence-transformer cosine similarity (meaning-based)
      2. Hybrid    — BM25 + TF-IDF combined (exact-term precision + IDF weight)
      3. TF-IDF    — cosine similarity with IDF weighting
      4. Keyword   — simple word-overlap, bare-minimum fallback
    """

    CHUNK_SENTENCES = 4    # sentences per chunk; 4 works well for semantic models
    # (too large dilutes the embedding; too small loses context)
    OVERLAP_SENTENCES = 1   # sentence overlap between adjacent chunks
    MIN_CHUNK_LENGTH = 20   # chars — skip chunks shorter than this (headers, page numbers, etc.)

    # Default similarity thresholds per backend
    SEMANTIC_MIN_SCORE: float = 0.20   # cosine similarity; 0.20+ is meaningfully related
    HYBRID_MIN_SCORE: float = 0.04     # normalised BM25+TF-IDF combined score
    KEYWORD_MIN_SCORE: float = 0.10    # fraction of query words that must appear

    # Hybrid combination weights (must sum to 1.0)
    _TFIDF_WEIGHT: float = 0.5   # TF-IDF cosine: rewards rare, document-specific terms
    _BM25_WEIGHT: float = 0.5    # BM25: rewards exact token matches

    def __init__(self) -> None:
        self.chunks: List[DocumentChunk] = []
        self.backend_name: str = "keyword"
        self._stop_words = set(nltk_stopwords.words("english"))

        # Semantic backend (populated externally via set_embed_model / set_embeddings)
        self._embed_model = None     # SentenceTransformer
        self._embeddings: Optional[np.ndarray] = None  # shape (N, D)

        # Keyword backends (built synchronously when a file is loaded)
        self._bm25 = None
        self._tfidf_vec = None
        self._tfidf_mat = None

    # ── External setters (called from QThread signals) ───────────────────────

    @property
    def embed_model(self):
        """The loaded SentenceTransformer model, or None."""
        return self._embed_model

    def set_embed_model(self, model) -> None:
        self._embed_model = model
        if model is not None and self.backend_name in ("keyword", "tfidf", "hybrid"):
            self.backend_name = "semantic (indexing…)"

    def set_embeddings(self, embeddings: Optional[np.ndarray]) -> None:
        if embeddings is not None:
            self._embeddings = embeddings
            self.backend_name = "semantic"
        else:
            # Embedding failed — revert to best available keyword backend
            if self._bm25 is not None and self._tfidf_vec is not None:
                self.backend_name = "hybrid"
            elif self._tfidf_vec is not None:
                self.backend_name = "tfidf"
            else:
                self.backend_name = "keyword"

    # ── Public API ──────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.chunks.clear()
        self._embeddings = None
        self._bm25 = None
        self._tfidf_vec = None
        self._tfidf_mat = None
        self.backend_name = "keyword"

    def load_file(self, path: str) -> int:
        """Load a file and rebuild keyword indices; return chunk count added."""
        ext = Path(path).suffix.lower()
        try:
            if ext == ".pdf":
                new_chunks = self._load_pdf(path)
            elif ext == ".docx":
                new_chunks = self._load_docx(path)
            else:
                new_chunks = self._load_txt(path)
        except Exception as exc:
            print(f"[DocumentManager] Error loading {path}: {exc}")
            return 0

        if new_chunks:
            self.chunks.extend(new_chunks)
            self._rebuild_keyword_index()
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return (chunk, score) pairs ranked by relevance, best backend first."""
        if not self.chunks or not query.strip():
            return []

        if self._embeddings is not None and self._embed_model is not None:
            return self._search_semantic(
                query, top_k,
                min_score if min_score is not None else self.SEMANTIC_MIN_SCORE,
            )

        if self._bm25 is not None and self._tfidf_vec is not None:
            return self._search_hybrid(
                query, top_k,
                min_score if min_score is not None else self.HYBRID_MIN_SCORE,
            )

        if self._tfidf_vec is not None:
            return self._search_tfidf(
                query, top_k,
                min_score if min_score is not None else self.HYBRID_MIN_SCORE,
            )

        return self._search_keyword(
            query, top_k,
            min_score if min_score is not None else self.KEYWORD_MIN_SCORE,
        )

    # ── File loading ─────────────────────────────────────────────────────────

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

    def _load_txt(self, path: str) -> List[DocumentChunk]:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return self._chunk_text(fh.read(), path)

    def _load_pdf(self, path: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(path) as pdf:
                for pnum, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        chunks.extend(self._chunk_text(text, path, pnum))
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
                        chunks.extend(self._chunk_text(text, path, pnum))
            return chunks
        except Exception:
            pass
        return self._load_txt(path)

    def _load_docx(self, path: str) -> List[DocumentChunk]:
        try:
            from docx import Document  # type: ignore
            doc = Document(path)
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return self._chunk_text(full_text, path)
        except Exception:
            return self._load_txt(path)

    # ── Keyword index ────────────────────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(
            w for w in word_tokenize(text)
            if w.isalpha() and w not in self._stop_words
        )

    def _tokenize(self, text: str) -> List[str]:
        return self._preprocess(text).split()

    def _rebuild_keyword_index(self) -> None:
        """Build TF-IDF and BM25 indices synchronously (fast, keyword-level)."""
        texts = [c.text for c in self.chunks]
        best = "keyword"

        if _SKLEARN:
            try:
                proc = [self._preprocess(t) for t in texts]
                self._tfidf_vec = TfidfVectorizer(
                    ngram_range=(1, 2), max_features=8_000, sublinear_tf=True
                )
                self._tfidf_mat = self._tfidf_vec.fit_transform(proc)
                best = "tfidf"
            except Exception as exc:
                print(f"[TF-IDF] {exc}")

        if _BM25:
            try:
                corpus = [self._tokenize(t) for t in texts]
                self._bm25 = BM25Okapi(corpus)
                if best == "tfidf":
                    best = "hybrid"
            except Exception as exc:
                print(f"[BM25] {exc}")

        # Only update backend_name if semantic is not already active
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
        # TF-IDF scores (normalised to [0, 1])
        pq = self._preprocess(query)
        if pq.strip():
            q_vec = self._tfidf_vec.transform([pq])
            tfidf_raw = cosine_similarity(q_vec, self._tfidf_mat)[0]
        else:
            tfidf_raw = np.zeros(len(self.chunks))
        tfidf_norm = tfidf_raw / (tfidf_raw.max() or 1.0)

        # BM25 scores (normalised)
        tokens = self._tokenize(query)
        if tokens:
            bm25_raw = np.array(self._bm25.get_scores(tokens))
        else:
            bm25_raw = np.zeros(len(self.chunks))
        bm25_norm = bm25_raw / (bm25_raw.max() or 1.0)

        # Weighted combination — BM25 rewards exact matches; TF-IDF rewards rarity
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
        return [(self.chunks[i], float(scores[i])) for i in idx if scores[i] >= min_score]

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


# ─────────────────────────────────────────────────────────────────────────────
# Audio Transcription Thread
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionThread(QThread):
    """
    Continuously captures microphone audio and emits transcribed text.

    Backends tried in order:
      1. faster-whisper  (fast, local, recommended)
      2. openai-whisper  (local, slightly slower)
      3. SpeechRecognition + Google (requires internet)
    """

    new_text = pyqtSignal(str)
    status_update = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    SAMPLE_RATE = 16_000
    CHUNK_SECONDS = 4   # seconds of audio fed to each transcription call
    STEP_SECONDS = 2    # how far to advance the sliding window (yields 2 s overlap)

    def __init__(self, model_size: str = "base", parent=None) -> None:
        super().__init__(parent)
        self.model_size = model_size
        self._stop_event = threading.Event()
        self._audio_q: queue.Queue = queue.Queue()
        self._model = None
        self._backend: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        device = _best_device()

        # faster-whisper uses CTranslate2 which only supports "cpu" and "cuda".
        # MPS (Apple Silicon) is not supported — fall back to CPU for this backend.
        fw_device = "cpu" if device == "mps" else device
        fw_compute = "int8" if fw_device == "cpu" else "float16"

        # 1. faster-whisper
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self.status_update.emit(
                f"Loading faster-whisper ({self.model_size}) on {fw_device}…"
            )
            self._model = WhisperModel(
                self.model_size, device=fw_device, compute_type=fw_compute
            )
            self._backend = "faster_whisper"
            self.status_update.emit(f"faster-whisper ready  [{fw_device}]  ✔")
            return True
        except ImportError:
            pass
        except Exception as exc:
            print(f"[faster-whisper load error] {exc}")

        # 2. openai-whisper (supports MPS via PyTorch natively)
        try:
            import whisper  # type: ignore
            self.status_update.emit(
                f"Loading openai-whisper ({self.model_size}) on {device}…"
            )
            self._model = whisper.load_model(self.model_size, device=device)
            self._backend = "openai_whisper"
            self.status_update.emit(f"openai-whisper ready  [{device}]  ✔")
            return True
        except ImportError:
            pass
        except Exception as exc:
            print(f"[openai-whisper load error] {exc}")

        # 3. SpeechRecognition (Google, needs internet)
        try:
            import speech_recognition as sr  # type: ignore
            self._model = sr.Recognizer()
            self._backend = "speech_recognition"
            self.status_update.emit("SpeechRecognition (Google) ready  ✔")
            return True
        except ImportError:
            pass

        self.error_signal.emit(
            "No speech recognition backend found.\n\n"
            "Install at least one of:\n"
            "  pip install faster-whisper\n"
            "  pip install openai-whisper\n"
            "  pip install SpeechRecognition\n\n"
            "You can still use the Manual Input box for testing."
        )
        return False

    # ── Transcription ────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._backend == "faster_whisper":
            segments, _ = self._model.transcribe(
                audio, language="en", vad_filter=True
            )
            return " ".join(seg.text for seg in segments).strip()

        if self._backend == "openai_whisper":
            result = self._model.transcribe(audio, language="en")
            return result.get("text", "").strip()

        if self._backend == "speech_recognition":
            import speech_recognition as sr  # type: ignore
            import io
            import wave

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            buf.seek(0)
            audio_data = sr.AudioData(buf.read(), self.SAMPLE_RATE, 2)
            try:
                return self._model.recognize_google(audio_data)
            except Exception:
                return ""

        return ""

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        if not self._load_model():
            return

        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            self.error_signal.emit(
                "sounddevice not found.\n"
                "Install with:  pip install sounddevice\n\n"
                "On macOS you may also need:  brew install portaudio"
            )
            return

        chunk_samples = int(self.SAMPLE_RATE * self.CHUNK_SECONDS)
        step_samples = int(self.SAMPLE_RATE * self.STEP_SECONDS)
        buffer = np.zeros(0, dtype=np.float32)

        self.status_update.emit("🎙 Listening…")

        def _callback(indata, frames, time_info, status):
            if status:
                print(f"[audio] {status}")
            self._audio_q.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=int(self.SAMPLE_RATE * 0.25),
                callback=_callback,
            ):
                while not self._stop_event.is_set():
                    try:
                        block = self._audio_q.get(timeout=0.3)
                        buffer = np.concatenate([buffer, block.flatten()])

                        if len(buffer) >= chunk_samples:
                            audio_chunk = buffer[:chunk_samples].copy()
                            # Slide window forward by step_samples
                            buffer = buffer[step_samples:]

                            text = self._transcribe(audio_chunk)
                            if text and len(text.strip()) > 2:
                                self.new_text.emit(text.strip())

                    except queue.Empty:
                        continue
                    except Exception as exc:
                        print(f"[TranscriptionThread] {exc}")
        except Exception as exc:
            self.error_signal.emit(
                f"Audio device error: {exc}\n\n"
                "Make sure your microphone is connected and accessible.\n"
                "On macOS: System Preferences → Security & Privacy → Microphone."
            )

        self.status_update.emit("⏹ Stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────

STYLESHEET = """
/* ── Base ── */
QMainWindow, QWidget {
    background-color: #1a1b26;
    color: #c0caf5;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", "SF Pro Display", sans-serif;
    font-size: 13px;
}

/* ── Group boxes ── */
QGroupBox {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    margin-top: 12px;
    padding: 16px 10px 10px 10px;
    font-weight: 600;
    font-size: 14px;
    color: #7aa2f7;
    background-color: rgba(255,255,255,0.02);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
}

/* ── Text areas ── */
QTextEdit {
    background-color: #16161e;
    color: #c0caf5;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    font-size: 13px;
    padding: 8px;
    selection-background-color: #364a82;
    line-height: 1.5;
}
QTextEdit:focus { border-color: #7aa2f7; }

/* ── Buttons ── */
QPushButton {
    background-color: #24283b;
    color: #c0caf5;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2f3549;
    border-color: #7aa2f7;
}
QPushButton:pressed { background-color: #3b4261; }
QPushButton:disabled {
    color: #3b4261;
    border-color: rgba(255,255,255,0.04);
    background-color: #1a1b26;
}

QPushButton#startBtn {
    background-color: #9ece6a;
    color: #1a1b26;
    font-weight: bold;
    border: none;
    padding: 8px 20px;
    font-size: 13px;
}
QPushButton#startBtn:hover { background-color: #73daca; }
QPushButton#startBtn:disabled { background-color: #24283b; color: #3b4261; }

QPushButton#stopBtn {
    background-color: #f7768e;
    color: #1a1b26;
    font-weight: bold;
    border: none;
    padding: 8px 16px;
}
QPushButton#stopBtn:hover { background-color: #ff9e64; }
QPushButton#stopBtn:disabled { background-color: #24283b; color: #3b4261; }

/* ── Labels ── */
QLabel { color: #c0caf5; font-size: 13px; }
QLabel#titleLabel {
    color: #7aa2f7;
    font-size: 22px;
    font-weight: bold;
    letter-spacing: 0.5px;
}
QLabel#subLabel    { color: #565f89; font-size: 12px; }
QLabel#docLabel    { color: #9ece6a; font-size: 12px; font-weight: 500; }
QLabel#statusLabel { color: #9ece6a; font-size: 12px; font-weight: bold; }
QLabel#backendLabel {
    color: #e0af68;
    font-size: 11px;
    font-weight: bold;
    padding: 3px 10px;
    border-radius: 10px;
    background-color: rgba(224,175,104,0.12);
}
QLabel#chunkLabel {
    color: #565f89;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 8px;
    background-color: rgba(255,255,255,0.04);
}

/* ── Combo boxes ── */
QComboBox {
    background-color: #24283b;
    color: #c0caf5;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 12px;
}
QComboBox::drop-down { border: none; }
QComboBox:hover { border-color: #7aa2f7; }
QComboBox QAbstractItemView {
    background-color: #24283b;
    color: #c0caf5;
    selection-background-color: #3b4261;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px;
}

/* ── Splitter ── */
QSplitter::handle {
    background-color: rgba(255,255,255,0.04);
    width: 3px;
    height: 3px;
    border-radius: 1px;
}

/* ── Status bar ── */
QStatusBar {
    background-color: #16161e;
    color: #565f89;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 11px;
    padding: 2px 8px;
}

/* ── Scroll bars ── */
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    border-radius: 3px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: rgba(255,255,255,0.12);
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(255,255,255,0.20);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { height: 0; }

/* ── Tooltips ── */
QToolTip {
    background-color: #24283b;
    color: #c0caf5;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}
"""

# Highlight colour palette (cycled across matched keywords)
# Each tuple: (foreground, background) — bolder backgrounds for readability
_HIGHLIGHT_COLOURS = [
    ("#ff9e64", "rgba(255,158,100,0.18)"),  # orange
    ("#9ece6a", "rgba(158,206,106,0.18)"),  # green
    ("#7dcfff", "rgba(125,207,255,0.18)"),  # cyan
    ("#e0af68", "rgba(224,175,104,0.18)"),  # yellow
    ("#bb9af7", "rgba(187,154,247,0.18)"),  # purple
    ("#7aa2f7", "rgba(122,162,247,0.18)"),  # blue
    ("#f7768e", "rgba(247,118,142,0.18)"),  # pink
]


# ─────────────────────────────────────────────────────────────────────────────
# Document Reference Card widget
# ─────────────────────────────────────────────────────────────────────────────

class ReferenceCard(QFrame):
    """Displays one document chunk match with highlighted keywords."""

    # Score thresholds for colour grading
    _SCORE_COLOURS = [
        (0.55, "#9ece6a", "rgba(158,206,106,0.10)"),  # high → green
        (0.30, "#e0af68", "rgba(224,175,104,0.10)"),   # medium → amber
        (0.00, "#f7768e", "rgba(247,118,142,0.10)"),   # low → pink
    ]

    # File-extension → emoji mapping
    _TYPE_ICONS = {
        ".pdf": "📕", ".docx": "📘", ".doc": "📘",
        ".txt": "📄", ".md": "📝", ".csv": "📊",
    }

    def __init__(
        self,
        chunk: DocumentChunk,
        score: float,
        keywords: List[str],
        backend_name: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        pct = int(score * 100)

        # Determine colour for this score
        accent, accent_bg = "#565f89", "rgba(86,95,137,0.08)"
        for threshold, colour, bg in self._SCORE_COLOURS:
            if score >= threshold:
                accent, accent_bg = colour, bg
                break

        ext = Path(chunk.source_file).suffix.lower()
        icon = self._TYPE_ICONS.get(ext, "📄")

        # ── Card container styling ──
        self.setObjectName("refCard")
        self.setStyleSheet(f"""
            QFrame#refCard {{
                background-color: #1f2335;
                border: 1px solid rgba(255,255,255,0.06);
                border-left: 3px solid {accent};
                border-radius: 10px;
                margin: 2px 0px;
            }}
            QFrame#refCard:hover {{
                border-color: {accent};
                background-color: #24283b;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # ── Header row: source badge + score pill ──
        header = QHBoxLayout()
        header.setSpacing(8)

        # Source file badge
        src_lbl = QLabel(f"{icon}  {chunk.display_source}")
        src_lbl.setStyleSheet(
            "color: #7aa2f7; font-weight: 600; font-size: 12px; border: none;"
            "padding: 2px 0px;"
        )

        # Score pill
        score_lbl = QLabel(f"  {pct}%  ")
        score_lbl.setAlignment(Qt.AlignCenter)
        score_lbl.setStyleSheet(
            f"color: {accent}; font-size: 12px; font-weight: bold; border: none;"
            f"background-color: {accent_bg}; border-radius: 8px;"
            f"padding: 2px 10px; min-width: 36px;"
        )

        header.addWidget(src_lbl)
        header.addStretch()
        # Backend name label (small, muted)
        if backend_name:
            be_lbl = QLabel(backend_name)
            be_lbl.setStyleSheet(
                "color: #565f89; font-size: 10px; border: none; "
                "padding: 1px 6px; font-style: italic;"
            )
            header.addWidget(be_lbl)
        header.addWidget(score_lbl)
        layout.addLayout(header)

        # Score bar (thin visual indicator) — scale pct (0–100) to ~250 px max
        bar_container = QFrame()
        bar_container.setFixedHeight(4)
        bar_container.setStyleSheet(
            "background-color: rgba(255,255,255,0.04); border-radius: 2px; border: none;"
        )
        bar_fill = QFrame(bar_container)
        fill_width = max(4, int(pct * 2.5))  # 100% → 250 px, 0% → 4 px minimum
        bar_fill.setFixedSize(fill_width, 4)
        bar_fill.setStyleSheet(
            f"background-color: {accent}; border-radius: 2px; border: none;"
        )
        layout.addWidget(bar_container)

        # ── Text body ──
        body = QTextEdit()
        body.setReadOnly(True)
        body.setMaximumHeight(100)
        body.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255,255,255,0.02);
                border: none;
                border-radius: 6px;
                font-size: 12px;
                padding: 6px 8px;
                line-height: 1.6;
            }
        """)
        body.setHtml(self._highlight(chunk.text, keywords))
        layout.addWidget(body)

    @staticmethod
    def _highlight(text: str, keywords: List[str]) -> str:
        """HTML-escape text then wrap keyword matches in coloured spans."""
        html = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        for i, kw in enumerate(keywords[: len(_HIGHLIGHT_COLOURS)]):
            if not kw or len(kw) < 2:
                continue
            fg, bg = _HIGHLIGHT_COLOURS[i % len(_HIGHLIGHT_COLOURS)]
            html = re.sub(
                re.compile(re.escape(kw), re.IGNORECASE),
                (
                    f'<span style="color:{fg};font-weight:bold;'
                    f'background-color:{bg};border-radius:3px;'
                    f'padding:0 2px;">'
                    r"\g<0></span>"
                ),
                html,
            )
        return (
            f'<span style="color:#a9b1d6;font-size:12px;'
            f'line-height:1.6">{html}</span>'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class InterviewAssistant(QMainWindow):
    """
    Two-panel interview assistant:
      Left  — live rolling transcript of what is being said
      Right — top-ranked document sections relevant to recent speech

    Search quality upgrades automatically as backends become available:
      Startup  → TF-IDF + BM25 (immediate, keyword-level)
      ~5-15 s  → Semantic embeddings (meaning-based, much more accurate)
    """

    _BUFFER_MAX_WORDS = 120         # how many recent words are kept for searching
    _THREAD_STOP_TIMEOUT_MS = 4_000  # ms to wait for the transcription thread to finish

    def __init__(self) -> None:
        super().__init__()
        self._doc_manager = DocumentManager()
        self._loaded_files: List[str] = []
        self._transcript_words: List[str] = []
        self._transcription_thread: Optional[TranscriptionThread] = None
        self._model_loader: Optional[ModelLoaderThread] = None
        self._embed_indexer: Optional[EmbeddingIndexThread] = None
        self._embed_needed: bool = False  # docs loaded before model was ready

        self._build_ui()
        self.setStyleSheet(STYLESHEET)

        # Debounce timer — triggers a search 600 ms after the last new text
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._run_search)
        self._pending_query: str = ""

        # Begin loading the semantic embedding model in the background
        self._start_model_loading()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Interview Assistant — Live Reference Finder")
        self.setMinimumSize(1100, 720)
        self.resize(1340, 880)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 14, 16, 8)
        root_layout.setSpacing(10)

        # Title bar ──────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_lbl = QLabel("🎙  Interview Assistant")
        title_lbl.setObjectName("titleLabel")
        self._status_indicator = QLabel("⚪  Ready")
        self._status_indicator.setObjectName("statusLabel")
        self._status_indicator.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # Search backend pill (updates as model loads)
        self._backend_label = QLabel("⏳ Loading model…")
        self._backend_label.setObjectName("backendLabel")
        self._backend_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        title_row.addWidget(self._backend_label)
        title_row.addSpacing(10)
        title_row.addWidget(self._status_indicator)
        root_layout.addLayout(title_row)

        # Document toolbar ───────────────────────────────────────────────────
        root_layout.addLayout(self._build_doc_toolbar())

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: rgba(255,255,255,0.06); border: none;")
        root_layout.addWidget(sep)

        # Main split pane ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([420, 750])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, stretch=1)

        # Bottom controls ────────────────────────────────────────────────────
        root_layout.addLayout(self._build_controls())

        # Status bar ─────────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(
            "Load documents (PDF / DOCX / TXT) then click  ▶ Start Listening"
        )

    def _build_doc_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)

        load_btn = QPushButton("📂  Load Documents")
        load_btn.setToolTip(
            "Load resume, cover letter, portfolio, or any text document\n"
            "Supported: PDF, DOCX, TXT"
        )
        load_btn.clicked.connect(self._load_documents)

        clear_btn = QPushButton("🗑  Clear All")
        clear_btn.setToolTip("Remove all loaded documents")
        clear_btn.clicked.connect(self._clear_documents)

        self._doc_label = QLabel("No documents loaded")
        self._doc_label.setObjectName("subLabel")

        self._chunk_label = QLabel("")
        self._chunk_label.setObjectName("chunkLabel")

        bar.addWidget(load_btn)
        bar.addWidget(clear_btn)
        bar.addSpacing(12)
        bar.addWidget(self._doc_label)
        bar.addStretch()
        bar.addWidget(self._chunk_label)
        return bar

    def _build_left_panel(self) -> QGroupBox:
        grp = QGroupBox("Live Transcript")
        ly = QVBoxLayout(grp)
        ly.setContentsMargins(10, 16, 10, 10)
        ly.setSpacing(10)

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setFont(QFont("Segoe UI", 13))
        self._transcript.setPlaceholderText(
            "Transcribed speech appears here in real time.\n\n"
            "Click  ▶ Start Listening  to begin, or use the\n"
            "Manual Input box below to test without a microphone."
        )
        ly.addWidget(self._transcript, stretch=1)

        # Manual input sub-group
        manual_grp = QGroupBox("Manual Input  (test without microphone)")
        manual_grp.setStyleSheet(
            "QGroupBox { font-size: 11px; color: #565f89; "
            "border-color: rgba(255,255,255,0.06); margin-top: 8px; padding-top: 8px; }"
        )
        ml = QHBoxLayout(manual_grp)
        ml.setContentsMargins(8, 10, 8, 8)
        ml.setSpacing(8)
        self._manual_input = QTextEdit()
        self._manual_input.setMaximumHeight(56)
        self._manual_input.setPlaceholderText(
            "Type text here and press Search to find matching document sections…"
        )
        search_btn = QPushButton("🔍 Search")
        search_btn.setMinimumWidth(90)
        search_btn.clicked.connect(self._manual_search)
        ml.addWidget(self._manual_input)
        ml.addWidget(search_btn)
        ly.addWidget(manual_grp)

        return grp

    def _build_right_panel(self) -> QGroupBox:
        grp = QGroupBox("Document References")
        ly = QVBoxLayout(grp)
        ly.setContentsMargins(10, 16, 10, 10)
        ly.setSpacing(8)

        # Info row
        info_row = QHBoxLayout()
        info_row.setSpacing(6)
        self._query_label = QLabel("Matching: —")
        self._query_label.setObjectName("subLabel")
        self._query_label.setStyleSheet(
            "color: #565f89; font-size: 12px; font-style: italic;"
        )
        self._count_label = QLabel("")
        self._count_label.setObjectName("subLabel")
        info_row.addWidget(self._query_label)
        info_row.addStretch()
        info_row.addWidget(self._count_label)
        ly.addLayout(info_row)

        # Scrollable card area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._results_widget = QWidget()
        self._results_widget.setStyleSheet("background: transparent;")
        self._results_layout = QVBoxLayout(self._results_widget)
        self._results_layout.setContentsMargins(0, 0, 4, 0)
        self._results_layout.setSpacing(8)
        self._results_layout.addStretch()
        scroll.setWidget(self._results_widget)
        ly.addWidget(scroll, stretch=1)

        # Placeholder label (shown when no results)
        self._placeholder = QLabel(
            "📂  Load documents and start speaking\n"
            "to see live references here"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #3b4261; font-size: 15px; font-style: italic;"
            "padding: 40px; line-height: 1.8;"
        )
        ly.addWidget(self._placeholder)

        return grp

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)

        bar.addWidget(QLabel("Whisper model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["tiny", "base", "small", "medium"])
        self._model_combo.setCurrentText("base")
        self._model_combo.setToolTip(
            "tiny   — fastest, lowest accuracy\n"
            "base   — good balance  (recommended)\n"
            "small  — more accurate, slower\n"
            "medium — best accuracy, needs more RAM"
        )
        self._model_combo.setFixedWidth(90)
        bar.addWidget(self._model_combo)

        bar.addSpacing(12)
        bar.addWidget(QLabel("Max results:"))
        self._results_combo = QComboBox()
        self._results_combo.addItems(["3", "5", "7", "10"])
        self._results_combo.setCurrentText("5")
        self._results_combo.setFixedWidth(56)
        bar.addWidget(self._results_combo)

        bar.addStretch()

        clear_t_btn = QPushButton("Clear Transcript")
        clear_t_btn.clicked.connect(self._clear_transcript)
        bar.addWidget(clear_t_btn)

        self._start_btn = QPushButton("▶  Start Listening")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.setMinimumWidth(148)
        self._start_btn.clicked.connect(self._start_listening)
        bar.addWidget(self._start_btn)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setMinimumWidth(90)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_listening)
        bar.addWidget(self._stop_btn)

        return bar

    # ── Semantic model loading ────────────────────────────────────────────────

    def _start_model_loading(self) -> None:
        self._model_loader = ModelLoaderThread(parent=self)
        self._model_loader.model_ready.connect(self._on_model_ready)
        self._model_loader.status_update.connect(self._on_status)
        self._model_loader.start()

    def _on_model_ready(self, model, device: str) -> None:
        self._doc_manager.set_embed_model(model)
        if model is not None:
            self._backend_label.setText(f"🧠 Semantic  [{device}]")
            self._backend_label.setStyleSheet(
                "color: #9ece6a; font-size: 11px; font-weight: bold;"
                "padding: 3px 10px; border-radius: 10px;"
                "background-color: rgba(158,206,106,0.12);"
            )
            # If documents were loaded before the model was ready, index them now
            if self._embed_needed and self._doc_manager.chunks:
                self._start_embedding_index()
        else:
            self._update_backend_label()
        self._embed_needed = False

    def _start_embedding_index(self) -> None:
        if not self._doc_manager.chunks or self._doc_manager.embed_model is None:
            return
        # Cancel any running indexer
        if self._embed_indexer and self._embed_indexer.isRunning():
            self._embed_indexer.quit()
            self._embed_indexer.wait(2_000)
        self._embed_indexer = EmbeddingIndexThread(
            self._doc_manager.embed_model,
            list(self._doc_manager.chunks),
            parent=self,
        )
        self._embed_indexer.embeddings_ready.connect(self._on_embeddings_ready)
        self._embed_indexer.status_update.connect(self._on_status)
        self._embed_indexer.start()
        self._backend_label.setText("⏳ Indexing…")
        self._backend_label.setStyleSheet(
            "color: #e0af68; font-size: 11px; font-weight: bold;"
            "padding: 3px 10px; border-radius: 10px;"
            "background-color: rgba(224,175,104,0.12);"
        )

    def _on_embeddings_ready(self, embeddings) -> None:
        self._doc_manager.set_embeddings(embeddings)
        n = len(self._doc_manager.chunks)
        if self._doc_manager.backend_name == "semantic":
            # Embeddings computed successfully — show chunk count and device
            device = _best_device()
            self._backend_label.setText(f"🧠 Semantic  [{device}]  {n} chunks")
            self._backend_label.setStyleSheet(
                "color: #9ece6a; font-size: 11px; font-weight: bold;"
                "padding: 3px 10px; border-radius: 10px;"
                "background-color: rgba(158,206,106,0.12);"
            )
            self._status_bar.showMessage(
                f"Semantic index ready — {n} chunks indexed for meaning-based search  ✔"
            )
        else:
            # Embedding failed — show the fallback backend
            self._update_backend_label()
            self._status_bar.showMessage(
                f"Embedding failed — using {self._doc_manager.backend_name} search"
            )

    def _update_backend_label(self) -> None:
        name = self._doc_manager.backend_name
        if "semantic" in name:
            icon, colour, bg = "🧠", "#9ece6a", "rgba(158,206,106,0.12)"
        elif "hybrid" in name:
            icon, colour, bg = "📊", "#e0af68", "rgba(224,175,104,0.12)"
        elif "tfidf" in name:
            icon, colour, bg = "📈", "#bb9af7", "rgba(187,154,247,0.12)"
        else:
            icon, colour, bg = "⌨️", "#565f89", "rgba(86,95,137,0.12)"
        self._backend_label.setText(f"{icon} {name.capitalize()}")
        self._backend_label.setStyleSheet(
            f"color: {colour}; font-size: 11px; font-weight: bold;"
            f"padding: 3px 10px; border-radius: 10px;"
            f"background-color: {bg};"
        )

    # ── Document management ──────────────────────────────────────────────────

    def _load_documents(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Candidate Documents",
            "",
            "Documents (*.pdf *.docx *.txt);;"
            "PDF (*.pdf);;Word (*.docx);;Text (*.txt);;All Files (*)",
        )
        if not files:
            return
        for path in files:
            n = self._doc_manager.load_file(path)
            if n:
                self._loaded_files.append(path)
                self._status_bar.showMessage(
                    f"Loaded '{Path(path).name}'  ({n} chunks)"
                )
        self._refresh_doc_label()
        self._update_backend_label()

        # Trigger semantic indexing if the model is already loaded
        if self._doc_manager.embed_model is not None:
            self._start_embedding_index()
        else:
            self._embed_needed = True
            self._status_bar.showMessage(
                "Documents loaded — waiting for semantic model to finish…"
            )

    def _clear_documents(self) -> None:
        self._doc_manager.clear()
        self._loaded_files.clear()
        self._refresh_doc_label()
        self._clear_results()
        self._status_bar.showMessage("All documents cleared.")
        self._update_backend_label()

    def _refresh_doc_label(self) -> None:
        if not self._loaded_files:
            self._doc_label.setText("No documents loaded")
            self._doc_label.setObjectName("subLabel")
            self._chunk_label.setText("")
        else:
            names = [Path(f).name for f in self._loaded_files]
            display = ", ".join(names[:3])
            if len(names) > 3:
                display += f"  +{len(names) - 3} more"
            self._doc_label.setText(f"📑  {display}")
            self._doc_label.setObjectName("docLabel")
            total = len(self._doc_manager.chunks)
            self._chunk_label.setText(f"{total} indexed chunks")
        self._doc_label.style().unpolish(self._doc_label)
        self._doc_label.style().polish(self._doc_label)

    # ── Listening ────────────────────────────────────────────────────────────

    def _start_listening(self) -> None:
        model = self._model_combo.currentText()
        self._transcription_thread = TranscriptionThread(
            model_size=model, parent=self
        )
        self._transcription_thread.new_text.connect(self._on_transcription)
        self._transcription_thread.status_update.connect(self._on_status)
        self._transcription_thread.error_signal.connect(self._on_error)
        self._transcription_thread.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_indicator.setText("🔴  Listening")
        self._status_indicator.setStyleSheet(
            "color: #f7768e; font-weight: bold; font-size: 12px;"
        )

    def _stop_listening(self) -> None:
        if self._transcription_thread:
            self._transcription_thread.stop()
            self._transcription_thread.wait(self._THREAD_STOP_TIMEOUT_MS)
            self._transcription_thread = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_indicator.setText("⚪  Stopped")
        self._status_indicator.setStyleSheet(
            "color: #565f89; font-size: 12px;"
        )

    # ── Transcription signal handlers ────────────────────────────────────────

    def _on_transcription(self, text: str) -> None:
        if not text.strip():
            return

        # Append to rolling transcript display
        self._transcript.moveCursor(QTextCursor.End)
        self._transcript.insertHtml(
            f'<span style="color:#c0caf5">{text} </span>'
        )
        self._transcript.moveCursor(QTextCursor.End)

        # Maintain rolling word buffer for searching
        self._transcript_words.extend(text.split())
        if len(self._transcript_words) > self._BUFFER_MAX_WORDS:
            self._transcript_words = self._transcript_words[-self._BUFFER_MAX_WORDS:]

        # Schedule a debounced search
        self._pending_query = " ".join(self._transcript_words)
        self._search_timer.start(600)

    def _on_status(self, msg: str) -> None:
        self._status_bar.showMessage(msg)

    def _on_error(self, msg: str) -> None:
        self._stop_listening()
        QMessageBox.critical(self, "Transcription Error", msg)

    # ── Manual search ────────────────────────────────────────────────────────

    def _manual_search(self) -> None:
        query = self._manual_input.toPlainText().strip()
        if not query:
            return
        self._transcript.moveCursor(QTextCursor.End)
        self._transcript.insertHtml(
            f'<br><span style="color:#7aa2f7;font-style:italic">'
            f"[Manual: {query}]</span><br>"
        )
        self._pending_query = query
        self._run_search()

    # ── Search & results ─────────────────────────────────────────────────────

    def _run_search(self) -> None:
        query = self._pending_query
        if not query or not self._doc_manager.chunks:
            return

        top_k = int(self._results_combo.currentText())
        results = self._doc_manager.search(query, top_k=top_k)
        keywords = self._extract_keywords(query)

        kw_preview = " · ".join(keywords[:6]) if keywords else "—"
        self._query_label.setText(f'Matching:  "{kw_preview}"')
        self._update_results(results, keywords)

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract significant content words from the query for highlighting."""
        _stop = set(nltk_stopwords.words("english"))
        clean = text.lower().translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(clean)
        # Keep any non-stopword word longer than 2 characters
        kws = [w for w in words if w.isalpha() and len(w) > 2 and w not in _stop]
        seen: set = set()
        unique: List[str] = []
        for w in kws:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:10]

    def _update_results(
        self,
        results: List[Tuple[DocumentChunk, float]],
        keywords: List[str],
    ) -> None:
        self._clear_results()
        if not results:
            self._placeholder.show()
            self._count_label.setText("No matches")
            return

        self._placeholder.hide()
        backend = self._doc_manager.backend_name
        self._count_label.setText(
            f"{len(results)} reference(s) found  ·  {backend}"
        )
        self._count_label.setStyleSheet(
            "color: #7aa2f7; font-size: 11px; font-weight: 500;"
        )

        for chunk, score in results:
            card = ReferenceCard(
                chunk, score, keywords,
                backend_name=backend,
            )
            self._results_layout.insertWidget(
                self._results_layout.count() - 1, card
            )

    def _clear_results(self) -> None:
        while self._results_layout.count() > 1:
            item = self._results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._count_label.setText("")

    def _clear_transcript(self) -> None:
        self._transcript.clear()
        self._transcript_words.clear()
        self._clear_results()

    # ── Window close ────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._stop_listening()
        if self._embed_indexer and self._embed_indexer.isRunning():
            self._embed_indexer.quit()
            self._embed_indexer.wait(2_000)
        if self._model_loader and self._model_loader.isRunning():
            self._model_loader.quit()
            self._model_loader.wait(2_000)
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Interview Assistant")
    app.setApplicationVersion("1.0.0")

    window = InterviewAssistant()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

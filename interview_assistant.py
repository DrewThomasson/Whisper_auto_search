#!/usr/bin/env python3
"""
Interview Assistant — Live Speech-to-Document Reference Finder
==============================================================
As an interviewee speaks, this tool automatically surfaces relevant sections
from their resume and supplemental documents in real time, so the interviewer
always has context without needing to pre-read everything.

Usage:
    python interview_assistant.py

Requirements:
    pip install -r requirements.txt
"""

import os
import sys
import re
import time
import queue
import string
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

# ── Qt ──────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QFrame, QScrollArea,
    QSplitter, QStatusBar, QGroupBox, QComboBox, QSizePolicy,
    QMessageBox, QSpacerItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QTextCursor

# ── NLP ──────────────────────────────────────────────────────────────────────
import nltk

for _ds in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng"):
    try:
        nltk.download(_ds, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# ── Optional scikit-learn ─────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

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
    start_sent: int = 0

    @property
    def display_source(self) -> str:
        name = Path(self.source_file).name
        if self.page_num is not None:
            return f"{name} · p.{self.page_num}"
        return name


# ─────────────────────────────────────────────────────────────────────────────
# Document Manager
# ─────────────────────────────────────────────────────────────────────────────

class DocumentManager:
    """Loads documents (PDF / DOCX / TXT) and runs fast TF-IDF searches."""

    CHUNK_SENTENCES = 5   # sentences per chunk
    OVERLAP_SENTENCES = 2  # overlap between adjacent chunks

    def __init__(self) -> None:
        self.chunks: List[DocumentChunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._stop_words = set(nltk_stopwords.words("english"))

    # ── Public API ──────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.chunks.clear()
        self._vectorizer = None
        self._tfidf_matrix = None

    def load_file(self, path: str) -> int:
        """Load a file; return the number of new chunks added."""
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
            self._rebuild_index()
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.04,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return (chunk, score) pairs ranked by relevance."""
        if not self.chunks or not query.strip():
            return []
        if _SKLEARN_AVAILABLE and self._vectorizer is not None:
            return self._search_tfidf(query, top_k, min_score)
        return self._search_keyword(query, top_k, min_score)

    # ── Loading helpers ──────────────────────────────────────────────────────

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
            chunk_sents = sentences[i : i + self.CHUNK_SENTENCES]
            chunk_text = " ".join(chunk_sents).strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        source_file=source,
                        text=chunk_text,
                        chunk_index=len(self.chunks) + len(chunks),
                        page_num=page_num,
                        start_sent=i,
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
        except ImportError:
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
        except ImportError:
            pass
        # Last-resort: treat as plain text
        return self._load_txt(path)

    def _load_docx(self, path: str) -> List[DocumentChunk]:
        try:
            from docx import Document  # type: ignore
            doc = Document(path)
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return self._chunk_text(full_text, path)
        except ImportError:
            return self._load_txt(path)

    # ── Indexing & search ────────────────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        return " ".join(
            w for w in tokens if w.isalpha() and w not in self._stop_words
        )

    def _rebuild_index(self) -> None:
        if not self.chunks or not _SKLEARN_AVAILABLE:
            return
        try:
            processed = [self._preprocess(c.text) for c in self.chunks]
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8_000,
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(processed)
        except Exception as exc:
            print(f"[DocumentManager] Index build error: {exc}")

    def _search_tfidf(
        self, query: str, top_k: int, min_score: float
    ) -> List[Tuple[DocumentChunk, float]]:
        pq = self._preprocess(query)
        if not pq.strip():
            return []
        q_vec = self._vectorizer.transform([pq])
        scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.chunks[i], float(scores[i]))
            for i in indices
            if scores[i] >= min_score
        ]

    def _search_keyword(
        self, query: str, top_k: int, min_score: float
    ) -> List[Tuple[DocumentChunk, float]]:
        qwords = set(self._preprocess(query).split())
        if not qwords:
            return []
        results: List[Tuple[DocumentChunk, float]] = []
        for chunk in self.chunks:
            cwords = set(self._preprocess(chunk.text).split())
            if not cwords:
                continue
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
        # 1. faster-whisper
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self.status_update.emit(
                f"Loading faster-whisper ({self.model_size}) — first run may download model…"
            )
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
            self._backend = "faster_whisper"
            self.status_update.emit("faster-whisper ready  ✔")
            return True
        except ImportError:
            pass
        except Exception as exc:
            print(f"[faster-whisper load error] {exc}")

        # 2. openai-whisper
        try:
            import whisper  # type: ignore
            self.status_update.emit(
                f"Loading openai-whisper ({self.model_size})…"
            )
            self._model = whisper.load_model(self.model_size)
            self._backend = "openai_whisper"
            self.status_update.emit("openai-whisper ready  ✔")
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
                "Install with:  pip install sounddevice"
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

        self.status_update.emit("⏹ Stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────

STYLESHEET = """
/* ── Base ── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
}

/* ── Group boxes ── */
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    font-size: 13px;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* ── Text areas ── */
QTextEdit {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 6px;
    font-size: 13px;
    padding: 6px;
    selection-background-color: #585b70;
}
QTextEdit:focus { border-color: #89b4fa; }

/* ── Buttons ── */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
}
QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { color: #45475a; border-color: #313244; background-color: #1e1e2e; }

QPushButton#startBtn {
    background-color: #a6e3a1;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton#startBtn:hover { background-color: #94e2d5; }
QPushButton#startBtn:disabled { background-color: #313244; color: #45475a; }

QPushButton#stopBtn {
    background-color: #f38ba8;
    color: #1e1e2e;
    font-weight: bold;
    border: none;
}
QPushButton#stopBtn:hover { background-color: #eba0ac; }
QPushButton#stopBtn:disabled { background-color: #313244; color: #45475a; }

/* ── Labels ── */
QLabel { color: #cdd6f4; font-size: 13px; }
QLabel#titleLabel { color: #89b4fa; font-size: 20px; font-weight: bold; }
QLabel#subLabel    { color: #6c7086; font-size: 12px; }
QLabel#docLabel    { color: #a6e3a1; font-size: 12px; }
QLabel#statusLabel { color: #a6e3a1; font-size: 12px; font-weight: bold; }

/* ── Combo boxes ── */
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
}
QComboBox::drop-down { border: none; }
QComboBox:hover { border-color: #89b4fa; }
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #585b70;
    border: 1px solid #45475a;
}

/* ── Splitter ── */
QSplitter::handle {
    background-color: #313244;
    width: 4px;
    height: 4px;
}

/* ── Status bar ── */
QStatusBar {
    background-color: #181825;
    color: #6c7086;
    border-top: 1px solid #313244;
    font-size: 11px;
}

/* ── Scroll bars ── */
QScrollBar:vertical {
    background: #1e1e2e;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { height: 0; }
"""

# Highlight colour palette (cycled across matched keywords)
_HIGHLIGHT_COLOURS = [
    ("#fab387", "rgba(250,179,135,.15)"),  # peach
    ("#a6e3a1", "rgba(166,227,161,.15)"),  # green
    ("#89dceb", "rgba(137,220,235,.15)"),  # sky
    ("#f9e2af", "rgba(249,226,175,.15)"),  # yellow
    ("#cba6f7", "rgba(203,166,247,.15)"),  # mauve
    ("#89b4fa", "rgba(137,180,250,.15)"),  # blue
    ("#f38ba8", "rgba(243,139,168,.15)"),  # red
]


# ─────────────────────────────────────────────────────────────────────────────
# Document Reference Card widget
# ─────────────────────────────────────────────────────────────────────────────

class ReferenceCard(QFrame):
    """Displays one document chunk match with highlighted keywords."""

    def __init__(
        self,
        chunk: DocumentChunk,
        score: float,
        keywords: List[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("refCard")
        self.setStyleSheet("""
            QFrame#refCard {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 10px;
                margin: 3px 2px;
            }
            QFrame#refCard:hover { border-color: #89b4fa; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        src_lbl = QLabel(f"📄  {chunk.display_source}")
        src_lbl.setStyleSheet(
            "color: #89b4fa; font-weight: bold; font-size: 12px; border: none;"
        )
        pct = int(score * 100)
        if pct > 55:
            score_colour = "#a6e3a1"
        elif pct > 30:
            score_colour = "#f9e2af"
        else:
            score_colour = "#f38ba8"
        score_lbl = QLabel(f"{pct}% match")
        score_lbl.setStyleSheet(
            f"color: {score_colour}; font-size: 11px; border: none;"
        )
        header.addWidget(src_lbl)
        header.addStretch()
        header.addWidget(score_lbl)
        layout.addLayout(header)

        # Text body
        body = QTextEdit()
        body.setReadOnly(True)
        body.setMaximumHeight(110)
        body.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e2e;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                padding: 4px;
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
                    f'background-color:{bg};">'
                    r"\g<0></span>"
                ),
                html,
            )
        return f'<span style="color:#cdd6f4;font-size:12px;line-height:1.5">{html}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class InterviewAssistant(QMainWindow):
    """
    Two-panel interview assistant:
      Left  — live rolling transcript of what is being said
      Right — top-ranked document sections relevant to recent speech
    """

    _BUFFER_MAX_WORDS = 120       # how many recent words are kept for searching
    _THREAD_STOP_TIMEOUT_MS = 4_000  # ms to wait for the transcription thread to finish

    def __init__(self) -> None:
        super().__init__()
        self._doc_manager = DocumentManager()
        self._loaded_files: List[str] = []
        self._transcript_words: List[str] = []
        self._transcription_thread: Optional[TranscriptionThread] = None

        self._build_ui()
        self.setStyleSheet(STYLESHEET)

        # Debounce timer — triggers a search 600 ms after the last new text
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._run_search)
        self._pending_query: str = ""

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Interview Assistant — Live Reference Finder")
        self.setMinimumSize(1100, 720)
        self.resize(1300, 840)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 6)
        root_layout.setSpacing(8)

        # Title bar ──────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_lbl = QLabel("🎙  Interview Assistant")
        title_lbl.setObjectName("titleLabel")
        self._status_indicator = QLabel("⚪  Ready")
        self._status_indicator.setObjectName("statusLabel")
        self._status_indicator.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        title_row.addWidget(self._status_indicator)
        root_layout.addLayout(title_row)

        # Document toolbar ───────────────────────────────────────────────────
        root_layout.addLayout(self._build_doc_toolbar())

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #313244;")
        root_layout.addWidget(sep)

        # Main split pane ────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([400, 700])
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
        bar.setSpacing(8)

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
        self._chunk_label.setObjectName("subLabel")

        bar.addWidget(load_btn)
        bar.addWidget(clear_btn)
        bar.addSpacing(10)
        bar.addWidget(self._doc_label)
        bar.addStretch()
        bar.addWidget(self._chunk_label)
        return bar

    def _build_left_panel(self) -> QGroupBox:
        grp = QGroupBox("Live Transcript")
        ly = QVBoxLayout(grp)
        ly.setContentsMargins(8, 14, 8, 8)
        ly.setSpacing(8)

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
            "QGroupBox { font-size: 11px; color: #6c7086; "
            "border-color: #313244; margin-top: 6px; padding-top: 6px; }"
        )
        ml = QHBoxLayout(manual_grp)
        ml.setContentsMargins(6, 8, 6, 6)
        self._manual_input = QTextEdit()
        self._manual_input.setMaximumHeight(56)
        self._manual_input.setPlaceholderText(
            "Type text here and press Search to find matching document sections…"
        )
        search_btn = QPushButton("Search")
        search_btn.setFixedWidth(72)
        search_btn.clicked.connect(self._manual_search)
        ml.addWidget(self._manual_input)
        ml.addWidget(search_btn)
        ly.addWidget(manual_grp)

        return grp

    def _build_right_panel(self) -> QGroupBox:
        grp = QGroupBox("Document References")
        ly = QVBoxLayout(grp)
        ly.setContentsMargins(8, 14, 8, 8)
        ly.setSpacing(6)

        # Info row
        info_row = QHBoxLayout()
        self._query_label = QLabel("Matching: —")
        self._query_label.setObjectName("subLabel")
        self._count_label = QLabel("")
        self._count_label.setObjectName("subLabel")
        info_row.addWidget(self._query_label)
        info_row.addStretch()
        info_row.addWidget(self._count_label)
        ly.addLayout(info_row)

        # Scrollable card area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        self._results_widget = QWidget()
        self._results_layout = QVBoxLayout(self._results_widget)
        self._results_layout.setContentsMargins(0, 0, 0, 0)
        self._results_layout.setSpacing(6)
        self._results_layout.addStretch()
        scroll.setWidget(self._results_widget)
        ly.addWidget(scroll, stretch=1)

        # Placeholder label (shown when no results)
        self._placeholder = QLabel(
            "Load documents and start speaking\nto see live references here"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #45475a; font-size: 15px; font-style: italic;"
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
                    f"Loaded '{Path(path).name}'  ({n} indexed chunks)"
                )
        self._refresh_doc_label()

    def _clear_documents(self) -> None:
        self._doc_manager.clear()
        self._loaded_files.clear()
        self._refresh_doc_label()
        self._clear_results()
        self._status_bar.showMessage("All documents cleared.")

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
        # Force style refresh
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
            "color: #f38ba8; font-weight: bold; font-size: 12px;"
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
            "color: #6c7086; font-size: 12px;"
        )

    # ── Transcription signal handlers ────────────────────────────────────────

    def _on_transcription(self, text: str) -> None:
        if not text.strip():
            return

        # Append to rolling transcript display
        self._transcript.moveCursor(QTextCursor.End)
        self._transcript.insertHtml(
            f'<span style="color:#cdd6f4">{text} </span>'
        )
        self._transcript.moveCursor(QTextCursor.End)

        # Maintain rolling word buffer for searching
        self._transcript_words.extend(text.split())
        if len(self._transcript_words) > self._BUFFER_MAX_WORDS:
            self._transcript_words = self._transcript_words[
                -self._BUFFER_MAX_WORDS :
            ]

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
            f'<br><span style="color:#89b4fa;font-style:italic">'
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
        """Extract the most meaningful nouns / adjectives from a string."""
        _stop = set(nltk_stopwords.words("english"))
        clean = text.lower().translate(str.maketrans("", "", string.punctuation))
        try:
            tokens = word_tokenize(clean)
            tagged = pos_tag(tokens)
            kws = [
                w
                for w, t in tagged
                if t in ("NN", "NNS", "NNP", "NNPS", "JJ", "VBG")
                and w not in _stop
                and len(w) > 2
            ]
        except Exception:
            kws = [w for w in clean.split() if w not in _stop and len(w) > 2]
        # Deduplicate, preserve order
        seen: set = set()
        unique: List[str] = []
        for kw in kws:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
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
        self._count_label.setText(f"{len(results)} reference(s) found")

        # Insert cards before the trailing stretch
        for chunk, score in results:
            card = ReferenceCard(chunk, score, keywords)
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

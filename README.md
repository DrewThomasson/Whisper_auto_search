# Whisper Auto Search

Live speech-to-document reference matching — ideal for interviewers who want instant context from a candidate's resume and documents as the conversation unfolds.

---

## 🌐 Web Application  *(recommended — new)*

A fully browser-based redesign of the Interview Assistant.  
**Whisper transcription runs directly in your browser via WebGPU** (or WASM fallback) — no GPU required on the server.  
The Python backend handles only document indexing and semantic search.

![Web Application screenshot](https://github.com/user-attachments/assets/c74b5e36-af79-4eb5-bad1-2906383fcdda)

### Features

| Feature | Detail |
|---|---|
| **In-browser Whisper** | Speech-to-text runs via [Transformers.js](https://huggingface.co/docs/transformers.js) + **WebGPU** (Chrome 113+, Edge 113+) with automatic WASM fallback. No API key or internet required after model download. |
| **Server-side semantic search** | `all-MiniLM-L6-v2` on the backend — 4-tier search: semantic → hybrid BM25+TF-IDF → TF-IDF → keyword. |
| **Multi-document support** | Upload PDF, DOCX, TXT, MD, CSV files via button or drag-and-drop. |
| **Keyword highlighting** | Matched terms highlighted in the result cards. |
| **Relevance scores** | Each card shows a % match with a colour-graded bar. |
| **Debounced search** | Results update 600 ms after new speech — not on every word. |
| **Manual input** | Text box for testing without a microphone. |
| **Dockerised** | Single command to run anywhere. |

### Quick start (Docker — recommended)

```bash
docker compose up --build
```

Then open **http://localhost:8000** in Chrome or Edge (WebGPU required for hardware-accelerated transcription; other browsers fall back to WASM automatically).

### Quick start (local Python)

```bash
cd webapp
pip install -r requirements.txt
uvicorn app:app --reload
```

Open **http://localhost:8000**.

### How it works

```
Browser                                 Server (Python / FastAPI)
──────────────────────────────────────  ─────────────────────────────────────
Microphone → AudioWorklet (PCM)         POST /api/upload  → DocumentManager
         → Transformers.js Whisper      POST /api/search  → TF-IDF / BM25 /
              (WebGPU or WASM)                              sentence-transformers
         → transcribed text ──────────► (JSON results)
         ◄─────────────────────────────
Display reference cards with highlights
```

### Whisper model guide

| Model | Download | Speed | Accuracy | Browser RAM |
|---|---|---|---|---|
| tiny  | ~40 MB | fastest | good   | ~200 MB |
| base  | ~75 MB | fast    | better | ~350 MB |
| small | ~240 MB | moderate | best  | ~900 MB |

Models are downloaded from HuggingFace on first use and cached in the browser (IndexedDB).

### Browser requirements

| Browser | WebGPU | WASM fallback |
|---|---|---|
| Chrome 113+ / Edge 113+ | ✅ Full WebGPU | ✅ |
| Firefox | ⚠ Flag required | ✅ |
| Safari 18+ (macOS 15+) | ✅ | ✅ |

---

## 🖥 Desktop App  *(original)*

A polished PyQt5 desktop application.  Uses `faster-whisper` for local transcription.

```bash
pip install -r requirements.txt
python interview_assistant.py
```

1. Click **📂 Load Documents** and select the candidate's resume, cover letter, or any other files (PDF / DOCX / TXT).
2. Click **▶ Start Listening**.
3. As the candidate speaks, relevant document sections appear automatically on the right panel.

---

## Run_auto_search_gui.py  *(original prototype)*

The original prototype. Listens via the `whisper.cpp` `./stream` binary and searches a TXT file for relevant noun-matched sections.

To run: set up [whisper.cpp](https://github.com/ggerganov/whisper.cpp) according to its README, then:

```bash
python Run_auto_search_gui.py
```

The test document is set to *Alice in Wonderland* to avoid copyright issues.

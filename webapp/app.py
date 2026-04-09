"""
app.py — FastAPI backend for the Whisper Auto-Search web application.

Endpoints:
  GET  /                      Serve the web UI (index.html)
  POST /api/upload            Upload one or more documents (PDF/DOCX/TXT)
  DELETE /api/documents       Clear all loaded documents
  POST /api/search            Search loaded documents with a query
  GET  /api/status            Return backend & document status
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from document_manager import DocumentManager

# ─────────────────────────────────────────────────────────────────────────────
# Application setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Whisper Auto-Search", version="2.0.0")

# Allow all origins so that the browser's fetch() works even when the front-end
# is opened directly (e.g. via file://).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared document manager (one search index for all clients).
doc_manager = DocumentManager()

# Temporary directory that stores uploaded files for the lifetime of the process.
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="whisper_upload_"))

# ─────────────────────────────────────────────────────────────────────────────
# Static file serving
# ─────────────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ─────────────────────────────────────────────────────────────────────────────
# API models
# ─────────────────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────────────────────────────────────

# Allowed document extensions
_ALLOWED = {".pdf", ".docx", ".txt", ".md", ".csv"}


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)) -> JSONResponse:
    """Accept one or more document files and add them to the search index."""
    results = []
    for upload in files:
        original_name = Path(upload.filename or "document").name
        ext = Path(original_name).suffix.lower()
        if ext not in _ALLOWED:
            results.append(
                {
                    "filename": original_name,
                    "status": "error",
                    "message": f"Unsupported file type '{ext}'. "
                    f"Supported: {', '.join(sorted(_ALLOWED))}",
                    "chunks": 0,
                }
            )
            continue

        # Save to temp dir with a unique name to avoid collisions.
        safe_stem = uuid.uuid4().hex
        dest = UPLOAD_DIR / f"{safe_stem}{ext}"
        try:
            content = await upload.read()
            dest.write_bytes(content)
            n = doc_manager.load_file(str(dest), display_name=original_name)
            results.append(
                {
                    "filename": original_name,
                    "status": "ok",
                    "message": f"Loaded {n} chunks",
                    "chunks": n,
                }
            )
        except Exception as exc:
            # Log the full exception server-side; return only a safe summary to the client.
            print(f"[upload] Error processing '{original_name}': {exc}")
            results.append(
                {
                    "filename": original_name,
                    "status": "error",
                    "message": "Failed to process file. Check server logs for details.",
                    "chunks": 0,
                }
            )

    return JSONResponse(
        content={
            "files": results,
            "total_chunks": doc_manager.status["chunk_count"],
            "backend": doc_manager.status["backend"],
        }
    )


@app.delete("/api/documents")
async def clear_documents() -> JSONResponse:
    """Remove all loaded documents and reset the search index."""
    doc_manager.clear()
    # Clean up uploaded files as well.
    for f in UPLOAD_DIR.iterdir():
        try:
            f.unlink()
        except OSError:
            pass
    return JSONResponse(content={"status": "ok", "message": "Documents cleared"})


@app.post("/api/search")
async def search(body: SearchRequest) -> JSONResponse:
    """Search loaded documents for the most relevant chunks."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    kwargs: dict = {"top_k": body.top_k}
    if body.min_score is not None:
        kwargs["min_score"] = body.min_score

    hits = doc_manager.search(body.query, **kwargs)

    return JSONResponse(
        content={
            "query": body.query,
            "backend": doc_manager.status["backend"],
            "results": [
                {
                    "chunk": chunk.to_dict(),
                    "score": round(score, 4),
                    "score_pct": min(100, round(score * 100)),
                }
                for chunk, score in hits
            ],
        }
    )


@app.get("/api/status")
async def status() -> JSONResponse:
    """Return the current state of the document index and search backend."""
    return JSONResponse(content=doc_manager.status)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (development / Docker)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )

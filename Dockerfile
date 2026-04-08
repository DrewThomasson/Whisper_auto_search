# ── Stage 1: build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/install \
        -r /tmp/requirements.txt

# ── Stage 2: runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim

# System libraries needed at runtime:
#   - libportaudio2        → sounddevice (audio capture)
#   - libgl1, libglib2.0-0 → OpenCV / general native libs
#   - Qt5 libs + xcb       → PyQt5 GUI rendering over X11
#   - pulseaudio-utils     → PulseAudio client for mic passthrough
#   - libsndfile1          → soundfile (audio I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libfontconfig1 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxcb-cursor0 \
        libegl1 \
        pulseaudio-utils \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Pre-download NLTK data so the first run is fast
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# Set the working directory
WORKDIR /app

# Copy application source
COPY . /app

# Set environment variables for Qt to use X11
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

ENTRYPOINT ["python", "interview_assistant.py"]

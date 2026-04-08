# Whisper Auto Search

Two tools for live speech-to-document reference matching — ideal for interviewers who want instant context from a candidate's resume and documents as the conversation unfolds.

---

## 🎙 Interview Assistant  *(new — recommended)*

A polished, fully self-contained desktop app that surfaces relevant sections from a candidate's resume and supplemental documents **in real time as they speak** — so you always have context without needing to pre-read anything.

![Interview Assistant screenshot](https://github.com/user-attachments/assets/c74b5e36-af79-4eb5-bad1-2906383fcdda)

### Features

| Feature | Detail |
|---|---|
| **Live transcription** | Runs locally via `faster-whisper` (tiny → medium models). No API key or internet required. |
| **Multi-document support** | Load PDF, DOCX, and TXT files simultaneously. Documents are chunked and indexed with TF-IDF for sub-second search. |
| **Keyword highlighting** | Matched keywords are highlighted in colour inside each reference card so you can skim results instantly. |
| **Relevance scores** | Each card shows a percentage match so you know how strongly a section relates to what was just said. |
| **Debounced search** | Results update smoothly 600 ms after speech is detected — not on every word. |
| **Manual input** | A text box lets you test the search without a microphone. |
| **Clean dark UI** | Modern two-panel layout (live transcript ∣ document references). |

### Quick start

```bash
pip install -r requirements.txt
python interview_assistant.py
```

1. Click **📂 Load Documents** and select the candidate's resume, cover letter, or any other files (PDF / DOCX / TXT).
2. Click **▶ Start Listening**.
3. As the candidate speaks, relevant document sections appear automatically on the right panel.

### Transcription backends (tried in order)

1. **`faster-whisper`** — recommended; fast, fully local, no internet.
2. **`openai-whisper`** — alternative local option (`pip install openai-whisper`).
3. **`SpeechRecognition` + Google** — fallback, requires internet.

### Model size guide

| Model | Speed | Accuracy | RAM |
|---|---|---|---|
| tiny | fastest | lowest | ~1 GB |
| base | fast | good | ~1 GB |
| small | moderate | better | ~2 GB |
| medium | slow | best | ~5 GB |

---

## 🐳 Docker

Run the Interview Assistant inside a container — no need to install Python or any dependencies on the host.

### Prerequisites

* **Docker** and **Docker Compose** installed on the host.
* A Linux desktop with an X11 display server (Wayland users: enable XWayland).
* Allow the container to access your display:

  ```bash
  xhost +local:docker
  ```

### Build & run

```bash
# Build the image
docker compose build

# Run the container
docker compose up
```

Or without Compose:

```bash
docker build -t interview-assistant .
docker run --rm -it \
    -e DISPLAY=$DISPLAY \
    -e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${XDG_RUNTIME_DIR}/pulse/native:/run/user/1000/pulse/native \
    -v ./documents:/app/documents \
    --device /dev/snd \
    --network host \
    interview-assistant
```

> **Tip:** Place your PDF / DOCX / TXT files in a `documents/` folder next to the `docker-compose.yml`. They will be available inside the container at `/app/documents`.

### GPU support (optional)

To use NVIDIA GPU acceleration for Whisper, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and add the following to the service in `docker-compose.yml`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## Run_auto_search_gui.py  *(original)*

The original prototype. Listens via the `whisper.cpp` `./stream` binary and searches a TXT file for relevant noun-matched sections.

To run: set up [whisper.cpp](https://github.com/ggerganov/whisper.cpp) according to its README, then:

```bash
python Run_auto_search_gui.py
```

The test document is set to *Alice in Wonderland* to avoid copyright issues.

/**
 * app.js — Whisper Auto-Search Web Application
 *
 * Architecture:
 *  • Whisper transcription runs in the browser via Transformers.js (WebGPU or WASM).
 *  • Audio capture uses the Web Audio API + AudioWorklet for zero-latency PCM.
 *  • Document indexing and search run on the Python/FastAPI backend.
 *  • Transcribed text is sent to the backend every 600 ms (debounced).
 */

import {
  pipeline,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

// ── Constants ────────────────────────────────────────────────────────────────

/** Maps UI model name → HuggingFace ONNX model id */
const MODEL_IDS = {
  tiny:   'onnx-community/whisper-tiny.en',
  base:   'onnx-community/whisper-base.en',
  small:  'onnx-community/whisper-small.en',
};

const SAMPLE_RATE    = 16_000;
const CHUNK_SECONDS  = 4;
const STEP_SECONDS   = 2;
const DEBOUNCE_MS    = 600;
const MAX_WORDS      = 120;     // rolling transcript window for search

// ── Detect WebGPU ────────────────────────────────────────────────────────────

const HAS_WEBGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;

// ── App state ─────────────────────────────────────────────────────────────────

let transcriber   = null;       // Transformers.js pipeline
let audioContext  = null;
let mediaStream   = null;
let workletNode   = null;
let sourceNode    = null;

let isListening        = false;
let transcriptWords    = [];    // rolling word buffer
let debounceTimer      = null;
let pendingQuery       = '';
let statusPollerTimer  = null;

// ── DOM references (populated in init()) ─────────────────────────────────────

let elTranscript, elManualInput, elResults, elPlaceholder,
    elStatusBar, elBackendPill, elStatusIndicator, elDocLabel,
    elChunkLabel, elQueryLabel, elCountLabel, elDropZone,
    elModelSelect, elResultsSelect,
    btnStart, btnStop, btnSearch, btnLoad, btnClear, btnClearTranscript;

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Cache DOM elements
  elTranscript       = document.getElementById('transcript');
  elManualInput      = document.getElementById('manual-input');
  elResults          = document.getElementById('results-list');
  elPlaceholder      = document.getElementById('results-placeholder');
  elStatusBar        = document.getElementById('status-bar');
  elBackendPill      = document.getElementById('backend-pill');
  elStatusIndicator  = document.getElementById('status-indicator');
  elDocLabel         = document.getElementById('doc-label');
  elChunkLabel       = document.getElementById('chunk-label');
  elQueryLabel       = document.getElementById('query-label');
  elCountLabel       = document.getElementById('count-label');
  elDropZone         = document.getElementById('drop-zone');
  elModelSelect      = document.getElementById('model-select');
  elResultsSelect    = document.getElementById('results-select');
  btnStart           = document.getElementById('btn-start');
  btnStop            = document.getElementById('btn-stop');
  btnSearch          = document.getElementById('btn-search');
  btnLoad            = document.getElementById('btn-load');
  btnClear           = document.getElementById('btn-clear');
  btnClearTranscript = document.getElementById('btn-clear-transcript');

  // Wire events
  btnStart.addEventListener('click',          onStartListening);
  btnStop.addEventListener('click',           onStopListening);
  btnSearch.addEventListener('click',         onManualSearch);
  btnLoad.addEventListener('click',           onBrowseFiles);
  btnClear.addEventListener('click',          onClearDocuments);
  btnClearTranscript.addEventListener('click', onClearTranscript);

  elManualInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onManualSearch(); }
  });

  // Drag-and-drop upload
  elDropZone.addEventListener('dragover',  e => { e.preventDefault(); elDropZone.classList.add('drag-over'); });
  elDropZone.addEventListener('dragleave', () => elDropZone.classList.remove('drag-over'));
  elDropZone.addEventListener('drop',      e => { e.preventDefault(); elDropZone.classList.remove('drag-over'); handleFileUpload(e.dataTransfer.files); });

  // WebGPU status badge
  if (!HAS_WEBGPU) {
    const badge = document.getElementById('webgpu-badge');
    if (badge) {
      badge.textContent  = '⚠ WebGPU unavailable — using WASM';
      badge.className    = 'badge badge-warn';
    }
  }

  // Begin polling backend status (model ready, chunk count, etc.)
  pollStatus();
  statusPollerTimer = setInterval(pollStatus, 3000);

  setStatus('Ready — load documents to begin', 'idle');
});

// ─────────────────────────────────────────────────────────────────────────────
// Whisper model loading
// ─────────────────────────────────────────────────────────────────────────────

async function loadWhisperModel() {
  const modelKey  = elModelSelect.value;
  const modelId   = MODEL_IDS[modelKey] || MODEL_IDS.tiny;
  const device    = HAS_WEBGPU ? 'webgpu' : 'wasm';
  const dtype     = HAS_WEBGPU ? 'fp16' : 'q8';

  setStatus(`Loading Whisper ${modelKey} on ${device.toUpperCase()}…`, 'loading');
  btnStart.disabled = true;

  // Show progress in the status bar
  env.allowLocalModels = false;

  try {
    transcriber = await pipeline(
      'automatic-speech-recognition',
      modelId,
      {
        device,
        dtype,
        progress_callback: (p) => {
          if (p.status === 'progress' && p.total) {
            const pct = Math.round((p.loaded / p.total) * 100);
            setStatus(`Downloading ${p.file} … ${pct}%`, 'loading');
          } else if (p.status === 'initiate') {
            setStatus(`Initialising ${p.file}…`, 'loading');
          }
        },
      }
    );
    setStatus(
      `Whisper ${modelKey} ready on ${device.toUpperCase()} ✔`,
      'ready'
    );
    return true;
  } catch (err) {
    console.error('[Whisper load]', err);
    setStatus(`Failed to load model: ${err.message}`, 'error');
    btnStart.disabled = false;
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio capture
// ─────────────────────────────────────────────────────────────────────────────

async function startAudioCapture() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount:      1,
        sampleRate:        SAMPLE_RATE,
        echoCancellation:  true,
        noiseSuppression:  true,
        autoGainControl:   true,
      },
    });
  } catch (err) {
    setStatus(`Microphone access denied: ${err.message}`, 'error');
    return false;
  }

  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  sourceNode   = audioContext.createMediaStreamSource(mediaStream);

  try {
    await audioContext.audioWorklet.addModule('/static/audio-processor.js');
  } catch (err) {
    setStatus(`AudioWorklet error: ${err.message}`, 'error');
    return false;
  }

  workletNode = new AudioWorkletNode(audioContext, 'audio-processor', {
    processorOptions: {
      sampleRate:    SAMPLE_RATE,
      chunkSeconds:  CHUNK_SECONDS,
      stepSeconds:   STEP_SECONDS,
    },
  });

  workletNode.port.onmessage = async (event) => {
    if (!isListening) return;
    const { audio } = event.data;
    if (!audio) return;
    await processAudioChunk(audio);
  };

  sourceNode.connect(workletNode);
  return true;
}

function stopAudioCapture() {
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (sourceNode)  { sourceNode.disconnect();  sourceNode  = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (audioContext && audioContext.state !== 'closed') {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transcription
// ─────────────────────────────────────────────────────────────────────────────

async function processAudioChunk(audio) {
  if (!transcriber) return;
  try {
    const result = await transcriber(audio, {
      language:      'english',
      task:          'transcribe',
      return_timestamps: false,
    });
    const text = (result.text || '').trim();
    if (text && text.length > 2) {
      onNewTranscription(text);
    }
  } catch (err) {
    console.warn('[Transcription]', err);
  }
}

function onNewTranscription(text) {
  // Append to rolling transcript display
  elTranscript.value += (elTranscript.value ? ' ' : '') + text;
  elTranscript.scrollTop = elTranscript.scrollHeight;

  // Update the rolling word buffer used for search
  const words = text.split(/\s+/).filter(Boolean);
  transcriptWords.push(...words);
  if (transcriptWords.length > MAX_WORDS) {
    transcriptWords = transcriptWords.slice(transcriptWords.length - MAX_WORDS);
  }

  // Debounce search
  pendingQuery = transcriptWords.join(' ');
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => runSearch(pendingQuery), DEBOUNCE_MS);
}

// ─────────────────────────────────────────────────────────────────────────────
// Button handlers
// ─────────────────────────────────────────────────────────────────────────────

async function onStartListening() {
  if (isListening) return;

  // Load Whisper model if not yet loaded (or if the model selection changed).
  if (!transcriber) {
    const ok = await loadWhisperModel();
    if (!ok) return;
  }

  const ok = await startAudioCapture();
  if (!ok) return;

  isListening = true;
  btnStart.disabled = true;
  btnStop.disabled  = false;
  elStatusIndicator.textContent = '🔴  Listening…';
  elStatusIndicator.className   = 'badge badge-live';
  setStatus('🎙 Listening — speak now…', 'live');
}

function onStopListening() {
  if (!isListening) return;
  isListening = false;
  stopAudioCapture();

  btnStart.disabled = false;
  btnStop.disabled  = true;
  elStatusIndicator.textContent = '⚪  Stopped';
  elStatusIndicator.className   = 'badge';
  setStatus('Stopped', 'idle');
}

function onManualSearch() {
  const q = elManualInput.value.trim();
  if (!q) return;
  // Add to transcript display
  elTranscript.value += (elTranscript.value ? '\n[Manual] ' : '[Manual] ') + q;
  elTranscript.scrollTop = elTranscript.scrollHeight;
  runSearch(q);
}

function onClearTranscript() {
  elTranscript.value = '';
  transcriptWords    = [];
  pendingQuery       = '';
  clearResultsPanel();
}

async function onClearDocuments() {
  try {
    await fetch('/api/documents', { method: 'DELETE' });
    clearResultsPanel();
    elDocLabel.textContent   = 'No documents loaded';
    elChunkLabel.textContent = '';
    setStatus('Documents cleared', 'idle');
    await pollStatus();
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'error');
  }
}

function onBrowseFiles() {
  const input = document.createElement('input');
  input.type     = 'file';
  input.multiple = true;
  input.accept   = '.pdf,.docx,.txt,.md,.csv';
  input.addEventListener('change', () => handleFileUpload(input.files));
  input.click();
}

// ─────────────────────────────────────────────────────────────────────────────
// File upload
// ─────────────────────────────────────────────────────────────────────────────

async function handleFileUpload(fileList) {
  if (!fileList || fileList.length === 0) return;

  const formData = new FormData();
  for (const file of fileList) formData.append('files', file);

  setStatus(`Uploading ${fileList.length} file(s)…`, 'loading');
  btnLoad.disabled = true;

  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    const ok  = data.files.filter(f => f.status === 'ok');
    const bad = data.files.filter(f => f.status !== 'ok');
    const names = ok.map(f => `${f.filename} (${f.chunks} chunks)`).join(', ');

    if (ok.length) {
      setStatus(`Loaded: ${names}`, 'ready');
      elDocLabel.textContent   = `${ok.length} file(s) loaded`;
      elChunkLabel.textContent = `${data.total_chunks} chunks`;
    }
    if (bad.length) {
      const msgs = bad.map(f => `${f.filename}: ${f.message}`).join('; ');
      setStatus(`Upload errors — ${msgs}`, 'error');
    }
  } catch (err) {
    setStatus(`Upload failed: ${err.message}`, 'error');
  } finally {
    btnLoad.disabled = false;
    await pollStatus();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Search
// ─────────────────────────────────────────────────────────────────────────────

async function runSearch(query) {
  if (!query || !query.trim()) return;

  const topK = parseInt(elResultsSelect.value, 10) || 5;

  try {
    const resp = await fetch('/api/search', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, top_k: topK }),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    renderResults(data);
  } catch (err) {
    console.warn('[Search]', err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Results rendering
// ─────────────────────────────────────────────────────────────────────────────

const HIGHLIGHT_COLOURS = [
  ['#ff9e64', 'rgba(255,158,100,0.18)'],
  ['#9ece6a', 'rgba(158,206,106,0.18)'],
  ['#7dcfff', 'rgba(125,207,255,0.18)'],
  ['#e0af68', 'rgba(224,175,104,0.18)'],
  ['#bb9af7', 'rgba(187,154,247,0.18)'],
  ['#7aa2f7', 'rgba(122,162,247,0.18)'],
  ['#f7768e', 'rgba(247,118,142,0.18)'],
];

const FILE_ICONS = { pdf: '📕', docx: '📘', doc: '📘', txt: '📄', md: '📝', csv: '📊' };

function renderResults(data) {
  // Update labels
  const q = data.query.length > 60 ? data.query.slice(0, 57) + '…' : data.query;
  elQueryLabel.textContent = `Matching: "${q}"`;
  elCountLabel.textContent = `${data.results.length} result(s) · ${data.backend}`;

  clearResultsPanel();

  if (data.results.length === 0) {
    elPlaceholder.style.display = 'block';
    elPlaceholder.textContent   = '🔍 No matching sections found';
    return;
  }

  elPlaceholder.style.display = 'none';

  // Extract keywords from query (non-stop short words)
  const keywords = data.query
    .split(/\s+/)
    .filter(w => w.length >= 3)
    .slice(0, HIGHLIGHT_COLOURS.length);

  for (const hit of data.results) {
    elResults.appendChild(buildCard(hit, keywords, data.backend));
  }
}

function buildCard(hit, keywords, backend) {
  const { chunk, score, score_pct } = hit;
  const ext  = (chunk.display_source.split('.').pop() || '').toLowerCase();
  const icon = FILE_ICONS[ext] || '📄';

  // Score colour
  let accent = '#f7768e';
  if (score_pct >= 55) accent = '#9ece6a';
  else if (score_pct >= 30) accent = '#e0af68';

  const card = document.createElement('div');
  card.className = 'ref-card';
  card.style.setProperty('--accent', accent);

  // Header
  const header = document.createElement('div');
  header.className = 'ref-card-header';

  const src = document.createElement('span');
  src.className   = 'ref-card-source';
  src.textContent = `${icon}  ${chunk.display_source}`;

  const meta = document.createElement('span');
  meta.className = 'ref-card-meta';

  const be = document.createElement('span');
  be.className   = 'ref-card-backend';
  be.textContent = backend;

  const pill = document.createElement('span');
  pill.className   = 'ref-card-score';
  pill.style.color = accent;
  pill.textContent = `${score_pct}%`;

  meta.append(be, pill);
  header.append(src, meta);

  // Score bar
  const barWrap = document.createElement('div');
  barWrap.className = 'ref-card-bar';
  const barFill = document.createElement('div');
  barFill.className = 'ref-card-bar-fill';
  barFill.style.width       = `${Math.min(100, score_pct)}%`;
  barFill.style.background  = accent;
  barWrap.appendChild(barFill);

  // Body text with keyword highlights
  const body = document.createElement('div');
  body.className = 'ref-card-body';
  body.innerHTML = highlightKeywords(chunk.text, keywords);

  card.append(header, barWrap, body);
  return card;
}

function highlightKeywords(text, keywords) {
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  keywords.slice(0, HIGHLIGHT_COLOURS.length).forEach((kw, i) => {
    if (!kw || kw.length < 2) return;
    const [fg, bg] = HIGHLIGHT_COLOURS[i % HIGHLIGHT_COLOURS.length];
    const re = new RegExp(kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    html = html.replace(
      re,
      m => `<mark style="color:${fg};background:${bg};border-radius:3px;padding:0 2px;font-weight:600">${m}</mark>`
    );
  });

  return html;
}

function clearResultsPanel() {
  elResults.innerHTML     = '';
  elPlaceholder.style.display = 'block';
  elPlaceholder.textContent   = '📂  Load documents and start speaking\nto see live references here';
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend status polling
// ─────────────────────────────────────────────────────────────────────────────

async function pollStatus() {
  try {
    const resp = await fetch('/api/status');
    if (!resp.ok) return;
    const s = await resp.json();
    updateBackendPill(s.backend);
    if (s.chunk_count > 0) {
      elChunkLabel.textContent = `${s.chunk_count} chunks`;
    }
  } catch { /* backend not yet ready */ }
}

function updateBackendPill(backend) {
  const map = {
    semantic:   ['🧠', 'pill-semantic'],
    hybrid:     ['📊', 'pill-hybrid'],
    tfidf:      ['📈', 'pill-tfidf'],
    keyword:    ['⌨️',  'pill-keyword'],
  };
  const key   = Object.keys(map).find(k => backend.includes(k)) || 'keyword';
  const [icon, cls] = map[key];
  elBackendPill.textContent = `${icon} ${backend}`;
  elBackendPill.className   = `badge ${cls}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

function setStatus(msg, state = 'idle') {
  elStatusBar.textContent  = msg;
  elStatusBar.className    = `status-bar status-${state}`;
}

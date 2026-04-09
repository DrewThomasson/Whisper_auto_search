/**
 * audio-processor.js — AudioWorklet processor for real-time PCM capture.
 *
 * Buffers raw float32 samples from the microphone and posts fixed-size chunks
 * to the main thread using a sliding window (same parameters as the original
 * Python implementation: 4 s window, 2 s step).
 *
 * Must be loaded via:
 *   await audioContext.audioWorklet.addModule('/static/audio-processor.js');
 */

class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this._sampleRate  = opts.sampleRate   || 16000;
    this._chunkSecs   = opts.chunkSeconds || 4;
    this._stepSecs    = opts.stepSeconds  || 2;
    this._chunkSize   = Math.round(this._sampleRate * this._chunkSecs);
    this._stepSize    = Math.round(this._sampleRate * this._stepSecs);

    // Growable buffer — we only allocate new arrays when we post chunks.
    this._buffer = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    // Append new samples to the buffer.
    const merged = new Float32Array(this._buffer.length + channel.length);
    merged.set(this._buffer);
    merged.set(channel, this._buffer.length);
    this._buffer = merged;

    // Emit complete chunks via sliding window.
    while (this._buffer.length >= this._chunkSize) {
      // Transfer ownership of the typed array for zero-copy messaging.
      const chunk = this._buffer.slice(0, this._chunkSize);
      this.port.postMessage({ audio: chunk });
      this._buffer = this._buffer.slice(this._stepSize);
    }

    return true; // keep processor alive
  }
}

registerProcessor('audio-processor', AudioProcessor);

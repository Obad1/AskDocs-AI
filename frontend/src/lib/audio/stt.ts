// STT engine for AskDocs AI v2.0 (spec §4.5, §8.3).
// Primary: Whisper via @xenova/transformers (Transformers.js), running locally
// in-browser. Fallback: Vosk-browser if Whisper fails to initialize.
// Zero API keys, zero network beyond the (cacheable) local model download.

import type { MODELID } from "../../types/schema";

export interface STTResult {
  text: string;
}

export interface STTEngine {
  listen: (onTranscript: (text: string) => void) => Promise<() => void>;
  stop: () => void;
}

interface WhisperPipeline {
  (audio: Float32Array, opts: Record<string, unknown>): Promise<{ text: string }>;
}

const SAMPLE_RATE = 16000;

export class WhisperSTTEngine implements STTEngine {
  private pipeline: WhisperPipeline | null = null;
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private audioCtx: AudioContext | null = null;
  private stopped = false;

  constructor(private modelId: MODELID) {}

  private async ensurePipeline(): Promise<WhisperPipeline> {
    if (this.pipeline) return this.pipeline;
    const transformers = await import(
      /* @vite-ignore */ "@xenova/transformers"
    );
    const modelName = this._normalizeModel(this.modelId);
    const pipe = await (transformers as any).pipeline(
      "automatic-speech-recognition",
      modelName,
      { quantized: true },
    );
    this.pipeline = pipe as WhisperPipeline;
    return this.pipeline;
  }

  private _normalizeModel(id: MODELID): string {
    // Catalog ids look like "whisper-tiny.en"; Xenova HF ids are "Xenova/whisper-tiny.en".
    if (id.startsWith("Xenova/")) return id;
    if (id.startsWith("whisper-")) return `Xenova/${id}`;
    return "Xenova/whisper-tiny.en";
  }

  async listen(onTranscript: (text: string) => void): Promise<() => void> {
    this.stopped = false;
    const pipe = await this.ensurePipeline().catch(async (e) => {
      console.warn("[STT] Whisper init failed, falling back to Vosk:", e);
      const vosk = new VoskSTTEngine(this.modelId);
      return (await vosk.listen(onTranscript)) as unknown as WhisperPipeline;
    });

    // If the fallback returned a stop fn (Vosk path), return it directly.
    if (typeof (pipe as any) === "function" && (pipe as any).__isStop) {
      return (pipe as any) as () => void;
    }

    this.audioCtx = new (window.AudioContext ||
      (window as any).webkitAudioContext)();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.recorder = new MediaRecorder(this.stream);

    this.recorder.ondataavailable = async (ev: BlobEvent) => {
      if (this.stopped || ev.data.size === 0) return;
      const buf = await ev.data.arrayBuffer();
      const audioBuf = await this.audioCtx!.decodeAudioData(buf);
      // Downsample to 16kHz mono for Whisper.
      const samples = this._resample(audioBuf, SAMPLE_RATE);
      try {
        const out = await pipe(samples, {
          chunk_length_s: 30,
          stride_length_s: 5,
          language: "english",
          task: "transcribe",
        });
        if (out?.text) onTranscript(out.text.trim());
      } catch (err) {
        console.error("[STT] transcription error:", err);
      }
    };

    this.recorder.start(1500);
    return () => this.stop();
  }

  private _resample(audioBuf: AudioBuffer, targetRate: number): Float32Array {
    const channel = audioBuf.getChannelData(0);
    const ratio = audioBuf.sampleRate / targetRate;
    const newLen = Math.floor(channel.length / ratio);
    const out = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      out[i] = channel[Math.floor(i * ratio)];
    }
    return out;
  }

  stop(): void {
    this.stopped = true;
    try {
      this.recorder?.stop();
    } catch {
      /* noop */
    }
    this.stream?.getTracks().forEach((t) => t.stop());
    this.audioCtx?.close().catch(() => {});
  }
}

// Vosk-browser fallback. Loaded lazily; gracefully degrades if unavailable.
export class VoskSTTEngine implements STTEngine {
  private stopped = false;
  private stream: MediaStream | null = null;
  private audioCtx: AudioContext | null = null;
  private recognizer: any = null;

  constructor(private modelId: MODELID) {}

  async listen(onTranscript: (text: string) => void): Promise<() => void> {
    this.stopped = false;
    let vosk: any;
    try {
      // @ts-ignore - optional runtime-only dependency (vosk-browser)
      const voskModule = "vosk-browser";
      vosk = await import(/* @vite-ignore */ voskModule);
    } catch (e) {
      throw new Error(
        "Neither Whisper nor Vosk STT is available in this build: " + e,
      );
    }

    const modelUrl =
      "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip";
    const model = await vosk.createModel(modelUrl);
    this.recognizer = new vosk.Recognizer({ model, sampleRate: SAMPLE_RATE });

    this.audioCtx = new (window.AudioContext ||
      (window as any).webkitAudioContext)();
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = this.audioCtx.createMediaStreamSource(this.stream);
    const processor = (this.audioCtx as any).createScriptProcessor(
      4096,
      1,
      1,
    );
    processor.onaudioprocess = (e: AudioProcessingEvent) => {
      if (this.stopped) return;
      const input = e.inputBuffer.getChannelData(0);
      const resampled = this._to16k(input, this.audioCtx!.sampleRate);
      if (this.recognizer.acceptWaveform(resampled)) {
        const res = this.recognizer.result();
        if (res?.text) onTranscript(res.text.trim());
      }
    };
    source.connect(processor);
    processor.connect(this.audioCtx.destination);

    const stop = () => {
      this.stopped = true;
      try {
        source.disconnect();
        processor.disconnect();
      } catch {
        /* noop */
      }
      this.stream?.getTracks().forEach((t) => t.stop());
      this.audioCtx?.close().catch(() => {});
      try {
        this.recognizer?.free();
      } catch {
        /* noop */
      }
    };
    (stop as any).__isStop = true;
    return stop;
  }

  private _to16k(input: Float32Array, rate: number): Float32Array {
    const ratio = rate / SAMPLE_RATE;
    const len = Math.floor(input.length / ratio);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = input[Math.floor(i * ratio)];
    return out;
  }

  stop(): void {
    this.stopped = true;
  }
}

let singleton: STTEngine | null = null;
export function getSTTEngine(modelId: MODELID): STTEngine {
  if (!singleton || (singleton as any).modelId !== modelId) {
    singleton = new WhisperSTTEngine(modelId);
    (singleton as any).modelId = modelId;
  }
  return singleton;
}

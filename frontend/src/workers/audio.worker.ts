// Audio worker (spec §3.7, §4.5). Off-thread Speech-to-Text and Text-to-Speech
// using local, zero-auth models (Transformers.js Whisper + piper-tts). Exposed
// via Comlink. The main-thread UI may also use the lib/audio helpers directly;
// this worker exists to keep heavy transcription/synthesis off the UI thread.

import * as Comlink from "comlink";
import { pipeline, env } from "@xenova/transformers";

env.allowLocalModels = false;
// Self-hosted ONNX runtime .wasm (see embedding.worker.ts note).
env.backends.onnx.wasm.wasmPaths = "/ort/";
env.backends.onnx.wasm.numThreads = 1;

const SAMPLE_RATE = 16000;

async function whisperTranscribe(
  audio: Float32Array,
  modelId = "Xenova/whisper-base.en",
): Promise<string> {
  const pipe = await pipeline("automatic-speech-recognition", modelId, {
    quantized: true,
  });
  const out: any = await pipe(audio, {
    chunk_length_s: 30,
    stride_length_s: 5,
    language: "english",
    task: "transcribe",
  });
  return typeof out === "string" ? out : out?.text ?? "";
}

async function piperSynthesize(text: string, voice: string): Promise<ArrayBuffer> {
  // "piper-tts" does not exist on npm (verified E404), so this path can never
  // load a WASM synthesizer. Reject clearly instead of failing unhandled; the
  // main-thread TTSEngine already falls back to the Web Speech API.
  throw new Error(
    `Piper WASM is not available in this build (piper-tts is not installable). ` +
      `Voice "${voice}" unavailable; use the main-thread TTS (Web Speech) engine.`,
  );
}

const api = {
  async transcribe(audio: ArrayBuffer, modelId?: string): Promise<string> {
    const samples = new Float32Array(audio);
    return whisperTranscribe(samples, modelId ?? "Xenova/whisper-base.en");
  },
  async synthesize(text: string, voice: string): Promise<ArrayBuffer> {
    return piperSynthesize(text, voice);
  },
};

export type AudioWorkerApi = typeof api;
Comlink.expose(api);

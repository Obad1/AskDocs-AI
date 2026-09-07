// Audio worker (spec §3.7, §4.5). Off-thread Speech-to-Text and Text-to-Speech
// using local, zero-auth models (Transformers.js Whisper + piper-tts). Exposed
// via Comlink. The main-thread UI may also use the lib/audio helpers directly;
// this worker exists to keep heavy transcription/synthesis off the UI thread.

import * as Comlink from "comlink";
import { pipeline, env } from "@xenova/transformers";

env.allowLocalModels = false;
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
  // @ts-ignore - optional runtime-only dependency (piper-tts)
  const piperModule = "piper-tts";
  const mod = await import(/* @vite-ignore */ piperModule);
  const VOICE_PATHS: Record<string, string> = {
    "en_US-lessac-medium": "en/en_US-lessac-medium/en_US-lessac-medium.onnx",
    "en_US-lessac-low": "en/en_US-lessac-low/en_US-lessac-low.onnx",
    "en_US-libritts-high": "en/en_US-libritts-high/en_US-libritts-high.onnx",
    "en_US-multi-high": "en/en_US-multi-high/en_US-multi-high.onnx",
  };
  const rel = VOICE_PATHS[voice] ?? VOICE_PATHS["en_US-lessac-medium"];
  const base = "https://huggingface.co/rhasspy/piper-voices/resolve/main";
  const onnxUrl = `${base}/${rel}`;
  const jsonUrl = onnxUrl.replace(/\.onnx$/, ".onnx.json");
  const voiceObj = await mod.PiperVoice.fromConfig(onnxUrl, jsonUrl, () => {});
  const wav: Uint8Array = await voiceObj.synthesize(text, { lengthScale: 1 });
  return wav.buffer.slice(
    wav.byteOffset,
    wav.byteOffset + wav.byteLength,
  ) as ArrayBuffer;
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

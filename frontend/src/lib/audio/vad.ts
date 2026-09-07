// VAD (Voice Activity Detection) wrapper for AskDocs AI v2.0 (spec §3.7, §4.5).
// Uses @ricky0123/vad-web (Silero WASM) locally in-browser. Detects when the
// user starts/stops speaking so the VoiceInterrupter can pause TTS, capture a
// query via STT, retrieve from local RAG, and speak the answer.
//
// Zero API keys, zero network beyond the (cacheable) Silero WASM fetch.

export interface VADHandlers {
  onSpeechStart?: () => void;
  onSpeechEnd?: (audio: Float32Array) => void;
  onError?: (err: Error) => void;
}

export interface VADHandle {
  stop: () => void;
}

interface MyVAD {
  start: () => Promise<void>;
  stop: () => Promise<void>;
}

export async function startVAD(handlers: VADHandlers): Promise<VADHandle> {
  let vad: any;
  try {
    vad = await import(/* @vite-ignore */ "@ricky0123/vad-web");
  } catch (e) {
    const err = new Error("@ricky0123/vad-web is not available: " + e);
    handlers.onError?.(err);
    return { stop: () => {} };
  }

  let myvad: MyVAD | null = null;
  let stopped = false;

  try {
    myvad = await vad.MicVAD.new({
      modelURL:
        "https://huggingface.co/ricky0123/vad/resolve/main/silero_vad.onnx",
      onSpeechStart: () => {
        if (!stopped) handlers.onSpeechStart?.();
      },
      onSpeechEnd: (audio: Float32Array) => {
        if (!stopped) handlers.onSpeechEnd?.(audio);
      },
    });
    if (!myvad) return { stop: () => {} };
    await myvad.start();
  } catch (e) {
    const err = e instanceof Error ? e : new Error(String(e));
    handlers.onError?.(err);
    return { stop: () => {} };
  }

  return {
    stop: () => {
      stopped = true;
      try {
        myvad?.stop();
      } catch {
        /* noop */
      }
    },
  };
}

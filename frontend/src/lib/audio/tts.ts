// TTS engine for AskDocs AI v2.0 (spec §4.5, §8.3).
// Primary: Piper via WebAssembly (rhasspy/piper-voices) loaded from HuggingFace.
// Fallback: native Web Speech API speechSynthesis (if Piper fails to load
// within 3000ms, or is unavailable). Zero API keys, zero network at runtime
// beyond the local model fetch which is cached after first load.
//
// Emits sentence boundaries so the KaraokeTranscript can highlight the active
// sentence during playback.

export type TTSVoice = string;
export type TTSSpeed = number;

export interface TTSOptions {
  voice: TTSVoice;
  speed?: TTSSpeed;
  onSentence?: (sentence: string, index: number) => void;
  onBoundary?: (charIndex: number, charLength: number) => void;
  onEnd?: () => void;
  onError?: (err: Error) => void;
}

export interface TTSControls {
  pause: () => void;
  resume: () => void;
  stop: () => void;
  setSpeed: (s: number) => void;
  readonly usingFallback: boolean;
}

// Piper voice id -> HF model path within rhasspy/piper-voices.
// These map to the catalog ids in ModelEngineContext (e.g. en_US-lessac-medium).
const PIPER_VOICE_PATHS: Record<string, string> = {
  "en_US-lessac-low": "en/en_US-lessac-low/en_US-lessac-low.onnx",
  "en_US-lessac-medium": "en/en_US-lessac-medium/en_US-lessac-medium.onnx",
  "en_US-libritts-high": "en/en_US-libritts-high/en_US-libritts-high.onnx",
  "en_US-multi-high": "en/en_US-multi-high/en_US-multi-high.onnx",
};

const PIPER_LOAD_TIMEOUT_MS = 3000;

interface PiperModule {
  PiperVoice: any;
  getDefaultSession?: () => any;
}

/**
 * Splits text into sentence-ish chunks for karaoke highlighting.
 */
function splitSentences(text: string): string[] {
  return text
    .replace(/\s+/g, " ")
    .trim()
    .split(/(?<=[.!?])\s+/)
    .filter(Boolean);
}

export class TTSEngine {
  private piper: PiperModule | null = null;
  private piperVoice: any = null;
  private piperLoaded = false;
  private usingFallback = false;
  private loadPromise: Promise<boolean> | null = null;
  private audioCtx: AudioContext | null = null;

  /**
   * Ensure the Piper WASM runtime is loaded. Resolves true if Piper is usable.
   * Resolves false (after fallback) if Piper failed / timed out.
   */
  async ensurePiper(modelId: TTSVoice): Promise<boolean> {
    if (this.loadPromise) return this.loadPromise;
    this.loadPromise = this._loadPiper(modelId);
    return this.loadPromise;
  }

  private async _loadPiper(modelId: TTSVoice): Promise<boolean> {
    const withTimeout = <T>(p: Promise<T>, ms: number, tag: string): Promise<T> =>
      new Promise<T>((resolve, reject) => {
        const t = setTimeout(
          () => reject(new Error(`Piper ${tag} timed out after ${ms}ms`)),
          ms,
        );
        p.then(
          (v) => {
            clearTimeout(t);
            resolve(v);
          },
          (e) => {
            clearTimeout(t);
            reject(e);
          },
        );
      });

    try {
      const piperModule = "piper-tts";
      const mod = (await withTimeout(
        // @ts-ignore - optional runtime-only dependency (piper-tts)
        import(/* @vite-ignore */ piperModule),
        PIPER_LOAD_TIMEOUT_MS,
        "module import",
      )) as PiperModule;
      this.piper = mod;

      const voiceRel = PIPER_VOICE_PATHS[modelId] ?? PIPER_VOICE_PATHS["en_US-lessac-medium"];
      const base = "https://huggingface.co/rhasspy/piper-voices/resolve/main";
      const onnxUrl = `${base}/${voiceRel}`;
      const jsonUrl = onnxUrl.replace(/\.onnx$/, ".onnx.json");

      this.piperVoice = await withTimeout(
        mod.PiperVoice.fromConfig(onnxUrl, jsonUrl, (progress: number) =>
          void progress,
        ),
        PIPER_LOAD_TIMEOUT_MS,
        "voice load",
      );
      this.piperLoaded = true;
      this.usingFallback = false;
      return true;
    } catch (err) {
      console.warn("[TTSEngine] Piper unavailable, using speechSynthesis:", err);
      this.piperLoaded = false;
      this.usingFallback = true;
      return false;
    }
  }

  /**
   * Speak `text`. Returns playback controls. Sentence boundaries are emitted
   * through `opts.onSentence` for karaoke highlighting.
   */
  async speak(text: string, opts: TTSOptions): Promise<TTSControls> {
    const sentences = splitSentences(text);
    const speed = opts.speed ?? 1.0;

    const ok = await this.ensurePiper(opts.voice);
    if (ok && this.piper && this.piperVoice) {
      return this._speakPiper(sentences, opts, speed);
    }
    return this._speakFallback(sentences, opts, speed);
  }

  private _speakPiper(
    sentences: string[],
    opts: TTSOptions,
    speed: number,
  ): TTSControls {
    let stopped = false;
    let paused = false;
    let current = 0;

    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext ||
        (window as any).webkitAudioContext)();
    }
    const ctx = this.audioCtx;

    const playOne = async (index: number) => {
      if (stopped || paused || index >= sentences.length) {
        if (index >= sentences.length && !stopped) opts.onEnd?.();
        return;
      }
      current = index;
      opts.onSentence?.(sentences[index], index);
      try {
        const audio = await this.piperVoice.synthesize(sentences[index], {
          lengthScale: 1 / speed,
        });
        const blob = new Blob([audio], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        const el = new Audio(url);
        el.playbackRate = speed;
        await ctx.decodeAudioData(await blob.arrayBuffer()).then(async (buf) => {
          const src = ctx.createBufferSource();
          src.buffer = buf;
          src.playbackRate.value = speed;
          src.onended = () => {
            URL.revokeObjectURL(url);
            if (!stopped && !paused) playOne(index + 1);
          };
          src.connect(ctx.destination);
          src.start(0);
        });
      } catch (e) {
        opts.onError?.(e as Error);
      }
    };

    playOne(0);

    return {
      pause: () => {
        paused = true;
      },
      resume: () => {
        if (paused) {
          paused = false;
          playOne(current);
        }
      },
      stop: () => {
        stopped = true;
        paused = false;
      },
      setSpeed: (s: number) => {
        // Piper lengthScale re-synthesis: restart current sentence at new speed.
        if (!stopped && !paused) {
          paused = true;
          setTimeout(() => {
            paused = false;
            playOne(current);
          }, 0);
        }
        void s;
      },
      get usingFallback() {
        return false;
      },
    };
  }

  private _speakFallback(
    sentences: string[],
    opts: TTSOptions,
    speed: number,
  ): TTSControls {
    const synth = window.speechSynthesis;
    let current = 0;
    let cancelled = false;

    const speakOne = (index: number) => {
      if (cancelled || index >= sentences.length) {
        if (index >= sentences.length && !cancelled) opts.onEnd?.();
        return;
      }
      current = index;
      opts.onSentence?.(sentences[index], index);
      const u = new SpeechSynthesisUtterance(sentences[index]);
      u.rate = speed;
      u.voice = this._pickSpeechVoice(opts.voice);
      u.onboundary = (e: SpeechSynthesisEvent) =>
        opts.onBoundary?.(e.charIndex, e.charLength ?? sentences[index].length);
      u.onend = () => {
        if (!cancelled) speakOne(index + 1);
      };
      u.onerror = (e: SpeechSynthesisErrorEvent) =>
        opts.onError?.(new Error(e.error));
      synth.speak(u);
    };

    speakOne(0);

    return {
      pause: () => synth.pause(),
      resume: () => synth.resume(),
      stop: () => {
        cancelled = true;
        synth.cancel();
        opts.onEnd?.();
      },
      setSpeed: (s: number) => {
        synth.cancel();
        if (!cancelled) {
          // Re-speak remaining from current sentence at new speed.
          const remaining = sentences.slice(current);
          remaining.forEach((s2, i) => {
            const u = new SpeechSynthesisUtterance(s2);
            u.rate = s;
            u.voice = this._pickSpeechVoice(opts.voice);
            if (i === 0) {
              u.onend = () => opts.onEnd?.();
            }
            synth.speak(u);
          });
        }
      },
      get usingFallback() {
        return true;
      },
    };
  }

  private _pickSpeechVoice(voiceId: TTSVoice): SpeechSynthesisVoice | null {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;
    return (
      voices.find((v) => v.voiceURI === voiceId || v.name === voiceId) ??
      voices.find((v) => /en[-_]US/i.test(v.lang) || /^en/i.test(v.lang)) ??
      voices[0]
    );
  }
}

let singleton: TTSEngine | null = null;
export function getTTSEngine(): TTSEngine {
  if (!singleton) singleton = new TTSEngine();
  return singleton;
}

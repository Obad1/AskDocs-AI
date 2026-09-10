import React, { createContext, useContext, useState } from "react";
import { useModelEngine } from "./ModelEngineContext";

// Podcast / voice mode state (spec §3.7). Holds TTS playback, karaoke sync,
// and voice-interrupter (VAD) coordination. TTS engine itself lives in
// services/audio (built by the audio subsystem agent).

interface AudioCtx {
  isPlaying: boolean;
  speed: number;
  voice: string;
  currentSentenceIndex: number;
  transcript: string[];
  voiceInterruptActive: boolean;
  setIsPlaying: (b: boolean) => void;
  setSpeed: (s: number) => void;
  setVoice: (v: string) => void;
  setTranscript: (t: string[]) => void;
  setCurrentSentence: (i: number) => void;
  setVoiceInterrupt: (b: boolean) => void;
}

const Ctx = createContext<AudioCtx | null>(null);

export function AudioProvider({ children }: { children: React.ReactNode }) {
  const { state } = useModelEngine();
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [voice, setVoice] = useState(state.tts_model);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [currentSentenceIndex, setCurrentSentence] = useState(0);
  const [voiceInterruptActive, setVoiceInterrupt] = useState(false);

  return (
    <Ctx.Provider
      value={{
        isPlaying,
        speed,
        voice,
        currentSentenceIndex,
        transcript,
        voiceInterruptActive,
        setIsPlaying,
        setSpeed,
        setVoice,
        setTranscript,
        setCurrentSentence,
        setVoiceInterrupt,
      }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useAudio(): AudioCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useAudio must be used within AudioProvider");
  return c;
}

"""YouTube fetch via yt-dlp (spec Section 4.1).

Uses ONLY public metadata / caption endpoints of yt-dlp - no YouTube Data API
key, no authentication. Strategy (multimodal cascade for video):
  1. Try to fetch auto-generated / manual captions as text.
  2. If none are available, download the bestaudio track locally and return its
     path so the caller can run faster-whisper (see media_transcriber).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional


def fetch_youtube(url: str, workdir: Optional[str] = None) -> dict:
    """Fetch a YouTube URL.

    Returns ``{audio_path, captions, title}``. ``captions`` may be empty, in
    which case ``audio_path`` points to a downloaded audio file for transcription.
    """
    from yt_dlp import YoutubeDL

    base = workdir or tempfile.mkdtemp(prefix="askdocs-yt-")
    Path(base).mkdir(parents=True, exist_ok=True)

    # Pass 1: probe for captions without downloading media.
    probe_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "quiet": True,
        "no_warnings": True,
    }
    title = None
    captions = ""
    try:
        with YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title")
            for key in ("subtitles", "automatic_captions"):
                subs = info.get(key) or {}
                for lang in ("en", "en-US", "en-GB"):
                    tracks = subs.get(lang)
                    if tracks:
                        # Prefer json3 / srv3; fall back to any listed format.
                        fmt = next(
                            (t for t in tracks if t.get("ext") in ("json3", "srv3", "vtt")),
                            tracks[0],
                        )
                        captions = _render_captions(fmt)
                        if captions:
                            break
                if captions:
                    break
    except Exception:
        # Probe failures are non-fatal; fall through to audio download.
        pass

    audio_path: Optional[str] = None
    if not captions:
        audio_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(base, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(audio_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = title or info.get("title")
            audio_path = ydl.prepare_filename(info)
            # prepare_filename may not have the real extension; resolve.
            if not (audio_path and os.path.exists(audio_path)):
                matches = list(Path(base).glob(f"{info.get('id', '*')}.*"))
                audio_path = str(matches[0]) if matches else None

    return {"audio_path": audio_path, "captions": captions, "title": title}


def _render_captions(track: dict) -> str:
    """Best-effort flatten of a yt-dlp subtitle track to plain text."""
    url = track.get("url")
    if not url:
        return ""
    try:
        import urllib.request

        raw = urllib.request.urlopen(url, timeout=20).read().decode("utf-8", "ignore")
        # Very light VTT/srv3 cleanup: drop timing + tags.
        lines = [
            ln.strip()
            for ln in raw.splitlines()
            if ln.strip()
            and not ln.strip().isdigit()
            and "-->" not in ln
            and not ln.startswith("<")
            and not ln.startswith("WEBVTT")
        ]
        return "\n".join(lines)
    except Exception:
        return ""

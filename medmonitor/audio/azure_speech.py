"""Transcrição de áudio — Azure AI Speech (Speech to Text) com fallback offline.

Ordem de tentativa:
  1. Azure Speech (``AZURE_SPEECH_KEY`` + ``AZURE_SPEECH_REGION``) — reconhecimento contínuo pt-BR;
  2. faster-whisper local (se instalado);
  3. transcrição de referência ``<arquivo>.txt`` ao lado do ``.wav`` (dados de demonstração).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..config import env, offline_forced

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    engine: str
    language: str = "pt-BR"
    segments: list[dict] = field(default_factory=list)   # [{offset_s, duration_s, text, confidence}]


def azure_available() -> bool:
    return bool(env("AZURE_SPEECH_KEY")) and not offline_forced()


def transcribe_azure(wav: Path, language: str | None = None) -> Transcript:
    import azure.cognitiveservices.speech as speechsdk

    cfg = speechsdk.SpeechConfig(subscription=env("AZURE_SPEECH_KEY"), region=env("AZURE_SPEECH_REGION", "brazilsouth"))
    cfg.speech_recognition_language = language or env("AZURE_SPEECH_LANGUAGE", "pt-BR")
    cfg.output_format = speechsdk.OutputFormat.Detailed
    cfg.request_word_level_timestamps()
    audio = speechsdk.audio.AudioConfig(filename=str(wav))
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)

    segments, done, errors = [], threading.Event(), []

    def on_recognized(evt):
        r = evt.result
        if r.reason == speechsdk.ResultReason.RecognizedSpeech and r.text:
            conf = None
            try:
                import json
                conf = json.loads(r.json)["NBest"][0]["Confidence"]
            except Exception:  # noqa: BLE001
                pass
            segments.append({"offset_s": r.offset / 1e7, "duration_s": r.duration / 1e7, "text": r.text,
                             "confidence": conf})

    def on_canceled(evt):
        if evt.reason == speechsdk.CancellationReason.Error:
            errors.append(evt.error_details)
        done.set()

    rec.recognized.connect(on_recognized)
    rec.session_stopped.connect(lambda _: done.set())
    rec.canceled.connect(on_canceled)
    rec.start_continuous_recognition()
    done.wait(timeout=600)
    rec.stop_continuous_recognition()
    if errors:
        raise RuntimeError(f"Azure Speech: {errors[0]}")
    return Transcript(" ".join(s["text"] for s in segments), "azure-speech", cfg.speech_recognition_language, segments)


def transcribe_local(wav: Path) -> Transcript:
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segs, _ = model.transcribe(str(wav), language="pt")
        segs = [{"offset_s": s.start, "duration_s": s.end - s.start, "text": s.text.strip(), "confidence": None}
                for s in segs]
        return Transcript(" ".join(s["text"] for s in segs), "faster-whisper", "pt", segs)
    except ImportError:
        pass
    ref = Path(wav).with_suffix(".txt")
    if ref.exists():
        return Transcript(ref.read_text(encoding="utf-8").strip(), "transcricao-referencia (offline)")
    return Transcript("", "indisponivel")


def transcribe(wav: Path) -> Transcript:
    if azure_available():
        try:
            return transcribe_azure(wav)
        except Exception as e:  # noqa: BLE001
            log.warning("Falha no Azure Speech (%s) — usando fallback offline", e)
    return transcribe_local(wav)

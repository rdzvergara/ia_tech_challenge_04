"""Áudios sintéticos de consulta (fala do paciente) com degradações vocais controladas.

Usa o sintetizador `espeak-ng` (voz pt-BR) e aplica, por processamento de sinal:
  * fadiga/dificuldade respiratória: fala lenta, pausas longas, sons de respiração (inspirações audíveis);
  * disartria: instabilidade de frequência (jitter), de amplitude (shimmer), ruído (baixa HNR) e
    perda de nitidez articulatória (filtro passa-baixa).
Um arquivo `.txt` com a transcrição de referência é salvo ao lado de cada `.wav` (usado no modo offline).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

SR = 16000

SCRIPTS = {
    "P001": ("Bom dia, doutor. Estou me sentindo bem melhor hoje. Dormi bem a noite toda e a dor da cirurgia "
             "diminuiu bastante. Consegui caminhar pelo corredor com a enfermeira e já estou comendo normalmente."),
    "P002": ("Doutor, eu estou com muita falta de ar. Sinto um cansaço enorme, não consigo nem levantar da cama. "
             "Tive febre de madrugada e uma dor no peito quando respiro fundo. Estou ficando com tontura."),
    "P003": ("Doutora, o meu braço esquerdo continua fraco e com formigamento. Minha família disse que estou com a "
             "fala enrolada desde ontem. Às vezes sinto confusão para encontrar as palavras."),
}


def _espeak(text: str, speed: int, pitch: int = 45) -> np.ndarray:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise RuntimeError("espeak-ng não encontrado. Instale com: sudo apt-get install espeak-ng")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "s.wav"
        subprocess.run([exe, "-v", "pt-br", "-s", str(speed), "-p", str(pitch), "-w", str(out), text],
                       check=True, capture_output=True)
        y, sr = sf.read(out)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)


def _breath(rng, dur=0.7, level=0.06):
    n = int(dur * SR)
    noise = rng.normal(0, 1, n)
    b, a = butter(2, [300 / (SR / 2), 2500 / (SR / 2)], btype="band")
    x = lfilter(b, a, noise)
    env = np.sin(np.linspace(0, np.pi, n)) ** 1.5
    return (level * x / (np.abs(x).max() + 1e-9) * env).astype(np.float32)


def _split_sentences(text):
    return [s.strip() + "." for s in text.replace("?", ".").split(".") if s.strip()]


def _speak_with_pauses(text, speed, rng, pause_s, breath=False, pitch=45):
    parts = []
    for s in _split_sentences(text):
        parts.append(_espeak(s, speed, pitch))
        gap = np.zeros(int(pause_s * SR), np.float32)
        if breath:
            br = _breath(rng, dur=0.75, level=0.09)
            st = max(0, (len(gap) - len(br)) // 2)
            gap[st:st + len(br)] += br[: len(gap) - st]
        parts.append(gap)
    return np.concatenate(parts)


def _dysarthria(y, rng):
    n = len(y)
    # jitter: modulação de frequência aleatória via reamostragem com taxa variável (≈ ±4%)
    rate = 1 + 0.04 * np.convolve(rng.normal(0, 1, n), np.ones(80) / 80, mode="same") * 8
    pos = np.cumsum(rate); pos = pos / pos[-1] * (n - 1)
    y = np.interp(pos, np.arange(n), y).astype(np.float32)
    # shimmer: modulação de amplitude por ciclo (~8 ms)
    seg = int(0.008 * SR)
    amp = np.repeat(1 + 0.35 * rng.normal(0, 1, n // seg + 1), seg)[:n]
    y = y * np.clip(amp, 0.2, 2.0)
    # articulação "borrada": passa-baixa em 2.2 kHz
    b, a = butter(4, 2200 / (SR / 2), btype="low")
    y = lfilter(b, a, y)
    # soprosidade (reduz HNR)
    y = y + rng.normal(0, 0.02, n) * (np.abs(y) > 0.01)
    return y.astype(np.float32)


def generate_consultation_audio(pid: str, out_dir: Path, seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    text = SCRIPTS[pid]
    if pid == "P001":
        y = _speak_with_pauses(text, speed=165, rng=rng, pause_s=0.35)
    elif pid == "P002":
        y = _speak_with_pauses(text, speed=105, rng=rng, pause_s=1.4, breath=True, pitch=35) * 0.6
    else:
        y = _dysarthria(_speak_with_pauses(text, speed=115, rng=rng, pause_s=0.7, pitch=40), rng)
    y = y + rng.normal(0, 0.002, len(y)).astype(np.float32)  # ruído ambiente
    y = 0.9 * y / (np.abs(y).max() + 1e-9)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"consulta_{pid}.wav"
    sf.write(wav, y, SR)
    wav.with_suffix(".txt").write_text(text, encoding="utf-8")
    return wav

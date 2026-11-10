"""Biomarcadores vocais extraídos do sinal de áudio (librosa + numpy).

* Temporais: taxa de fala (sílabas/s), taxa de articulação, razão de pausas, pausas longas;
* Fonatórios: F0 médio/variabilidade, jitter (aprox. por quadro), shimmer, HNR (autocorrelação);
* Articulatórios: proporção de energia em alta frequência (> 2,5 kHz) — nitidez das consoantes;
* Respiratórios: eventos de respiração audível nos intervalos de silêncio (por minuto).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np
from scipy.signal import find_peaks

FRAME, HOP = 400, 160   # 25 ms / 10 ms @16 kHz


@dataclass
class VoiceFeatures:
    duration_s: float
    speech_time_s: float
    speech_rate_sps: float
    articulation_rate_sps: float
    pause_ratio: float
    long_pauses: int
    f0_mean_hz: float
    f0_std_st: float
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float
    hf_energy_ratio: float
    breath_events_per_min: float

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _segments(mask):
    segs, s = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and s is None:
            s = i
        elif not m and s is not None:
            segs.append((s, i)); s = None
    return segs


def extract(y: np.ndarray, sr: int) -> VoiceFeatures:
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr = 16000
    y = y.astype(np.float32)
    dur = len(y) / sr
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-9)
    floor = np.percentile(db, 10)
    energetic = db > max(floor + 15, db.max() - 45)
    fps = sr / HOP

    # F0 (pYIN) — usado também para separar fala (vozeada) de respiração/ruído
    f0, vflag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    n = min(len(f0), len(energetic))
    f0, vflag, energetic, db, rms = f0[:n], vflag[:n], energetic[:n], db[:n], rms[:n]
    near_voice = np.convolve(vflag.astype(float), np.ones(25), mode="same") > 0   # ±120 ms de um quadro vozeado
    speech = _open(_close(energetic & near_voice, 15), 6)
    speech_time = speech.sum() / fps
    idx = np.where(speech)[0]
    span = (idx[-1] - idx[0] + 1) / fps if len(idx) else dur
    pauses = [(e - s) / fps for s, e in _segments(~speech) if len(idx) and idx[0] < s and e < idx[-1]]
    pause_ratio = float(sum(pauses) / span) if span else 0.0
    long_pauses = int(sum(p > 0.6 for p in pauses))

    # sílabas: picos do envelope de energia na banda da voz (300–3000 Hz) dentro da fala
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP, win_length=FRAME)) ** 2
    S = S[:, :n]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    band = S[(freqs >= 300) & (freqs <= 3000)].sum(0)
    env = np.convolve(10 * np.log10(band + 1e-10), np.ones(5) / 5, mode="same")
    pk, _ = find_peaks(np.where(speech, env, env.min()), prominence=4, distance=int(0.09 * fps))
    n_syl = len(pk)
    speech_rate = n_syl / span if span else 0.0
    artic_rate = n_syl / speech_time if speech_time else 0.0

    hf = S[freqs > 2500].sum(0)
    tot = S.sum(0) + 1e-12
    hf_ratio = float(np.mean(hf[speech] / tot[speech])) if speech.any() else 0.0

    vflag = vflag & speech
    fv = f0[vflag]
    f0_mean = float(np.nanmean(fv)) if len(fv) else 0.0
    f0_std_st = float(np.nanstd(12 * np.log2(fv / np.nanmedian(fv)))) if len(fv) > 2 else 0.0
    # jitter/shimmer: perturbação de curto prazo = resíduo em relação à tendência local (média móvel de 5 quadros)
    rms_short = librosa.feature.rms(y=y, frame_length=160, hop_length=HOP)[0][:n]
    jit, shim = [], []
    for s, e in _segments(vflag):
        if e - s < 8:
            continue
        T = 1.0 / f0[s:e]
        Tt = np.convolve(T, np.ones(5) / 5, mode="valid")
        jit.append(np.mean(np.abs(T[2:-2] - Tt)) / np.mean(T))
        A = rms_short[s:e]
        At = np.convolve(A, np.ones(5) / 5, mode="valid")
        shim.append(np.mean(np.abs(A[2:-2] - At)) / (np.mean(A) + 1e-9))
    jitter = 100 * float(np.median(jit)) if jit else 0.0
    shimmer = 100 * float(np.median(shim)) if shim else 0.0
    hnr = []
    frames = librosa.util.frame(np.pad(y, (0, 1024)), frame_length=1024, hop_length=HOP)
    for i in np.where(vflag)[0][::3]:
        fr = frames[:, i] * np.hanning(1024)
        ac = np.correlate(fr, fr, "full")[1023:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lo, hi = int(sr / 400), int(sr / 60)
        r = float(np.clip(ac[lo:hi].max(), 1e-4, 0.9999))
        hnr.append(10 * np.log10(r / (1 - r)))
    hnr_db = float(np.median(hnr)) if hnr else 0.0

    # respiração audível: energia acima do ruído de fundo nos silêncios, ruído de banda larga, > 250 ms
    flat = librosa.feature.spectral_flatness(S=S)[0]
    breath = (~speech) & (~near_voice) & (db > floor + 10) & (flat > 0.001)
    breath = _open(_close(breath, 5), 5)
    n_breath = sum(1 for s, e in _segments(breath) if (e - s) / fps > 0.25)
    breath_pm = n_breath / (dur / 60)

    return VoiceFeatures(dur, float(speech_time), speech_rate, artic_rate, pause_ratio, long_pauses, f0_mean, f0_std_st,
                         jitter, shimmer, hnr_db, hf_ratio, breath_pm)


def _close(m, k):
    """Fechamento morfológico 1D (preenche lacunas < k)."""
    m = m.copy()
    for s, e in _segments(~m):
        if e - s < k and s > 0 and e < len(m):
            m[s:e] = True
    return m


def _open(m, k):
    """Abertura morfológica 1D (remove trechos < k)."""
    m = m.copy()
    for s, e in _segments(m):
        if e - s < k:
            m[s:e] = False
    return m

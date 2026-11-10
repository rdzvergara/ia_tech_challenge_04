"""Séries temporais sintéticas de sinais vitais (1 amostra/min) com anomalias injetadas."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _base(n: int, rng: np.random.Generator, hr=75, sbp=120, dbp=78, spo2=97, rr=15, temp=36.7):
    t = np.arange(n)
    circ = np.sin(2 * np.pi * t / (24 * 60))  # ritmo circadiano
    ar = np.zeros(n)
    for i in range(1, n):  # ruído AR(1) para dar realismo
        ar[i] = 0.9 * ar[i - 1] + rng.normal(0, 1)
    ar /= ar.std() + 1e-9
    return {
        "hr": hr + 4 * circ + 2.0 * ar + rng.normal(0, 1.2, n),
        "sbp": sbp + 5 * circ + 3.0 * ar + rng.normal(0, 2.0, n),
        "dbp": dbp + 3 * circ + 2.0 * ar + rng.normal(0, 1.5, n),
        "spo2": spo2 + 0.5 * ar * 0.5 + rng.normal(0, 0.5, n),
        "rr": rr + 0.6 * ar + rng.normal(0, 0.6, n),
        "temp": temp + 0.25 * circ + 0.05 * ar + rng.normal(0, 0.03, n),
    }


def _ramp(n, start, dur, amount):
    x = np.zeros(n)
    end = min(n, start + dur)
    x[start:end] = np.linspace(0, amount, end - start)
    x[end:] = amount
    return x


def generate_vitals(hours: int = 48, seed: int = 42, start="2026-11-08 00:00") -> tuple[pd.DataFrame, list[dict]]:
    """Retorna (dataframe, ground_truth). Ground truth lista os eventos injetados."""
    rng = np.random.default_rng(seed)
    n = hours * 60
    ts = pd.date_range(start, periods=n, freq="min")
    frames, truth = [], []

    # P001 — estável. Apenas um artefato de sensor (1 amostra) que NÃO deve virar alerta.
    v = _base(n, rng)
    v["spo2"][600] = 72.0
    truth.append({"patient_id": "P001", "event": "artefato_sensor_spo2", "minute": 600, "should_alert": False})
    frames.append(pd.DataFrame({"timestamp": ts, "patient_id": "P001", **v}))

    # P002 — deterioração séptica progressiva a partir de 30h (taquicardia, hipotensão, febre, taquipneia, dessaturação)
    v = _base(n, rng, hr=82, sbp=124, spo2=96, rr=17, temp=37.0)
    s = 30 * 60
    v["hr"] += _ramp(n, s, 8 * 60, 48)
    v["sbp"] += _ramp(n, s + 120, 8 * 60, -42)
    v["dbp"] += _ramp(n, s + 120, 8 * 60, -22)
    v["rr"] += _ramp(n, s, 8 * 60, 13)
    v["temp"] += _ramp(n, s - 120, 6 * 60, 2.4)
    v["spo2"] += _ramp(n, s + 240, 6 * 60, -9)
    truth.append({"patient_id": "P002", "event": "deterioracao_septica", "minute": s - 120, "should_alert": True})  # início da febre
    frames.append(pd.DataFrame({"timestamp": ts, "patient_id": "P002", **v}))

    # P003 — pico hipertensivo sustentado (40 min) às 20h + episódio de bradicardia (15 min) às 40h
    v = _base(n, rng, hr=70, sbp=135, dbp=85)
    a = 20 * 60
    v["sbp"][a:a + 40] += 62 + rng.normal(0, 3, 40)
    v["dbp"][a:a + 40] += 25 + rng.normal(0, 2, 40)
    b = 40 * 60
    v["hr"][b:b + 15] -= 33 + rng.normal(0, 1, 15)
    truth.append({"patient_id": "P003", "event": "crise_hipertensiva", "minute": a, "should_alert": True})
    truth.append({"patient_id": "P003", "event": "bradicardia", "minute": b, "should_alert": True})
    frames.append(pd.DataFrame({"timestamp": ts, "patient_id": "P003", **v}))

    df = pd.concat(frames, ignore_index=True)
    df["spo2"] = df["spo2"].clip(upper=100)
    for c in ("hr", "sbp", "dbp", "rr"):
        df[c] = df[c].round(0)
    df["spo2"] = df["spo2"].round(0)
    df["temp"] = df["temp"].round(1)
    return df, truth

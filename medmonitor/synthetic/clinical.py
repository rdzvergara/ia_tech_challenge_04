"""Prescrições e padrões de movimentação (sensor de leito/acelerômetro) sintéticos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_prescriptions(start="2026-11-08 08:00") -> tuple[pd.DataFrame, list[dict]]:
    t0 = pd.Timestamp(start)
    rows = []

    def add(pid, h, drug, dose, unit, freq, route, prescriber="Dr. Silva"):
        rows.append(dict(timestamp=t0 + pd.Timedelta(hours=h), patient_id=pid, drug=drug, dose=dose,
                         unit=unit, doses_per_day=freq, route=route, prescriber=prescriber))

    # P001 — prescrição estável
    for h in (0, 24):
        add("P001", h, "dipirona", 1000, "mg", 4, "IV")
        add("P001", h, "enoxaparina", 40, "mg", 1, "SC")
        add("P001", h, "paracetamol", 750, "mg", 3, "VO")

    # P002 — erro de casa decimal na enoxaparina (40 -> 400 mg) + vancomicina iniciada + morfina dobrando
    add("P002", 0, "enoxaparina", 40, "mg", 1, "SC")
    add("P002", 0, "dipirona", 1000, "mg", 4, "IV")
    add("P002", 0, "morfina", 4, "mg", 6, "IV")
    add("P002", 24, "enoxaparina", 400, "mg", 1, "SC", "Dr. Costa")          # anomalia
    add("P002", 24, "dipirona", 1000, "mg", 4, "IV")
    add("P002", 24, "morfina", 10, "mg", 6, "IV", "Dr. Costa")               # aumento > 2x em droga de alta vigilância
    add("P002", 30, "vancomicina", 1000, "mg", 2, "IV", "Dr. Costa")

    # P003 — metoprolol suspenso abruptamente (betabloqueador) + insulina com dose acima do limite
    add("P003", 0, "metoprolol", 50, "mg", 2, "VO")
    add("P003", 0, "insulina_regular", 10, "UI", 3, "SC")
    add("P003", 24, "insulina_regular", 60, "UI", 3, "SC", "Dr. Lima")      # 180 UI/dia — acima do limite
    # metoprolol não reaparece na prescrição de 24h -> suspensão abrupta

    truth = [
        {"patient_id": "P002", "event": "dose_10x_enoxaparina", "should_alert": True},
        {"patient_id": "P002", "event": "morfina_aumento_2.5x", "should_alert": True},
        {"patient_id": "P003", "event": "insulina_acima_limite", "should_alert": True},
        {"patient_id": "P003", "event": "suspensao_abrupta_betabloqueador", "should_alert": True},
    ]
    return pd.DataFrame(rows), truth


def generate_mobility(hours: int = 48, seed: int = 7, start="2026-11-08 00:00") -> tuple[pd.DataFrame, list[dict]]:
    """Índice de atividade (0..1, por minuto, de acelerômetro/sensor de leito) e flag de paciente fora do leito."""
    rng = np.random.default_rng(seed)
    n = hours * 60
    ts = pd.date_range(start, periods=n, freq="min")
    hour = ts.hour.values
    day = ((hour >= 7) & (hour < 22)).astype(float)
    frames, truth = [], []

    def mk(pid, act, out):
        frames.append(pd.DataFrame({"timestamp": ts, "patient_id": pid,
                                    "activity": np.clip(act, 0, 1).round(3), "out_of_bed": out.astype(int)}))

    # P001 — padrão normal: ativo de dia, dorme à noite
    act = 0.05 + 0.25 * day + rng.gamma(1.0, 0.05, n)
    out = (rng.random(n) < 0.04 * day)
    mk("P001", act, out)

    # P002 — hipoatividade progressiva (letargia) a partir de 26h + agitação noturna com saídas do leito na 2ª noite
    act = 0.05 + 0.22 * day + rng.gamma(1.0, 0.05, n)
    act[26 * 60:] *= 0.25
    out = (rng.random(n) < 0.03 * day)
    out[26 * 60:] = False
    for m in (47 * 60 + 5, 47 * 60 + 18, 47 * 60 + 31, 47 * 60 + 47):  # 23h05–23h50 da 2ª noite (delirium?)
        out[m:m + 3] = True
        act[m:m + 3] = 0.8
    truth.append({"patient_id": "P002", "event": "hipoatividade_letargia", "minute": 26 * 60, "should_alert": True})
    truth.append({"patient_id": "P002", "event": "agitacao_noturna", "minute": 47 * 60, "should_alert": True})
    mk("P002", act, out)

    # P003 — queda: pico de aceleração seguido de imobilidade fora do leito às 15h do 2º dia
    act = 0.05 + 0.18 * day + rng.gamma(1.0, 0.05, n)
    out = (rng.random(n) < 0.03 * day)
    f = 39 * 60
    act[f] = 1.0
    act[f + 1:f + 25] = 0.0
    out[f:f + 25] = True
    truth.append({"patient_id": "P003", "event": "queda", "minute": f, "should_alert": True})
    mk("P003", act, out)

    return pd.concat(frames, ignore_index=True), truth

"""Avaliação contra o ground truth dos dados sintéticos (detecção, falsos positivos e latência)."""
from __future__ import annotations

import pandas as pd

START = pd.Timestamp("2026-11-08 00:00")

# evento do ground truth -> (modalidade, tipos de achado que contam como detecção, filtro por droga)
MAP = {
    "artefato_sensor_spo2": ("vitals", {"spo2_low", "spo2_abrupt_change", "multivariate_anomaly"}, None),
    "deterioracao_septica": ("vitals", None, None),
    "crise_hipertensiva": ("vitals", {"sbp_high", "sbp_abrupt_change", "dbp_abrupt_change"}, None),
    "bradicardia": ("vitals", {"hr_low", "hr_abrupt_change"}, None),
    "dose_10x_enoxaparina": ("prescriptions", {"dose_jump", "dose_above_max"}, "enoxaparina"),
    "morfina_aumento_2.5x": ("prescriptions", {"dose_jump"}, "morfina"),
    "insulina_acima_limite": ("prescriptions", {"dose_above_max"}, "insulina_regular"),
    "suspensao_abrupta_betabloqueador": ("prescriptions", {"abrupt_discontinuation"}, "metoprolol"),
    "hipoatividade_letargia": ("mobility", {"hypoactivity", "atypical_mobility_pattern"}, None),
    "agitacao_noturna": ("mobility", {"night_agitation"}, None),
    "queda": ("mobility", {"fall_suspected"}, None),
    "sessao_normal": ("video", None, None),
    "rom_reduzida_esquerda+assimetria+compensacao_tronco+tremor+queda": ("video", None, None),
}

AUDIO_TRUTH = [
    {"patient_id": "P001", "event": "fala_normal", "kinds": None, "should_alert": False},
    {"patient_id": "P002", "event": "fadiga+dificuldade_respiratoria+termos_criticos",
     "kinds": {"voice_fadiga", "voice_respiratorio", "critical_terms"}, "should_alert": True},
    {"patient_id": "P003", "event": "disartria+termos_criticos", "kinds": {"voice_disartria", "critical_terms"},
     "should_alert": True},
]


def evaluate(findings, truth: dict) -> pd.DataFrame:
    rows = []
    events = [dict(e, modality_key=k) for k, lst in truth.items() for e in lst]
    for e in events:
        mod, kinds, drug = MAP[e["event"]]
        cand = [f for f in findings if f.patient_id == e["patient_id"] and f.modality == mod
                and (kinds is None or f.kind in kinds)
                and (drug is None or f.evidence.get("droga") == drug)]
        latency = None
        if "minute" in e:
            t_ev = START + pd.Timedelta(minutes=e["minute"])
            win = 15 if not e["should_alert"] else 24 * 60
            cand = [f for f in cand if t_ev - pd.Timedelta(minutes=1) <= pd.Timestamp(f.timestamp) <= t_ev + pd.Timedelta(minutes=win)]
            if cand:
                latency = (min(pd.Timestamp(f.timestamp) for f in cand) - t_ev).total_seconds() / 60
        detected = len(cand) > 0
        rows.append({"paciente": e["patient_id"], "modalidade": mod, "evento": e["event"],
                     "esperado_alerta": e["should_alert"], "detectado": detected,
                     "correto": detected == e["should_alert"],
                     "latencia_min": None if latency is None else round(latency, 1),
                     "achados": len(cand)})
    for e in AUDIO_TRUTH:
        cand = [f for f in findings if f.patient_id == e["patient_id"] and f.modality == "audio"
                and (e["kinds"] is None or f.kind in e["kinds"])]
        kinds_found = {f.kind for f in cand}
        detected = (kinds_found >= e["kinds"]) if e["kinds"] else bool(cand)
        rows.append({"paciente": e["patient_id"], "modalidade": "audio", "evento": e["event"],
                     "esperado_alerta": e["should_alert"], "detectado": detected,
                     "correto": detected == e["should_alert"], "latencia_min": None, "achados": len(cand)})
    return pd.DataFrame(rows)

"""Fusão multimodal tardia (late fusion) e escalonamento de risco por paciente.

Cada modalidade produz achados (Finding) com score 0..1. Para cada paciente, numa janela deslizante:

    m_mod  = min(1, max(score) + 0.05 · (nº de tipos distintos − 1))        # força da evidência por modalidade
    risco  = 0.6 · max_mod(m_mod) + 0.4 · Σ w_mod · m_mod + bônus · (nº modalidades com m ≥ 0,5 − 1)

* o termo ``max`` garante que um evento grave isolado (ex.: queda) nunca seja diluído;
* a média ponderada e o bônus de corroboração elevam o risco quando fontes independentes concordam
  (ex.: SpO2 caindo + paciente relatando falta de ar + respiração audível na consulta).

Um alerta é emitido quando o nível de risco do paciente sobe (escalonamento) ou quando, no mesmo nível,
surge evidência de uma nova modalidade após o período de cooldown.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .schemas import Alert, Finding, Severity

ACTIONS = [
    ({"news2_high", "sbp_low", "temp_high", "hr_high", "rr_high"},
     "Suspeita de deterioração clínica/sepse: avaliar à beira-leito, colher lactato e hemoculturas e "
     "considerar protocolo de sepse (antibiótico na 1ª hora)."),
    ({"spo2_low", "spo2_trend_down", "voice_respiratorio"},
     "Dessaturação/dificuldade respiratória: avaliar via aérea, ofertar O2 conforme protocolo, gasometria."),
    ({"dose_above_max", "dose_jump"},
     "Possível erro de prescrição em medicamento de alta vigilância: SUSPENDER administração e revisar com "
     "farmácia clínica e prescritor."),
    ({"abrupt_discontinuation"}, "Revisar suspensão do medicamento (necessidade de desmame) com o prescritor."),
    ({"fall_detected", "fall_suspected"},
     "Queda: avaliar trauma e estado neurológico, aplicar protocolo pós-queda e reforçar prevenção."),
    ({"voice_disartria", "tremor_l", "reduced_rom_l", "asymmetry"},
     "Sinais neurológicos (disartria/assimetria motora): aplicar NIHSS e considerar acionamento do protocolo AVC."),
    ({"sbp_high"}, "Crise hipertensiva: confirmar aferição manual, avaliar lesão de órgão-alvo."),
    ({"hr_low"}, "Bradicardia: ECG de 12 derivações, revisar medicações cronotrópicas negativas."),
    ({"night_agitation", "hypoactivity"}, "Alteração de comportamento motor: rastrear delirium (CAM-ICU)."),
]

ESCALATION_TEXT = {
    Severity.CRITICAL: "Acionar Time de Resposta Rápida imediatamente.",
    Severity.HIGH: "Avaliação médica em até 30 minutos.",
    Severity.MEDIUM: "Reavaliação pela enfermagem na próxima hora; manter vigilância.",
    Severity.LOW: "Registrar e acompanhar.",
    Severity.INFO: "Sem ação necessária.",
}

MODALITY_PT = {"vitals": "sinais vitais", "audio": "áudio", "video": "vídeo", "prescriptions": "prescrições",
               "mobility": "movimentação"}


class RiskFusionEngine:
    def __init__(self, cfg: dict, window_hours: float = 12, cooldown_hours: float = 1, reminder_hours: float = 4):
        f = cfg["fusion"]
        self.w = f["weights"]
        self.esc = f["escalation"]
        self.bonus = f["corroboration_bonus"]
        self.window = pd.Timedelta(hours=window_hours)
        self.cooldown = pd.Timedelta(hours=cooldown_hours)
        self.reminder = pd.Timedelta(hours=reminder_hours)
        self.buffer: dict[str, list[Finding]] = defaultdict(list)
        self.last: dict[str, tuple[Severity, pd.Timestamp, set]] = {}
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def modality_scores(self, findings: list[Finding]) -> dict[str, float]:
        by = defaultdict(list)
        for f in findings:
            by[f.modality].append(f)
        return {m: min(1.0, max(x.score for x in fs) + 0.05 * (len({x.kind for x in fs}) - 1)) for m, fs in by.items()}

    def risk(self, findings: list[Finding]) -> tuple[float, dict]:
        ms = self.modality_scores(findings)
        if not ms:
            return 0.0, ms
        wmean = sum(self.w.get(m, 0.1) * s for m, s in ms.items())
        strong = sum(1 for s in ms.values() if s >= 0.5)
        r = 0.6 * max(ms.values()) + 0.4 * wmean + self.bonus * max(0, strong - 1)
        return min(1.0, r), ms

    def severity(self, r: float) -> Severity:
        return Severity.from_score(r, self.esc["critical"], self.esc["high"], self.esc["medium"])

    # ------------------------------------------------------------------
    def ingest(self, f: Finding) -> Alert | None:
        ts = pd.Timestamp(f.timestamp)
        buf = self.buffer[f.patient_id]
        buf.append(f)
        buf[:] = [x for x in buf if ts - pd.Timestamp(x.timestamp) <= self.window]
        r, ms = self.risk(buf)
        sev = self.severity(r)
        self.history.append({"timestamp": ts, "patient_id": f.patient_id, "risk": r, "severity": sev.name, **ms})
        if sev < Severity.MEDIUM:
            return None
        prev = self.last.get(f.patient_id)
        mods = set(ms)
        # emite se: 1º alerta | escalonou | nova modalidade corroborando (após cooldown) | lembrete periódico
        emit = prev is None or sev > prev[0] or \
            (ts - prev[1] >= self.cooldown and bool(mods - prev[2])) or \
            (ts - prev[1] >= self.reminder and sev >= prev[0])
        if not emit:
            return None
        self.last[f.patient_id] = (sev, ts, mods | (prev[2] if prev else set()))
        return self.build_alert(f.patient_id, buf, r, sev, ms, ts)

    def build_alert(self, pid, findings, r, sev, ms, ts=None) -> Alert:
        top = sorted({(x.kind): x for x in findings}.values(), key=lambda x: -x.score)[:6]
        kinds = {x.kind for x in findings}
        actions = [a for keys, a in ACTIONS if keys & kinds]
        mods = sorted(ms, key=lambda m: -ms[m])
        title = f"[{sev.name}] Paciente {pid} — risco {r:.2f} ({', '.join(MODALITY_PT[m] for m in mods)})"
        summary = "; ".join(x.description for x in top[:4])
        al = Alert(pid, sev, r, title, summary, ESCALATION_TEXT[sev] + (" " + " ".join(actions[:3]) if actions else ""),
                   top, mods)
        if ts is not None:
            al.created_at = pd.Timestamp(ts).isoformat()
        return al

    def run(self, findings: list[Finding]) -> list[Alert]:
        alerts = []
        for f in sorted(findings, key=lambda x: pd.Timestamp(x.timestamp)):
            a = self.ingest(f)
            if a:
                alerts.append(a)
        return alerts

    def patient_summary(self, findings: list[Finding], patients=()) -> dict[str, dict]:
        out = {}
        by = defaultdict(list, {p: [] for p in patients})
        for f in findings:
            by[f.patient_id].append(f)
        for pid, fs in by.items():
            r, ms = self.risk(fs)
            out[pid] = {"risk": round(r, 3), "severity": self.severity(r).name,
                        "modalities": {m: round(s, 3) for m, s in ms.items()}, "n_findings": len(fs)}
        return out

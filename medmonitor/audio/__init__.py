"""Análise de áudio de consultas: biomarcadores vocais + Azure Speech to Text + Azure Text Analytics."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa

from ..schemas import Finding
from .azure_speech import Transcript, transcribe
from .azure_text import TextInsights, analyze_text
from .features import VoiceFeatures, extract


@dataclass
class AudioReport:
    patient_id: str
    file: str
    features: VoiceFeatures
    transcript: Transcript
    insights: TextInsights
    marker_flags: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self):
        return {"patient_id": self.patient_id, "file": self.file, "features": self.features.to_dict(),
                "transcript": asdict(self.transcript), "insights": asdict(self.insights),
                "marker_flags": self.marker_flags, "findings": [f.to_dict() for f in self.findings]}


def _ratio_over(v, thr):   # quanto passou do limite superior (0..1)
    return max(0.0, min(1.0, (v - thr) / (abs(thr) + 1e-9)))


def _ratio_under(v, thr):  # quanto ficou abaixo do limite inferior (0..1)
    return max(0.0, min(1.0, (thr - v) / (abs(thr) + 1e-9)))


def score_voice(f: VoiceFeatures, cfg: dict) -> dict:
    fa, dy, re_ = cfg["fatigue"], cfg["dysarthria"], cfg["respiratory"]
    flags = {
        "fadiga": {
            "fala_lenta": _ratio_under(f.speech_rate_sps, fa["min_speech_rate_sps"]),
            "pausas_excessivas": _ratio_over(f.pause_ratio, fa["max_pause_ratio"]),
        },
        "disartria": {
            "jitter_elevado": _ratio_over(f.jitter_pct, dy["max_jitter_pct"]),
            "shimmer_elevado": _ratio_over(f.shimmer_pct, dy["max_shimmer_pct"]),
            "hnr_baixo": _ratio_under(f.hnr_db, dy["min_hnr_db"]),
            "articulacao_imprecisa": _ratio_under(f.hf_energy_ratio, dy["min_hf_energy_ratio"]),
        },
        "respiratorio": {
            "respiracao_audivel": _ratio_over(f.breath_events_per_min, re_["max_breath_events_per_min"]),
        },
    }
    return flags


def analyze_audio(wav: Path, patient_id: str, settings: dict) -> AudioReport:
    cfg = settings["audio"]
    y, sr = librosa.load(str(wav), sr=cfg.get("sample_rate", 16000))
    feats = extract(y, sr)
    tr = transcribe(Path(wav))
    ins = analyze_text(tr.text, cfg["critical_terms"], tr.language)
    flags = score_voice(feats, cfg)
    rep = AudioReport(patient_id, str(wav), feats, tr, ins, flags)

    labels = {"fadiga": "Sinais vocais de fadiga (fala lentificada / pausas longas)",
              "disartria": "Sinais vocais compatíveis com disartria (instabilidade fonatória / articulação imprecisa)",
              "respiratorio": "Dificuldade respiratória audível (inspirações frequentes entre frases)"}
    for group, markers in flags.items():
        hits = {k: v for k, v in markers.items() if v > 0}
        if not hits:
            continue
        frac = len(hits) / len(markers)
        if group != "respiratorio" and frac < 0.5:
            continue   # exige concordância de pelo menos metade dos marcadores do grupo
        score = min(0.8, 0.25 + 0.25 * frac + 0.5 * float(max(hits.values())))
        rep.findings.append(Finding(patient_id, "audio", f"voice_{group}", score, labels[group],
                                    evidence={"marcadores": {k: round(v, 2) for k, v in hits.items()},
                                              **{k: v for k, v in feats.to_dict().items() if k != "duration_s"}},
                                    source="biomarcadores acústicos (librosa/pYIN)"))
    if ins.critical_terms:
        score = min(0.8, 0.3 + 0.12 * len(ins.critical_terms) + (0.1 if ins.sentiment == "negative" else 0))
        rep.findings.append(Finding(patient_id, "audio", "critical_terms",
                                    score, "Termos clínicos críticos na fala: " + ", ".join(ins.critical_terms),
                                    evidence={"termos": ins.critical_terms, "sentimento": ins.sentiment,
                                              "sentiment_scores": ins.sentiment_scores,
                                              "entidades_saude": ins.health_entities[:15]},
                                    source=f"{tr.engine} + {ins.engine}"))
    elif ins.sentiment == "negative":
        rep.findings.append(Finding(patient_id, "audio", "negative_sentiment", 0.25,
                                    "Sentimento negativo predominante no relato do paciente",
                                    evidence={"sentiment_scores": ins.sentiment_scores}, source=ins.engine))
    return rep


def save_audio_report(rep: AudioReport, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"audio_{rep.patient_id}.json").write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2,
                                                                     default=float), encoding="utf-8")

"""Estruturas de dados compartilhadas entre as modalidades."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def from_score(cls, score: float, critical=0.75, high=0.5, medium=0.3) -> "Severity":
        if score >= critical:
            return cls.CRITICAL
        if score >= high:
            return cls.HIGH
        if score >= medium:
            return cls.MEDIUM
        if score > 0.1:
            return cls.LOW
        return cls.INFO


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Finding:
    """Achado (anomalia/sinal de risco) produzido por um analisador de uma modalidade."""

    patient_id: str
    modality: str            # vitals | audio | video | prescriptions | mobility
    kind: str                # ex.: "spo2_low", "dysarthria", "trunk_compensation"
    score: float             # 0..1 — intensidade/confiança do achado
    description: str
    timestamp: str = field(default_factory=_now)
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = ""         # modelo/técnica que gerou o achado

    @property
    def severity(self) -> Severity:
        return Severity.from_score(self.score)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return d


@dataclass
class Alert:
    """Alerta consolidado (pós-fusão) enviado à equipe médica."""

    patient_id: str
    severity: Severity
    risk_score: float
    title: str
    summary: str
    recommended_action: str
    findings: list[Finding] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "created_at": self.created_at,
            "patient_id": self.patient_id,
            "severity": self.severity.name,
            "risk_score": round(self.risk_score, 3),
            "title": self.title,
            "summary": self.summary,
            "recommended_action": self.recommended_action,
            "modalities": self.modalities,
            "findings": [f.to_dict() for f in self.findings],
        }

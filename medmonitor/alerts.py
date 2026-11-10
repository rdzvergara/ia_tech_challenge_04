"""Despacho de alertas para a equipe médica.

Canais:
  * ``console`` — saída colorida no terminal (central de monitoramento);
  * ``jsonl``   — trilha de auditoria (um alerta por linha), consumida pelo dashboard;
  * ``webhook`` — Microsoft Teams (Incoming Webhook / Adaptive Card), Slack ou Azure Logic Apps,
                  configurado em ``ALERT_WEBHOOK_URL``. Em produção, a Logic App pode encaminhar para
                  SMS/pager/prontuário (HL7 FHIR ``Flag``/``Communication``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import env, resolve
from .schemas import Alert, Severity

log = logging.getLogger(__name__)

COLORS = {Severity.CRITICAL: "\033[1;97;41m", Severity.HIGH: "\033[1;31m", Severity.MEDIUM: "\033[33m",
          Severity.LOW: "\033[36m", Severity.INFO: "\033[37m"}
RESET = "\033[0m"


def teams_card(a: Alert) -> dict:
    color = {"CRITICAL": "attention", "HIGH": "warning", "MEDIUM": "accent"}.get(a.severity.name, "default")
    facts = [{"title": f.modality, "value": f.description[:180]} for f in a.findings[:5]]
    return {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json", "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {"type": "TextBlock", "text": a.title, "weight": "Bolder", "size": "Medium", "color": color,
                     "wrap": True},
                    {"type": "TextBlock", "text": f"Ação recomendada: {a.recommended_action}", "wrap": True},
                    {"type": "FactSet", "facts": facts},
                    {"type": "TextBlock", "text": f"Alerta {a.alert_id} • {a.created_at}", "isSubtle": True,
                     "size": "Small"},
                ],
            },
        }],
    }


class AlertManager:
    def __init__(self, cfg: dict, quiet: bool = False):
        c = cfg["alerts"]
        self.channels = c["channels"]
        self.jsonl = resolve(c["jsonl_path"])
        self.webhook = env("ALERT_WEBHOOK_URL")
        self.quiet = quiet
        self.sent: list[Alert] = []

    def reset(self):
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl.write_text("", encoding="utf-8")

    def dispatch(self, a: Alert):
        self.sent.append(a)
        if "console" in self.channels and not self.quiet:
            print(f"{COLORS[a.severity]} {a.severity.name:8s} {RESET} {a.created_at}  {a.title}")
            print(f"           ↳ {a.summary[:220]}")
            print(f"           ↳ AÇÃO: {a.recommended_action[:220]}")
        if "jsonl" in self.channels:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(a.to_dict(), ensure_ascii=False, default=float) + "\n")
        if "webhook" in self.channels and self.webhook and a.severity >= Severity.HIGH:
            try:
                import requests
                requests.post(self.webhook, json=teams_card(a), timeout=10).raise_for_status()
            except Exception as e:  # noqa: BLE001
                log.warning("Falha ao enviar webhook: %s", e)

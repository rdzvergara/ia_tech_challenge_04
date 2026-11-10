"""Análise do texto transcrito — Azure AI Language (Text Analytics) com fallback offline.

Azure: análise de sentimento (com opinion mining), extração de frases-chave e
Text Analytics for Health (entidades clínicas: sintomas, partes do corpo, medicamentos, negação).
Offline: léxico clínico/sentimento em português (regras) com a mesma interface de saída.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from ..config import env, offline_forced

log = logging.getLogger(__name__)


@dataclass
class TextInsights:
    engine: str
    sentiment: str                       # positive | neutral | negative | mixed
    sentiment_scores: dict
    key_phrases: list[str] = field(default_factory=list)
    health_entities: list[dict] = field(default_factory=list)   # {text, category, negated}
    critical_terms: list[str] = field(default_factory=list)


def azure_available() -> bool:
    return bool(env("AZURE_LANGUAGE_KEY") and env("AZURE_LANGUAGE_ENDPOINT")) and not offline_forced()


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def find_critical_terms(text: str, terms: list[str]) -> list[str]:
    t = _strip(text)
    return [term for term in terms if _strip(term) in t]


def analyze_azure(text: str, critical_terms: list[str], language="pt-BR") -> TextInsights:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    client = TextAnalyticsClient(env("AZURE_LANGUAGE_ENDPOINT"), AzureKeyCredential(env("AZURE_LANGUAGE_KEY")))
    lang = language.split("-")[0] if language else "pt"
    docs = [{"id": "1", "language": "pt-BR" if lang == "pt" else lang, "text": text}]
    sent = client.analyze_sentiment(docs, show_opinion_mining=True)[0]
    kp = client.extract_key_phrases(docs)[0]
    health = []
    try:
        poller = client.begin_analyze_healthcare_entities([{"id": "1", "language": "pt", "text": text}])
        for doc in poller.result():
            if doc.is_error:
                continue
            for ent in doc.entities:
                negated = bool(ent.assertion and getattr(ent.assertion, "certainty", None) in ("negative",
                                                                                                "negativePossible"))
                health.append({"text": ent.text, "category": ent.category, "negated": negated,
                               "confidence": round(ent.confidence_score, 2)})
    except Exception as e:  # noqa: BLE001 — TA4H pode não estar habilitado no recurso/região
        log.warning("Text Analytics for Health indisponível: %s", e)
    return TextInsights(
        engine="azure-language",
        sentiment=sent.sentiment,
        sentiment_scores={"positive": sent.confidence_scores.positive, "neutral": sent.confidence_scores.neutral,
                          "negative": sent.confidence_scores.negative},
        key_phrases=list(kp.key_phrases) if not kp.is_error else [],
        health_entities=health,
        critical_terms=find_critical_terms(text, critical_terms),
    )


# ------------------------------------------------------------------ fallback offline (léxico)
_HEALTH_LEXICON = {
    "SymptomOrSign": ["falta de ar", "cansaco", "febre", "dor no peito", "dor", "tontura", "formigamento",
                      "fraco", "fraqueza", "fala enrolada", "confusao", "desmaio", "tosse", "nausea", "sangramento"],
    "BodyStructure": ["braco esquerdo", "braco direito", "peito", "cabeca", "perna", "cirurgia"],
    "Diagnosis": ["pneumonia", "avc", "sepse", "infarto"],
}
_NEG = ["nao", "nem", "sem", "nunca"]
_POS = ["melhor", "bem", "normalmente", "diminuiu", "consegui", "otimo", "tranquilo"]
_NEG_SENT = ["muita", "enorme", "nao consigo", "piorou", "pior", "fraco", "dor", "falta de ar", "tontura",
             "confusao", "febre", "cansaco", "formigamento", "enrolada"]


def analyze_offline(text: str, critical_terms: list[str]) -> TextInsights:
    t = _strip(text)
    pos = sum(len(re.findall(rf"\b{w}\b", t)) for w in _POS)
    neg = sum(len(re.findall(rf"\b{w}\b", t)) for w in _NEG_SENT)
    tot = pos + neg + 1
    scores = {"positive": round(pos / tot, 2), "negative": round(neg / tot, 2), "neutral": round(1 / tot, 2)}
    sentiment = "positive" if pos > neg * 1.5 else "negative" if neg > pos * 1.5 else "mixed" if pos and neg else "neutral"
    ents = []
    for cat, words in _HEALTH_LEXICON.items():
        for w in sorted(words, key=len, reverse=True):
            for m in re.finditer(rf"\b{w}\b", t):
                if any(e["start"] <= m.start() < e["end"] for e in ents):
                    continue
                window = t[max(0, m.start() - 25):m.start()]
                negated = any(re.search(rf"\b{n}\b", window) for n in ("sem", "nega", "nenhum"))
                ents.append({"text": w, "category": cat, "negated": negated, "start": m.start(), "end": m.end()})
    for e in ents:
        e.pop("start"); e.pop("end")
    kp = list(dict.fromkeys(e["text"] for e in ents))
    return TextInsights("lexico-offline", sentiment, scores, kp, ents, find_critical_terms(text, critical_terms))


def analyze_text(text: str, critical_terms: list[str], language="pt-BR") -> TextInsights:
    if azure_available() and text:
        try:
            return analyze_azure(text, critical_terms, language)
        except Exception as e:  # noqa: BLE001
            log.warning("Falha no Azure Language (%s) — usando fallback offline", e)
    return analyze_offline(text, critical_terms)

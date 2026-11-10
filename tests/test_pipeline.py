"""Testes automatizados (pytest). Executar a partir da raiz: ``pytest -q``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medmonitor.anomaly.mobility import analyze_mobility  # noqa: E402
from medmonitor.anomaly.prescriptions import analyze_prescriptions  # noqa: E402
from medmonitor.anomaly.vitals import StreamingVitalsMonitor, news2  # noqa: E402
from medmonitor.audio.azure_text import analyze_offline  # noqa: E402
from medmonitor.config import load_settings  # noqa: E402
from medmonitor.fusion import RiskFusionEngine  # noqa: E402
from medmonitor.schemas import Finding, Severity  # noqa: E402
from medmonitor.synthetic.clinical import generate_mobility, generate_prescriptions  # noqa: E402
from medmonitor.synthetic.pose import session_keypoints  # noqa: E402
from medmonitor.synthetic.vitals import generate_vitals  # noqa: E402
from medmonitor.video.movement import PhysioAnalyzer  # noqa: E402
from medmonitor.video.pose import PoseSequence  # noqa: E402

S = load_settings()


@pytest.fixture(scope="module")
def vitals_findings():
    df, _ = generate_vitals(hours=48)
    return StreamingVitalsMonitor(S).run(df)


def test_news2_reference_values():
    normal = dict(hr=75, sbp=120, spo2=98, rr=16, temp=36.8)
    assert news2(normal) == 0
    sepsis = dict(hr=135, sbp=85, spo2=88, rr=30, temp=39.5)
    assert news2(sepsis) >= 7


def test_stable_patient_has_no_vital_alerts(vitals_findings):
    # P001 tem um artefato isolado de SpO2 (1 amostra) que deve ser filtrado pela persistência
    assert not [f for f in vitals_findings if f.patient_id == "P001"]


def test_sepsis_detected_by_trend_before_hard_limits(vitals_findings):
    p2 = [f for f in vitals_findings if f.patient_id == "P002"]
    trend = min(pd.Timestamp(f.timestamp) for f in p2 if "_trend_" in f.kind)
    hard = min(pd.Timestamp(f.timestamp) for f in p2 if f.kind.endswith(("_high", "_low")) and "news2" not in f.kind)
    assert trend < hard, "CUSUM deveria antecipar a deterioração antes dos limites clínicos"


def test_bradycardia_and_hypertension_detected(vitals_findings):
    kinds = {f.kind for f in vitals_findings if f.patient_id == "P003"}
    assert "hr_low" in kinds and "sbp_high" in kinds


def test_prescription_anomalies():
    df, _ = generate_prescriptions()
    F = analyze_prescriptions(df, S)
    assert any(f.kind == "dose_jump" and f.evidence["droga"] == "enoxaparina" and f.score > 0.75 for f in F)
    assert any(f.kind == "abrupt_discontinuation" and f.evidence["droga"] == "metoprolol" for f in F)
    assert not [f for f in F if f.patient_id == "P001"]


def test_mobility_fall_and_agitation():
    df, _ = generate_mobility()
    F = analyze_mobility(df, S)
    assert any(f.patient_id == "P003" and f.kind == "fall_suspected" for f in F)
    assert any(f.patient_id == "P002" and f.kind == "night_agitation" for f in F)
    assert not [f for f in F if f.patient_id == "P001"]


def _seq(abnormal):
    return PoseSequence(session_keypoints(abnormal), 30.0, (640, 640), "synthetic")


def test_physio_normal_session_clean():
    a = PhysioAnalyzer(S).fit_reference(_seq(False))
    rep = a.analyze(_seq(False), "P001")
    assert rep.findings == []
    assert rep.metrics["repeticoes"] >= 5


def test_physio_abnormal_session_flags():
    a = PhysioAnalyzer(S).fit_reference(_seq(False))
    rep = a.analyze(_seq(True), "P003")
    kinds = {f.kind for f in rep.findings}
    assert {"fall_detected", "reduced_rom_l", "asymmetry", "trunk_compensation", "tremor_l"} <= kinds
    assert 4.0 <= rep.metrics["tremor_esq_freq_hz"] <= 6.0
    assert rep.metrics["queda_t_s"] == pytest.approx(21.7, abs=0.6)


def test_offline_text_analytics():
    ins = analyze_offline("Estou com muita falta de ar e dor no peito, não consigo dormir.",
                          S["audio"]["critical_terms"])
    assert ins.sentiment == "negative"
    assert {"falta de ar", "dor no peito"} <= set(ins.critical_terms)
    assert any(e["category"] == "SymptomOrSign" for e in ins.health_entities)


def test_fusion_corroboration_raises_risk():
    fe = RiskFusionEngine(S)
    one = [Finding("X", "vitals", "spo2_low", 0.6, "")]
    two = one + [Finding("X", "audio", "voice_respiratorio", 0.6, "")]
    assert fe.risk(two)[0] > fe.risk(one)[0]
    # evento grave isolado não pode ser diluído
    fall = [Finding("X", "video", "fall_detected", 0.95, "")]
    assert fe.severity(fe.risk(fall)[0]) >= Severity.HIGH


def test_audio_features_distinguish_dysarthria():
    soundfile = pytest.importorskip("soundfile")
    from medmonitor.audio import analyze_audio
    wav = ROOT / "data" / "audio" / "consulta_P003.wav"
    if not wav.exists():
        pytest.skip("rode scripts/generate_data.py antes")
    rep = analyze_audio(wav, "P003", S)
    assert any(f.kind == "voice_disartria" for f in rep.findings)
    _ = soundfile, np

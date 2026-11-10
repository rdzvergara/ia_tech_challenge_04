"""Orquestrador do fluxo multimodal ponta a ponta.

    sessões (áudio/vídeo) ─┐
    sinais vitais (stream) ─┼─► analisadores por modalidade ─► achados ─► fusão de risco ─► alertas ─► equipe
    prescrições / mobilidade┘                                    (Finding)    (late fusion)   (Teams/JSONL)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .alerts import AlertManager
from .anomaly.mobility import analyze_mobility
from .anomaly.prescriptions import analyze_prescriptions
from .anomaly.vitals import StreamingVitalsMonitor
from .audio import analyze_audio, save_audio_report
from .audio.azure_speech import azure_available as speech_on
from .audio.azure_text import azure_available as language_on
from .config import load_settings, resolve
from .evaluation import evaluate
from .fusion import RiskFusionEngine
from .plots import plot_risk, plot_vitals
from .video import analyze_physio_session, analyze_scene
from .video.pose import OpenPoseJsonReader

log = logging.getLogger("medmonitor")


def _stamp(findings, ts):
    for f in findings:
        f.timestamp = pd.Timestamp(ts).isoformat()
    return findings


def run_all(data_dir: str | Path = "data", out_dir: str | Path = "outputs", pose_backend: str = "openpose_json",
            annotate: bool = True, quiet: bool = False, scene: bool = True) -> dict:
    S = load_settings()
    data, out = resolve(data_dir), resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings = {}
    findings = []
    sessions = json.loads((data / "sessions.json").read_text())

    def say(msg):
        if not quiet:
            print(msg)

    say(f"== MedMonitor | Azure Speech: {'ON' if speech_on() else 'offline'} | "
        f"Azure Language: {'ON' if language_on() else 'offline'} | pose: {pose_backend}")

    # 1) Áudio --------------------------------------------------------------
    t = time.time()
    audio_reports = {}
    for s in [s for s in sessions if s["type"] == "audio"]:
        rep = analyze_audio(data / s["file"], s["patient_id"], S)
        save_audio_report(rep, out / "audio")
        findings += _stamp(rep.findings, s["timestamp"])
        audio_reports[s["patient_id"]] = rep
        say(f"[áudio] {s['patient_id']}: {rep.transcript.engine} | sentimento={rep.insights.sentiment} | "
            f"termos={rep.insights.critical_terms} | achados={[f.kind for f in rep.findings]}")
    timings["audio_s"] = round(time.time() - t, 1)

    # 2) Vídeo --------------------------------------------------------------
    t = time.time()
    video_reports = {}
    ref_s = next((s for s in sessions if s.get("reference")), None)
    reference = None
    if ref_s and pose_backend == "openpose_json":
        reference = OpenPoseJsonReader().read(data / ref_s["openpose_json"])
    for s in [s for s in sessions if s["type"] == "physio"]:
        if reference is None and pose_backend != "openpose_json" and s.get("reference"):
            from .video.pose import estimate_pose
            reference = estimate_pose(data / s["file"], pose_backend, S)
        rep = analyze_physio_session(data / s["file"], s["patient_id"], S, out / "video", pose_backend,
                                     data / s.get("openpose_json", ""), reference, annotate)
        findings += _stamp(rep.findings, s["timestamp"])
        video_reports[s["patient_id"]] = rep
        say(f"[vídeo] {s['patient_id']}: {rep.metrics} | achados={[f.kind for f in rep.findings]}")
    scene_summary = None
    if scene:
        for s in [s for s in sessions if s["type"] == "scene"]:
            _, scene_summary = analyze_scene(data / s["file"], s["patient_id"], S, out / "video")
            say(f"[cena/YOLOv8] {s['patient_id']}: classes={scene_summary['classes_detectadas']}")
    timings["video_s"] = round(time.time() - t, 1)

    # 3) Sinais vitais em fluxo, prescrições e mobilidade ------------------------
    t = time.time()
    vit = pd.read_csv(data / "vitals.csv", parse_dates=["timestamp"])
    mon = StreamingVitalsMonitor(S)
    vf = mon.run(vit)
    findings += vf
    rx = analyze_prescriptions(pd.read_csv(data / "prescriptions.csv"), S)
    mob = analyze_mobility(pd.read_csv(data / "mobility.csv"), S)
    findings += rx + mob
    timings["vitals_rx_mobility_s"] = round(time.time() - t, 1)
    say(f"[vitais] {len(vf)} achados | [prescrições] {len(rx)} | [mobilidade] {len(mob)}")

    # 4) Fusão + alertas (replay em ordem temporal = tempo real) -----------------
    fusion = RiskFusionEngine(S)
    am = AlertManager(S, quiet=quiet)
    am.reset()
    say("\n== Alertas emitidos à equipe médica (ordem cronológica) ==")
    alerts = fusion.run(findings)
    for a in alerts:
        am.dispatch(a)
    summary = fusion.patient_summary(findings, sorted(vit.patient_id.unique()))
    history = pd.DataFrame(fusion.history)

    # 5) Artefatos -------------------------------------------------------------
    (out / "findings.json").write_text(json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=1,
                                                  default=float), encoding="utf-8")
    (out / "patient_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    history.to_csv(out / "risk_history.csv", index=False)
    for pid in sorted(vit.patient_id.unique()):
        plot_vitals(vit, findings, alerts, S["vitals"]["hard_limits"], pid, out / f"vitais_{pid}.png")
    if len(history):
        plot_risk(history, out / "risco_multimodal.png", sorted(vit.patient_id.unique()))

    gt_path = data / "ground_truth.json"
    ev = None
    if gt_path.exists():
        ev = evaluate(findings, json.loads(gt_path.read_text(encoding="utf-8")))
        ev.to_csv(out / "avaliacao_ground_truth.csv", index=False)
    timings["total_s"] = round(time.time() - t0, 1)
    result = {"alerts": alerts, "findings": findings, "summary": summary, "evaluation": ev, "timings": timings,
              "audio": audio_reports, "video": video_reports, "scene": scene_summary}
    _write_results_md(out, result)
    say(f"\n== Resumo por paciente ==\n{json.dumps(summary, ensure_ascii=False, indent=1)}")
    if ev is not None:
        say(f"\n== Avaliação vs ground truth: {int(ev.correto.sum())}/{len(ev)} corretos ==")
    say(f"Tempos: {timings}")
    return result


def _write_results_md(out: Path, r: dict):
    L = ["# Resultados da execução", "",
         "Gerado automaticamente por `python -m medmonitor.cli run-all`.", "",
         "## Risco final por paciente", "", "| Paciente | Risco | Nível | Modalidades (score) | Achados |",
         "|---|---|---|---|---|"]
    for pid, s in sorted(r["summary"].items()):
        mods = ", ".join(f"{m} {v:.2f}" for m, v in sorted(s["modalities"].items(), key=lambda x: -x[1]))
        L.append(f"| {pid} | {s['risk']:.2f} | {s['severity']} | {mods} | {s['n_findings']} |")
    L += ["", f"## Alertas emitidos ({len(r['alerts'])})", "", "| Horário | Paciente | Nível | Risco | Modalidades | Resumo |",
          "|---|---|---|---|---|---|"]
    for a in r["alerts"]:
        L.append(f"| {a.created_at[5:16].replace('T', ' ')} | {a.patient_id} | {a.severity.name} | {a.risk_score:.2f} | "
                 f"{', '.join(a.modalities)} | {a.summary[:160]} |")
    if r["evaluation"] is not None:
        ev = r["evaluation"]
        L += ["", f"## Avaliação contra o ground truth — {int(ev.correto.sum())}/{len(ev)} corretos", "",
              ev.to_markdown(index=False) if hasattr(ev, "to_markdown") else ev.to_string(index=False)]
    L += ["", "## Tempos de processamento (s)", "", f"`{r['timings']}`", ""]
    (out / "RESULTADOS.md").write_text("\n".join(L), encoding="utf-8")

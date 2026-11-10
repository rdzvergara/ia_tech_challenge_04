"""Interface de linha de comando.

Exemplos:
    python -m medmonitor.cli run-all
    python -m medmonitor.cli audio data/audio/consulta_P002.wav --patient P002
    python -m medmonitor.cli video meu_video.mp4 --patient P010 --backend yolov8
    python -m medmonitor.cli stream --speed 600
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from .config import load_settings, resolve


def cmd_run_all(a):
    from .pipeline import run_all
    run_all(a.data, a.out, a.backend, annotate=not a.no_video, quiet=a.quiet, scene=not a.no_scene)


def cmd_audio(a):
    from .audio import analyze_audio, save_audio_report
    rep = analyze_audio(Path(a.file), a.patient, load_settings())
    save_audio_report(rep, resolve(a.out) / "audio")
    print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2, default=float))


def cmd_video(a):
    from .video import analyze_physio_session
    from .video.pose import OpenPoseJsonReader, estimate_pose
    S = load_settings()
    ref = None
    if a.reference:
        ref = OpenPoseJsonReader().read(Path(a.reference)) if Path(a.reference).is_dir() else \
            estimate_pose(Path(a.reference), a.backend, S)
    rep = analyze_physio_session(Path(a.file), a.patient, S, resolve(a.out) / "video", a.backend,
                                 Path(a.json_dir) if a.json_dir else None, ref)
    print(json.dumps(rep.metrics, ensure_ascii=False, indent=2))
    for f in rep.findings:
        print(f"[{f.severity.name}] {f.description}")
    print(f"Relatório: {resolve(a.out) / 'video' / f'video_{a.patient}_relatorio.md'}")


def cmd_stream(a):
    """Simula a chegada de sinais vitais em tempo real e imprime alertas conforme ocorrem."""
    from .alerts import AlertManager
    from .anomaly.vitals import StreamingVitalsMonitor
    from .fusion import RiskFusionEngine
    S = load_settings()
    df = pd.read_csv(resolve(a.data) / "vitals.csv", parse_dates=["timestamp"]).sort_values("timestamp")
    if a.patient:
        df = df[df.patient_id == a.patient]
    mon, fus, am = StreamingVitalsMonitor(S), RiskFusionEngine(S), AlertManager(S)
    am.reset()
    step = 60 / a.speed   # 1 amostra = 1 min de tempo clínico
    for rec in df.to_dict("records"):
        for f in mon.update(rec):
            al = fus.ingest(f)
            if al:
                am.dispatch(al)
        if step > 0.0005:
            time.sleep(step / max(1, df.patient_id.nunique()))


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="medmonitor", description="Monitoramento multimodal hospitalar")
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("run-all", help="executa o fluxo multimodal completo")
    r.add_argument("--data", default="data"); r.add_argument("--out", default="outputs")
    r.add_argument("--backend", default="openpose_json", choices=["openpose_json", "yolov8", "openpose_caffe"])
    r.add_argument("--no-video", action="store_true", help="não gera os vídeos anotados")
    r.add_argument("--no-scene", action="store_true", help="pula a cena real (YOLOv8)")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run_all)

    au = sub.add_parser("audio", help="analisa um áudio de consulta")
    au.add_argument("file"); au.add_argument("--patient", default="PX"); au.add_argument("--out", default="outputs")
    au.set_defaults(fn=cmd_audio)

    v = sub.add_parser("video", help="analisa um vídeo de fisioterapia")
    v.add_argument("file"); v.add_argument("--patient", default="PX"); v.add_argument("--out", default="outputs")
    v.add_argument("--backend", default="yolov8", choices=["yolov8", "openpose_json", "openpose_caffe"])
    v.add_argument("--json-dir", help="diretório de saída do OpenPose (backend openpose_json)")
    v.add_argument("--reference", help="vídeo ou diretório OpenPose de uma sessão normal de referência")
    v.set_defaults(fn=cmd_video)

    s = sub.add_parser("stream", help="simula monitoramento de sinais vitais em tempo real")
    s.add_argument("--data", default="data"); s.add_argument("--patient")
    s.add_argument("--speed", type=float, default=3600, help="fator de aceleração (3600 = 1 h por segundo)")
    s.set_defaults(fn=cmd_stream)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

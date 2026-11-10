"""Análise de vídeo clínico: pose (OpenPose / YOLOv8-pose), objetos e áreas críticas (YOLOv8), relatórios."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import resolve
from .movement import MovementReport, PhysioAnalyzer
from .pose import PoseSequence, estimate_pose
from .report import annotate_video, plot_angles, write_markdown


def analyze_physio_session(video: Path, patient_id: str, settings: dict, out_dir: Path,
                           backend: str | None = None, json_dir: Path | None = None,
                           reference: PoseSequence | None = None, annotate: bool = True) -> MovementReport:
    backend = backend or settings["video"]["pose_backend"]
    seq = estimate_pose(video, backend, settings, json_dir)
    analyzer = PhysioAnalyzer(settings)
    if reference is not None:
        analyzer.fit_reference(reference)
    rep = analyzer.analyze(seq, patient_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"video_{patient_id}"
    plot_angles(rep, out_dir / f"{stem}_angulos.png", settings["video"]["physio"]["target_rom_deg"][0])
    if annotate and video and Path(video).exists():
        annotate_video(Path(video), seq, rep, out_dir / f"{stem}_anotado.mp4")
    write_markdown(rep, out_dir / f"{stem}_relatorio.md", images=[f"{stem}_angulos.png"])
    (out_dir / f"{stem}_metricas.json").write_text(json.dumps(
        {"metrics": rep.metrics, "events": rep.events, "findings": [f.to_dict() for f in rep.findings]},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return rep


def analyze_scene(video: Path, patient_id: str, settings: dict, out_dir: Path, stride: int = 3):
    """Cena real: YOLOv8 (objetos + zonas críticas) e YOLOv8-pose sobre pixels reais."""
    import cv2
    import numpy as np

    from .detector import ObjectDetector
    from .pose import SKELETON, YoloPoseEstimator
    v = settings["video"]
    det = ObjectDetector(str(resolve(v["detector_model"])), v["critical_objects"], v["critical_zones"])
    res = det.run(video, patient_id, stride=stride)
    pose = YoloPoseEstimator(str(resolve(v["pose_model"])))
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    polys = det.zone_polys(w, h)
    vw = cv2.VideoWriter(str(out_dir / f"cena_{patient_id}_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
    i = j = 0
    snapshot = None
    n_people = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            for poly in polys.values():
                cv2.polylines(frame, [poly.astype(np.int32)], True, (0, 0, 255), 2)
            for d in res.frames[j]:
                x0, y0, x1, y1 = map(int, d["box"])
                col = (0, 0, 255) if d["cls"] in v["critical_objects"] else (255, 160, 0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), col, 2)
                cv2.putText(frame, f"{d['cls']} {d['conf']:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1, cv2.LINE_AA)
            people = pose.estimate_all(frame)
            n_people.append(len(people))
            for kp in people:
                for a, b in SKELETON:
                    if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                        cv2.line(frame, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), (0, 230, 255), 2)
            vw.write(frame)
            if snapshot is None and len(people) and j > 5:
                snapshot = frame.copy()
            j += 1
        i += 1
    cap.release(); vw.release()
    from .report import to_h264
    to_h264(out_dir / f"cena_{patient_id}_anotado.mp4")
    if snapshot is not None:
        cv2.imwrite(str(out_dir / f"cena_{patient_id}_frame.jpg"), snapshot)
    summary = {"classes_detectadas": res.class_counts, "eventos": res.events,
               "pessoas_com_pose_media": round(float(np.mean(n_people)) if n_people else 0, 2),
               "findings": [f.to_dict() for f in res.findings]}
    (out_dir / f"cena_{patient_id}_deteccoes.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                               encoding="utf-8")
    return res, summary

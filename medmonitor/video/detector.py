"""Detecção de objetos e áreas críticas com YOLOv8.

* Detecta pessoas e objetos (classes COCO) por frame;
* Verifica invasão de zonas críticas configuráveis (ex.: campo estéril, área restrita);
* Sinaliza objetos de risco (instrumentos cortantes: ``knife``, ``scissors``);
* Contagem de instrumentos início × fim (indício de item retido / não devolvido em cirurgia).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..schemas import Finding


@dataclass
class DetectionResult:
    frames: list[list[dict]] = field(default_factory=list)   # por frame: [{cls, conf, box}]
    events: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)


def _in_poly(pt, poly):
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


class ObjectDetector:
    def __init__(self, model_path: str, critical_objects: list[str], zones: list[dict], conf=0.35):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.critical = set(critical_objects)
        self.zones = zones
        self.conf = conf

    def detect_frame(self, frame) -> list[dict]:
        r = self.model(frame, conf=self.conf, verbose=False)[0]
        out = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append({"cls": self.names[int(c)], "conf": float(s), "box": [float(v) for v in b]})
        return out

    def zone_polys(self, w, h):
        return {z["name"]: np.array(z["polygon"]) * [w, h] for z in self.zones}

    def run(self, video: Path, patient_id: str, stride: int = 3) -> DetectionResult:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        polys = self.zone_polys(w, h)
        res = DetectionResult()
        counts = Counter()
        zone_hits = Counter()
        critical_hits = Counter()
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                dets = self.detect_frame(frame)
                t = i / fps
                for d in dets:
                    counts[d["cls"]] += 1
                    x0, y0, x1, y1 = d["box"]
                    foot = ((x0 + x1) / 2, y1)
                    for zn, poly in polys.items():
                        if d["cls"] == "person" and _in_poly(foot, poly):
                            if zone_hits[zn] == 0:
                                res.events.append({"t": round(t, 2), "type": "zona_critica",
                                                   "detail": f"pessoa dentro da zona '{zn}'"})
                            zone_hits[zn] += 1
                    if d["cls"] in self.critical:
                        if critical_hits[d["cls"]] == 0:
                            res.events.append({"t": round(t, 2), "type": "objeto_critico",
                                               "detail": f"{d['cls']} detectado (conf {d['conf']:.2f})"})
                        critical_hits[d["cls"]] += 1
                res.frames.append(dets)
            i += 1
        cap.release()
        res.class_counts = dict(counts)
        n = max(1, len(res.frames))
        for zn, c in zone_hits.items():
            res.findings.append(Finding(patient_id, "video", "critical_zone_intrusion", min(0.6, 0.2 + c / n),
                                        f"Presença na zona crítica '{zn}' em {100*c/n:.0f}% dos quadros analisados",
                                        evidence={"zona": zn, "quadros": c}, source="YOLOv8"))
        for cls, c in critical_hits.items():
            res.findings.append(Finding(patient_id, "video", "critical_object", 0.35,
                                        f"Objeto de risco '{cls}' detectado em {c} quadros",
                                        evidence={"classe": cls, "quadros": c}, source="YOLOv8"))
        # contagem de instrumentos: primeiros vs últimos 10% dos quadros
        k = max(1, n // 10)
        first = Counter(d["cls"] for f in res.frames[:k] for d in f if d["cls"] in self.critical)
        last = Counter(d["cls"] for f in res.frames[-k:] for d in f if d["cls"] in self.critical)
        for cls in first:
            if last[cls] < first[cls] * 0.5:
                res.findings.append(Finding(patient_id, "video", "instrument_count_mismatch", 0.6,
                                            f"Contagem de '{cls}' caiu do início para o fim do procedimento",
                                            evidence={"inicio": first[cls], "fim": last[cls]}, source="YOLOv8"))
        return res

"""Estimação de pose — backends intercambiáveis normalizados para o esqueleto COCO-17.

Backends
--------
* ``yolov8``         — Ultralytics YOLOv8-pose sobre o vídeo bruto (padrão; roda em CPU).
* ``openpose_json``  — lê a saída do OpenPose (``--write_json``, BODY_25 ou COCO-18) já processada.
* ``openpose_caffe`` — executa o modelo OpenPose COCO (Caffe) via ``cv2.dnn`` (requer os pesos oficiais).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

COCO_NAMES = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
              "lhip", "rhip", "lknee", "rknee", "lank", "rank"]
K = {n: i for i, n in enumerate(COCO_NAMES)}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

# BODY_25 -> COCO-17
BODY25_TO_COCO = {0: 0, 16: 1, 15: 2, 18: 3, 17: 4, 5: 5, 2: 6, 6: 7, 3: 8, 7: 9, 4: 10, 12: 11, 9: 12,
                  13: 13, 10: 14, 14: 15, 11: 16}
# COCO-18 (OpenPose Caffe) -> COCO-17
COCO18_TO_COCO = {0: 0, 15: 1, 14: 2, 17: 3, 16: 4, 5: 5, 2: 6, 6: 7, 3: 8, 7: 9, 4: 10, 11: 11, 8: 12,
                  12: 13, 9: 14, 13: 15, 10: 16}


@dataclass
class PoseSequence:
    keypoints: np.ndarray          # (T, 17, 3) -> x, y, confiança
    fps: float
    frame_size: tuple[int, int]    # (w, h)
    backend: str
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.keypoints)

    def masked(self, min_conf=0.3) -> np.ndarray:
        """Coordenadas com NaN onde a confiança é baixa, interpoladas no tempo."""
        xy = self.keypoints[..., :2].copy()
        xy[self.keypoints[..., 2] < min_conf] = np.nan
        T = len(xy)
        idx = np.arange(T)
        for j in range(17):
            for c in range(2):
                col = xy[:, j, c]
                ok = ~np.isnan(col)
                if ok.sum() >= 2:
                    xy[:, j, c] = np.interp(idx, idx[ok], col[ok])
        return xy


def _video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, (w, h), n


class YoloPoseEstimator:
    def __init__(self, model_path: str = "yolov8n-pose.pt", conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf

    def estimate(self, video: Path, stride: int = 1) -> PoseSequence:
        fps, size, _ = _video_info(video)
        cap = cv2.VideoCapture(str(video))
        out, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                out.append(self.estimate_frame(frame))
            i += 1
        cap.release()
        kp = np.stack(out) if out else np.zeros((0, 17, 3))
        return PoseSequence(kp, fps / stride, size, "yolov8-pose")

    def estimate_frame(self, frame) -> np.ndarray:
        """Retorna os keypoints (17x3) da pessoa de maior área no quadro (paciente)."""
        r = self.model(frame, conf=self.conf, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return np.zeros((17, 3))
        boxes = r.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        j = int(np.argmax(areas))
        xy = r.keypoints.xy.cpu().numpy()[j]
        cf = r.keypoints.conf.cpu().numpy()[j] if r.keypoints.conf is not None else np.ones(17)
        return np.concatenate([xy, cf[:, None]], axis=1)

    def estimate_all(self, frame) -> list[np.ndarray]:
        r = self.model(frame, conf=self.conf, verbose=False)[0]
        if r.keypoints is None:
            return []
        xy = r.keypoints.xy.cpu().numpy()
        cf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
        return [np.concatenate([xy[i], cf[i][:, None]], axis=1) for i in range(len(xy))]


class OpenPoseJsonReader:
    """Lê o diretório de saída do OpenPose (um JSON por frame)."""

    def read(self, json_dir: Path, fps: float = 30.0, frame_size=(640, 640)) -> PoseSequence:
        files = sorted(Path(json_dir).glob("*_keypoints.json"),
                       key=lambda p: int(re.findall(r"(\d+)_keypoints", p.name)[0]))
        seq = []
        for f in files:
            people = json.loads(f.read_text()).get("people", [])
            kp = np.zeros((17, 3))
            if people:
                # escolhe a pessoa com maior soma de confiança
                arrs = [np.array(p["pose_keypoints_2d"]).reshape(-1, 3) for p in people]
                a = max(arrs, key=lambda x: x[:, 2].sum())
                mapping = BODY25_TO_COCO if len(a) == 25 else COCO18_TO_COCO
                for src, dst in mapping.items():
                    kp[dst] = a[src]
            seq.append(kp)
        return PoseSequence(np.stack(seq), fps, frame_size, "openpose-json", {"source": str(json_dir)})


class OpenPoseCaffeEstimator:
    """OpenPose (modelo COCO, Caffe) via OpenCV DNN — estimação de pessoa única.

    Pesos: ``pose_iter_440000.caffemodel`` + ``pose_deploy_linevec.prototxt`` (repositório oficial
    CMU-Perceptual-Computing-Lab/openpose, pasta models/pose/coco). Coloque-os em ``models/openpose/``.
    """

    def __init__(self, proto: Path, weights: Path, in_size=368, thr=0.1):
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.in_size, self.thr = in_size, thr

    def estimate_frame(self, frame) -> np.ndarray:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.in_size, self.in_size), (0, 0, 0), swapRB=False)
        self.net.setInput(blob)
        out = self.net.forward()
        kp = np.zeros((17, 3))
        for src, dst in COCO18_TO_COCO.items():
            hm = out[0, src]
            _, conf, _, pt = cv2.minMaxLoc(hm)
            if conf > self.thr:
                kp[dst] = [w * pt[0] / out.shape[3], h * pt[1] / out.shape[2], conf]
        return kp

    def estimate(self, video: Path, stride: int = 1) -> PoseSequence:
        fps, size, _ = _video_info(video)
        cap = cv2.VideoCapture(str(video))
        out, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                out.append(self.estimate_frame(frame))
            i += 1
        cap.release()
        return PoseSequence(np.stack(out), fps / stride, size, "openpose-caffe")


def estimate_pose(video: Path | None, backend: str, settings: dict, json_dir: Path | None = None) -> PoseSequence:
    from ..config import resolve
    vcfg = settings["video"]
    if backend == "openpose_json":
        fps, size = 30.0, (640, 640)
        if video and Path(video).exists():
            fps, size, _ = _video_info(Path(video))
        return OpenPoseJsonReader().read(json_dir, fps, size)
    if backend == "openpose_caffe":
        d = resolve(settings["paths"]["models_dir"]) / "openpose"
        return OpenPoseCaffeEstimator(d / "pose_deploy_linevec.prototxt", d / "pose_iter_440000.caffemodel").estimate(
            video, vcfg.get("frame_stride", 1))
    return YoloPoseEstimator(str(resolve(vcfg["pose_model"]))).estimate(video, vcfg.get("frame_stride", 1))

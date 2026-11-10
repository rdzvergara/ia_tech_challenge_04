"""Sessões de fisioterapia sintéticas (abdução bilateral de ombro).

Gera, para cada sessão:
  * diretório de keypoints no formato de saída do OpenPose (`--write_json`, modelo BODY_25);
  * vídeo MP4 renderizado (manequim) para visualização/overlay.

Convenção: paciente de frente para a câmera — o lado ESQUERDO do paciente aparece à DIREITA da imagem.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 640, 30

# Índices COCO-17
COCO = dict(nose=0, leye=1, reye=2, lear=3, rear=4, lsho=5, rsho=6, lelb=7, relb=8, lwri=9, rwri=10,
            lhip=11, rhip=12, lknee=13, rknee=14, lank=15, rank=16)
# COCO-17 -> BODY_25
COCO_TO_BODY25 = {0: 0, 6: 2, 8: 3, 10: 4, 5: 5, 7: 6, 9: 7, 12: 9, 14: 10, 16: 11, 11: 12, 13: 13, 15: 14,
                  2: 15, 1: 16, 4: 17, 3: 18}


def _rot(p, c, ang):
    s, co = np.sin(ang), np.cos(ang)
    d = p - c
    return c + np.array([co * d[0] - s * d[1], s * d[0] + co * d[1]])


def _skeleton(theta_l, theta_r, lean_deg, tremor, drop, tilt_deg, cx=320.0):
    """Retorna keypoints COCO-17 (17x2) para ângulos de abdução (graus) e inclinação de tronco."""
    hip_y = 360.0
    hipc = np.array([cx, hip_y])
    k = np.zeros((17, 2))
    k[COCO["lhip"]] = [cx + 38, hip_y]
    k[COCO["rhip"]] = [cx - 38, hip_y]
    k[COCO["lknee"]] = [cx + 42, hip_y + 115]
    k[COCO["rknee"]] = [cx - 42, hip_y + 115]
    k[COCO["lank"]] = [cx + 45, hip_y + 230]
    k[COCO["rank"]] = [cx - 45, hip_y + 230]
    sh_y = hip_y - 170
    k[COCO["lsho"]] = [cx + 58, sh_y]
    k[COCO["rsho"]] = [cx - 58, sh_y]
    k[COCO["nose"]] = [cx, sh_y - 62]
    k[COCO["leye"]] = [cx + 13, sh_y - 72]
    k[COCO["reye"]] = [cx - 13, sh_y - 72]
    k[COCO["lear"]] = [cx + 28, sh_y - 66]
    k[COCO["rear"]] = [cx - 28, sh_y - 66]
    up, lo = 92.0, 88.0
    for side, th, sgn in (("l", theta_l, 1), ("r", theta_r, -1)):
        t = np.radians(th)
        d = np.array([sgn * np.sin(t), np.cos(t)])
        sh = k[COCO[side + "sho"]]
        k[COCO[side + "elb"]] = sh + up * d
        k[COCO[side + "wri"]] = sh + (up + lo) * d
    k[COCO["lwri"]] += tremor
    k[COCO["lelb"]] += tremor * 0.4
    # inclinação lateral do tronco (compensação): rotaciona parte superior em torno do centro do quadril
    upper = [COCO[n] for n in ("nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri")]
    for i in upper:
        k[i] = _rot(k[i], hipc, np.radians(lean_deg))
    # queda: corpo inteiro rotaciona em torno dos tornozelos e desce
    if tilt_deg or drop:
        pivot = (k[COCO["lank"]] + k[COCO["rank"]]) / 2
        for i in range(17):
            k[i] = _rot(k[i], pivot, np.radians(tilt_deg))
        k[:, 1] += drop
        k[:, 0] += abs(tilt_deg) / 82 * 240   # mantém o corpo caído dentro do quadro
    return k


def _profile(t, period, peak, rest=10.0):
    """Ângulo de abdução ao longo do tempo (repetições senoidais)."""
    ph = (t % period) / period
    return rest + (peak - rest) * (0.5 - 0.5 * np.cos(2 * np.pi * ph))


def session_keypoints(abnormal: bool, seconds: float = 24.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    out = np.zeros((n, 17, 3))
    period = 4.0
    for f in range(n):
        t = f / FPS
        if abnormal:
            th_r = _profile(t, period, 162)
            th_l = _profile(t, period, 95)                     # ROM reduzida (hemiparesia esquerda)
            lean = -16 * (th_r - 10) / (162 - 10)               # compensação de tronco para a direita
            tremor = np.array([6 * np.sin(2 * np.pi * 5.0 * t), 4 * np.sin(2 * np.pi * 5.0 * t + 1)])
            # perda de equilíbrio/queda aos 21 s
            fall_t = 21.0
            prog = np.clip((t - fall_t) / 0.7, 0, 1)
            tilt, drop = -82 * prog ** 2, 12 * prog ** 2
        else:
            th_r = _profile(t, period, 168)
            th_l = _profile(t, period, 166)
            lean, tremor, tilt, drop = 1.5 * np.sin(2 * np.pi * t / period), np.zeros(2), 0.0, 0.0
        k = _skeleton(th_l, th_r, lean, tremor, drop, tilt)
        k += rng.normal(0, 1.2, k.shape)                        # ruído de estimação
        out[f, :, :2] = k
        out[f, :, 2] = np.clip(rng.normal(0.88, 0.05, 17), 0.5, 0.99)
    return out


def write_openpose_json(kps: np.ndarray, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f, k in enumerate(kps):
        b25 = np.zeros((25, 3))
        for c, b in COCO_TO_BODY25.items():
            b25[b] = k[c]
        b25[1, :2] = (k[5, :2] + k[6, :2]) / 2; b25[1, 2] = min(k[5, 2], k[6, 2])     # Neck
        b25[8, :2] = (k[11, :2] + k[12, :2]) / 2; b25[8, 2] = min(k[11, 2], k[12, 2])  # MidHip
        doc = {"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": b25.round(3).ravel().tolist()}]}
        (out_dir / f"{stem}_{f:012d}_keypoints.json").write_text(json.dumps(doc))


def render_video(kps: np.ndarray, path: Path, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    skin, shirt, pants = (150, 180, 225), (160, 110, 50), (70, 70, 70)
    for f, k in enumerate(kps):
        img = np.full((H, W, 3), (232, 236, 240), np.uint8)
        cv2.rectangle(img, (0, 600), (W, H), (200, 205, 210), -1)  # chão
        P = {n: tuple(int(v) for v in k[i, :2]) for n, i in COCO.items()}
        L = lambda a, b, c, t: cv2.line(img, P[a], P[b], c, t, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([P["lsho"], P["rsho"], P["rhip"], P["lhip"]], np.int32)], shirt)
        for s in "lr":
            L(s + "hip", s + "knee", pants, 30); L(s + "knee", s + "ank", pants, 26)
            L(s + "sho", s + "elb", shirt, 24); L(s + "elb", s + "wri", skin, 18)
            cv2.circle(img, P[s + "wri"], 11, skin, -1, cv2.LINE_AA)
        neck = tuple(int(v) for v in (k[5, :2] + k[6, :2]) / 2)
        cv2.line(img, neck, P["nose"], skin, 18)
        cv2.circle(img, P["nose"], 30, skin, -1, cv2.LINE_AA)
        cv2.putText(img, f"{title}  t={f / FPS:5.1f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def make_real_frames_clip(images: list[Path], path: Path, seconds_each: float = 3.0):
    """Clipe curto (pan/zoom) a partir de fotos reais, para validar YOLOv8/YOLOv8-pose em pixels reais."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        n = int(seconds_each * FPS)
        for i in range(n):
            z = 1.0 + 0.15 * i / n
            cw, ch = int(w / z), int(h / z)
            x0 = int((w - cw) * i / max(n - 1, 1)); y0 = (h - ch) // 2
            crop = img[y0:y0 + ch, x0:x0 + cw]
            s = min(W / cw, H / ch)
            r = cv2.resize(crop, (int(cw * s), int(ch * s)))
            canvas = np.zeros((H, W, 3), np.uint8)
            oy, ox = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
            canvas[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r
            vw.write(canvas)
    vw.release()

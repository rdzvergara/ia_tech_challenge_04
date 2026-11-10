"""Relatórios automáticos de vídeo: vídeo anotado, gráfico de ângulos e relatório Markdown."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .movement import MovementReport  # noqa: E402
from .pose import SKELETON, PoseSequence  # noqa: E402

SEV_COLOR = {"CRITICAL": (40, 40, 220), "HIGH": (30, 120, 240), "MEDIUM": (0, 190, 240), "LOW": (60, 170, 60)}


def to_h264(path: Path):
    """Converte para H.264 (reprodutível em navegadores) se o ffmpeg estiver disponível."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_suffix(".h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(tmp)], capture_output=True)
    if r.returncode == 0:
        tmp.replace(path)


def annotate_video(video: Path, seq: PoseSequence, rep: MovementReport, out: Path, detections=None, zones=None):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or seq.fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    abd_l, abd_r = rep.series.get("abd_l", []), rep.series.get("abd_r", [])
    events = sorted(rep.events, key=lambda e: e["t"])
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < len(seq):
            kp = seq.keypoints[i]
            for a, b in SKELETON:
                if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                    cv2.line(frame, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), (0, 220, 255), 2, cv2.LINE_AA)
            for x, y, c in kp:
                if c > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 90, 255), -1, cv2.LINE_AA)
        if zones:
            for poly in zones.values():
                cv2.polylines(frame, [poly.astype(np.int32)], True, (0, 0, 255), 2)
        if detections and i < len(detections):
            for d in detections[i]:
                x0, y0, x1, y1 = map(int, d["box"])
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 160, 0), 2)
                cv2.putText(frame, f"{d['cls']} {d['conf']:.2f}", (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1, cv2.LINE_AA)
        t = i / fps
        panel = [f"t={t:5.1f}s"]
        if i < len(abd_l):
            panel.append(f"abd E={abd_l[i]:5.0f}  D={abd_r[i]:5.0f}")
        for k, line in enumerate(panel):
            cv2.putText(frame, line, (10, h - 40 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        active = [e for e in events if e["t"] <= t <= e["t"] + 2.5]
        for k, e in enumerate(active[:3]):
            color = (40, 40, 220) if e["type"] == "queda" else (0, 140, 255)
            cv2.rectangle(frame, (8, 40 + 26 * k), (w - 8, 62 + 26 * k), color, -1)
            cv2.putText(frame, f"ALERTA: {e['type']} - {e['detail']}"[:80], (14, 56 + 26 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()
    to_h264(out)


def plot_angles(rep: MovementReport, out: Path, target=150):
    t = np.array(rep.series["t"])
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=110)
    ax.plot(t, rep.series["abd_r"], label="Abdução ombro direito", color="#2a78d6", lw=1.6)
    ax.plot(t, rep.series["abd_l"], label="Abdução ombro esquerdo", color="#eb6834", lw=1.6)
    ax.plot(t, np.abs(rep.series["lean"]), label="Inclinação do tronco", color="#868e96", lw=1.2, ls="--")
    ax.axhline(target, color="#2b8a3e", lw=1, ls=":", label=f"Meta ROM ({target}°)")
    for e in rep.events:
        if e["type"] == "queda":
            ax.axvline(e["t"], color="#c92a2a", lw=1.5)
            ax.text(e["t"], 175, " queda", color="#c92a2a", fontsize=8, va="top")
    ax.set_xlabel("tempo (s)"); ax.set_ylabel("graus"); ax.set_ylim(-5, 185)
    ax.set_title(f"Paciente {rep.patient_id} — cinemática da sessão")
    ax.legend(fontsize=7, loc="lower right", ncol=2); ax.grid(alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def write_markdown(rep: MovementReport, out: Path, extra: dict | None = None, images: list[str] | None = None):
    L = [f"# Relatório automático de vídeo — paciente {rep.patient_id}", "",
         f"- Duração analisada: **{rep.duration_s:.1f} s** ({rep.n_frames} quadros)",
         f"- Backend de pose: **{rep.backend}**", "", "## Métricas", "", "| Métrica | Valor |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rep.metrics.items()]
    L += ["", "## Desvios / falhas detectados", ""]
    if rep.findings:
        L += ["| Severidade | Achado | Fonte |", "|---|---|---|"]
        L += [f"| {f.severity.name} | {f.description} | {f.source} |"
              for f in sorted(rep.findings, key=lambda f: -f.score)]
    else:
        L.append("Nenhum desvio relevante — sessão dentro do padrão esperado.")
    L += ["", "## Linha do tempo de eventos", ""]
    L += [f"- **{e['t']:.1f}s** — {e['type']}: {e['detail']}" for e in rep.events] or ["- (sem eventos)"]
    if extra:
        L += ["", "## Detecção de objetos (YOLOv8)", "", f"```\n{extra}\n```"]
    for im in images or []:
        L += ["", f"![]({im})"]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")

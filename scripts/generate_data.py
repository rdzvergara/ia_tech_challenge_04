"""Gera todo o conjunto de dados sintéticos de demonstração em data/.

Uso:  python scripts/generate_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medmonitor.synthetic import PATIENTS  # noqa: E402
from medmonitor.synthetic.audio import generate_consultation_audio  # noqa: E402
from medmonitor.synthetic.clinical import generate_mobility, generate_prescriptions  # noqa: E402
from medmonitor.synthetic.pose import (make_real_frames_clip, render_video, session_keypoints,  # noqa: E402
                                       write_openpose_json)
from medmonitor.synthetic.vitals import generate_vitals  # noqa: E402


def main():
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    truth = {}

    vit, t = generate_vitals()
    vit.to_csv(data / "vitals.csv", index=False); truth["vitals"] = t
    rx, t = generate_prescriptions()
    rx.to_csv(data / "prescriptions.csv", index=False); truth["prescriptions"] = t
    mob, t = generate_mobility()
    mob.to_csv(data / "mobility.csv", index=False); truth["mobility"] = t
    print(f"[ok] vitals.csv ({len(vit)} linhas), prescriptions.csv ({len(rx)}), mobility.csv ({len(mob)})")

    for pid in PATIENTS:
        p = generate_consultation_audio(pid, data / "audio")
        print(f"[ok] {p.relative_to(ROOT)}")

    sessions = {"P001": False, "P003": True}
    for pid, abnormal in sessions.items():
        kps = session_keypoints(abnormal)
        stem = f"fisio_{pid}"
        write_openpose_json(kps, data / "video" / f"{stem}_openpose", stem)
        render_video(kps, data / "video" / f"{stem}.mp4", f"Fisioterapia {pid}")
        print(f"[ok] data/video/{stem}.mp4 + keypoints OpenPose ({len(kps)} frames)")
    truth["video"] = [
        {"patient_id": "P001", "event": "sessao_normal", "should_alert": False},
        {"patient_id": "P003", "event": "rom_reduzida_esquerda+assimetria+compensacao_tronco+tremor+queda",
         "should_alert": True},
    ]

    imgs = [data / "zidane.jpg", data / "bus.jpg"]
    if all(p.exists() for p in imgs):
        make_real_frames_clip(imgs, data / "video" / "sala_real_yolo.mp4")
        print("[ok] data/video/sala_real_yolo.mp4 (pixels reais para YOLOv8)")

    # agenda das sessões gravadas (consultas e fisioterapia) — define o instante de cada mídia no fluxo
    sessions = [
        {"patient_id": "P001", "type": "audio", "file": "audio/consulta_P001.wav", "timestamp": "2026-11-09T09:00:00"},
        {"patient_id": "P002", "type": "audio", "file": "audio/consulta_P002.wav", "timestamp": "2026-11-09T09:30:00"},
        {"patient_id": "P003", "type": "audio", "file": "audio/consulta_P003.wav", "timestamp": "2026-11-09T10:00:00"},
        {"patient_id": "P001", "type": "physio", "file": "video/fisio_P001.mp4",
         "openpose_json": "video/fisio_P001_openpose", "timestamp": "2026-11-09T11:00:00", "reference": True},
        {"patient_id": "P003", "type": "physio", "file": "video/fisio_P003.mp4",
         "openpose_json": "video/fisio_P003_openpose", "timestamp": "2026-11-09T14:58:00"},
        {"patient_id": "LEITO-07", "type": "scene", "file": "video/sala_real_yolo.mp4", "timestamp": "2026-11-09T12:00:00"},
    ]
    (data / "sessions.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    (data / "patients.json").write_text(json.dumps(PATIENTS, ensure_ascii=False, indent=2), encoding="utf-8")
    (data / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] ground_truth.json")


if __name__ == "__main__":
    main()

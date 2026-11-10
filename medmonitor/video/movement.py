"""Análise biomecânica de sessões de fisioterapia a partir de keypoints (COCO-17).

Métricas
--------
* Ângulo de abdução de ombro (esq./dir.) por frame e amplitude de movimento (ROM) por repetição;
* Assimetria entre lados; inclinação lateral do tronco (compensação);
* Tremor: energia do punho na banda 3–7 Hz (filtro Butterworth) normalizada pelo comprimento do tronco;
* Queda: descida rápida do centro do quadril + tronco próximo da horizontal;
* Padrão anômalo frame-a-frame: Isolation Forest treinado numa sessão de referência normal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import IsolationForest

from ..schemas import Finding
from .pose import K, PoseSequence


def _angle(v1, v2):
    """Ângulo (graus) entre vetores (T,2)."""
    dot = (v1 * v2).sum(-1)
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / n, -1, 1)))


def kinematics(seq: PoseSequence, min_conf=0.3) -> dict[str, np.ndarray]:
    xy = seq.masked(min_conf)
    sh_c = (xy[:, K["lsho"]] + xy[:, K["rsho"]]) / 2
    hip_c = (xy[:, K["lhip"]] + xy[:, K["rhip"]]) / 2
    trunk = sh_c - hip_c  # aponta para cima (y negativo)
    vertical = np.tile([0.0, -1.0], (len(xy), 1))
    lean = _angle(trunk, vertical) * np.sign(trunk[:, 0] + 1e-9)
    abd = {}
    for s in ("l", "r"):
        arm = xy[:, K[s + "elb"]] - xy[:, K[s + "sho"]]
        side = xy[:, K[s + "hip"]] - xy[:, K[s + "sho"]]
        abd[s] = _angle(arm, side)
    torso_len = np.nanmedian(np.linalg.norm(trunk, axis=-1))
    body_h = np.nanmedian(np.abs(((xy[:, K["lank"]] + xy[:, K["rank"]]) / 2)[:, 1] - xy[:, K["nose"], 1]))
    return {"abd_l": abd["l"], "abd_r": abd["r"], "lean": lean, "hip_y": hip_c[:, 1],
            "hip_c": hip_c, "xy": xy, "torso_len": torso_len, "body_h": body_h}


def _frame_features(kin) -> np.ndarray:
    f = np.stack([kin["abd_l"], kin["abd_r"], kin["abd_l"] - kin["abd_r"], kin["lean"]], axis=1)
    vel = np.vstack([np.zeros((1, f.shape[1])), np.diff(f, axis=0)])
    return np.nan_to_num(np.hstack([f, vel]))


@dataclass
class MovementReport:
    patient_id: str
    duration_s: float
    n_frames: int
    backend: str
    metrics: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    series: dict = field(default_factory=dict)


class PhysioAnalyzer:
    def __init__(self, cfg: dict):
        self.cfg = cfg["video"]["physio"]
        self.min_conf = cfg["video"].get("min_keypoint_conf", 0.3)
        self.iforest: IsolationForest | None = None

    # -------- referência normal (aprendizado não supervisionado do "padrão esperado")
    def fit_reference(self, ref: PoseSequence):
        X = _frame_features(kinematics(ref, self.min_conf))
        self.iforest = IsolationForest(n_estimators=200, contamination=0.01, random_state=0).fit(X)
        return self

    def analyze(self, seq: PoseSequence, patient_id: str) -> MovementReport:
        c = self.cfg
        fps = seq.fps
        kin = kinematics(seq, self.min_conf)
        T = len(seq)
        rep = MovementReport(patient_id, T / fps, T, seq.backend)
        valid = ~np.isnan(kin["hip_y"])

        # ---------- queda (avaliada primeiro: frames pós-queda são excluídos da análise do exercício)
        w = max(1, int(c["fall_window_s"] * fps))
        hip_y = np.nan_to_num(kin["hip_y"], nan=np.nanmedian(kin["hip_y"]))
        drop = (hip_y[w:] - hip_y[:-w]) / (kin["body_h"] + 1e-9)
        fall_idx = np.where((drop > c["fall_drop_ratio"]) & (np.abs(kin["lean"][w:]) > 45))[0]
        fall_t = None
        end = T
        if len(fall_idx):
            f0 = int(fall_idx[0])
            fall_t = (f0 + w) / fps   # instante em que a descida se completa
            end = f0
            rep.events.append({"t": round(fall_t, 2), "type": "queda", "detail":
                               f"centro do quadril desceu {drop[f0]*100:.0f}% da altura corporal em {c['fall_window_s']}s"})
            rep.findings.append(Finding(patient_id, "video", "fall_detected", 0.95,
                                        f"Queda detectada durante a sessão em t={fall_t:.1f}s",
                                        evidence={"t_s": round(fall_t, 2), "drop_ratio": round(float(drop[f0]), 2)},
                                        source="pose keypoints + regra cinemática"))

        sl = slice(0, end)
        al, ar, lean = kin["abd_l"][sl], kin["abd_r"][sl], kin["lean"][sl]

        # ---------- repetições e ROM
        def reps(sig):
            sig = np.nan_to_num(sig)
            pk, _ = find_peaks(sig, prominence=30, distance=int(1.5 * fps))
            return pk, sig[pk]
        pl, rl = reps(al)
        pr, rr = reps(ar)
        rom_l = float(np.median(rl)) if len(rl) else float(np.nanmax(al))
        rom_r = float(np.median(rr)) if len(rr) else float(np.nanmax(ar))
        n_reps = int(max(len(pl), len(pr)))
        asym = abs(rom_l - rom_r)
        lean_peak = float(np.nanpercentile(np.abs(lean), 95))

        # ---------- tremor (punho esquerdo e direito)
        b, a = butter(3, [c["tremor_band_hz"][0] / (fps / 2), c["tremor_band_hz"][1] / (fps / 2)], btype="band")
        tremor = {}
        for s in ("l", "r"):
            wr = np.nan_to_num(kin["xy"][sl, K[s + "wri"]])
            if len(wr) > 30:
                bp = filtfilt(b, a, wr, axis=0)
                rms = float(np.sqrt((bp ** 2).sum(1).mean()))
                spec = np.abs(np.fft.rfft(wr[:, 0] - wr[:, 0].mean()))
                fr = np.fft.rfftfreq(len(wr), 1 / fps)
                band = (fr >= c["tremor_band_hz"][0]) & (fr <= c["tremor_band_hz"][1])
                fdom = float(fr[band][np.argmax(spec[band])]) if band.any() else 0.0
                tremor[s] = (rms / (kin["torso_len"] + 1e-9), fdom)
            else:
                tremor[s] = (0.0, 0.0)

        rep.metrics = {
            "repeticoes": n_reps,
            "rom_esquerdo_graus": round(rom_l, 1),
            "rom_direito_graus": round(rom_r, 1),
            "assimetria_graus": round(asym, 1),
            "inclinacao_tronco_p95_graus": round(lean_peak, 1),
            "tremor_esq_indice": round(float(tremor["l"][0]) * 100, 2),
            "tremor_esq_freq_hz": round(tremor["l"][1], 1),
            "tremor_dir_indice": round(float(tremor["r"][0]) * 100, 2),
            "queda_t_s": None if fall_t is None else round(fall_t, 2),
            "frames_validos_pct": round(float(100 * valid.mean()), 1),
        }

        lo = c["target_rom_deg"][0]
        for side, rom, name in (("l", rom_l, "esquerdo"), ("r", rom_r, "direito")):
            if rom < lo:
                deficit = (lo - rom) / lo
                rep.findings.append(Finding(patient_id, "video", f"reduced_rom_{side}", min(0.9, 0.3 + deficit),
                                            f"Amplitude de abdução do ombro {name} reduzida: {rom:.0f}° (meta ≥{lo}°)",
                                            evidence={"rom_deg": round(rom, 1), "meta_deg": lo},
                                            source="pose keypoints"))
        if asym > c["max_asymmetry_deg"]:
            rep.findings.append(Finding(patient_id, "video", "asymmetry", min(0.85, 0.3 + asym / 150),
                                        f"Assimetria entre membros de {asym:.0f}° (limite {c['max_asymmetry_deg']}°)",
                                        evidence={"asym_deg": round(asym, 1)}, source="pose keypoints"))
        if lean_peak > c["max_trunk_lean_deg"]:
            rep.findings.append(Finding(patient_id, "video", "trunk_compensation", min(0.7, 0.25 + lean_peak / 60),
                                        f"Compensação de tronco: inclinação lateral de {lean_peak:.0f}° "
                                        f"(limite {c['max_trunk_lean_deg']}°)",
                                        evidence={"lean_p95_deg": round(lean_peak, 1)}, source="pose keypoints"))
        for s, name in (("l", "esquerdo"), ("r", "direito")):
            idx, fdom = tremor[s]
            if idx > 0.015:
                rep.findings.append(Finding(patient_id, "video", f"tremor_{s}", min(0.65, 0.25 + idx * 10),
                                            f"Tremor no membro superior {name} (~{fdom:.1f} Hz, índice {idx*100:.1f}%)",
                                            evidence={"freq_hz": fdom, "indice_pct": round(idx * 100, 2)},
                                            source="pose keypoints + filtro passa-banda 3–7 Hz"))

        # ---------- padrão anômalo vs. referência (Isolation Forest)
        if self.iforest is not None:
            X = _frame_features(kin)[sl]
            pred = self.iforest.predict(X)
            frac = float((pred == -1).mean())
            rep.metrics["frames_fora_do_padrao_pct"] = round(100 * frac, 1)
            segs = _segments(pred == -1, fps, min_len_s=0.5)
            for s0, s1 in segs[:10]:
                rep.events.append({"t": round(s0, 2), "type": "fora_do_padrao",
                                   "detail": f"movimento fora do padrão de referência ({s0:.1f}s–{s1:.1f}s)"})
            if frac > 0.10:
                rep.findings.append(Finding(patient_id, "video", "pattern_deviation", min(0.8, 0.2 + frac),
                                            f"{frac*100:.0f}% dos frames fora do padrão da sessão de referência",
                                            evidence={"pct_frames": round(100 * frac, 1), "segmentos": len(segs)},
                                            source="Isolation Forest (referência normal)"))

        rep.series = {"t": (np.arange(T) / fps).tolist(), "abd_l": np.nan_to_num(kin["abd_l"]).tolist(),
                      "abd_r": np.nan_to_num(kin["abd_r"]).tolist(), "lean": np.nan_to_num(kin["lean"]).tolist()}
        rep.events.sort(key=lambda e: e["t"])
        return rep


def _segments(mask: np.ndarray, fps: float, min_len_s=0.5):
    segs, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - start) / fps >= min_len_s:
                segs.append((start / fps, i / fps))
            start = None
    return segs

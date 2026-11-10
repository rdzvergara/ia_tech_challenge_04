"""Detecção de anomalias em sinais vitais em tempo real (processamento amostra a amostra).

Camadas complementares (cada uma com persistência e cooldown para evitar fadiga de alarme):
  1. Limites clínicos absolutos (hard limits);
  2. NEWS2 — National Early Warning Score 2 (escore agregado de deterioração);
  3. Z-score robusto (mediana/MAD em janela deslizante) — picos e quedas abruptas;
  4. CUSUM sobre o desvio em relação à linha de base do paciente — tendências lentas (ex.: sepse);
  5. Isolation Forest multivariado treinado na linha de base do próprio paciente — combinações atípicas.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..schemas import Finding

SIGNALS = ["hr", "sbp", "dbp", "spo2", "rr", "temp"]
# Desvio-padrão mínimo de variabilidade clinicamente irrelevante (evita que linhas de base muito
# estáveis transformem oscilações fisiológicas — ex.: ritmo circadiano — em "anomalias")
SD_FLOOR = {"hr": 6.0, "sbp": 8.0, "dbp": 6.0, "spo2": 1.5, "rr": 2.0, "temp": 0.3}
LABEL = {"hr": "FC", "sbp": "PAS", "dbp": "PAD", "spo2": "SpO2", "rr": "FR", "temp": "Temperatura"}
UNIT = {"hr": "bpm", "sbp": "mmHg", "dbp": "mmHg", "spo2": "%", "rr": "irpm", "temp": "°C"}


def news2(s: dict) -> int:
    """NEWS2 simplificado (sem nível de consciência e O2 suplementar)."""
    sc = 0
    rr, sp, sbp, hr, t = s["rr"], s["spo2"], s["sbp"], s["hr"], s["temp"]
    sc += 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3
    sc += 3 if sp <= 91 else 2 if sp <= 93 else 1 if sp <= 95 else 0
    sc += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3
    sc += 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    sc += 3 if t <= 35.0 else 1 if t <= 36.0 else 0 if t <= 38.0 else 1 if t <= 39.0 else 2
    return sc


@dataclass
class _PatientState:
    window: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=60)))
    baseline: list = field(default_factory=list)
    base_stats: dict | None = None
    cusum_pos: dict = field(default_factory=lambda: defaultdict(float))
    cusum_neg: dict = field(default_factory=lambda: defaultdict(float))
    persist: dict = field(default_factory=lambda: defaultdict(int))
    last_alert: dict = field(default_factory=dict)
    iforest: IsolationForest | None = None
    n: int = 0


class StreamingVitalsMonitor:
    def __init__(self, cfg: dict):
        c = cfg["vitals"]
        self.hard = c["hard_limits"]
        self.win = c["zscore_window"]
        self.zthr = c["zscore_threshold"]
        self.persistence = c["persistence"]
        self.cooldown = pd.Timedelta(minutes=c["cooldown_minutes"])
        self.if_cfg = c["isolation_forest"]
        self.baseline_n = int(self.if_cfg["baseline_hours"] * 60 / c.get("sampling_minutes", 1))
        self.k, self.h = c["cusum"]["k"], c["cusum"]["h"]
        self.state: dict[str, _PatientState] = {}
        self.trace: list[dict] = []   # anotação por amostra (para gráficos/dashboard)

    # ------------------------------------------------------------------
    def _emit(self, st, pid, ts, kind, score, desc, evidence, source, out):
        """Aplica persistência + cooldown antes de emitir um achado."""
        st.persist[kind] += 1
        if st.persist[kind] < self.persistence:
            return
        last = st.last_alert.get(kind)
        if last is not None and ts - last < self.cooldown:
            return
        st.last_alert[kind] = ts
        out.append(Finding(pid, "vitals", kind, float(score), desc, ts.isoformat(), evidence, source))

    def update(self, sample: dict) -> list[Finding]:
        pid = sample["patient_id"]
        ts = pd.Timestamp(sample["timestamp"])
        st = self.state.setdefault(pid, _PatientState(window=defaultdict(lambda: deque(maxlen=self.win))))
        st.n += 1
        out: list[Finding] = []
        active: set[str] = set()
        vals = {s: float(sample[s]) for s in SIGNALS}

        # 1) limites clínicos
        for s, (lo, hi) in self.hard.items():
            v = vals[s]
            if v < lo or v > hi:
                kind = f"{s}_{'low' if v < lo else 'high'}"
                active.add(kind)
                dist = (lo - v) / lo if v < lo else (v - hi) / hi
                self._emit(st, pid, ts, kind, min(0.9, 0.55 + 2 * dist),
                           f"{LABEL[s]} {'baixa' if v < lo else 'elevada'}: {v:g} {UNIT[s]} (limite {lo}–{hi})",
                           {"valor": v, "limite": [lo, hi]}, "limite clínico", out)

        # 2) NEWS2
        n2 = news2(vals)
        if n2 >= 5:
            kind = "news2_high"
            active.add(kind)
            self._emit(st, pid, ts, kind, 0.85 if n2 >= 7 else 0.6,
                       f"NEWS2 = {n2} ({'risco alto — resposta de emergência' if n2 >= 7 else 'risco médio — avaliação urgente'})",
                       {"news2": n2, **vals}, "NEWS2", out)

        # 3) z-score robusto (janela deslizante) — antes de atualizar a janela
        zs = {}
        for s in SIGNALS:
            w = st.window[s]
            if len(w) >= 20:
                arr = np.asarray(w)
                med = np.median(arr)
                mad = 1.4826 * np.median(np.abs(arr - med)) + 1e-6
                mad = max(mad, {"hr": 2, "sbp": 3, "dbp": 2, "spo2": 0.7, "rr": 1, "temp": 0.1}[s])
                z = (vals[s] - med) / mad
                zs[s] = z
                if abs(z) > self.zthr:
                    kind = f"{s}_abrupt_change"
                    active.add(kind)
                    self._emit(st, pid, ts, kind, min(0.6, 0.3 + abs(z) / 40),
                               f"Mudança abrupta em {LABEL[s]}: {vals[s]:g} {UNIT[s]} (z robusto = {z:+.1f})",
                               {"valor": vals[s], "mediana_janela": float(med), "z": round(float(z), 2)},
                               "z-score robusto (mediana/MAD)", out)
            w.append(vals[s])

        # 4) linha de base + CUSUM + Isolation Forest
        if st.base_stats is None:
            st.baseline.append([vals[s] for s in SIGNALS])
            if len(st.baseline) >= self.baseline_n:
                B = np.array(st.baseline)
                floor = np.array([SD_FLOOR[s] for s in SIGNALS])
                st.base_stats = {"mean": B.mean(0), "std": np.maximum(B.std(0), floor)}
                st.iforest = IsolationForest(n_estimators=100, contamination=self.if_cfg["contamination"],
                                             random_state=0).fit(B)
        else:
            mu, sd = st.base_stats["mean"], st.base_stats["std"]
            x = np.array([vals[s] for s in SIGNALS])
            z = (x - mu) / sd
            for i, s in enumerate(SIGNALS):
                st.cusum_pos[s] = max(0.0, st.cusum_pos[s] + z[i] - self.k)
                st.cusum_neg[s] = max(0.0, st.cusum_neg[s] - z[i] - self.k)
                for direction, val in (("up", st.cusum_pos[s]), ("down", st.cusum_neg[s])):
                    # só reporta tendências clinicamente adversas
                    adverse = (s in ("hr", "rr", "temp") and direction == "up") or \
                              (s in ("spo2", "sbp") and direction == "down")
                    if adverse and val > self.h:
                        kind = f"{s}_trend_{direction}"
                        active.add(kind)
                        self._emit(st, pid, ts, kind, 0.45,
                                   f"Tendência sustentada de {'alta' if direction == 'up' else 'queda'} em {LABEL[s]} "
                                   f"(atual {vals[s]:g} {UNIT[s]}; base {mu[i]:.1f})",
                                   {"cusum": round(val, 1), "baseline": round(float(mu[i]), 1), "valor": vals[s]},
                                   "CUSUM vs. linha de base", out)
            # Isolation Forest: só avaliado quando algum sinal se afasta >3 DP da base (gate barato)
            if np.abs(z).max() > 3 and \
                    st.iforest.score_samples(x.reshape(1, -1))[0] < st.iforest.offset_:
                if_score = float(-st.iforest.score_samples(x.reshape(1, -1))[0])
                kind = "multivariate_anomaly"
                active.add(kind)
                top = [SIGNALS[i] for i in np.argsort(-np.abs(z))[:3]]
                self._emit(st, pid, ts, kind, 0.4,
                           "Combinação atípica de sinais vitais vs. linha de base: " +
                           ", ".join(f"{LABEL[s]} {vals[s]:g}" for s in top),
                           {"if_score": round(if_score, 3), "z": {s: round(float(z[SIGNALS.index(s)]), 1) for s in top}},
                           "Isolation Forest (linha de base do paciente)", out)
        # zera persistência das condições que não estão mais ativas
        for k in list(st.persist):
            if k not in active:
                st.persist[k] = 0
        self.trace.append({"timestamp": ts, "patient_id": pid, "news2": n2, "n_findings": len(out)})
        return out

    def run(self, df: pd.DataFrame) -> list[Finding]:
        """Reproduz o DataFrame como um fluxo em tempo real (ordenado por timestamp)."""
        out = []
        for rec in df.sort_values("timestamp").to_dict("records"):
            out.extend(self.update(rec))
        return out

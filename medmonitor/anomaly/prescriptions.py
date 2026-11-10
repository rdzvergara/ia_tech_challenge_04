"""Detecção de alterações inesperadas na evolução das prescrições.

Regras (farmacovigilância) + estatística:
  * dose diária acima do máximo de referência (peso maior para medicamentos de alta vigilância);
  * salto de dose entre prescrições consecutivas (> N×) — ex.: erro de casa decimal (≈10×);
  * suspensão abrupta de medicamentos que exigem desmame (ex.: betabloqueadores, clonidina, corticoides);
  * dose atípica em relação à distribuição hospitalar do mesmo fármaco (z-score robusto do log da dose).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import Finding

TAPER_REQUIRED = {"metoprolol", "propranolol", "atenolol", "clonidina", "prednisona", "hidrocortisona",
                  "clonazepam", "diazepam"}


def analyze_prescriptions(df: pd.DataFrame, cfg: dict) -> list[Finding]:
    c = cfg["prescriptions"]
    max_daily = c["max_daily_dose_mg"]
    high_alert = set(c["high_alert_drugs"])
    ratio_thr = c["max_dose_change_ratio"]
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["daily"] = df["dose"] * df["doses_per_day"]
    out: list[Finding] = []

    # distribuição hospitalar por fármaco (log-dose diária)
    pop = df.groupby("drug")["daily"].apply(lambda s: np.log(s.values))

    for pid, g in df.sort_values("timestamp").groupby("patient_id"):
        snapshots = [s for _, s in g.groupby("timestamp")]
        prev = None
        for snap in snapshots:
            ts = snap["timestamp"].iloc[0]
            for r in snap.itertuples():
                ha = r.drug in high_alert
                lim = max_daily.get(r.drug)
                if lim and r.daily > lim:
                    over = r.daily / lim
                    out.append(Finding(pid, "prescriptions", "dose_above_max", min(0.95, 0.55 + 0.1 * over + (0.15 if ha else 0)),
                                       f"{r.drug}: dose diária {r.daily:g} {r.unit} acima do máximo de referência "
                                       f"({lim} {r.unit}/dia){' — MEDICAMENTO DE ALTA VIGILÂNCIA' if ha else ''}",
                                       ts.isoformat(), {"droga": r.drug, "dose_diaria": r.daily, "maximo": lim,
                                                        "prescritor": r.prescriber}, "regra: dose máxima"))
                if prev is not None:
                    p = prev[prev["drug"] == r.drug]
                    if len(p):
                        ratio = r.daily / float(p["daily"].iloc[0])
                        if ratio >= ratio_thr or ratio <= 1 / ratio_thr:
                            decimal = 8 <= ratio <= 12 or 1 / 12 <= ratio <= 1 / 8
                            out.append(Finding(pid, "prescriptions", "dose_jump",
                                               min(0.95, 0.45 + 0.08 * abs(np.log2(ratio)) + (0.2 if ha else 0) + (0.15 if decimal else 0)),
                                               f"{r.drug}: dose mudou {ratio:.1f}× em relação à prescrição anterior "
                                               f"({float(p['daily'].iloc[0]):g} → {r.daily:g} {r.unit}/dia)"
                                               + (" — possível erro de casa decimal" if decimal else ""),
                                               ts.isoformat(), {"droga": r.drug, "razao": round(ratio, 2),
                                                                "prescritor": r.prescriber}, "regra: variação de dose"))
                # z-score robusto vs. hospital
                vals = pop.get(r.drug)
                if vals is not None and len(vals) >= 4:
                    med = np.median(vals); mad = 1.4826 * np.median(np.abs(vals - med))
                    if mad > 0:
                        z = (np.log(r.daily) - med) / mad
                        if abs(z) > 3.5:
                            out.append(Finding(pid, "prescriptions", "atypical_dose", 0.35,
                                               f"{r.drug}: dose atípica frente às prescrições do hospital (z={z:+.1f})",
                                               ts.isoformat(), {"droga": r.drug, "z": round(float(z), 2)},
                                               "z-score robusto (log-dose)"))
            # snapshot só com fármacos novos = acréscimo (não substitui a prescrição vigente)
            addition = prev is not None and not (set(snap["drug"]) & set(prev["drug"]))
            if prev is not None and not addition:
                stopped = set(prev["drug"]) - set(snap["drug"])
                for d in sorted(stopped & TAPER_REQUIRED):
                    out.append(Finding(pid, "prescriptions", "abrupt_discontinuation", 0.55,
                                       f"{d}: suspenso abruptamente sem desmame (risco de efeito rebote)",
                                       ts.isoformat(), {"droga": d}, "regra: suspensão abrupta"))
            prev = pd.concat([prev, snap]) if addition else snap
    return out

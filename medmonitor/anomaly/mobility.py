"""Padrões de movimentação do paciente durante a internação (acelerômetro / sensor de leito).

* Queda suspeita: pico de aceleração seguido de imobilidade prolongada fora do leito;
* Hipoatividade: atividade diurna muito abaixo da linha de base do próprio paciente (letargia/delirium hipoativo);
* Agitação noturna: múltiplas saídas do leito durante a noite (risco de queda/delirium hiperativo);
* Padrão horário atípico: Isolation Forest sobre features horárias (média, variabilidade, tempo fora do leito).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..schemas import Finding


def _is_night(h, night):
    a, b = night
    return (h >= a) | (h < b)


def analyze_mobility(df: pd.DataFrame, cfg: dict) -> list[Finding]:
    c = cfg["mobility"]
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out: list[Finding] = []
    for pid, g in df.sort_values("timestamp").groupby("patient_id"):
        g = g.set_index("timestamp")
        act, oob = g["activity"].values, g["out_of_bed"].values
        ts = g.index

        # queda suspeita
        for i in np.where(act >= 0.95)[0]:
            after = act[i + 1:i + 11]
            if len(after) == 10 and after.max() < 0.03 and oob[i:i + 11].all():
                out.append(Finding(pid, "mobility", "fall_suspected", 0.9,
                                   "Queda suspeita: impacto seguido de imobilidade fora do leito por ≥10 min",
                                   ts[i].isoformat(), {"pico_atividade": float(act[i])}, "regra: impacto + imobilidade"))

        hourly = g.resample("1h").agg({"activity": ["mean", "std"], "out_of_bed": "sum"})
        hourly.columns = ["act_mean", "act_std", "oob_min"]
        hourly["hour"] = hourly.index.hour
        day = ~_is_night(hourly["hour"], c["night_hours"])

        # hipoatividade vs. linha de base (primeiras 24h, período diurno)
        base = hourly[day & (hourly.index < hourly.index[0] + pd.Timedelta(hours=24))]["act_mean"].mean()
        roll = hourly["act_mean"].where(day).rolling(c["window_hours"], min_periods=c["window_hours"]).mean()
        low = roll < 0.5 * base
        if low.any():
            t0 = low[low].index[0]
            out.append(Finding(pid, "mobility", "hypoactivity", 0.5,
                               f"Hipoatividade: atividade diurna {roll[t0]:.2f} vs. base {base:.2f} "
                               f"(janela {c['window_hours']} h)", t0.isoformat(),
                               {"atividade": round(float(roll[t0]), 3), "base": round(float(base), 3)},
                               "comparação com linha de base"))

        # agitação noturna: saídas do leito (bordas de subida) à noite numa janela
        exits = pd.Series(np.diff(np.r_[0, oob]) == 1, index=ts)
        night_exits = exits[_is_night(ts.hour, c["night_hours"])]
        cnt = night_exits.rolling(f"{c['window_hours']}h").sum()
        hit = cnt[cnt >= c["night_bed_exit_threshold"]]
        if len(hit):
            out.append(Finding(pid, "mobility", "night_agitation", 0.55,
                               f"Agitação noturna: {int(hit.iloc[0])} saídas do leito em {c['window_hours']} h "
                               "(risco de queda / delirium)", hit.index[0].isoformat(),
                               {"saidas": int(hit.max())}, "regra: saídas noturnas do leito"))

        # Isolation Forest horário (treinado nas primeiras 24 h)
        X = hourly[["act_mean", "act_std", "oob_min"]].fillna(0).assign(
            hs=np.sin(2 * np.pi * hourly["hour"] / 24), hc=np.cos(2 * np.pi * hourly["hour"] / 24))
        train = X.iloc[:24]
        if len(train) >= 12:
            iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(train)
            pred = iso.predict(X.iloc[24:])
            n_anom = int((pred == -1).sum())
            if n_anom >= 4:
                first = X.iloc[24:].index[np.where(pred == -1)[0][0]]
                out.append(Finding(pid, "mobility", "atypical_mobility_pattern", 0.35,
                                   f"{n_anom} horas com padrão de movimentação atípico vs. 1º dia de internação",
                                   first.isoformat(), {"horas_atipicas": n_anom}, "Isolation Forest (features horárias)"))
    return out

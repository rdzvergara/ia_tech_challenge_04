"""Baixa sinais vitais reais do PhysioNet (BIDMC PPG and Respiration Dataset) e roda o monitor de anomalias.

Fonte: https://physionet.org/content/bidmc/1.0.0/  (acesso aberto, licença ODC-By)
Cada registro "Numerics" traz ~8 min a 1 Hz com HR, PULSE, RESP e SpO2 de pacientes de UTI.

Uso:
    python scripts/download_physionet.py --records 10 --run

Observação: o BIDMC não tem pressão arterial nem temperatura; essas colunas são preenchidas com valores
fisiológicos constantes (marcados em ``imputed``) apenas para manter o formato de entrada do monitor.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BASE = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_{:02d}_Numerics.csv"


def fetch(rec: int) -> pd.DataFrame:
    r = requests.get(BASE.format(rec), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in df.columns if c.startswith("time"))
    out = pd.DataFrame({
        "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(df[tcol], unit="s"),
        "patient_id": f"BIDMC{rec:02d}",
        "hr": df.get("hr"), "spo2": df.get("spo2"), "rr": df.get("resp"),
    })
    out["sbp"], out["dbp"], out["temp"], out["imputed"] = 120.0, 80.0, 36.8, "sbp,dbp,temp"
    return out.dropna(subset=["hr", "spo2", "rr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=5, help="quantidade de registros (1..53)")
    ap.add_argument("--run", action="store_true", help="executa o monitor de sinais vitais nos dados baixados")
    a = ap.parse_args()
    dest = ROOT / "data" / "physionet"
    dest.mkdir(parents=True, exist_ok=True)
    frames = []
    for rec in range(1, a.records + 1):
        try:
            frames.append(fetch(rec)); print(f"[ok] bidmc_{rec:02d}")
        except Exception as e:  # noqa: BLE001
            print(f"[erro] bidmc_{rec:02d}: {e}")
    if not frames:
        sys.exit("Nenhum registro baixado (verifique o acesso a physionet.org).")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(dest / "vitals_bidmc.csv", index=False)
    print(f"Salvo: {dest / 'vitals_bidmc.csv'} ({len(df)} amostras a 1 Hz)")

    if a.run:
        from medmonitor.anomaly.vitals import StreamingVitalsMonitor
        from medmonitor.config import load_settings
        S = load_settings()
        S = {**S, "vitals": {**S["vitals"], "zscore_window": 60, "persistence": 5, "cooldown_minutes": 2,
                             "isolation_forest": {"contamination": 0.01, "baseline_hours": 2 / 60}}}
        F = StreamingVitalsMonitor(S).run(df)
        for f in F:
            print(f"{f.patient_id} {f.timestamp} [{f.severity.name}] {f.description}")
        print(f"Total de achados: {len(F)}")


if __name__ == "__main__":
    main()

"""Gráficos de resultados (PNG) para o relatório técnico."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .anomaly.vitals import LABEL, UNIT, news2  # noqa: E402

SERIES = "#2a78d6"
CRIT, SERIOUS = "#d03b3b", "#ec835a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_vitals(df: pd.DataFrame, findings, alerts, hard_limits: dict, pid: str, out: Path):
    g = df[df.patient_id == pid].sort_values("timestamp").copy()
    g["news2"] = [news2(r) for r in g[["hr", "sbp", "spo2", "rr", "temp"]].to_dict("records")]
    sigs = ["hr", "sbp", "spo2", "rr", "temp", "news2"]
    fig, axes = plt.subplots(len(sigs), 1, figsize=(10, 9), dpi=110, sharex=True)
    al = [a for a in alerts if a.patient_id == pid and a.severity.name in ("HIGH", "CRITICAL")]
    for ax, s in zip(axes, sigs):
        _style(ax)
        ax.plot(g["timestamp"], g[s], color=SERIES, lw=1.0)
        if s in hard_limits:
            for v in hard_limits[s]:
                ax.axhline(v, color=MUTED, lw=0.8, ls=":")
        if s == "news2":
            ax.axhline(5, color=MUTED, lw=0.8, ls=":")
        ax.set_ylabel(f"{LABEL.get(s, 'NEWS2')}\n{UNIT.get(s, 'pontos')}", fontsize=7, color=INK)
        for a in al:
            ax.axvline(pd.Timestamp(a.created_at), color=CRIT if a.severity.name == "CRITICAL" else SERIOUS,
                       lw=1.0, alpha=0.8)
    first = {}
    for f in findings:
        if f.patient_id == pid and f.modality == "vitals":
            first.setdefault(f.kind, pd.Timestamp(f.timestamp))
    if first:
        t0 = min(first.values())
        axes[0].annotate("1ª anomalia", xy=(t0, axes[0].get_ylim()[1]), fontsize=7, color=INK,
                         xytext=(4, -10), textcoords="offset points")
    axes[0].set_title(f"Paciente {pid} — sinais vitais (linhas verticais: alertas HIGH/CRITICAL; "
                      "pontilhado: limites clínicos)", fontsize=9, color=INK, loc="left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_risk(history: pd.DataFrame, out: Path, patients=()):
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
    _style(ax)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for (pid, g), c in zip(history.groupby("patient_id"), colors):
        g = g.sort_values("timestamp")
        ax.step(g["timestamp"], g["risk"], where="post", color=c, lw=2, label=pid)
        ax.text(g["timestamp"].iloc[-1], g["risk"].iloc[-1], f" {pid}", color=INK, fontsize=8, va="center")
    quiet = [p for p in patients if p not in set(history["patient_id"])]
    if quiet:
        ax.plot(history["timestamp"].agg(["min", "max"]), [0, 0], color="#1baf7a", lw=2,
                label=", ".join(quiet) + " (sem achados)")
    for y, lbl in ((0.75, "CRITICAL"), (0.5, "HIGH"), (0.3, "MEDIUM")):
        ax.axhline(y, color=MUTED, lw=0.8, ls=":")
        ax.text(ax.get_xlim()[0], y, f" {lbl}", fontsize=7, color=MUTED, va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("risco fundido", fontsize=8, color=INK)
    ax.set_title("Risco multimodal fundido por paciente (atualizado a cada novo achado; janela de 12 h)", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

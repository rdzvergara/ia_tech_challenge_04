"""Central de monitoramento (Streamlit).

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT, DATA = ROOT / "outputs", ROOT / "data"

st.set_page_config(page_title="MedMonitor — Central", layout="wide")
st.title("MedMonitor · Central de monitoramento multimodal")

with st.sidebar:
    st.header("Execução")
    if st.button("▶ Rodar pipeline completo", use_container_width=True):
        from medmonitor.pipeline import run_all
        with st.spinner("Processando áudio, vídeo, sinais vitais, prescrições e mobilidade..."):
            run_all(quiet=True)
        st.success("Concluído")
    from medmonitor.audio.azure_speech import azure_available as sp
    from medmonitor.audio.azure_text import azure_available as la
    st.caption(f"Azure Speech: **{'conectado' if sp() else 'offline'}**  \nAzure Language: **{'conectado' if la() else 'offline'}**")

if not (OUT / "patient_summary.json").exists():
    st.info("Nenhum resultado ainda. Clique em **Rodar pipeline completo** na barra lateral.")
    st.stop()

summary = json.loads((OUT / "patient_summary.json").read_text(encoding="utf-8"))
patients = json.loads((DATA / "patients.json").read_text(encoding="utf-8")) if (DATA / "patients.json").exists() else {}
alerts = [json.loads(line) for line in (OUT / "alerts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "🟢"}

cols = st.columns(len(summary))
for c, (pid, s) in zip(cols, sorted(summary.items())):
    info = patients.get(pid, {})
    c.metric(f"{ICON[s['severity']]} {pid} · {info.get('nome', '')}", f"{s['risk']:.2f}", s["severity"], delta_color="off")
    c.caption(info.get("perfil", ""))

tab_al, tab_vit, tab_au, tab_vid, tab_res = st.tabs(["🚨 Alertas", "📈 Sinais vitais", "🎙️ Áudio", "🎥 Vídeo", "📊 Resultados"])

with tab_al:
    sev = st.multiselect("Severidade", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM"])
    for a in reversed(alerts):
        if a["severity"] not in sev:
            continue
        with st.expander(f"{ICON[a['severity']]} {a['created_at'].replace('T', ' ')} — {a['title']}"):
            st.markdown(f"**Ação recomendada:** {a['recommended_action']}")
            st.dataframe(pd.DataFrame(a["findings"])[["modality", "severity", "description", "source", "timestamp"]],
                         hide_index=True, use_container_width=True)

with tab_vit:
    pid = st.selectbox("Paciente", sorted(summary), key="vit")
    img = OUT / f"vitais_{pid}.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
    if (OUT / "risco_multimodal.png").exists():
        st.image(str(OUT / "risco_multimodal.png"), use_container_width=True)

with tab_au:
    pid = st.selectbox("Paciente", sorted(summary), key="au")
    f = OUT / "audio" / f"audio_{pid}.json"
    wav = DATA / "audio" / f"consulta_{pid}.wav"
    if wav.exists():
        st.audio(str(wav))
    if f.exists():
        r = json.loads(f.read_text(encoding="utf-8"))
        st.markdown(f"**Transcrição** ({r['transcript']['engine']}): _{r['transcript']['text']}_")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Sentimento:** {r['insights']['sentiment']} ({r['insights']['engine']})")
        c1.markdown("**Termos críticos:** " + (", ".join(r["insights"]["critical_terms"]) or "—"))
        c1.dataframe(pd.DataFrame(r["insights"]["health_entities"]), hide_index=True)
        c2.dataframe(pd.Series(r["features"], name="valor"), use_container_width=True)
        for x in r["findings"]:
            st.warning(f"[{x['severity']}] {x['description']}")

with tab_vid:
    vids = sorted((OUT / "video").glob("*_anotado.mp4"))
    if vids:
        v = st.selectbox("Vídeo", vids, format_func=lambda p: p.name)
        st.video(str(v))
    for md in sorted((OUT / "video").glob("video_*_relatorio.md")):
        with st.expander(md.name):
            st.markdown(md.read_text(encoding="utf-8").split("![](")[0])
            png = md.with_name(md.name.replace("_relatorio.md", "_angulos.png"))
            if png.exists():
                st.image(str(png))

with tab_res:
    if (OUT / "RESULTADOS.md").exists():
        st.markdown((OUT / "RESULTADOS.md").read_text(encoding="utf-8"))

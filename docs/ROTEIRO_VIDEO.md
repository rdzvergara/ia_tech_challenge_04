# Roteiro do vídeo de demonstração (≤ 15 min)

Gravação sugerida: OBS Studio (tela + webcam no canto), 1080p. Publicar no YouTube como **não listado**.
Antes de gravar: `python -m medmonitor.cli run-all` uma vez (para os arquivos de `outputs/` existirem) e,
se possível, preencher o `.env` com as chaves do Azure para mostrar a integração real.

| # | Tempo | Cena | O que mostrar / falar |
|---|---|---|---|
| 1 | 0:00–1:00 | Apresentação | Grupo, desafio da Fase 4: monitorar pacientes com vídeo, áudio, texto e sinais vitais para identificar risco precoce. Mostrar o diagrama Mermaid do README. |
| 2 | 1:00–2:30 | Arquitetura | Percorrer `medmonitor/`: um analisador por modalidade → contrato `Finding` → fusão → alertas. Explicar por que a fusão é tardia (cada fonte chega num ritmo diferente). |
| 3 | 2:30–3:30 | Dados | `python scripts/generate_data.py`. Mostrar os 3 pacientes (P001 controle, P002 sepse, P003 pós-AVC) e o `ground_truth.json`. Citar o script do PhysioNet para dados reais. |
| 4 | 3:30–6:00 | **Vídeo** | Abrir `data/video/fisio_P001.mp4` (normal) e `fisio_P003.mp4`. Rodar `python -m medmonitor.cli video data/video/fisio_P003.mp4 --patient P003 --backend openpose_json --json-dir data/video/fisio_P003_openpose --reference data/video/fisio_P001_openpose`. Mostrar o vídeo anotado (esqueleto + faixa de ALERTA na queda), o gráfico de ângulos e o relatório Markdown gerado. Em seguida, `outputs/video/cena_LEITO-07_anotado.mp4`: YOLOv8 detectando pessoas/objetos em imagens reais e zona crítica. **Opcional e recomendado:** gravar 20 s de você fazendo o exercício de abdução de ombro e rodar com `--backend yolov8` ao vivo. |
| 5 | 6:00–8:30 | **Áudio + Azure** | Tocar `consulta_P001.wav` e `consulta_P003.wav`. Rodar `python -m medmonitor.cli audio data/audio/consulta_P003.wav --patient P003`. Mostrar no JSON: `"engine": "azure-speech"` (transcrição), sentimento/entidades do **Azure Text Analytics for Health**, termos críticos e biomarcadores (jitter, shimmer, HNR) que indicam disartria. Mostrar rapidamente os recursos no portal Azure (Speech e Language) e o `.env.example`. |
| 6 | 8:30–10:30 | **Anomalias em tempo real** | `python -m medmonitor.cli stream --patient P002 --speed 7200` — alertas surgindo enquanto os dados "chegam". Abrir `outputs/vitais_P002.png`: CUSUM detecta a tendência às 05:50, ~3 h antes de cruzar os limites clínicos (08:48). Mostrar que o artefato de SpO2 de P001 não gerou alarme (persistência). Prescrições: erro decimal da enoxaparina; mobilidade: queda e agitação noturna. |
| 7 | 10:30–12:30 | **Fusão e alerta à equipe** | `python -m medmonitor.cli run-all`: alertas coloridos no console. Abrir `streamlit run dashboard/app.py`: cartões de risco, aba Alertas (expandir o CRITICAL de P003 às 14:58 que une queda no vídeo + disartria + prescrição), abas Áudio e Vídeo. Se configurado, mostrar o Adaptive Card chegando no **Microsoft Teams**. |
| 8 | 12:30–14:00 | Resultados | `outputs/RESULTADOS.md`: 16/16 eventos corretos, 344 achados → 11 alertas, gráfico de risco multimodal. `pytest -q` (11 testes passando). |
| 9 | 14:00–15:00 | Encerramento | Limitações (dados sintéticos, calibração por paciente, fine-tuning para instrumentais cirúrgicos) e arquitetura de produção no Azure (Event Hubs, Functions, Azure ML, Logic Apps, Key Vault). |

## Dicas

- Deixe os comandos prontos num arquivo de texto para copiar e colar (evita digitação no vídeo).
- Aumente a fonte do terminal (≥ 16 pt) e use o tema claro do dashboard.
- Corte esperas: o `run-all` leva ~45 s; grave e acelere esse trecho na edição.
- Se não tiver chaves Azure, mostre o modo offline e explique o fallback, mas o requisito pede a integração —
  vale criar os recursos no nível gratuito (F0) só para a gravação (ver `docs/AZURE_SETUP.md`).

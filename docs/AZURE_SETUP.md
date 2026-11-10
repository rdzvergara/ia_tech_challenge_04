# Configuração dos serviços Azure

O MedMonitor usa dois recursos do **Azure AI services** (antigo Cognitive Services). O nível gratuito (F0)
é suficiente para a demonstração.

## 1. Azure AI Speech (Speech to Text)

1. Portal Azure → **Create a resource** → *Speech* (Azure AI services).
2. Região: `Brazil South` (ou `East US`); tipo de preço **Free F0**.
3. Após criar: **Keys and Endpoint** → copie `KEY 1` e a região.
4. No `.env`:
   ```
   AZURE_SPEECH_KEY=<KEY 1>
   AZURE_SPEECH_REGION=brazilsouth
   AZURE_SPEECH_LANGUAGE=pt-BR
   ```

Implementação: `medmonitor/audio/azure_speech.py` — `SpeechRecognizer` com reconhecimento **contínuo**
(áudios longos), saída *Detailed* (confiança por segmento) e timestamps por palavra.

## 2. Azure AI Language (Text Analytics)

1. Portal Azure → **Create a resource** → *Language service*.
   Em *Custom features* não é preciso marcar nada.
2. Região com suporte a **Text Analytics for Health** (ex.: `East US`, `West Europe`); preço **Free F0**.
3. **Keys and Endpoint** → copie `KEY 1` e o `Endpoint`.
4. No `.env`:
   ```
   AZURE_LANGUAGE_KEY=<KEY 1>
   AZURE_LANGUAGE_ENDPOINT=https://<nome>.cognitiveservices.azure.com/
   ```

Implementação: `medmonitor/audio/azure_text.py`
- `analyze_sentiment(show_opinion_mining=True)` — sentimento do relato do paciente;
- `extract_key_phrases` — frases-chave;
- `begin_analyze_healthcare_entities` — entidades clínicas (sintomas, estruturas corporais, medicamentos)
  com detecção de negação/certeza. Se o recurso/região não suportar, o restante continua funcionando.

## 3. Alertas no Microsoft Teams (opcional)

1. No canal do Teams: **⋯ → Workflows → "Post to a channel when a webhook request is received"**
   (ou *Incoming Webhook* nos conectores clássicos).
2. Copie a URL gerada para `ALERT_WEBHOOK_URL` no `.env`.
3. Alertas **HIGH** e **CRITICAL** chegam como *Adaptive Card* com título, ação recomendada e evidências.

Em produção, a URL pode apontar para uma **Azure Logic App** que roteia por severidade
(SMS/pager para CRITICAL, Teams para HIGH) e registra no prontuário (FHIR `Flag`/`Communication`).

## 4. Testando

```bash
python -m medmonitor.cli audio data/audio/consulta_P002.wav --patient P002
```

A saída deve mostrar `"engine": "azure-speech"` na transcrição e `"engine": "azure-language"` nos insights.
Para forçar o modo offline mesmo com chaves: `MEDMONITOR_OFFLINE=true`.

## Arquitetura de produção sugerida

| Função | Serviço Azure |
|---|---|
| Ingestão de sinais vitais em tempo real | Azure IoT Hub / Event Hubs |
| Execução do monitor em streaming | Azure Functions ou Container Apps (consumidor do Event Hub) |
| Armazenamento de vídeos/áudios | Blob Storage (gatilho de processamento) |
| Inferência de vídeo (YOLOv8/OpenPose) em GPU | Azure Machine Learning (endpoint gerenciado) |
| Transcrição e análise de texto | Azure AI Speech + Azure AI Language |
| Roteamento de alertas | Logic Apps → Teams / SMS / prontuário |
| Segredos | Key Vault (substitui o `.env`) |
| Observabilidade e auditoria | Application Insights + Log Analytics |

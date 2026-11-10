# Resultados da execução

Gerado automaticamente por `python -m medmonitor.cli run-all`.

## Risco final por paciente

| Paciente | Risco | Nível | Modalidades (score) | Achados |
|---|---|---|---|---|
| P001 | 0.00 | INFO |  | 0 |
| P002 | 1.00 | CRITICAL | vitals 1.00, prescriptions 1.00, audio 0.90, mobility 0.65 | 321 |
| P003 | 1.00 | CRITICAL | video 1.00, vitals 1.00, prescriptions 0.96, audio 0.90, mobility 0.90 | 23 |

## Alertas emitidos (11)

| Horário | Paciente | Nível | Risco | Modalidades | Resumo |
|---|---|---|---|---|---|
| 11-08 20:02 | P003 | HIGH | 0.52 | vitals | PAS elevada: 192 mmHg (limite 85–180) |
| 11-09 05:50 | P002 | MEDIUM | 0.36 | vitals, mobility | Tendência sustentada de alta em Temperatura (atual 38.1 °C; base 37.2); 20 horas com padrão de movimentação atípico vs. 1º dia de internação |
| 11-09 08:00 | P002 | CRITICAL | 0.79 | prescriptions, vitals, mobility | enoxaparina: dose diária 400 mg acima do máximo de referência (200 mg/dia) — MEDICAMENTO DE ALTA VIGILÂNCIA; Tendência sustentada de alta em Temperatura (atual  |
| 11-09 08:00 | P003 | CRITICAL | 0.78 | vitals, prescriptions | insulina_regular: dose diária 180 UI acima do máximo de referência (150 UI/dia) — MEDICAMENTO DE ALTA VIGILÂNCIA; PAS elevada: 184 mmHg (limite 85–180); Mudança |
| 11-09 09:30 | P002 | CRITICAL | 1.00 | prescriptions, vitals, audio, mobility | enoxaparina: dose diária 400 mg acima do máximo de referência (200 mg/dia) — MEDICAMENTO DE ALTA VIGILÂNCIA; morfina: dose mudou 2.5× em relação à prescrição an |
| 11-09 10:00 | P003 | CRITICAL | 0.78 | prescriptions, audio | insulina_regular: dose mudou 6.0× em relação à prescrição anterior (30 → 180 UI/dia); insulina_regular: dose diária 180 UI acima do máximo de referência (150 UI |
| 11-09 13:41 | P002 | CRITICAL | 1.00 | vitals, prescriptions, audio, mobility | enoxaparina: dose diária 400 mg acima do máximo de referência (200 mg/dia) — MEDICAMENTO DE ALTA VIGILÂNCIA; NEWS2 = 12 (risco alto — resposta de emergência); T |
| 11-09 14:58 | P003 | CRITICAL | 0.96 | prescriptions, video, audio | Queda detectada durante a sessão em t=21.7s; insulina_regular: dose mudou 6.0× em relação à prescrição anterior (30 → 180 UI/dia); insulina_regular: dose diária |
| 11-09 16:02 | P003 | CRITICAL | 1.00 | video, prescriptions, audio, mobility, vitals | Queda detectada durante a sessão em t=21.7s; Queda suspeita: impacto seguido de imobilidade fora do leito por ≥10 min; FC baixa: 33 bpm (limite 40–130); insulin |
| 11-09 17:41 | P002 | CRITICAL | 1.00 | vitals, prescriptions, audio, mobility | enoxaparina: dose diária 400 mg acima do máximo de referência (200 mg/dia) — MEDICAMENTO DE ALTA VIGILÂNCIA; NEWS2 = 13 (risco alto — resposta de emergência); T |
| 11-09 21:41 | P002 | CRITICAL | 0.76 | vitals | NEWS2 = 13 (risco alto — resposta de emergência); SpO2 baixa: 86 % (limite 90–101); FR elevada: 29 irpm (limite 8–28); PAS baixa: 82 mmHg (limite 85–180) |

## Avaliação contra o ground truth — 16/16 corretos

| paciente   | modalidade    | evento                                                           | esperado_alerta   | detectado   | correto   |   latencia_min |   achados |
|:-----------|:--------------|:-----------------------------------------------------------------|:------------------|:------------|:----------|---------------:|----------:|
| P001       | vitals        | artefato_sensor_spo2                                             | False             | False       | True      |            nan |         0 |
| P002       | vitals        | deterioracao_septica                                             | True              | True        | True      |            110 |       312 |
| P003       | vitals        | crise_hipertensiva                                               | True              | True        | True      |              2 |         5 |
| P003       | vitals        | bradicardia                                                      | True              | True        | True      |              2 |         2 |
| P002       | prescriptions | dose_10x_enoxaparina                                             | True              | True        | True      |            nan |         2 |
| P002       | prescriptions | morfina_aumento_2.5x                                             | True              | True        | True      |            nan |         1 |
| P003       | prescriptions | insulina_acima_limite                                            | True              | True        | True      |            nan |         1 |
| P003       | prescriptions | suspensao_abrupta_betabloqueador                                 | True              | True        | True      |            nan |         1 |
| P002       | mobility      | hipoatividade_letargia                                           | True              | True        | True      |            120 |         2 |
| P002       | mobility      | agitacao_noturna                                                 | True              | True        | True      |             31 |         1 |
| P003       | mobility      | queda                                                            | True              | True        | True      |              0 |         1 |
| P001       | video         | sessao_normal                                                    | False             | False       | True      |            nan |         0 |
| P003       | video         | rom_reduzida_esquerda+assimetria+compensacao_tronco+tremor+queda | True              | True        | True      |            nan |         6 |
| P001       | audio         | fala_normal                                                      | False             | False       | True      |            nan |         0 |
| P002       | audio         | fadiga+dificuldade_respiratoria+termos_criticos                  | True              | True        | True      |            nan |         3 |
| P003       | audio         | disartria+termos_criticos                                        | True              | True        | True      |            nan |         2 |

## Tempos de processamento (s)

`{'audio_s': 3.1, 'video_s': 23.6, 'vitals_rx_mobility_s': 19.5, 'total_s': 47.5}`

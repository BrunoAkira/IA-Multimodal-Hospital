# example_data/

Amostras pequenas de bases públicas, incluídas para que o projeto rode sem downloads grandes:

```bash
export HOSPITAL_AI_DATASETS=example_data
python scripts/download_example_data.py      # base de fala (não redistribuída aqui)
python main.py all
python main.py video-validation
```

| Pasta | Conteúdo | Origem | Licença |
|---|---|---|---|
| `clinical/` | Sinais vitais (só os itens usados), prescrições e transferências | MIMIC-IV Clinical Database Demo v2.2 (PhysioNet) | Open Data Commons ODbL v1.0 |
| `motion_capture/` | Alguns exercícios de um participante (`.amc`) | UL-RED — University of Liverpool Rehabilitation Exercise Dataset | CC BY 4.0 |
| `exercise_videos/` | Gabarito dos vídeos escolhidos e ângulos das articulações já calculados | REHAB24-6 (Černek, Sedmidubsky, Budikova, SISAP 2024) | CC BY-NC 4.0 |
| `physio_videos/` | Trechos curtos de vídeo (640 px, 15 quadros/s) | REHAB24-6 | CC BY-NC 4.0 |
| `speech/` | Baixado por `scripts/download_example_data.py` (não versionado) | TORGO Database (Rudzicz et al., 2012) | Uso acadêmico, conforme os autores |

As amostras são subconjuntos das bases originais, sem alterações nos valores. Referências completas
no README principal.

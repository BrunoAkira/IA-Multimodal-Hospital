# data/

Pasta única dos dados de entrada do Hospital AI, com uma subpasta por tipo de dado.
O repositório traz amostras pequenas de bases públicas, sem alteração nos valores, para que o
projeto rode logo depois do clone. Dados novos (bases completas ou dados próprios no mesmo
formato) entram nas mesmas subpastas; os formatos estão no README principal, seção 4.

| Pasta | Amostra incluída | Origem | Licença |
|---|---|---|---|
| `clinical/` | Sinais vitais (só os itens usados), prescrições e transferências de 100 pacientes | MIMIC-IV Clinical Database Demo v2.2 (PhysioNet) | Open Data Commons ODbL v1.0 |
| `speech/` | Não incluída: baixada por `python scripts/download_speech_data.py` | TORGO Database (Rudzicz et al., 2012) | Uso acadêmico, conforme os autores |
| `motion_capture/` | Exercícios de um participante (`S01`, arquivos `.amc`) | UL-RED — University of Liverpool Rehabilitation Exercise Dataset | CC BY 4.0 |
| `exercise_videos/` | Gabarito (`Segmentation.csv`) e ângulos das articulações já calculados de 8 vídeos (`angles/`) | REHAB24-6 (Černek, Sedmidubsky, Budikova, SISAP 2024) | CC BY-NC 4.0 |
| `physio_videos/` | Dois trechos de 20 s (`*_example.mp4`) | REHAB24-6 | CC BY-NC 4.0 |
| `consultations/` | 12 roteiros de consulta em inglês, escritos para o projeto | Hospital AI | — |

Arquivos grandes adicionados localmente (vídeos, áudios `.wav`, `.parquet`, participantes além do
`S01`, vídeos em `exercise_videos/Ex*/`) ficam fora do Git pelo `.gitignore`.

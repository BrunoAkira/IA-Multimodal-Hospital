# datasets/

Pasta dos dados de entrada. É ignorada pelo Git: **dados clínicos e de pacientes nunca são versionados**.

Estrutura esperada (formatos e bases de exemplo no README principal, seção 4):

```
datasets/
  clinical/          hosp/  icu/                       # tabelas clínicas (CSV / CSV.gz)
  speech/            *.parquet                         # gravações de voz com texto e grupo
  motion_capture/    S01/ S02/ ...                     # captura de movimento (.amc)
  exercise_videos/   Ex1/ ... Ex6/  Segmentation.csv   # vídeos de exercícios + gabarito
```

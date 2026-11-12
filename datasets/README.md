# datasets/

Coloque aqui os datasets baixados. Esta pasta é ignorada pelo Git: **nunca versione dados clínicos**.

Estrutura esperada (detalhes e links no README principal, seção 4):

```
datasets/
  clinical/   hosp/  icu/            # base clínica de UTI (CSV.gz)
  audio/      *.parquet              # base de fala
  video/      S01/ S02/ ...          # captura de movimento (.amc)
  rehab24/    Ex1/ ... Ex6/  Segmentation.csv   # vídeos RGB de exercícios + gabarito
```

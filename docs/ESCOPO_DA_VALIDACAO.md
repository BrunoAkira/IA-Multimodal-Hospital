# Escopo e limites da validação

Este projeto é um protótipo acadêmico. Os resultados do README e do relatório técnico devem ser
interpretados dentro dos limites abaixo.

## Sinais vitais

A base clínica não traz rótulos de anomalia revisados por especialistas. A avaliação injeta desvios
artificiais em 2% das medições e compara as detecções com esses rótulos sintéticos. Valores reais
fora dos padrões também podem ser classificados como falsos positivos nesse cálculo. Portanto, a
precisão informada (19% a 41% no limiar 0,50) não é uma estimativa de precisão clínica.

O comando `python main.py clinical` agora grava uma curva de precisão, recall e F1 nos limiares de
score 0,50, 0,60, 0,70, 0,80 e 0,90 em `results/clinical_summary.json`. `python main.py summary`
mostra pontos representativos dessa curva. A curva ajuda a examinar o compromisso entre alertas e
recall, mas não deve ser usada para escolher um limiar final com o mesmo conjunto de avaliação.
Uma validação operacional exige dados separados e rótulos definidos por profissionais de saúde.

## Voz e vídeo

- A avaliação de disartria usa 8 falantes e 1.380 falas, com validação deixando um falante fora do
  treino. ROC-AUC 0,78 é um resultado inicial, não uma confirmação clínica.
- A validação de vídeo usa 8 vídeos e 123 repetições com gabarito. O resultado de agachamento tem
  ROC-AUC 0,79; abdução foi avaliada com apenas duas pessoas e não mostrou generalização.
- A detecção de ritmo na captura de movimento foi demonstrada com um participante.
- As 12 consultas roteirizadas são cenários de demonstração, não gravações clínicas reais.

## Fusão e tempo real

As bases públicas são de pessoas distintas. Para demonstrar a fusão, o projeto cria identificadores
de pacientes virtuais e alinha as modalidades em uma janela de 24 horas. Os alertas multimodais
resultantes demonstram o fluxo técnico, mas não são uma coorte de pacientes reais.

O processamento multimodal e a geração de alertas são executados em lote. O comando `monitor` e o
Live monitor do dashboard reproduzem sinais vitais históricos em ordem cronológica; não há ingestão
contínua de áudio, vídeo e sinais vitais nem fusão multimodal online.

## Azure

Azure AI Speech e Azure AI Language são chamados quando as credenciais estão configuradas e há cota.
Sem credenciais ou cota disponível, o pipeline usa alternativas locais. A execução demonstrativa
deve mostrar `Speech configurado: True` no `inspect` e confirmar o uso dos serviços no resultado do
comando `summary`.

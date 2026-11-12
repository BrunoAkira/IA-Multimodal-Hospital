# Hospital AI — Monitoramento multimodal de pacientes

O **Hospital AI** acompanha pacientes combinando quatro fontes de informação — **sinais
clínicos, voz, texto de consultas e vídeo de fisioterapia** — e transforma tudo em **alertas
priorizados** para a equipe de saúde, com a explicação de cada alerta.

## O que ele faz

| Área | O que detecta |
|---|---|
| 🩺 **Sinais clínicos** | Deterioração em sinais vitais (escore NEWS2, mudanças bruscas em relação ao histórico do próprio paciente, combinações raras de sinais), alterações inesperadas em prescrições (dose que dobra, troca de via, pedidos repetidos) e padrões incomuns de movimentação entre setores, como a volta à UTI em menos de 48 h |
| 🗣️ **Voz** | Sinais de disartria, cansaço e dificuldade respiratória na fala, comparando cada gravação com a voz habitual da pessoa |
| 📝 **Consultas** | Transcrição do áudio, extração de sintomas, diagnósticos e medicamentos (separando o que foi negado, como "no fever") e sentimento do paciente |
| 🎥 **Fisioterapia em vídeo** | Postura e ângulos das articulações, contagem de repetições, execução incorreta, assimetria e áreas de risco (objetos no chão, paciente fora da área do exercício) |
| 🚨 **Alertas** | Fusão de tudo por paciente em janela de 24 h, com níveis *medium*, *high* e *critical*, painel em tempo real e relatórios automáticos em HTML |

## Destaques

- **Roda no seu computador**, em CPU, com menos de 3 GB de RAM: os dados são lidos em pedaços
  e o vídeo é processado quadro a quadro.
- **Nuvem sem custo:** a transcrição e a análise de texto usam o Azure AI Speech e o Azure AI
  Language no plano gratuito, com cache (cada áudio ou texto é enviado uma única vez) e limite de
  uso mensal automático.
- **Explicável:** cada alerta traz o motivo e os valores que o justificam (ex.: `spo2 = 88.4`,
  mediana anterior 97).
- **Avaliado em dados reais:** em vídeos de fisioterapia com gabarito de fisioterapeuta, separa
  agachamentos corretos de incorretos com ROC-AUC de 0,79, testando sempre em pessoas que não
  estavam no treino.
- **Fácil de estender:** todos os módulos produzem eventos no mesmo formato; uma nova fonte de
  dados entra nos alertas e no painel sem mudar o resto.

> ⚠️ Projeto de pesquisa e demonstração. Não é um dispositivo médico e não deve orientar
> decisões clínicas.

📐 **Arquitetura detalhada:** [ARCHITECTURE.md](ARCHITECTURE.md)

### Início rápido

```bash
git clone <url-do-repositorio> hospital_ai && cd hospital_ai
python -m venv .venv && source .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python tests/make_sample_data.py && export HOSPITAL_AI_DATASETS=sample_datasets
python main.py all
python -m streamlit run dashboard/app.py
```

---

## Sumário

1. [Pré-requisitos](#1-pré-requisitos)
2. [Instalação (passo a passo em bash)](#2-instalação-passo-a-passo-em-bash)
3. [Primeiro teste com dados sintéticos](#3-primeiro-teste-com-dados-sintéticos)
4. [Amostras de dados: o que são, onde baixar e onde colocar](#4-amostras-de-dados-o-que-são-onde-baixar-e-onde-colocar)
5. [Azure no plano gratuito](#5-azure-no-plano-gratuito)
6. [Tutorial dos comandos](#6-tutorial-dos-comandos)
7. [Tutorial das funções (uso direto em Python)](#7-tutorial-das-funções-uso-direto-em-python)
8. [Variáveis de configuração](#8-variáveis-de-configuração)
9. [Arquivos gerados](#9-arquivos-gerados)
10. [Arquitetura](#10-arquitetura)
11. [Como os resultados são medidos](#11-como-os-resultados-são-medidos)
12. [Solução de problemas](#12-solução-de-problemas)
13. [Limitações conhecidas](#13-limitações-conhecidas)
14. [Licenças das ferramentas](#14-licenças-das-ferramentas)
15. [Referências](#15-referências)

---

## 1. Pré-requisitos

### Sistema e terminal

| Item | Requisito |
|---|---|
| Sistema operacional | Windows 10/11, Linux ou macOS |
| Terminal | **bash**. No Windows: **Git Bash** (https://git-scm.com) ou **MSYS2 UCRT64** |
| Python | **3.11 ou 3.12 (recomendado)**. Funciona de 3.10 a 3.14 (testado na 3.14 no Windows) |
| pip | Vem com o Python. Atualize com `python -m pip install --upgrade pip` |
| Memória RAM | Mínimo 8 GB. O pico típico do projeto fica abaixo de 3 GB (os dados são lidos em pedaços) |
| Disco | ~2 GB para as bibliotecas (o PyTorch é o maior) + o tamanho dos datasets (ver seção 4) |
| GPU | **Não precisa**. Tudo roda em CPU |
| Internet | Para instalar as bibliotecas, baixar modelos e datasets, e para chamar o Azure |
| Conta Azure | Gratuita, com os recursos **Speech** e **Language** no plano **Free F0** (seção 5) |

**Instalando o Python no Windows:** baixe em https://www.python.org/downloads/ e, no instalador,
marque **"Add python.exe to PATH"**. Depois feche e abra o terminal.

Confira:
```bash
python --version          # Python 3.11.x / 3.12.x (3.10–3.14 funcionam)
python -m pip --version
```
> No Linux/macOS o comando costuma ser `python3`. Depois de ativar o ambiente virtual (seção 2),
> `python` passa a funcionar em todos os sistemas.

### Bibliotecas Python (instaladas pelo `requirements.txt`)

| Biblioteca | Para que serve | Onde é usada |
|---|---|---|
| `numpy`, `scipy` | Cálculo numérico e processamento de sinal (voz, ângulos, DTW) | todos os módulos |
| `pandas` | Tabelas (dados clínicos, características de voz, repetições) | todos os módulos |
| `scikit-learn` | IsolationForest, regressão logística, Random Forest, validação | `clinical/`, `audio/`, `video/` |
| `joblib` | Salvar o classificador treinado | `audio/pipeline.py` |
| `pyarrow` | Ler a base de fala (`.parquet`) em lotes | `audio/torgo_loader.py` |
| `opencv-python` | Ler vídeos, rodar o OpenPose (módulo DNN), desenhar nos quadros | `video/vision.py` |
| `ultralytics` | YOLOv8 (objetos) e YOLOv8-pose (fallback do OpenPose). Instala o PyTorch | `video/vision.py` |
| `matplotlib` | Gráficos dos relatórios | `core/report.py`, `video/pipeline.py` |
| `jinja2` | Modelo HTML dos relatórios | `core/report.py` |
| `requests` | Chamadas REST ao Azure | `core/azure_client.py` |
| `python-dotenv` | Ler as chaves do Azure do arquivo `.env` | `config.py` |
| `streamlit`, `plotly` | Dashboard local e gráficos interativos | `dashboard/app.py` |

---

## 2. Instalação (passo a passo em bash)

```bash
# 1. Entre na pasta do projeto (onde estão main.py e requirements.txt)
cd ~/Downloads/hospital_ai          # ajuste para o seu caminho
ls main.py requirements.txt         # os dois devem aparecer

# 2. Crie o ambiente virtual (uma vez só)
python -m venv .venv

# 3. Ative o ambiente (TODA vez que abrir um terminal novo)
source .venv/Scripts/activate       # Windows (Git Bash / MSYS2)
# source .venv/bin/activate         # Linux / macOS
#   -> deve aparecer "(.venv)" no começo da linha

# 4. Instale as bibliotecas (uma vez só; leva alguns minutos por causa do PyTorch)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Baixe os modelos de visão (uma vez só)
python scripts/download_models.py
```

Verifique a instalação:
```bash
which python    # deve apontar para .../.venv/Scripts/python (ou .venv/bin/python)
python -c "import numpy, pandas, sklearn, cv2, pyarrow, ultralytics, streamlit, plotly; print('tudo instalado')"
```

---

## 3. Primeiro teste com dados sintéticos

Antes de baixar qualquer dataset, teste o projeto com **dados sintéticos** gerados nos mesmos
formatos dos datasets reais (não têm valor clínico):

```bash
source .venv/Scripts/activate
python tests/make_sample_data.py            # cria a pasta sample_datasets/
export HOSPITAL_AI_DATASETS=sample_datasets # usa a amostra (vale só para este terminal)
export AZURE_STT_TORGO_SAMPLES=0            # não gasta cota do Azure com áudio sintético

python main.py inspect
python main.py all
python -m streamlit run dashboard/app.py    # abra http://localhost:8501
```

Para voltar a usar os datasets reais:
```bash
unset HOSPITAL_AI_DATASETS AZURE_STT_TORGO_SAMPLES
```

---

## 4. Amostras de dados: o que são, onde baixar e onde colocar

O projeto foi construído e testado com quatro amostras públicas de dados, uma para cada tipo de
informação que ele analisa, mais um conjunto de consultas escritas para o projeto. **Todos os dados
são padronizados em inglês** (nomes de colunas, rótulos, eventos e alertas). Os nomes, links e
licenças de cada amostra estão nas [Referências](#15-referências), no fim deste documento.

| Amostra | O que contém | Para que é usada | O que baixar | Ref. |
|---|---|---|---|---|
| **Base clínica de UTI** | Registros desidentificados de 100 pacientes internados: sinais vitais medidos à beira do leito, prescrições de medicamentos e transferências entre setores do hospital | Detecção de anomalias em sinais vitais, prescrições e movimentação do paciente | O ZIP completo (~15 MB) | [1] |
| **Base de fala** | Gravações em inglês de pessoas com disartria (dificuldade na fala de origem neurológica) e de pessoas sem alteração, com o texto que cada uma leu | Análise de voz: disartria, cansaço e dificuldade respiratória; teste da transcrição | **1** arquivo `.parquet` (~370 MB) basta | [2] |
| **Base de captura de movimento** | Ângulos das articulações de 10 pessoas fazendo 22 exercícios de reabilitação, cada um em ritmo normal, rápido e lento | Movimento fora do padrão, comparando cada repetição com a execução de referência | 1 a 3 participantes (`S01.zip`...) | [3] |
| **Base de vídeos de exercícios** | Vídeos coloridos de 10 pessoas fazendo 6 exercícios de fisioterapia, com cada repetição marcada por um fisioterapeuta como correta ou incorreta | Reconhecimento de postura em vídeo real (OpenPose/YOLOv8) e avaliação de execução incorreta | `videos.zip`, `Segmentation.csv`, `Segmentation.txt` | [4] |
| **Consultas** | 12 roteiros de consulta em inglês, escritos para o projeto, com sintomas, termos críticos e tons diferentes | Transcrição (Speech to Text) e análise de texto (termos críticos e sentimento) | já incluídos em `data/consultations/` | — |

### Estrutura final das pastas

```
datasets/
  clinical/
    hosp/  prescriptions.csv.gz, transfers.csv.gz, ...   <- base clínica de UTI [1]
    icu/   chartevents.csv.gz, ...
  audio/
    train-00000-of-00004.parquet   <- base de fala [2]
  video/
    S01/ S02/ ...                  <- base de captura de movimento [3]
  rehab24/                         <- base de vídeos de exercícios [4] (vídeos + Segmentation.csv)
data/
  consultations/  consult_01.txt ... consult_12.txt  (+ consult_XX.wav)
  physio_videos/  squat_....mp4, arm_raise_....mp4   <- vídeos que serão analisados
```

### Comandos para organizar (bash)

```bash
mkdir -p datasets/clinical datasets/audio datasets/video datasets/rehab24

# Base clínica de UTI [1] (o Python extrai o zip; o Git Bash nem sempre tem unzip)
python -m zipfile -e ~/Downloads/mimic-iv-clinical-database-demo-2.2.zip /tmp/clinica
cp -r /tmp/clinica/*/hosp /tmp/clinica/*/icu datasets/clinical/

# Base de fala [2]
cp ~/Downloads/train-00000-of-00004.parquet datasets/audio/

# Base de captura de movimento [3] (repita para S02, S03...)
python -m zipfile -e ~/Downloads/S01.zip datasets/video/

# Base de vídeos de exercícios [4]
python -m zipfile -e ~/Downloads/videos.zip datasets/rehab24/
cp ~/Downloads/Segmentation.csv ~/Downloads/Segmentation.txt datasets/rehab24/

python main.py inspect      # confere tudo
```
> Os nomes dos zips são exemplos: use os nomes dos arquivos que você baixou.

### Vídeos para o OpenPose / YOLOv8 (`data/physio_videos/`)

A base de captura de movimento **não tem vídeo de pessoas** (os `.avi` dela mostram só o
esqueleto desenhado), então o reconhecimento visual roda em vídeos reais: da base de vídeos de
exercícios [4] ou gravados por você.

- Formatos aceitos: `.mp4`, `.avi`, `.mov`, `.mkv`.
- **O nome do arquivo começa pelo exercício**, para o sistema saber a amplitude esperada:

| Prefixo do arquivo | Articulação analisada | Amplitude mínima esperada |
|---|---|---|
| `squat_...` (ou `MiniSquat...`) | joelho | 50° |
| `sit_to_stand_...` (ou `SitToStand...`) | joelho | 60° |
| `knee_extension_...` | joelho | 50° |
| `arm_raise_...` (ou `ArmRaise...`, `ArmAbduction...`) | ombro | 100° |
| `elbow_flexion_...` | cotovelo | 90° |
| qualquer outro nome | joelho, quadril, ombro e cotovelo (análise genérica) | — |

Exemplo:
```bash
cp "datasets/rehab24/CAMINHO/DO/VIDEO.mp4" data/physio_videos/squat_rehab_01.mp4
```

**Para gravar com o celular:** câmera parada, corpo inteiro no quadro, 15–30 s, 5 repetições.
Sugestão: um vídeo correto e outro incorreto de propósito (agachando pouco, um lado mais baixo),
e um com um objeto no chão (garrafa, mochila) para testar a área de risco de queda.

### Áudio das consultas (`data/consultations/`)

Há 12 roteiros em inglês (`consult_01.txt` ... `consult_12.txt`). Grave a leitura de cada roteiro
(celular ou microfone, em WAV) e salve como `consult_XX.wav` na mesma pasta: é esse áudio que
passa pelo Azure Speech to Text e depois pelo Azure AI Language.

Enquanto um roteiro ainda não tiver gravação, o pipeline analisa o próprio texto do roteiro
(`transcript_source = script`).

---

## 5. Azure no plano gratuito

São usados **dois** serviços, os dois no plano **Free F0**:

| Serviço | Para quê | Limite gratuito | Limite de segurança do projeto |
|---|---|---|---|
| **Azure AI Speech** | Speech to Text | 5 h de áudio/mês | 4 h (14 400 s) |
| **Azure AI Language** | Sentimento + Text Analytics for health | 5 000 registros/mês (1 registro = até 1 000 caracteres) | 4 000 registros |

### Criar os recursos (portal.azure.com)

1. **Speech:** pesquise "Speech services" → *Create* → escolha a região (ex.: East US) →
   *Pricing tier* = **Free F0** → *Create*. Abra o recurso → **Keys and Endpoint** → copie
   **KEY 1** e **Location/Region**.
2. **Language:** pesquise "Language" → *Create* (pode continuar sem recursos extras) →
   *Pricing tier* = **Free F0** → *Create*. Em **Keys and Endpoint**, copie **KEY 1** e **Endpoint**.

> Se a opção F0 não aparecer, sua assinatura já tem um recurso gratuito daquele tipo: use as
> chaves dele.

### Configurar as chaves

```bash
cp .env.example .env
nano .env        # Ctrl+O, Enter para salvar; Ctrl+X para sair
```

```ini
AZURE_SPEECH_KEY=sua_chave_do_speech
AZURE_SPEECH_REGION=eastus
AZURE_LANGUAGE_KEY=sua_chave_do_language
AZURE_LANGUAGE_ENDPOINT=https://seu-recurso.cognitiveservices.azure.com
```
(sem aspas e sem espaços ao redor do `=`)

```bash
python main.py inspect     # deve mostrar "Speech configurado: True" e "Language configurado: True"
```

### Como o projeto evita gastos

- **Cache:** cada áudio ou texto é enviado **uma única vez**. As respostas ficam em `results/cache/`.
- **Contador mensal:** `results/cache/azure_usage.json` bloqueia as chamadas ao chegar no limite
  de segurança; até o mês seguinte, a análise local (dicionário de termos e de sentimento) assume.
- No plano F0 o Azure não cobra excedente: ao atingir o limite, ele apenas bloqueia.
- O `.env` está no `.gitignore`. **Não compartilhe esse arquivo.**

---

## 6. Tutorial dos comandos

Sempre a partir da pasta do projeto, com o ambiente ativo (`source .venv/Scripts/activate`).

| Comando | O que faz |
|---|---|
| `python main.py inspect` | Confere datasets, vídeos, consultas e chaves do Azure (não processa nada) |
| `python main.py clinical` | Anomalias clínicas (sinais vitais, prescrições, transferências) |
| `python main.py audio` | Voz + consultas (Speech to Text + Text Analytics) |
| `python main.py video` | Captura de movimento + vídeos (OpenPose/YOLOv8) + relatórios |
| `python main.py rehab24` | Avaliação em vídeo real contra o gabarito do fisioterapeuta (correta x incorreta) |
| `python main.py fusion` | Pacientes virtuais + fusão + alertas |
| `python main.py all` | `clinical` → `audio` → `video` → `fusion` |
| `python main.py monitor` | Simulação de tempo real dos sinais vitais no terminal |
| `python -m streamlit run dashboard/app.py` | Dashboard no navegador |

> Ordem obrigatória: `fusion`, `monitor` e o dashboard leem o que os outros comandos geraram.
> Na dúvida, rode `python main.py all`.

### 6.1 `inspect` — checklist antes de rodar

```bash
python main.py inspect
```
Mostra, para cada dataset, os arquivos encontrados e as colunas; quantos vídeos e gravações de
consulta existem; se as chaves do Azure estão configuradas e quanto da cota já foi usado.
Use sempre que mudar os dados de lugar.

### 6.2 `clinical` — anomalias em dados clínicos

```bash
python main.py clinical
```

| Etapa | Técnica | Eventos gerados |
|---|---|---|
| Sinais vitais (FC, PA, SpO₂, FR, temperatura) agregados por hora | Limites clínicos fixos | `spo2_low`, `heart_rate_high`, ... |
| | Escore NEWS2 (alerta precoce) | `news2_elevated` |
| | z-score robusto em fluxo (vs. histórico do próprio paciente) | `..._sudden_rise`, `..._sudden_drop` |
| | IsolationForest multivariado | `unusual_vital_pattern` |
| Prescrições | Mudança vs. pedido anterior + IsolationForest | `dose_jump`, `route_change`, `frequent_reorders`, `high_alert_start`, `order_burst`, `unusual_prescription_change` |
| Transferências | Cadeia de Markov + regras | `unusual_transfer`, `icu_readmission_48h`, `frequent_transfers` |

Saídas: `results/clinical_summary.json` (contagens + avaliação), `results/vitals_hourly.csv`
e os eventos em `results/events.jsonl`.

Como ler a avaliação no terminal:
```
'spo2': {'injected': 17, 'precision': 0.55, 'recall': 1.0}
```
→ foram injetadas 17 quedas artificiais de SpO₂ e o detector encontrou todas (recall 1.0).
A precisão é conservadora: os "falsos positivos" incluem anomalias reais que já existem nos dados.

### 6.3 `audio` — voz e consultas

```bash
python main.py audio                 # até 1500 falas da base de fala (padrão)
python main.py audio --rows 300      # teste rápido com 300 falas
```

O que acontece:
1. **Características da voz** de cada fala: velocidade de fala, pausas, pausas longas
   (respiração), F0, jitter, shimmer, HNR, volume, MFCC.
2. **Classificador de disartria** (regressão logística e Random Forest), validado em falantes
   que ele nunca ouviu (`GroupKFold` por falante). O melhor fica salvo em
   `models/dysarthria_classifier.joblib`.
3. **Indicadores de cansaço e de dificuldade respiratória**, comparando cada fala com a linha de
   base do próprio falante.
4. **Azure Speech to Text** numa amostra da base de fala (`AZURE_STT_TORGO_SAMPLES`, padrão 60),
   calculando a taxa de erro de palavras (WER) por grupo.
5. **Consultas:** `.wav` → Azure Speech to Text → Azure Text Analytics for health (termos
   críticos e negação, ex.: "no fever") + sentimento.

Eventos: `dysarthria_risk`, `fatigue_indicator`, `breathing_difficulty_indicator`,
`critical_term_mentioned`, `negative_sentiment`.

Saídas: `results/audio_summary.json`, `voice_features.csv`, `voice_indicators.csv`,
`consultations.json`, `torgo_azure_stt.csv` (se o Speech estiver ativo).

Conferir se o Azure foi usado:
```bash
python - <<'EOF'
import json
for c in json.load(open("results/consultations.json")):
    print(c["consultation_id"], c["transcript_source"], c["sentiment"], c["sentiment_source"],
          c["critical_terms"], "| negados:", c["negated_terms"])
EOF
```
`transcript_source`: `azure`/`cache` = Speech to Text usado; `script` = sem `.wav` (texto do roteiro).
`sentiment_source`: `azure_sentiment` = Azure AI Language; `local_lexicon` = análise local (cota do mês esgotada).

### 6.4 `video` — fisioterapia

```bash
python main.py video
```

**Parte 1 — captura de movimento (`datasets/video/`):** compara as 3 repetições de cada arquivo `R3`
(ritmo normal, rápido e lento) com a repetição de referência do arquivo `R1`:

| Medida | Significado |
|---|---|
| `duration_ratio` | > 1 mais lento, < 1 mais rápido que a referência |
| `rom_ratio` | amplitude de movimento; < 0,75 = amplitude reduzida |
| `dtw_shape` | diferença de forma da curva (Dynamic Time Warping) |
| `asymmetry_delta` | aumento da diferença entre os lados esquerdo e direito |
| `ml_score` | IsolationForest treinado só com repetições normais de outros participantes |

**Parte 2 — vídeos (`data/physio_videos/`):** a ~5 quadros por segundo:
- **OpenPose** (ou YOLOv8-pose, se os pesos do OpenPose não estiverem em `models/openpose/`)
  encontra os pontos do corpo e calcula os ângulos de joelho, quadril, ombro e cotovelo.
- **YOLOv8** detecta pessoas e objetos e verifica as **áreas críticas**.

Contagem de repetições: pico do ângulo no ombro; vale no cotovelo e quadril; no joelho, pela
altura do quadril (funciona com a câmera de frente). Exercícios de um lado só (ex.: abdução de um
braço) são detectados automaticamente e não geram alerta de assimetria.

Eventos: `pace_deviation`, `reduced_range_of_motion`, `left_right_asymmetry`,
`abnormal_movement_pattern`, `irregular_repetitions`, `possible_fall`,
`patient_left_exercise_area`, `obstacle_in_critical_area`, `patient_out_of_view`.

**Relatórios automáticos** (abra no navegador; para PDF, use Imprimir → Salvar como PDF):
```bash
ls results/reports/
start results/reports/physio_S01.html          # Windows
# xdg-open results/reports/physio_S01.html     # Linux
# open results/reports/physio_S01.html         # macOS
```
- `physio_SXX.html`: tabela de desvios por repetição + curvas vs. referência (captura de movimento).
- `video_<nome>.html`: resumo, desvios, gráfico de ângulos e quadros com o esqueleto desenhado.

Ajustar as áreas críticas ao enquadramento da câmera: edite `ZONES` no topo de
`src/video/vision.py` (coordenadas de 0 a 1; `(0,0)` = canto superior esquerdo):
```python
ZONES = {
    "exercise_area":   np.array([[0.15, 0.0], [0.85, 0.0], [0.85, 1.0], [0.15, 1.0]]),
    "fall_risk_floor": np.array([[0.0, 0.85], [1.0, 0.85], [1.0, 1.0], [0.0, 1.0]]),
}
```

### 6.4b `rehab24` — avaliação em vídeo real com gabarito

```bash
python main.py rehab24                                  # exercícios 1 e 6, até 4 vídeos cada
python main.py rehab24 --exercises 1,5,6 --max-videos 6
```

Usa os vídeos `datasets/rehab24/Ex<n>/PM_xxx-Camera17-30fps.mp4` e o `Segmentation.csv`
(não precisa copiar nada para `data/physio_videos/`).

| Id | Exercício | Articulação analisada |
|---|---|---|
| 1 | Arm abduction | ombro |
| 2 | Arm VW | ombro |
| 3 | Push-ups | cotovelo |
| 4 | Leg abduction | quadril |
| 5 | Leg lunge | joelho |
| 6 | Squats | joelho |

O que faz:
1. Roda só a estimativa de pose (sem o YOLOv8 de objetos, ~2x mais rápido) e guarda os
   ângulos em `results/rehab24_angles/` (rodar de novo é instantâneo).
2. Corta cada vídeo nas repetições do gabarito e mede: amplitude, ângulo máximo e mínimo,
   duração, velocidade, inclinação do tronco e movimento do lado que deveria ficar parado.
3. Mostra o quanto cada medida, sozinha, separa correta de incorreta (`single_feature_auc`:
   0,5 = acaso, 1,0 = perfeito) e treina uma regressão logística validada deixando uma pessoa
   de fora por vez.
4. Compara o número de repetições contadas automaticamente com o gabarito.
5. Gera `results/reports/rehab24_evaluation.html` (gráfico de cada vídeo com as repetições
   corretas em verde e incorretas em vermelho) e eventos `incorrect_execution` para a fusão.

Tempo: cerca de 1 a 2 minutos por vídeo na primeira vez.

### Zonas críticas por vídeo

Para ajustar as zonas a um vídeo específico, crie um JSON com o mesmo nome do vídeo:
```bash
cat > data/physio_videos/squat_rehab01.zones.json <<'EOF'
{"exercise_area": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "fall_risk_floor": null}
EOF
```
Coordenadas de 0 a 1 (`(0,0)` = canto superior esquerdo); `null` desliga a zona.

### 6.5 `fusion` — fusão multimodal e alertas

```bash
python main.py fusion
```
1. **Pacientes virtuais:** como os datasets são de pessoas diferentes, cada `VP001`, `VP002`...
   junta 1 paciente da base clínica + 1 falante da base de fala + 1 participante da captura de movimento + 1 consulta
   (+ 1 vídeo). O mapa fica em `results/virtual_patients.csv`.
2. **Fusão em janela de 24 h:** maior pontuação de cada modalidade, combinada por
   `risco = 1 − Π(1 − peso × pontuação)`.
3. **Alertas:** `medium` (≥ 0,5), `high` (≥ 0,8), `critical` (≥ 0,9 com 2+ modalidades fortes),
   sem repetir o mesmo paciente antes de 6 h, a menos que piore.

Ver os alertas no terminal:
```bash
python - <<'EOF'
import pandas as pd
a = pd.read_csv("results/alerts.csv")
print(a["level"].value_counts())
print(a[a["level"] == "critical"][["patient_id", "timestamp", "modalities", "top_events"]].head(10).to_string())
EOF
```

### 6.6 `monitor` — tempo real no terminal

```bash
python main.py monitor                    # 0,2 s por hora simulada
python main.py monitor --speed 0.05       # mais rápido
python main.py monitor --hours 500        # para depois de 500 horas simuladas
```
Reproduz os sinais vitais em ordem cronológica, como se chegassem do monitor à beira do leito,
e imprime cada alerta na hora (amarelo = medium, vermelho = high). `Ctrl+C` encerra.
Precisa ter rodado `python main.py clinical` antes.

### 6.7 Dashboard

```bash
python -m streamlit run dashboard/app.py
```
Abra http://localhost:8501. Páginas:

| Página | Conteúdo |
|---|---|
| Overview | Números gerais, eventos por modalidade, uso da cota do Azure |
| Alerts | Fila de alertas com filtros por nível e "só multimodais" |
| Patient timeline | Eventos de todas as modalidades + sinais vitais de um paciente |
| Clinical | Eventos clínicos e avaliação por injeção |
| Voice & text | Métricas do classificador, WER do Azure, características de voz, consultas |
| Physio video | Repetições da captura de movimento, vídeos analisados, download dos relatórios |
| Live monitor | Escolha um paciente e clique em ▶ Start: sinais vitais "chegando" em tempo real |

Depois de rodar o pipeline de novo, clique em **Reload results** na barra lateral.
`Ctrl+C` no terminal encerra o dashboard.

### 6.8 Scripts auxiliares

| Script | O que faz |
|---|---|
| `python scripts/download_models.py` | Baixa YOLOv8n, YOLOv8n-pose e OpenPose COCO para `models/` |
| `python tests/make_sample_data.py` | Cria `sample_datasets/` com dados sintéticos nos formatos reais |

---

## 7. Tutorial das funções (uso direto em Python)

Todas as funções podem ser usadas isoladamente, por exemplo num notebook ou num script.
Rode os exemplos a partir da pasta do projeto, com o ambiente ativo. Os blocos
`python - <<'EOF' ... EOF` executam o código Python direto no bash.

### 7.1 Núcleo (`src/core/`)

**`Event`, `save_events`, `load_events`, `clear_events`** — formato único de evento:
```bash
python - <<'EOF'
from src.core.events import Event, save_events, load_events
ev = Event(patient_id="mimic:10000032", timestamp="2150-01-01T10:00:00",
           modality="clinical", source="vitals", event_type="spo2_low",
           score=0.85, message="spo2 = 86.0")
print(ev.severity)            # "high" (calculado a partir do score)
print(len(load_events()), "eventos salvos em results/events.jsonl")
EOF
```
`save_events(lista)` anexa ao arquivo; `clear_events("audio")` apaga só os eventos de uma modalidade.

**Azure (`src/core/azure_client.py`):**

| Função | Retorno |
|---|---|
| `speech_enabled()` / `language_enabled()` | `True` se as chaves estão no `.env` |
| `to_wav16k_bytes(samples, sr)` | Converte um áudio em WAV 16 kHz mono (formato do Azure) |
| `speech_to_text(wav_bytes)` | `{"text", "confidence", "source"}` ou `None` (áudio de até 60 s) |
| `analyze_sentiment(text)` | `{"sentiment", "scores", "sentences"}` ou `None` |
| `analyze_health(text)` | `{"entities": [{"text", "category", "confidence", "assertion"}]}` ou `None` |
| `usage_summary()` | Uso do mês e limites de segurança |

```bash
python - <<'EOF'
from src.core import azure_client as az
print(az.usage_summary())
r = az.analyze_health("Patient denies chest pain but reports shortness of breath.")
if r: [print(e["text"], e["category"], e["assertion"]) for e in r["entities"]]
EOF
```
(gasta 1 registro da cota na primeira execução; nas próximas lê do cache)

**Relatórios (`src/core/report.py`):** `write_report(nome_arquivo, titulo, secoes)`, em que cada
seção é `{"heading": ..., "text": ..., "table": DataFrame, "images": [fig_to_b64(fig)]}`.

### 7.2 Clínico (`src/clinical/`)

| Função | O que faz |
|---|---|
| `loader.load_vitals()` | Lê os sinais vitais do `chartevents` (em pedaços) já com nomes em inglês |
| `loader.hourly_vitals(df)` | 1 linha por paciente por hora (mediana de cada sinal) |
| `loader.load_prescriptions()` / `loader.load_transfers()` | Prescrições e transferências padronizadas |
| `vitals.news2_score(linha)` | Escore NEWS2 de uma hora |
| `vitals.VitalStreamDetector().update(stay_id, vital, valor)` | Avalia **um** ponto novo → `(score, tipo, evidência)` |
| `vitals.detect_vital_events(hourly)` | Detector em fluxo + NEWS2 em toda a série |
| `vitals.detect_multivariate(hourly)` | IsolationForest nas combinações de sinais |
| `vitals.evaluate_with_injection(hourly)` | Precisão/recall com anomalias injetadas |
| `vitals.replay(hourly, speed_seconds, on_event)` | Simulação de tempo real |
| `prescriptions.detect_prescription_events(rx)` | Eventos de prescrição + tabela de características |
| `transfers.detect_transfer_events(tr)` | Eventos de transferência + matriz de probabilidades |

Exemplo — detector em tempo real, um ponto por vez:
```bash
python - <<'EOF'
from src.clinical.vitals import VitalStreamDetector
det = VitalStreamDetector()
for spo2 in [97, 96, 97, 98, 97, 96, 97, 88]:      # última leitura cai de repente
    score, tipo, ev = det.update("leito_1", "spo2", spo2)
    print(spo2, "->", round(score, 2), tipo, ev)
EOF
```

### 7.3 Áudio (`src/audio/`)

| Função | O que faz |
|---|---|
| `torgo_loader.iter_torgo(max_rows=...)` | Gera as falas da base de fala uma a uma: `{speaker_id, group, prompt, audio, sr, ...}` |
| `features.extract_features(audio, sr)` | Dicionário com ~28 características da voz |
| `voice_model.train_dysarthria_classifier(df)` | Treina e valida por falante; retorna métricas e modelo |
| `voice_model.vocal_indicators(df)` | `fatigue_index` e `breathing_index` por fala |
| `text_analysis.analyze_text(texto)` | Termos críticos + sentimento (Azure ou local) |
| `text_analysis.local_critical_terms(texto)` | Só a versão local (dicionário + negação) |
| `consultations.split_audio(audio, sr)` | Divide áudios longos em trechos de até 50 s |
| `consultations.transcribe(audio, sr)` | Transcreve com o Azure trecho a trecho |
| `pipeline.word_error_rate(ref, hyp)` | WER entre o texto esperado e o transcrito |

Exemplo — analisar um `.wav` qualquer:
```bash
python - <<'EOF'
import numpy as np
from scipy.io import wavfile
from src.audio.features import extract_features
sr, y = wavfile.read("data/consultations/consult_01.wav")   # troque pelo seu arquivo
y = y / np.iinfo(y.dtype).max if np.issubdtype(y.dtype, np.integer) else y
f = extract_features(y, sr)
for k in ("speech_rate_sps", "pause_ratio", "long_pauses", "f0_mean_hz", "jitter", "shimmer", "hnr_db"):
    print(f"{k:18} {f[k]:.3f}")
EOF
```

Exemplo — termos críticos de um texto (sem Azure):
```bash
python - <<'EOF'
from src.audio.text_analysis import local_critical_terms
for t in local_critical_terms("No fever. I have chest pain and I feel short of breath."):
    print(t["text"], "| negado:", t["negated"], "| peso:", t["weight"])
EOF
```

### 7.4 Vídeo (`src/video/`)

| Função | O que faz |
|---|---|
| `amc.parse_amc(caminho)` | Lê um `.amc` → DataFrame (quadros × ângulos) |
| `amc.index_ulred()` | Lista os `.amc` com participante, movimento e nº de repetições |
| `motion.build_repetition_table()` | Medidas de cada repetição vs. referência |
| `motion.score_repetitions(df)` | Regras + IsolationForest → pontuação e métricas |
| `motion.dtw_distance(a, b)` | Distância DTW entre duas curvas |
| `vision.PoseEstimator()` | OpenPose (ou YOLOv8-pose): `estimador(quadro)` → pontos do corpo |
| `vision.frame_angles(pontos)` | Ângulos de joelho, quadril, ombro, cotovelo e inclinação do tronco |
| `vision.analyze_angles(df, exercicio, fps)` | Amplitude, assimetria, repetições, queda |
| `vision.analyze_video(caminho)` | Vídeo completo: pose + YOLOv8 + zonas → resumo, achados, ângulos, fotos |

Exemplo — analisar um único vídeo sem rodar o pipeline todo:
```bash
python - <<'EOF'
from pathlib import Path
from src.video.vision import analyze_video
res = analyze_video(Path("data/physio_videos/squat_teste01.mp4"))
print(res["summary"])
for f in res["findings"]:
    print(f["event_type"], round(f["score"], 2), f["message"])
EOF
```

Exemplo — ler um arquivo de captura de movimento:
```bash
python - <<'EOF'
from src.video.amc import index_ulred, parse_amc
idx = index_ulred()
print(idx.head())
df = parse_amc(idx.iloc[0]["path"])
print(df.shape, list(df.columns)[:8])
EOF
```

### 7.5 Fusão (`src/fusion.py`)

| Função | O que faz |
|---|---|
| `build_virtual_patients(eventos)` | Monta os pacientes virtuais |
| `apply_virtual_ids(eventos, vp)` | Troca os ids de origem pelos virtuais e alinha os horários |
| `fuse_and_alert(eventos)` | Fusão em janela de 24 h + alertas com cooldown |
| `run_fusion()` | Tudo acima, salvando os arquivos em `results/` |

---

## 8. Variáveis de configuração

Coloque no `.env` ou use `export NOME=valor` no terminal (vale só para aquele terminal).

| Variável | Padrão | Para quê |
|---|---|---|
| `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` | vazio | Azure Speech |
| `AZURE_LANGUAGE_KEY` / `AZURE_LANGUAGE_ENDPOINT` | vazio | Azure Language |
| `SPEECH_SECONDS_BUDGET` | `14400` | Limite de segurança do Speech (segundos/mês) |
| `LANGUAGE_RECORDS_BUDGET` | `4000` | Limite de segurança do Language (registros/mês) |
| `AZURE_STT_TORGO_SAMPLES` | `60` | Quantas falas da base de fala vão para o Speech to Text (`0` desliga) |
| `MAX_TORGO_ROWS` | `1500` | Quantas falas da base de fala processar |
| `HOSPITAL_AI_DATASETS` | `datasets` | Pasta dos datasets (ex.: `sample_datasets`) |

Limiares e listas (limites clínicos, medicamentos de alto risco, pesos da fusão, janela, cooldown)
ficam em `src/config.py`, todos comentados.

---

## 9. Arquivos gerados

Tudo vai para `results/` (pode apagar a pasta para recomeçar do zero):

| Arquivo | Conteúdo |
|---|---|
| `events.jsonl` | Todos os eventos de todos os módulos (1 por linha) |
| `events_fused.csv` | Eventos com ids de pacientes virtuais e horários alinhados |
| `alerts.csv` | Alertas finais (nível, pontuação, modalidades, motivos, ação recomendada) |
| `virtual_patients.csv` | Quem compõe cada paciente virtual |
| `clinical_summary.json`, `audio_summary.json`, `video_summary.json`, `fusion_summary.json` | Métricas de cada módulo |
| `vitals_hourly.csv` | Sinais vitais por hora (usado no dashboard e no monitor) |
| `voice_features.csv`, `voice_indicators.csv` | Características e índices de voz por fala |
| `torgo_azure_stt.csv` | Transcrições do Azure + WER por fala |
| `consultations.json` | Transcrição, sentimento e termos de cada consulta |
| `physio_repetitions.csv` | Medidas e pontuação de cada repetição da captura de movimento |
| `video_angles_<nome>.csv` | Ângulos quadro a quadro de cada vídeo |
| `reports/*.html` | Relatórios automáticos |
| `cache/` | Respostas do Azure + `azure_usage.json` (**não apague no meio do mês**: o cache evita novas cobranças de cota) |

---

## 10. Arquitetura

A arquitetura completa (diagramas, formato `Event`, fusão, uso do Azure, memória) está em
[ARCHITECTURE.md](ARCHITECTURE.md). Onde fica cada funcionalidade:

| Funcionalidade | Onde está |
|---|---|
| Postura e ângulos em vídeo (OpenPose / YOLOv8-pose) | `src/video/vision.py` → `PoseEstimator`, `frame_angles` |
| Objetos e áreas críticas (YOLOv8) | `src/video/vision.py` → `analyze_video`, `ZONES` |
| Movimento fora do padrão e execução incorreta | `src/video/motion.py`, `analyze_angles`, `src/video/rehab24.py` |
| Relatórios automáticos de fisioterapia | `results/reports/physio_*.html`, `video_*.html`, `rehab24_evaluation.html` |
| Alterações na voz (disartria, cansaço, respiração) | `src/audio/features.py`, `src/audio/voice_model.py` |
| Transcrição de consultas (Azure Speech to Text) | `src/core/azure_client.py` → `speech_to_text` |
| Termos críticos e sentimento (Azure AI Language) | `src/audio/text_analysis.py`, `azure_client.analyze_health` |
| Anomalias em sinais vitais | `src/clinical/vitals.py` |
| Anomalias em prescrições | `src/clinical/prescriptions.py` |
| Movimentação entre setores | `src/clinical/transfers.py` |
| Fusão e alertas | `src/fusion.py` → `results/alerts.csv` |
| Painel e monitor em tempo real | `dashboard/app.py`, `python main.py monitor` |

---

## 11. Como os resultados são medidos

| Módulo | Como medimos | Onde ver |
|---|---|---|
| Sinais vitais | Injeção de anomalias artificiais → precisão e recall | `clinical_summary.json` |
| Voz | Classificador validado em falantes nunca vistos (`GroupKFold`) → acurácia, F1, ROC-AUC | `audio_summary.json` |
| Captura de movimento | Ritmo rápido/lento como rótulo → ROC-AUC e recall | `video_summary.json` |
| Speech to Text | WER em fala com e sem disartria | `audio_summary.json`, `torgo_azure_stt.csv` |

> Com os **dados de amostra** as métricas saem altas (ex.: acurácia 1,0), porque os dados
> sintéticos são fáceis. **Resultados representativos vêm dos datasets reais.**

---

## 12. Solução de problemas

| Problema | Solução |
|---|---|
| `source .venv/Scripts/activate: No such file or directory` | Você não está na pasta do projeto: `cd` até onde estão `main.py` e `.venv` |
| `streamlit: command not found` | Ative o ambiente e use `python -m streamlit run dashboard/app.py` |
| `(.venv)` não aparece | Cada terminal novo precisa de `source .venv/Scripts/activate` |
| `python: command not found` ou abre a Microsoft Store | Reinstale o Python marcando "Add python.exe to PATH" e reabra o terminal |
| Projeto usa a amostra sem querer | `unset HOSPITAL_AI_DATASETS` |
| `Tabela 'chartevents' não encontrada` | As pastas `hosp/` e `icu/` devem estar dentro de `datasets/clinical/` |
| `Nenhum .parquet do TORGO` (base de fala) | Coloque o `.parquet` direto em `datasets/audio/` |
| `Nenhum .amc do UL-RED` (captura de movimento) | Extraia os zips dentro de `datasets/video/` (devem aparecer `S01/`, `S02/`...) |
| Vídeo com 0 quadros analisados | O OpenCV não reconhece o formato do arquivo: exporte o vídeo novamente em MP4 (H.264) |
| `pose_visible_fraction` muito baixo | O vídeo não mostra uma pessoa real ou o corpo está cortado |
| `Estimador de pose: yolov8-pose` | O OpenPose não foi baixado (servidor da CMU fora do ar). Funciona igual; para usar o OpenPose, coloque `pose_iter_440000.caffemodel` em `models/openpose/` |
| Erro 401 do Azure | Chave errada no `.env` |
| Erro 404 do Azure | Região (Speech) ou endpoint (Language) errados |
| `Cota ... atingida` | O limite de segurança do mês foi alcançado; a análise local assume até o mês seguinte |
| Erro ao instalar `torch`/`ultralytics` | Use Python 3.11 ou 3.12: `rm -rf .venv && py -3.12 -m venv .venv` |

---

## 13. Limitações conhecidas

- Os datasets são de pessoas diferentes: a fusão usa **pacientes virtuais** (simulação declarada).
- Cansaço e dificuldade respiratória são **indicadores** relativos ao próprio falante, sem
  rótulo clínico.
- Jitter e shimmer são aproximações por quadro, não o cálculo ciclo a ciclo do Praat.
- "Tempo real" é a reprodução de dados históricos na ordem em que aconteceram.
- O Azure AI Anomaly Detector foi desativado em 01/10/2026; por isso a detecção de anomalias é local.

---

## 14. Licenças das ferramentas

| Ferramenta | Licença / condição |
|---|---|
| OpenPose (CMU) | Licença acadêmica / não comercial |
| YOLOv8 (Ultralytics) | AGPL-3.0 |
| Azure AI Speech / Language | Termos do Azure; plano gratuito F0 |
| Bibliotecas Python (numpy, pandas, scikit-learn, OpenCV, Streamlit...) | Licenças de código aberto de cada projeto |

---

## 15. Referências

### Amostras de dados

| # | Amostra (como chamada neste README) | Nome original e fonte | Licença |
|---|---|---|---|
| [1] | Base clínica de UTI | **MIMIC-IV Clinical Database Demo v2.2** — Johnson, A. et al., PhysioNet. https://physionet.org/content/mimic-iv-demo/2.2/ | Open Data Commons ODbL v1.0 |
| [2] | Base de fala | **TORGO Database** — Rudzicz, F., Namasivayam, A. K., Wolff, T., University of Toronto. Cópia em Parquet: https://huggingface.co/datasets/abnerh/TORGO-database | Conforme os autores (verificar antes de redistribuir) |
| [3] | Base de captura de movimento | **UL-RED — University of Liverpool Rehabilitation Exercise Dataset**, *Scientific Data* 12:761 (2025). https://datacat.liverpool.ac.uk/2729/ | CC BY 4.0 |
| [4] | Base de vídeos de exercícios | **REHAB24-6** — Černek, A., Sedmidubsky, J., Budikova, P. *REHAB24-6: Physical Therapy Dataset for Analyzing Pose Estimation Methods*. SISAP 2024. https://zenodo.org/records/13305826 | CC BY-NC 4.0 |

### Métodos e serviços

- Royal College of Physicians. *National Early Warning Score (NEWS) 2*, 2017 — escore de alerta precoce usado nos sinais vitais.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. *Isolation Forest*. IEEE ICDM, 2008 — detecção de anomalias sem rótulos.
- Cao, Z. et al. *OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields*. IEEE TPAMI, 2019.
- Ultralytics YOLOv8 — https://docs.ultralytics.com
- Azure AI Speech — https://learn.microsoft.com/azure/ai-services/speech-service/
- Azure AI Language, Text Analytics for health — https://learn.microsoft.com/azure/ai-services/language-service/text-analytics-for-health/overview
- Desativação do Azure AI Anomaly Detector (01/10/2026) — https://azure.microsoft.com/updates/ai-services-anomaly-detector-will-be-retired-on-1-october-2026

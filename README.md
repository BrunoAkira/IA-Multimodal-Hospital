# Hospital AI — Monitoramento multimodal de pacientes

O **Hospital AI** acompanha pacientes combinando quatro fontes de informação — **sinais
clínicos, voz, texto de consultas e vídeo de fisioterapia** — e transforma tudo em **alertas
priorizados** para a equipe de saúde, com a explicação de cada alerta.

Os módulos trabalham sobre os tipos de dado que um hospital já produz: tabelas de prontuário
(sinais vitais, prescrições, transferências), gravações de voz, áudio de consultas e vídeos de
sessões de fisioterapia. Para demonstrar e validar o funcionamento, o repositório usa bases
públicas como **exemplo** de cada tipo de dado (seção 4); qualquer fonte no mesmo formato pode
substituí-las.

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
- **Validado com dados reais de exemplo:** em vídeos de fisioterapia com gabarito de
  fisioterapeuta, separa agachamentos corretos de incorretos com ROC-AUC de 0,79, testando sempre
  em pessoas que não estavam no treino.
- **Fácil de estender:** todos os módulos produzem eventos no mesmo formato; uma nova fonte de
  dados entra nos alertas e no painel sem mudar o resto.

> ⚠️ Projeto de pesquisa e demonstração. Não é um dispositivo médico e não deve orientar
> decisões clínicas.

📐 **Arquitetura:** [ARCHITECTURE.md](ARCHITECTURE.md) · 📊 **Resultados:** [seção 11](#11-resultados-e-como-são-medidos) · 📄 **Relatório técnico:** [docs/relatorio_tecnico_hospital_ai.pdf](docs/relatorio_tecnico_hospital_ai.pdf) · 📖 **Termos técnicos:** [Glossário](#13-glossário)

## ⚡ Uso rápido

Tudo o que é preciso para rodar o projeto do zero, em ordem. Os comandos são para **bash**
(no Windows, Git Bash ou MSYS2 UCRT64). Tempo total: cerca de 20 minutos, a maior parte em downloads.

**Passo 1 — Conferir o Python** (3.11 ou 3.12; 3.10 a 3.14 funcionam)
```bash
python --version
```

**Passo 2 — Baixar o projeto**
```bash
git clone https://github.com/BrunoAkira/IA-Multimodal-Hospital.git
cd IA-Multimodal-Hospital
```

**Passo 3 — Criar e ativar o ambiente virtual**
```bash
python -m venv .venv
source .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
```
Deve aparecer `(.venv)` no começo da linha. Em todo terminal novo, entre na pasta e rode o `source` de novo.

**Passo 4 — Instalar as bibliotecas e os modelos de visão** (alguns minutos)
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
```

**Passo 5 — Configurar o Azure** (plano gratuito; como criar os recursos na [seção 5](#5-azure-no-plano-gratuito))
```bash
cp .env.example .env
nano .env        # preencha as 4 linhas AZURE_...; Ctrl+O, Enter, Ctrl+X
```
Sem as chaves o projeto também roda: a transcrição fica de fora e o texto é analisado localmente.

**Passo 6 — Apontar para os dados de exemplo e baixar a base de fala** (~740 MB, uma vez só)
```bash
export HOSPITAL_AI_DATASETS=example_data
python scripts/download_example_data.py
```

**Passo 7 — Conferir se está tudo no lugar**
```bash
python main.py inspect
```

**Passo 8 — Rodar a análise completa** (clínico → áudio → vídeo → fusão e alertas)
```bash
python main.py all
```

**Passo 9 — Validar o vídeo contra o gabarito do fisioterapeuta**
```bash
python main.py video-validation
```

**Passo 10 — Gerar os gráficos dos resultados**
```bash
python main.py figures
```

**Passo 11 — Ver os resultados**
```bash
python main.py summary                            # métricas principais de cada módulo
python main.py alerts --level critical            # alertas mais graves
start results/reports/video_validation.html       # relatório HTML (Linux: xdg-open | macOS: open)
explorer.exe "$(cygpath -w results/figures)"      # pasta dos gráficos (Windows)
python -m streamlit run dashboard/app.py          # painel em http://localhost:8501 (Ctrl+C encerra)
```

**O que você deve ver** (pequenas variações são normais):

| Passo | Resultado esperado |
|---|---|
| 7 · `inspect` | Bases com **OK**, `ângulos pré-calculados: 8`, 2 vídeos de exemplo, `Speech configurado: True` |
| 8 · clínico | `72,826 medições -> 12,085 horas-paciente`; recall de 0,80 a 0,93 |
| 8 · áudio | `1380 falas de 8 falantes`; ROC-AUC ~0,78; WER ~0,24 (controle) e ~0,71 (disartria) |
| 8 · vídeo | 2 vídeos com `pose_visible_fraction` 1.0 e relatórios `video_*.html` |
| 8 · fusão | ~14.500 eventos e ~2.050 alertas, 12 *critical* |
| 9 · `video-validation` | Agachamento com ROC-AUC 0,79; contagem a até 2 repetições do gabarito |
| 10 · `figures` | `10 gráficos em .../results/figures` |

Para analisar mais dados além dos exemplos, veja onde colocá-los na [seção 3.2](#32-usando-mais-dados).

---

## Sumário

- [⚡ Uso rápido](#-uso-rápido)
1. [Pré-requisitos](#1-pré-requisitos)
2. [Instalação (passo a passo em bash)](#2-instalação-passo-a-passo-em-bash)
3. [Primeiros testes](#3-primeiros-testes)
4. [Dados de entrada](#4-dados-de-entrada)
5. [Azure no plano gratuito](#5-azure-no-plano-gratuito)
6. [Tutorial dos comandos](#6-tutorial-dos-comandos)
7. [Tutorial das funções (uso direto em Python)](#7-tutorial-das-funções-uso-direto-em-python)
8. [Variáveis de configuração](#8-variáveis-de-configuração)
9. [Arquivos gerados](#9-arquivos-gerados)
10. [Arquitetura](#10-arquitetura)
11. [Resultados e como são medidos](#11-resultados-e-como-são-medidos)
12. [Licenças das ferramentas](#12-licenças-das-ferramentas)
13. [Glossário](#13-glossário)
14. [Referências](#14-referências)

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
| `pyarrow` | Ler a base de fala (`.parquet`) em lotes | `audio/speech_loader.py` |
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
# 1. Clone o repositório e entre na pasta (onde estão main.py e requirements.txt)
git clone https://github.com/BrunoAkira/IA-Multimodal-Hospital.git
cd IA-Multimodal-Hospital
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

## 3. Primeiros testes

### 3.1 Com os dados de exemplo incluídos no repositório

A pasta `example_data/` traz amostras pequenas de bases públicas reais, no formato que o sistema
lê (detalhes e licenças em `example_data/README.md`):

| Pasta | Conteúdo |
|---|---|
| `clinical/` | Sinais vitais, prescrições e transferências de 100 pacientes de UTI |
| `motion_capture/` | Alguns exercícios de reabilitação de um participante (`.amc`) |
| `exercise_videos/` | Gabarito de fisioterapeuta e ângulos das articulações já calculados de 8 vídeos |
| `physio_videos/` | Dois trechos curtos de vídeo de exercícios, para OpenPose/YOLOv8 |
| `speech/` | Base de fala, baixada por `scripts/download_example_data.py` (não é redistribuída) |

```bash
source .venv/Scripts/activate
export HOSPITAL_AI_DATASETS=example_data     # vale só para este terminal
python scripts/download_example_data.py      # base de fala (2 arquivos, ~740 MB, uma vez só)

python main.py inspect
python main.py all
python main.py video-validation
python main.py figures
python -m streamlit run dashboard/app.py     # abra http://localhost:8501
```

### 3.2 Usando mais dados

Os dados de exemplo são recortes pequenos das bases públicas da [seção 4.2](#42-bases-de-exemplo-usadas-na-validação).
Para analisar mais dados (as bases completas ou dados próprios no mesmo formato), coloque os
arquivos em `datasets/`, cada tipo na sua pasta:

| Tipo de dado | Pasta | Formato |
|---|---|---|
| Sinais vitais, prescrições e transferências | `datasets/clinical/hosp/` e `datasets/clinical/icu/` | Tabelas `.csv` ou `.csv.gz` |
| Gravações de voz | `datasets/speech/` | Arquivos `.parquet` |
| Captura de movimento | `datasets/motion_capture/` | Arquivos `.amc` (uma pasta por participante) |
| Vídeos de exercícios com gabarito | `datasets/exercise_videos/` | `Ex<n>/` com os vídeos + `Segmentation.csv` |
| Vídeos de fisioterapia a analisar | `data/physio_videos/` | `.mp4`, `.avi`, `.mov` ou `.mkv` |
| Consultas | `data/consultations/` | `consult_XX.txt` + `consult_XX.wav` |

O formato de cada arquivo está na [seção 4.1](#41-formatos-aceitos), e os comandos para organizar
as bases completas na [seção 4.4](#44-organizando-as-bases-de-exemplo-bash).

A pasta `datasets/` é a padrão. Para trocar de `example_data/` para ela, basta desfazer a variável
do passo 6 do Uso rápido; para usar outra pasta qualquer, aponte a variável para ela:

```bash
unset HOSPITAL_AI_DATASETS                       # volta a usar datasets/
export HOSPITAL_AI_DATASETS=/caminho/para/dados   # ou usa outra pasta com a mesma estrutura
python main.py inspect                            # confere o que foi encontrado
python main.py all
```

### 3.3 Verificação dos resultados

Depois de executar o [Uso rápido](#-uso-rápido), o comando abaixo mostra as principais métricas
de cada módulo, lidas dos arquivos de resultado, para comparar com os valores de referência da
[seção 11](#11-resultados-e-como-são-medidos):

```bash
python main.py summary
```

Um módulo que não aparece na saída ainda não foi executado.

### 3.4 Gráficos para o relatório

```bash
python main.py figures
```

Gera em `results/figures/` os gráficos usados no relatório técnico, todos calculados a partir dos
arquivos de resultado (seção 6.6b). Rode depois de `all` e `video-validation`.

---

## 4. Dados de entrada

Cada módulo lê um tipo de dado num formato definido. Qualquer fonte que siga esse formato pode
ser usada; as bases públicas da seção 4.2 servem como exemplo e como validação. **Todos os dados
são padronizados em inglês** (nomes de colunas, rótulos, eventos e alertas).

### 4.1 Formatos aceitos

| Módulo | Pasta | Formato esperado |
|---|---|---|
| Clínico | `datasets/clinical/` | Tabelas CSV (`.csv` ou `.csv.gz`): `chartevents` (`subject_id`, `stay_id`, `charttime`, `itemid`, `valuenum`), `prescriptions` (`hadm_id`, `starttime`, `drug`, `dose_val_rx`, `route`...) e `transfers` (`hadm_id`, `careunit`, `intime`, `outtime`). Os códigos de cada sinal vital ficam em `VITAL_ITEMIDS` (`src/config.py`), ajustáveis para outro sistema hospitalar |
| Voz | `datasets/speech/` | Arquivos `.parquet` com uma coluna de áudio (WAV), o texto falado, o id do falante e o grupo (com ou sem alteração na fala). Os nomes das colunas são reconhecidos automaticamente |
| Consultas | `data/consultations/` | Pares `consult_XX.txt` (roteiro `Doctor: ...` / `Patient: ...`) e `consult_XX.wav` (gravação) |
| Captura de movimento | `datasets/motion_capture/` | Arquivos `.amc` (ângulos das articulações por quadro), nomeados `{Movimento}R{repetições}S{sujeito}.amc`: `R1` = repetição de referência, `R3` = três repetições a avaliar |
| Vídeo de fisioterapia | `data/physio_videos/` ou `<pasta de datasets>/physio_videos/` | Vídeos `.mp4`, `.avi`, `.mov` ou `.mkv`; o prefixo do nome define o exercício (seção 4.5) |
| Vídeos com gabarito | `datasets/exercise_videos/` | Vídeos em `Ex<n>/` + um `Segmentation.csv` (separado por `;`) com `video_id`, `repetition_number`, `exercise_id`, `person_id`, `first_frame`, `last_frame`, `exercise_subtype` e `correctness` (1 = correta, 0 = incorreta) |

### 4.2 Bases de exemplo usadas na validação

| Base de exemplo | O que contém | O que demonstra | O que baixar | Ref. |
|---|---|---|---|---|
| **Base clínica de UTI** | Registros desidentificados de 100 pacientes internados: sinais vitais, prescrições e transferências entre setores | Detecção de anomalias em sinais vitais, prescrições e movimentação do paciente | O ZIP completo (~15 MB) | [1] |
| **Base de fala** | Gravações em inglês de pessoas com disartria (dificuldade na fala de origem neurológica) e sem alteração, com o texto lido | Análise de voz (disartria, cansaço, respiração) e qualidade da transcrição | 2 arquivos `.parquet` (~370 MB cada): `train-00000` (controle) e `train-00002` (com disartria) | [2] |
| **Base de captura de movimento** | Ângulos das articulações de 10 pessoas em 22 exercícios de reabilitação, cada um em ritmo normal, rápido e lento | Movimento fora do padrão em relação à execução de referência | 1 a 3 participantes (`S01.zip`...) | [3] |
| **Base de vídeos de exercícios** | Vídeos de 10 pessoas em 6 exercícios de fisioterapia, cada repetição marcada por fisioterapeuta como correta ou incorreta | Postura em vídeo real (OpenPose/YOLOv8) e detecção de execução incorreta | `videos.zip`, `Segmentation.csv`, `Segmentation.txt` | [4] |
| **Consultas** | 12 roteiros de consulta em inglês escritos para o projeto, com sintomas, termos críticos e tons diferentes | Transcrição (Speech to Text) e análise de texto | já incluídos em `data/consultations/` | — |

Nomes, autores, links e licenças estão nas [Referências](#14-referências).

### 4.3 Estrutura das pastas

```
datasets/
  clinical/
    hosp/  prescriptions.csv.gz, transfers.csv.gz, ...   <- base clínica de UTI [1]
    icu/   chartevents.csv.gz, ...
  speech/
    train-00000-of-00004.parquet                         <- base de fala [2]
  motion_capture/
    S01/ S02/ ...                                        <- base de captura de movimento [3]
  exercise_videos/
    Ex1/ ... Ex6/  Segmentation.csv                      <- base de vídeos de exercícios [4]
data/
  consultations/  consult_01.txt ... consult_12.txt  (+ consult_XX.wav)
  physio_videos/  squat_....mp4, arm_raise_....mp4   <- vídeos a analisar
```

### 4.4 Organizando as bases de exemplo (bash)

```bash
mkdir -p datasets/clinical datasets/speech datasets/motion_capture datasets/exercise_videos

# Base clínica de UTI [1] (o Python extrai o zip; o Git Bash nem sempre tem unzip)
python -m zipfile -e ~/Downloads/mimic-iv-clinical-database-demo-2.2.zip /tmp/clinica
cp -r /tmp/clinica/*/hosp /tmp/clinica/*/icu datasets/clinical/

# Base de fala [2]
cp ~/Downloads/train-0000[02]-of-00004.parquet datasets/speech/

# Base de captura de movimento [3] (repita para S02, S03...)
python -m zipfile -e ~/Downloads/S01.zip datasets/motion_capture/

# Base de vídeos de exercícios [4]
python -m zipfile -e ~/Downloads/videos.zip datasets/exercise_videos/
cp ~/Downloads/Segmentation.csv ~/Downloads/Segmentation.txt datasets/exercise_videos/

python main.py inspect      # confere tudo
```
> Os nomes dos zips são exemplos: use os nomes dos arquivos que você baixou.

### 4.5 Vídeos de fisioterapia (`data/physio_videos/`)

O reconhecimento visual (OpenPose/YOLOv8) roda sobre vídeos de pessoas: gravações de sessões de
fisioterapia, da base de vídeos de exercícios [4] ou feitas com o celular. A base de captura de
movimento [3] não serve aqui, porque seus vídeos mostram só o esqueleto desenhado.

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
cp "datasets/exercise_videos/CAMINHO/DO/VIDEO.mp4" data/physio_videos/squat_rehab_01.mp4
```

**Para gravar com o celular:** câmera parada, corpo inteiro no quadro, 15–30 s, 5 repetições.
Sugestão: um vídeo correto e outro incorreto de propósito (agachando pouco, um lado mais baixo),
e um com um objeto no chão (garrafa, mochila) para testar a área de risco de queda.

### 4.6 Áudio das consultas (`data/consultations/`)

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
| `python main.py video-validation` | Mede a precisão da análise de vídeo contra o gabarito do fisioterapeuta (correta x incorreta) |
| `python main.py fusion` | Pacientes virtuais + fusão + alertas |
| `python main.py all` | `clinical` → `audio` → `video` → `fusion` |
| `python main.py monitor` | Simulação de tempo real dos sinais vitais no terminal |
| `python main.py figures` | Gráficos dos resultados em `results/figures/` |
| `python main.py summary` | Métricas principais de cada módulo no terminal |
| `python main.py alerts` | Fila de alertas (`--level critical`, `--limit 20`) |
| `python main.py consultations` | Resultado da análise de cada consulta |
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
'spo2': {'injected': 240, 'precision': 0.407, 'recall': 0.933}
```
→ foram injetadas 240 anomalias artificiais de SpO₂ e o detector encontrou 93% delas.
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
   que ele nunca ouviu (validação "deixa um falante de fora"). O melhor fica salvo em
   `models/dysarthria_classifier.joblib`.
3. **Indicadores de cansaço e de dificuldade respiratória**, comparando cada fala com a linha de
   base do próprio falante.
4. **Azure Speech to Text** numa amostra da base de fala (`AZURE_STT_SPEECH_SAMPLES`, padrão 60),
   calculando a taxa de erro de palavras (WER) por grupo.
5. **Consultas:** `.wav` → Azure Speech to Text → Azure Text Analytics for health (termos
   críticos e negação, ex.: "no fever") + sentimento.

Eventos: `dysarthria_risk`, `fatigue_indicator`, `breathing_difficulty_indicator`,
`critical_term_mentioned`, `negative_sentiment`.

Saídas: `results/audio_summary.json`, `voice_features.csv`, `voice_indicators.csv`,
`consultations.json`, `speech_azure_stt.csv` (se o Speech estiver ativo).

Ver o resultado de cada consulta (origem da transcrição, sentimento e termos):
```bash
python main.py consultations
```
`transcript_source`: `azure`/`cache` = Speech to Text usado; `script` = sem `.wav` (texto do roteiro).
`sentiment_source`: `azure_sentiment` = Azure AI Language; `local_lexicon` = análise local (cota do mês esgotada).

### 6.4 `video` — fisioterapia

```bash
python main.py video
```

**Parte 1 — captura de movimento (`datasets/motion_capture/`):** compara as 3 repetições de cada arquivo `R3`
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

### 6.4b `video-validation` — validação da análise de vídeo

```bash
python main.py video-validation                                  # exercícios 1 e 6, até 4 vídeos cada
python main.py video-validation --exercises 1,5,6 --max-videos 6
```

Mede se o sistema reconhece execuções incorretas, comparando cada repetição com a marcação de
um fisioterapeuta. Lê os vídeos em `datasets/exercise_videos/Ex<n>/` (câmera `Camera17` no nome do
arquivo) e o gabarito `Segmentation.csv` (formato na seção 4.1). Na base de exemplo [4], os ids
de exercício são:

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
   ângulos em `results/video_validation_angles/` (rodar de novo é instantâneo).
2. Corta cada vídeo nas repetições do gabarito e mede: amplitude, ângulo máximo e mínimo,
   duração, velocidade, inclinação do tronco e movimento do lado que deveria ficar parado.
3. Mostra o quanto cada medida, sozinha, separa correta de incorreta (`single_feature_auc`:
   0,5 = acaso, 1,0 = perfeito) e treina uma regressão logística validada deixando uma pessoa
   de fora por vez.
4. Compara o número de repetições contadas automaticamente com o gabarito.
5. Gera `results/reports/video_validation.html` (gráfico de cada vídeo com as repetições
   corretas em verde e incorretas em vermelho) e eventos `incorrect_execution` para a fusão.

Tempo: cerca de 1 a 2 minutos por vídeo na primeira vez.

### Zonas críticas por vídeo

Para ajustar as zonas a um vídeo específico, crie ao lado dele um arquivo JSON com o mesmo nome
e a extensão `.zones.json` (ex.: `squat_rehab01.mp4` → `squat_rehab01.zones.json`):
```json
{"exercise_area": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], "fall_risk_floor": null}
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
   sem repetir o mesmo paciente antes de 12 h, a menos que piore. Eventos com pontuação abaixo de
   0,5 não entram na fusão, e *high* exige confirmação: dois tipos de sinal diferentes, NEWS2
   elevado ou mais de uma modalidade alterada.

Ver os alertas no terminal, dos mais graves para os menos graves:
```bash
python main.py alerts                              # contagem por nível + 10 primeiros alertas
python main.py alerts --level critical --limit 20  # só os critical
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

### 6.6b `figures` — gráficos dos resultados

```bash
python main.py figures
```

Lê os JSON de `results/` e salva um PNG por gráfico em `results/figures/`: fluxo do sistema,
recall e precisão por sinal vital, eventos por fonte, métricas e matrizes de confusão da voz,
erro da transcrição por grupo, poder de cada característica no vídeo, contagem de repetições,
alertas por nível e uso do Azure. Rode depois de `all` e `video-validation`; gráficos cujo
resultado ainda não existe são pulados.

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
| `python scripts/download_example_data.py` | Baixa a base de fala de exemplo para `example_data/speech/` |
| `python scripts/build_example_data.py` | Gera `example_data/` a partir das bases completas em `datasets/` (recortes pequenos de cada base) |

---

## 7. Tutorial das funções (uso direto em Python)

Todas as funções podem ser usadas isoladamente, num script ou notebook. A pasta `examples/` traz
um script pronto para cada caso; rode a partir da pasta do projeto, com o ambiente ativo.

| Script | O que demonstra |
|---|---|
| `python examples/event_example.py` | Criar um evento no formato único e ler os eventos gravados |
| `python examples/vital_stream_example.py` | Detector de sinais vitais recebendo uma leitura por vez |
| `python examples/text_terms_example.py "texto"` | Termos críticos e negação num texto (`--azure` usa o Text Analytics for health) |
| `python examples/voice_features_example.py audio.wav` | Características de voz de um arquivo WAV |
| `python examples/single_video_example.py video.mp4` | Análise completa de um único vídeo |
| `python examples/motion_capture_example.py` | Listar e ler os arquivos de captura de movimento |

Exemplo de uso de uma função num script próprio:
```python
from src.clinical.vitals import VitalStreamDetector

detector = VitalStreamDetector()
score, event_type, evidence = detector.update("bed_1", "spo2", 88)
```

### 7.1 Núcleo (`src/core/`)

| Função | O que faz |
|---|---|
| `events.Event(...)` | Evento no formato único; a gravidade é calculada a partir da pontuação |
| `events.save_events(lista)` | Anexa eventos a `results/events.jsonl` |
| `events.load_events()` | Lê todos os eventos gravados |
| `events.clear_events("audio")` | Apaga só os eventos de uma modalidade |
| `azure_client.speech_enabled()` / `language_enabled()` | `True` se as chaves estão no `.env` |
| `azure_client.to_wav16k_bytes(samples, sr)` | Converte um áudio em WAV 16 kHz mono (formato do Azure) |
| `azure_client.speech_to_text(wav_bytes)` | `{"text", "confidence", "source"}` ou `None` (áudio de até 60 s) |
| `azure_client.analyze_sentiment(text)` | `{"sentiment", "scores", "sentences"}` ou `None` |
| `azure_client.analyze_health(text)` | `{"entities": [{"text", "category", "confidence", "assertion"}]}` ou `None` |
| `azure_client.usage_summary()` | Uso do mês e limites de segurança |
| `report.write_report(nome, titulo, secoes)` | Relatório HTML; cada seção tem `heading`, `text`, `table` e `images` |

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

### 7.3 Áudio (`src/audio/`)

| Função | O que faz |
|---|---|
| `speech_loader.iter_speech(max_rows=...)` | Gera as falas da base de fala uma a uma: `{speaker_id, group, prompt, audio, sr, ...}` |
| `features.extract_features(audio, sr)` | Dicionário com ~28 características da voz |
| `voice_model.train_dysarthria_classifier(df)` | Treina e valida por falante; retorna métricas e modelo |
| `voice_model.vocal_indicators(df)` | `fatigue_index` e `breathing_index` por fala |
| `text_analysis.analyze_text(texto)` | Termos críticos + sentimento (Azure ou local) |
| `text_analysis.local_critical_terms(texto)` | Só a versão local (dicionário + negação) |
| `consultations.split_audio(audio, sr)` | Divide áudios longos em trechos de até 50 s |
| `consultations.transcribe(audio, sr)` | Transcreve com o Azure trecho a trecho |
| `pipeline.word_error_rate(ref, hyp)` | WER entre o texto esperado e o transcrito |

### 7.4 Vídeo (`src/video/`)

| Função | O que faz |
|---|---|
| `amc.parse_amc(caminho)` | Lê um `.amc` → DataFrame (quadros × ângulos) |
| `amc.index_motion_capture()` | Lista os `.amc` com participante, movimento e nº de repetições |
| `motion.build_repetition_table()` | Medidas de cada repetição vs. referência |
| `motion.score_repetitions(df)` | Regras + IsolationForest → pontuação e métricas |
| `motion.dtw_distance(a, b)` | Distância DTW entre duas curvas |
| `vision.PoseEstimator()` | OpenPose (ou YOLOv8-pose): `estimador(quadro)` → pontos do corpo |
| `vision.frame_angles(pontos)` | Ângulos de joelho, quadril, ombro, cotovelo e inclinação do tronco |
| `vision.analyze_angles(df, exercicio, fps)` | Amplitude, assimetria, repetições, queda |
| `vision.analyze_video(caminho)` | Vídeo completo: pose + YOLOv8 + zonas → resumo, achados, ângulos, fotos |

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
| `AZURE_STT_SPEECH_SAMPLES` | `60` | Quantas falas da base de fala vão para o Speech to Text (`0` desliga) |
| `MAX_SPEECH_ROWS` | `1500` | Quantas falas da base de fala processar |
| `HOSPITAL_AI_DATASETS` | `datasets` | Pasta dos datasets (ex.: `example_data`) |

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
| `speech_azure_stt.csv` | Transcrições do Azure + WER por fala |
| `consultations.json` | Transcrição, sentimento e termos de cada consulta |
| `physio_repetitions.csv` | Medidas e pontuação de cada repetição da captura de movimento |
| `video_angles_<nome>.csv` | Ângulos quadro a quadro de cada vídeo |
| `reports/*.html` | Relatórios automáticos |
| `figures/*.png` | Gráficos dos resultados (`python main.py figures`) |
| `cache/` | Respostas do Azure + `azure_usage.json` (**não apague no meio do mês**: o cache evita novas cobranças de cota) |

---

## 10. Arquitetura

A arquitetura completa (diagramas, formato `Event`, fusão, uso do Azure, memória) está em
[ARCHITECTURE.md](ARCHITECTURE.md). Onde fica cada funcionalidade:

| Funcionalidade | Onde está |
|---|---|
| Postura e ângulos em vídeo (OpenPose / YOLOv8-pose) | `src/video/vision.py` → `PoseEstimator`, `frame_angles` |
| Objetos e áreas críticas (YOLOv8) | `src/video/vision.py` → `analyze_video`, `ZONES` |
| Movimento fora do padrão e execução incorreta | `src/video/motion.py`, `analyze_angles`, `src/video/video_validation.py` |
| Relatórios automáticos de fisioterapia | `results/reports/physio_*.html`, `video_*.html`, `video_validation.html` |
| Alterações na voz (disartria, cansaço, respiração) | `src/audio/features.py`, `src/audio/voice_model.py` |
| Transcrição de consultas (Azure Speech to Text) | `src/core/azure_client.py` → `speech_to_text` |
| Termos críticos e sentimento (Azure AI Language) | `src/audio/text_analysis.py`, `azure_client.analyze_health` |
| Anomalias em sinais vitais | `src/clinical/vitals.py` |
| Anomalias em prescrições | `src/clinical/prescriptions.py` |
| Movimentação entre setores | `src/clinical/transfers.py` |
| Fusão e alertas | `src/fusion.py` → `results/alerts.csv` |
| Painel e monitor em tempo real | `dashboard/app.py`, `python main.py monitor` |

---

## 11. Resultados e como são medidos

| Módulo | Como medimos | Resultado com os dados de exemplo | Onde ver |
|---|---|---|---|
| Sinais vitais | Anomalias injetadas em 2% das medições de 100 pacientes (12.085 horas-paciente) | Recall de 80% a 93%; precisão de 19% a 41% | `clinical_summary.json` |
| Voz | Classificador de disartria testado sempre num falante fora do treino (8 falantes, 1.380 falas) | Regressão logística: ROC-AUC 0,78, acurácia 75%, 158 de 240 falas com disartria reconhecidas | `audio_summary.json` |
| Speech to Text | Taxa de erro de palavras (WER) em 60 falas | 24% sem alteração × 71% com disartria | `audio_summary.json`, `speech_azure_stt.csv` |
| Texto das consultas | 12 consultas roteirizadas | 30 eventos de termos críticos e sentimento | `consultations.json` |
| Vídeo × gabarito | Repetição correta × incorreta, testando sempre numa pessoa fora do treino (8 vídeos, 123 repetições) | Agachamento: ROC-AUC 0,79 (melhor sinal: inclinação do tronco, 0,82). Abdução: só 2 pessoas, o modelo não generaliza (0,12); o melhor sinal sozinho é o outro braço se mexer junto (0,79) | `video_validation.json`, `reports/video_validation.html` |
| Contagem de repetições | Contagem automática × gabarito | 127 contadas para 123, no máximo 2 de diferença por vídeo | `video_validation.json` |
| Captura de movimento | Ritmo rápido/lento como desvio (1 participante) | Regra de ritmo: 2 de 3 desvios, sem falso alarme | `video_summary.json`, `reports/physio_S01.html` |
| Fusão | Eventos de todos os módulos por paciente, janela de 24 h | 14.545 eventos → 2.051 alertas (12 *critical*, 1.126 *high*, 913 *medium*); 18 multimodais | `fusion_summary.json`, `alerts.csv` |
| Azure | Contador mensal | 256 s de áudio (1,4% da cota) e 24 registros de texto (0,5%) | `audio_summary.json` |

Todos esses números aparecem nos gráficos de `python main.py figures` e no relatório técnico.

---

## 12. Licenças das ferramentas

| Ferramenta | Licença / condição |
|---|---|
| OpenPose (CMU) | Licença acadêmica / não comercial |
| YOLOv8 (Ultralytics) | AGPL-3.0 |
| Azure AI Speech / Language | Termos do Azure; plano gratuito F0 |
| Bibliotecas Python (numpy, pandas, scikit-learn, OpenCV, Streamlit...) | Licenças de código aberto de cada projeto |

---

## 13. Glossário

| Termo | O que é | Onde aparece no projeto |
|---|---|---|
| **Alerta (medium / high / critical)** | Aviso para a equipe, gerado quando os eventos de um paciente somam risco suficiente. *Critical* exige dois ou mais tipos de dado alterados ao mesmo tempo | Central de alertas, painel |
| **Evento** | Registro padronizado de algo detectado (paciente, horário, tipo de dado, pontuação 0–1, explicação). É a "língua comum" entre os módulos | Todos os módulos |
| **Paciente virtual (VP001...)** | Combinação de registros de bases de exemplo diferentes num só paciente, para demonstrar a fusão. Com dados de um mesmo paciente real, não é necessário | Fusão |
| **Cooldown** | Tempo mínimo (12 h) antes de repetir o mesmo alerta para o mesmo paciente, a menos que piore | Central de alertas |
| **NEWS2** | *National Early Warning Score 2*: escore de alerta precoce que soma pontos para frequência cardíaca, pressão, saturação, respiração e temperatura fora do normal | Sinais vitais |
| **z-score robusto** | Quanto um valor se afasta do normal da própria pessoa, usando mediana e MAD em vez de média e desvio-padrão | Sinais vitais, índices de voz |
| **MAD** | *Median Absolute Deviation*: medida de variação baseada na mediana, pouco afetada por valores extremos | Sinais vitais, índices de voz |
| **Isolation Forest** | Modelo que encontra casos raros sem exemplos rotulados: o que fica isolado com poucos cortes aleatórios é anomalia | Sinais vitais, prescrições, repetições de exercício |
| **Cadeia de Markov** | Probabilidade de passar de um estado a outro; aqui, de um setor do hospital a outro | Transferências |
| **Regressão logística (LogReg)** | Modelo de classificação que dá a probabilidade de um caso pertencer a uma classe | Voz (disartria), vídeo (correto × incorreto) |
| **Random Forest (RF)** | Conjunto de árvores de decisão que votam na resposta | Voz (disartria) |
| **Deixa um de fora (leave-one-group-out)** | Validação em que o modelo é sempre testado em pessoas que não estavam no treino, para não "decorar" quem já viu | Voz, vídeo |
| **ROC-AUC** | Nota de 0 a 1 de quanto um modelo separa duas classes: 0,5 é acaso, 1,0 é perfeito | Resultados de voz e vídeo |
| **Precisão / Recall** | Recall: quanto das anomalias reais foi encontrado. Precisão: quanto dos alertas era anomalia de verdade | Avaliação dos sinais vitais |
| **Disartria** | Dificuldade na fala de origem neurológica (ex.: após AVC), com fala lenta, arrastada ou instável | Análise de voz |
| **F0** | Frequência fundamental: o tom da voz | Características de voz |
| **Jitter / Shimmer** | Pequenas variações de tom (jitter) e de volume (shimmer) entre ciclos da voz; altos em vozes instáveis | Características de voz |
| **HNR** | *Harmonics-to-Noise Ratio*: quanto a voz é "limpa" em relação ao ruído; baixo em voz rouca ou soprosa | Características de voz |
| **MFCC** | Coeficientes que resumem o timbre da voz, muito usados em reconhecimento de fala | Classificador de voz |
| **WER** | *Word Error Rate*: proporção de palavras erradas numa transcrição | Qualidade do Speech to Text |
| **Speech to Text** | Serviço do Azure que transforma áudio em texto | Consultas, base de fala |
| **Text Analytics for health** | Serviço do Azure que encontra sintomas, medicamentos e diagnósticos num texto e indica se foram negados | Consultas |
| **Negação (assertion)** | Marcação de que um termo foi negado pelo paciente (ex.: "no fever") e não deve gerar alerta | Consultas |
| **OpenPose** | Modelo de visão que encontra os pontos do corpo (ombros, cotovelos, joelhos...) numa imagem | Vídeo |
| **YOLOv8 / YOLOv8-pose** | Modelos de visão que detectam pessoas e objetos (YOLOv8) e os pontos do corpo (YOLOv8-pose) | Vídeo |
| **Amplitude de movimento (ROM)** | Diferença entre o maior e o menor ângulo de uma articulação durante a repetição | Vídeo, captura de movimento |
| **DTW** | *Dynamic Time Warping*: compara a forma de duas curvas mesmo quando uma é mais rápida | Captura de movimento |
| **Captura de movimento (.amc)** | Registro dos ângulos das articulações feito com sensores ou marcadores no corpo | Captura de movimento |
| **Plano F0** | Plano gratuito dos serviços do Azure, com limite mensal de uso | Azure |

---

## 14. Referências

### Bases de dados de exemplo

| # | Base (como chamada neste README) | Nome original e fonte | Licença |
|---|---|---|---|
| [1] | Base clínica de UTI | **MIMIC-IV Clinical Database Demo v2.2** — Johnson, A. et al., PhysioNet. https://physionet.org/content/mimic-iv-demo/2.2/ | Open Data Commons ODbL v1.0 |
| [2] | Base de fala | **TORGO Database** — Rudzicz, F., Namasivayam, A. K., Wolff, T., University of Toronto. Cópia em Parquet: https://huggingface.co/datasets/abnerh/TORGO-database | Conforme os autores (verificar antes de redistribuir) |
| [3] | Base de captura de movimento | **UL-RED — University of Liverpool Rehabilitation Exercise Dataset**, *Scientific Data* 12:761 (2025). https://datacat.liverpool.ac.uk/2729/ | CC BY 4.0 |
| [4] | Base de vídeos de exercícios | **REHAB24-6** — Černek, A., Sedmidubsky, J., Budikova, P. *REHAB24-6: Physical Therapy Dataset for Analyzing Pose Estimation Methods*. SISAP 2024. https://zenodo.org/records/13305826 | CC BY-NC 4.0 |

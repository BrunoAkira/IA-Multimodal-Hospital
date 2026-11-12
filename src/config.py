"""
Configuração central do projeto.

Tudo que é "ajustável" (caminhos, limiares, limites do Azure) fica aqui,
para que os módulos não tenham números mágicos espalhados pelo código.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # lê as chaves do Azure do arquivo .env (se existir)

# HOSPITAL_AI_DATASETS permite apontar para outra pasta (ex.: dados de amostra dos testes)
DATASETS = Path(os.getenv("HOSPITAL_AI_DATASETS", ROOT / "datasets"))
VIDEO_DIR = DATASETS / "video"          # UL-RED extraído (S01/, S02/, ...)
AUDIO_DIR = DATASETS / "audio"          # TORGO em Parquet
CLINICAL_DIR = DATASETS / "clinical"    # MIMIC-IV Demo (hosp/, icu/)

CONSULT_DIR = ROOT / "data" / "consultations"   # roteiros .txt + gravações .wav
PHYSIO_VIDEO_DIR = ROOT / "data" / "physio_videos"  # vídeos .mp4 gravados pelo grupo
MODELS_DIR = ROOT / "models"            # pesos do OpenPose / YOLO e modelos treinados
RESULTS_DIR = ROOT / "results"          # tudo que o pipeline gera
CACHE_DIR = RESULTS_DIR / "cache"       # respostas do Azure salvas (nunca chamamos 2x)
REPORTS_DIR = RESULTS_DIR / "reports"   # relatórios HTML

for _d in (RESULTS_DIR, CACHE_DIR, REPORTS_DIR, MODELS_DIR, PHYSIO_VIDEO_DIR, CONSULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Arquivos padronizados (em inglês) que cada módulo gera
EVENTS_FILE = RESULTS_DIR / "events.jsonl"   # todos os eventos de todos os módulos
ALERTS_FILE = RESULTS_DIR / "alerts.csv"     # alertas finais após a fusão

# ---------------------------------------------------------------------------
# Limites de memória (máquina com 24 GB: lemos tudo em pedaços)
# ---------------------------------------------------------------------------
CSV_CHUNK_ROWS = 200_000      # linhas por pedaço ao ler chartevents
PARQUET_BATCH_ROWS = 32       # áudios por lote ao ler o TORGO
MAX_TORGO_ROWS = int(os.getenv("MAX_TORGO_ROWS", "1500"))  # amostra de áudios processados

# ---------------------------------------------------------------------------
# MIMIC-IV: itemids dos sinais vitais em icu/chartevents
# (nomes padronizados em inglês -> lista de itemids equivalentes)
# ---------------------------------------------------------------------------
VITAL_ITEMIDS = {
    "heart_rate": [220045],
    "systolic_bp": [220179, 220050],   # não invasiva, arterial
    "diastolic_bp": [220180, 220051],
    "spo2": [220277],
    "resp_rate": [220210],
    "temperature_c": [223762],
    "temperature_f": [223761],         # convertido para °C no loader
}

# Faixas clínicas "duras" (fora disso já é alerta, independente do modelo)
VITAL_HARD_LIMITS = {
    "heart_rate": (40, 130),
    "systolic_bp": (90, 180),
    "diastolic_bp": (40, 110),
    "spo2": (90, 100),
    "resp_rate": (8, 30),
    "temperature_c": (35.0, 39.0),
}

# Faixas plausíveis: valores fora disso são erro de digitação/sensor e são descartados
VITAL_PLAUSIBLE = {
    "heart_rate": (10, 300),
    "systolic_bp": (30, 300),
    "diastolic_bp": (10, 200),
    "spo2": (40, 100),
    "resp_rate": (2, 80),
    "temperature_c": (25, 45),
}

ROBUST_Z_THRESHOLD = 4.0   # |z| robusto acima disso = mudança brusca
ROLLING_WINDOW = 24        # nº de medições anteriores usadas como "normal" do paciente

# Medicamentos de alto risco (início/alteração gera atenção extra)
HIGH_ALERT_DRUGS = [
    "insulin", "heparin", "warfarin", "enoxaparin", "morphine", "hydromorphone",
    "fentanyl", "oxycodone", "potassium chloride", "norepinephrine", "vasopressin",
    "propofol", "midazolam", "amiodarone", "digoxin",
]

# ---------------------------------------------------------------------------
# Azure (plano gratuito F0) — nunca passamos de ~80% da cota mensal
# ---------------------------------------------------------------------------
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY", "")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "").rstrip("/")

# F0 Speech: 5 h/mês -> usamos no máximo 4 h (14 400 s)
SPEECH_SECONDS_BUDGET = float(os.getenv("SPEECH_SECONDS_BUDGET", "14400"))
# F0 Language: 5 000 registros/mês (1 registro = até 1 000 caracteres) -> máx. 4 000
LANGUAGE_RECORDS_BUDGET = int(os.getenv("LANGUAGE_RECORDS_BUDGET", "4000"))
# Quantos áudios do TORGO mandar para o Azure STT (o resto fica só local)
AZURE_STT_TORGO_SAMPLES = int(os.getenv("AZURE_STT_TORGO_SAMPLES", "60"))

LANGUAGE = "en-US"   # todos os dados são padronizados em inglês

# ---------------------------------------------------------------------------
# Fusão e alertas
# ---------------------------------------------------------------------------
# Áudio/vídeo não têm data real: os eventos usam horários relativos a esta "época"
# e a fusão os reposiciona dentro da internação do paciente virtual.
RELATIVE_EPOCH = "2000-01-01T00:00:00"
FUSION_WINDOW_HOURS = 24       # eventos de modalidades diferentes nessa janela se somam
ALERT_COOLDOWN_HOURS = 6        # não repetir o mesmo alerta do mesmo paciente antes disso
MODALITY_WEIGHTS = {"clinical": 1.0, "audio": 0.7, "video": 0.6, "text": 0.8}

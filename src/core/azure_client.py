"""
Cliente Azure mínimo (REST) para os DOIS serviços exigidos na entrega:
  1. Azure AI Speech  -> Speech to Text (áudio curto, até 60 s por chamada)
  2. Azure AI Language -> Sentiment Analysis + Text Analytics for health

Proteções para não gastar NADA no plano gratuito (F0):
  * Cache em disco: a resposta de cada áudio/texto é salva em results/cache.
    Se o mesmo conteúdo for pedido de novo, lemos do cache (zero chamadas).
  * Controle de cota: um contador mensal (results/cache/azure_usage.json)
    bloqueia chamadas antes de chegar a ~80% do limite gratuito.
  * Sem chave configurada -> as funções retornam None e o projeto usa a
    alternativa local (o pipeline continua funcionando).

Usamos REST (biblioteca `requests`) em vez dos SDKs para manter poucas dependências.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
import wave
from datetime import date

import requests

from .. import config

USAGE_FILE = config.CACHE_DIR / "azure_usage.json"


# ---------------------------------------------------------------------------
# Cache e cota
# ---------------------------------------------------------------------------
def _cache_path(kind: str, payload: bytes):
    """Gera um nome de arquivo único a partir do conteúdo (hash SHA-256)."""
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return config.CACHE_DIR / f"{kind}_{digest}.json"


def _read_cache(kind: str, payload: bytes):
    p = _cache_path(kind, payload)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _write_cache(kind: str, payload: bytes, result):
    _cache_path(kind, payload).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def _load_usage() -> dict:
    """Uso do mês atual. Ao virar o mês, o contador zera sozinho."""
    month = date.today().strftime("%Y-%m")
    if USAGE_FILE.exists():
        usage = json.loads(USAGE_FILE.read_text())
        if usage.get("month") == month:
            return usage
    return {"month": month, "speech_seconds": 0.0, "language_records": 0}


def _save_usage(usage: dict):
    USAGE_FILE.write_text(json.dumps(usage, indent=2))


def usage_summary() -> dict:
    u = _load_usage()
    u["speech_budget_seconds"] = config.SPEECH_SECONDS_BUDGET
    u["language_budget_records"] = config.LANGUAGE_RECORDS_BUDGET
    return u


def speech_enabled() -> bool:
    return bool(config.AZURE_SPEECH_KEY and config.AZURE_SPEECH_REGION)


def language_enabled() -> bool:
    return bool(config.AZURE_LANGUAGE_KEY and config.AZURE_LANGUAGE_ENDPOINT)


# ---------------------------------------------------------------------------
# Speech to Text
# ---------------------------------------------------------------------------
def to_wav16k_bytes(samples, sample_rate: int) -> bytes:
    """Converte um array float (-1..1) em WAV PCM 16 bits mono 16 kHz (formato aceito pelo Azure)."""
    import numpy as np
    from scipy.signal import resample_poly

    y = np.asarray(samples, dtype=np.float32)
    if sample_rate != 16000:
        g = math.gcd(sample_rate, 16000)
        y = resample_poly(y, 16000 // g, sample_rate // g).astype(np.float32)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def speech_to_text(wav_bytes: bytes, language: str = config.LANGUAGE) -> dict | None:
    """
    Transcreve um WAV de até 60 s com a API REST de áudio curto.
    Retorna {"text": ..., "confidence": ..., "source": "azure"|"cache"} ou None.
    """
    cached = _read_cache("stt", wav_bytes + language.encode())
    if cached is not None:
        cached["source"] = "cache"
        return cached
    if not speech_enabled():
        return None

    # duração = (bytes de áudio) / (16000 amostras/s * 2 bytes)
    seconds = max(0.0, (len(wav_bytes) - 44) / 32000)
    if seconds > 60:
        raise ValueError("Áudio maior que 60 s: divida antes (ver audio.consultations.split_wav).")
    usage = _load_usage()
    if usage["speech_seconds"] + seconds > config.SPEECH_SECONDS_BUDGET:
        print("[azure] Cota de Speech do mês atingida (limite de segurança). Usando fallback local.")
        return None

    url = (f"https://{config.AZURE_SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/"
           f"conversation/cognitiveservices/v1?language={language}&format=detailed")
    headers = {
        "Ocp-Apim-Subscription-Key": config.AZURE_SPEECH_KEY,
        "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        "Accept": "application/json",
    }
    resp = requests.post(url, headers=headers, data=wav_bytes, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Contabiliza o uso mesmo se não reconheceu nada (o Azure cobra o áudio enviado)
    usage["speech_seconds"] += seconds
    _save_usage(usage)

    best = (data.get("NBest") or [{}])[0]
    result = {
        "text": data.get("DisplayText", ""),
        "status": data.get("RecognitionStatus"),
        "confidence": best.get("Confidence"),
    }
    _write_cache("stt", wav_bytes + language.encode(), result)
    result["source"] = "azure"
    return result


# ---------------------------------------------------------------------------
# Language: sentimento + Text Analytics for health
# ---------------------------------------------------------------------------
def _records(text: str) -> int:
    """1 registro de texto = até 1 000 caracteres."""
    return max(1, math.ceil(len(text) / 1000))


def _check_language_budget(n_records: int) -> bool:
    usage = _load_usage()
    if usage["language_records"] + n_records > config.LANGUAGE_RECORDS_BUDGET:
        print("[azure] Cota de Language do mês atingida (limite de segurança). Usando fallback local.")
        return False
    return True


def _add_language_usage(n_records: int):
    usage = _load_usage()
    usage["language_records"] += n_records
    _save_usage(usage)


def analyze_sentiment(text: str, language: str = "en") -> dict | None:
    """Sentimento do documento e de cada frase (positivo/neutro/negativo)."""
    key = f"sent|{language}|{text}".encode()
    cached = _read_cache("sentiment", key)
    if cached is not None:
        return cached
    if not language_enabled() or not _check_language_budget(_records(text)):
        return None

    url = f"{config.AZURE_LANGUAGE_ENDPOINT}/language/:analyze-text?api-version=2023-04-01"
    body = {
        "kind": "SentimentAnalysis",
        "analysisInput": {"documents": [{"id": "1", "language": language, "text": text}]},
        "parameters": {"opinionMining": False},
    }
    resp = requests.post(url, json=body, timeout=60,
                         headers={"Ocp-Apim-Subscription-Key": config.AZURE_LANGUAGE_KEY})
    resp.raise_for_status()
    _add_language_usage(_records(text))
    doc = resp.json()["results"]["documents"][0]
    result = {
        "sentiment": doc["sentiment"],
        "scores": doc["confidenceScores"],
        "sentences": [{"text": s["text"], "sentiment": s["sentiment"]} for s in doc["sentences"]],
    }
    _write_cache("sentiment", key, result)
    return result


def analyze_health(text: str, language: str = "en", poll_seconds: float = 2.0) -> dict | None:
    """
    Text Analytics for health: extrai entidades clínicas (sintomas, remédios,
    doses...) e a "assertion" (ex.: negado, hipotético).
    É uma operação assíncrona: enviamos o job e consultamos até terminar.
    """
    key = f"health|{language}|{text}".encode()
    cached = _read_cache("health", key)
    if cached is not None:
        return cached
    if not language_enabled() or not _check_language_budget(_records(text)):
        return None

    headers = {"Ocp-Apim-Subscription-Key": config.AZURE_LANGUAGE_KEY}
    url = f"{config.AZURE_LANGUAGE_ENDPOINT}/language/analyze-text/jobs?api-version=2023-04-01"
    body = {
        "analysisInput": {"documents": [{"id": "1", "language": language, "text": text}]},
        "tasks": [{"kind": "Healthcare", "taskName": "health"}],
    }
    resp = requests.post(url, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    _add_language_usage(_records(text))
    job_url = resp.headers["operation-location"]

    for _ in range(60):  # até ~2 minutos
        job = requests.get(job_url, headers=headers, timeout=60).json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(poll_seconds)
    if job["status"] != "succeeded":
        print(f"[azure] Job de saúde terminou com status {job['status']}")
        return None

    doc = job["tasks"]["items"][0]["results"]["documents"][0]
    result = {
        "entities": [
            {
                "text": e["text"],
                "category": e["category"],
                "confidence": e["confidenceScore"],
                "assertion": e.get("assertion", {}),
            }
            for e in doc["entities"]
        ]
    }
    _write_cache("health", key, result)
    return result

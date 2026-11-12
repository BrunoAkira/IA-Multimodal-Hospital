"""
Áudios de consultas médicas -> Azure Speech to Text -> Azure Text Analytics -> eventos.

Pasta data/consultations/:
    consult_01.txt   roteiro em inglês (Doctor: ... / Patient: ...)
    consult_01.wav   gravação desse roteiro (voz do grupo ou scripts/synthesize_consultations.py)

Se não houver .wav ou chave do Azure, usamos o próprio roteiro como "transcrição"
(fica marcado como transcript_source="script") para o resto do pipeline rodar.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
from scipy.io import wavfile

from .. import config
from ..core import azure_client
from ..core.events import Event
from .features import extract_features
from .text_analysis import analyze_text


def split_audio(y: np.ndarray, sr: int, max_s: float = 50.0) -> list[np.ndarray]:
    """
    Divide em trechos de até `max_s` segundos (a API de áudio curto aceita até 60 s).
    Corta no ponto de MENOR energia entre 35 s e 50 s, para não cortar palavras.
    """
    chunks, start = [], 0
    hop = int(0.05 * sr)
    while len(y) - start > max_s * sr:
        a, b = start + int(35 * sr), start + int(max_s * sr)
        seg = y[a:b]
        energy = np.convolve(seg ** 2, np.ones(hop) / hop, mode="same")
        cut = a + int(np.argmin(energy))
        chunks.append(y[start:cut])
        start = cut
    chunks.append(y[start:])
    return chunks


def transcribe(y: np.ndarray, sr: int) -> tuple[str, str]:
    """Transcreve com o Azure (trecho a trecho). Retorna (texto, origem)."""
    texts, origin = [], "azure"
    for chunk in split_audio(y, sr):
        res = azure_client.speech_to_text(azure_client.to_wav16k_bytes(chunk, sr))
        if res is None:
            return "", "unavailable"
        origin = res.get("source", origin)
        texts.append(res["text"])
    return " ".join(t for t in texts if t), origin


def patient_lines(script: str) -> str:
    """Do roteiro, só as falas do paciente (é a voz dele que interessa para sentimento)."""
    lines = [l.split(":", 1)[1].strip() for l in script.splitlines() if l.lower().startswith("patient:")]
    return " ".join(lines) or script


def run_consultations() -> tuple[list[Event], list[dict]]:
    events, rows = [], []
    base = datetime.fromisoformat(config.RELATIVE_EPOCH)
    for i, txt in enumerate(sorted(config.CONSULT_DIR.glob("consult_*.txt"))):
        cid = txt.stem.replace("consult_", "")
        pid = f"consult:{cid}"
        ts = (base + timedelta(hours=10 + i)).isoformat()   # horário relativo (a fusão reposiciona)
        script = txt.read_text(encoding="utf-8")
        wav = txt.with_suffix(".wav")

        transcript, origin, voice = "", "script", {}
        if wav.exists():
            sr, y = wavfile.read(wav)
            y = (y / np.iinfo(y.dtype).max if np.issubdtype(y.dtype, np.integer) else y).astype(np.float32)
            if y.ndim > 1:
                y = y.mean(axis=1)
            voice = extract_features(y, sr)          # a voz da consulta também é analisada
            transcript, origin = transcribe(y, sr)
        if not transcript:
            transcript, origin = patient_lines(script), "script"

        analysis = analyze_text(transcript)
        active = [t for t in analysis["terms"] if not t["negated"]]
        for t in active:
            if t["weight"] >= 0.45:
                events.append(Event(pid, ts, "text", "consultation_nlp", "critical_term_mentioned",
                                    t["weight"], message=f"Patient reports: '{t['text']}'",
                                    evidence={"category": t["category"], "source": t["source"]}))
        s = analysis["sentiment"]
        if s["sentiment"] == "negative":
            events.append(Event(pid, ts, "text", "consultation_nlp", "negative_sentiment",
                                0.3 + 0.4 * float(s["scores"].get("negative", 0.5)),
                                message="Negative sentiment in patient speech",
                                evidence={"source": s.get("source")}))
        rows.append({"consultation_id": cid, "transcript_source": origin, "transcript": transcript,
                     "sentiment": s["sentiment"], "sentiment_source": s.get("source"),
                     "critical_terms": [t["text"] for t in active],
                     "negated_terms": [t["text"] for t in analysis["terms"] if t["negated"]],
                     **{f"voice_{k}": v for k, v in voice.items() if not k.startswith("mfcc")}})
    return events, rows

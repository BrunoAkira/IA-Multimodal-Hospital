"""
Leitura da base de fala (Parquet do Hugging Face) em lotes pequenos e padronização em inglês.

Saída padronizada de cada fala:
    utterance_id, speaker_id, group ("dysarthric" | "control"), gender, prompt, audio, sr

O Parquet guarda o áudio como bytes de WAV. Lemos poucas linhas por vez
(config.PARQUET_BATCH_ROWS), então a memória fica baixa mesmo com arquivos de 400 MB.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np

from .. import config

# Nomes possíveis de cada coluna (a estrutura muda entre versões do dataset)
CANDIDATES = {
    "audio": ["audio", "speech", "wav"],
    "prompt": ["transcription", "text", "sentence", "prompt", "transcript"],
    "status": ["speech_status", "status", "label", "condition", "dysarthria"],
    "speaker": ["speaker_id", "speaker", "spk_id", "client_id"],
    "gender": ["gender", "sex"],
}


def _pick(columns: list[str], key: str) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in CANDIDATES[key]:
        if name in lower:
            return lower[name]
    return None


def decode_audio(value) -> tuple[np.ndarray, int]:
    """Converte {'bytes': ..., 'path': ...} (formato Hugging Face) em (array float, taxa)."""
    raw = value.get("bytes") if isinstance(value, dict) else value
    if raw is None and isinstance(value, dict) and value.get("array") is not None:
        return np.asarray(value["array"], dtype=np.float32), int(value["sampling_rate"])
    from scipy.io import wavfile          # a base de fala guarda o áudio em WAV
    sr, y = wavfile.read(io.BytesIO(raw))
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        y = y / float(np.iinfo(y.dtype).max)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def _speaker_from_path(path: str) -> str | None:
    """Na base de fala o id do falante aparece no caminho (F01, M03, FC02, MC01...)."""
    m = re.search(r"\b([FM]C?\d{2})\b", str(path).replace("/", " ").replace("_", " "))
    return m.group(1) if m else None


def _group(status, speaker: str | None) -> str:
    s = str(status).lower() if status is not None else ""
    if any(k in s for k in ("dysarth", "patient", "1", "true")) and "non" not in s:
        return "dysarthric"
    if any(k in s for k in ("healthy", "control", "normal", "0", "false", "non")):
        return "control"
    # sem rótulo: na base de fala, ids com "C" (FC01, MC02) são controles
    if speaker:
        return "control" if "C" in speaker[1:] else "dysarthric"
    return "unknown"


def iter_speech(path: Path | None = None, max_rows: int = config.MAX_SPEECH_ROWS):
    """Gera falas padronizadas uma a uma (generator: não carrega tudo na memória)."""
    import pyarrow.parquet as pq

    files = [path] if path else sorted(config.SPEECH_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum .parquet da base de fala em {config.SPEECH_DIR}")
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        cols = pf.schema_arrow.names
        c = {k: _pick(cols, k) for k in CANDIDATES}
        if c["audio"] is None:
            raise KeyError(f"Coluna de áudio não encontrada. Colunas: {cols}")
        for batch in pf.iter_batches(batch_size=config.PARQUET_BATCH_ROWS,
                                     columns=[v for v in c.values() if v]):
            for row in batch.to_pylist():
                audio_val = row[c["audio"]]
                path_hint = audio_val.get("path", "") if isinstance(audio_val, dict) else ""
                spk = row.get(c["speaker"]) if c["speaker"] else _speaker_from_path(path_hint)
                try:
                    y, sr = decode_audio(audio_val)
                except Exception as exc:
                    print(f"[audio] fala ignorada ({exc})")
                    continue
                yield {
                    "utterance_id": f"speech_{n:05d}",
                    "speaker_id": str(spk or "unknown"),
                    "group": _group(row.get(c["status"]) if c["status"] else None, spk),
                    "gender": str(row.get(c["gender"], "")).lower() if c["gender"] else "",
                    "prompt": str(row.get(c["prompt"], "")) if c["prompt"] else "",
                    "audio": y,
                    "sr": sr,
                }
                n += 1
                if n >= max_rows:
                    return

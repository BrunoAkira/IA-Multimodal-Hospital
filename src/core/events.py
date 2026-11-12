"""
Formato único de evento.

Cada módulo (clínico, áudio, vídeo, texto) só precisa produzir objetos `Event`.
A fusão e o dashboard leem apenas esse formato, então os módulos ficam
independentes uns dos outros.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..config import EVENTS_FILE

# Escala de gravidade usada em todo o projeto (em inglês)
SEVERITY_LEVELS = ["info", "low", "medium", "high"]


def severity_from_score(score: float) -> str:
    """Converte uma pontuação 0..1 em nível de gravidade."""
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.25:
        return "low"
    return "info"


@dataclass
class Event:
    patient_id: str          # id de origem (ex.: "clinical:10000032", "speech:F01", "mocap:S01")
    timestamp: str           # ISO 8601; para áudio/vídeo sem data usamos a data da sessão
    modality: str            # "clinical" | "audio" | "video" | "text"
    source: str              # sub-módulo (ex.: "vitals", "prescriptions", "speech_features")
    event_type: str          # ex.: "spo2_drop", "dose_jump", "dysarthria_risk"
    score: float             # 0..1 (quanto mais alto, mais anômalo)
    severity: str = ""       # preenchido automaticamente a partir do score
    message: str = ""        # explicação curta em inglês para a equipe
    evidence: dict = field(default_factory=dict)  # valores que justificam o evento

    def __post_init__(self):
        self.score = float(max(0.0, min(1.0, self.score)))
        if not self.severity:
            self.severity = severity_from_score(self.score)


def save_events(events: list[Event], path: Path = EVENTS_FILE, append: bool = True) -> Path:
    """Grava eventos em JSON Lines (uma linha por evento: fácil de anexar e ler em pedaços)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev), ensure_ascii=False, default=str) + "\n")
    return path


def load_events(path: Path = EVENTS_FILE) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clear_events(modality: str | None = None, path: Path = EVENTS_FILE):
    """Remove eventos antigos de uma modalidade (para reprocessar sem duplicar)."""
    if not path.exists():
        return
    if modality is None:
        path.unlink()
        return
    keep = [e for e in load_events(path) if e["modality"] != modality]
    with path.open("w", encoding="utf-8") as f:
        for e in keep:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

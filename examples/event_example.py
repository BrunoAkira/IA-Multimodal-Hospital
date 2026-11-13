"""
Exemplo: criar um evento no formato único do sistema e ler os eventos já gravados.

    python examples/event_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # permite importar src/ a partir de examples/

from src.core.events import Event, load_events

# um evento de saturação baixa; a gravidade é calculada a partir da pontuação
ev = Event(patient_id="clinical:10000032", timestamp="2150-01-01T10:00:00",
           modality="clinical", source="vitals", event_type="spo2_low",
           score=0.85, message="spo2 = 86.0", evidence={"spo2": 86.0})
print("Evento criado:", ev.event_type, "| pontuação", ev.score, "| gravidade", ev.severity)

events = load_events()
print(f"{len(events)} eventos gravados em results/events.jsonl")

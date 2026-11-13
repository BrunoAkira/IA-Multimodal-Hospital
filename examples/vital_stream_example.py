"""
Exemplo: detector de sinais vitais em tempo real, recebendo uma leitura por vez.

    python examples/vital_stream_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clinical.vitals import VitalStreamDetector

detector = VitalStreamDetector()
# leituras de SpO2 de um leito; a última cai de repente
for spo2 in [97, 96, 97, 98, 97, 96, 97, 88]:
    score, event_type, evidence = detector.update("bed_1", "spo2", spo2)
    print(f"spo2 {spo2:>3} -> pontuação {score:.2f} {event_type or 'normal'} {evidence}")

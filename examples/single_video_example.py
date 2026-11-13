"""
Exemplo: analisar um único vídeo (pose, ângulos, repetições e áreas críticas).

    python examples/single_video_example.py caminho/do/squat_video.mp4

O prefixo do nome do arquivo define o exercício (squat_, arm_raise_...; ver README, seção 4.5).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video.vision import analyze_video

if len(sys.argv) < 2:
    raise SystemExit("Uso: python examples/single_video_example.py caminho/do/video.mp4")

result = analyze_video(Path(sys.argv[1]))
print("Resumo:")
for key, value in result["summary"].items():
    print(f"  {key:24} {value}")
print("\nAchados:")
for f in result["findings"] or []:
    print(f"  {f['event_type']:28} {f['score']:.2f}  {f['message']}")
if not result["findings"]:
    print("  nenhum desvio encontrado")

"""
Exemplo: listar e ler os arquivos de captura de movimento (.amc).

    python examples/motion_capture_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video.amc import index_motion_capture, parse_amc

index = index_motion_capture()
if index.empty:
    raise SystemExit("Nenhum .amc encontrado. Confira a pasta com: python main.py inspect")
print(index.drop(columns=["path"]).head(10).to_string(index=False))

angles = parse_amc(index.iloc[0]["path"])
print(f"\nPrimeiro arquivo: {angles.shape[0]} quadros x {angles.shape[1]} ângulos")
print("Colunas:", ", ".join(list(angles.columns)[:8]), "...")

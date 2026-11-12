"""
Baixa a base de fala usada como exemplo (dois arquivos Parquet, ~370 MB cada) para <pasta>/speech/.

A base de fala é de uso acadêmico e não é redistribuída no repositório; este script a obtém
da cópia pública no Hugging Face. Destino padrão: example_data/speech/
(ou a pasta indicada em HOSPITAL_AI_DATASETS).
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

BASE = "https://huggingface.co/datasets/abnerh/TORGO-database/resolve/main/data/"
# dois arquivos da base: o primeiro tem falantes do grupo de controle, o terceiro tem falantes com disartria
FILES = ["train-00000-of-00004.parquet", "train-00002-of-00004.parquet"]


def progress(blocks, block_size, total):
    if total > 0:
        done = min(1.0, blocks * block_size / total)
        print(f"\r  {done:6.1%}  ({blocks * block_size / 1e6:,.0f} de {total / 1e6:,.0f} MB)", end="")


if __name__ == "__main__":
    # sem HOSPITAL_AI_DATASETS definido, baixa para example_data/speech/
    dest_dir = config.SPEECH_DIR if os.getenv("HOSPITAL_AI_DATASETS") else ROOT / "example_data" / "speech"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Já existe: {dest}")
            continue
        print(f"Baixando {name} para {dest_dir} ...")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(BASE + name, tmp, progress)
        tmp.replace(dest)
        print(f"\nPronto: {dest}")

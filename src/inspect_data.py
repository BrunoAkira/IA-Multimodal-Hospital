"""Inspeção: mostra o que foi encontrado em cada pasta de dados e o estado das chaves do Azure."""
from __future__ import annotations

import pandas as pd

from . import config
from .core import azure_client


def inspect_all():
    print(f"\nPasta de datasets: {config.DATASETS}")

    # --- MIMIC ---
    print("\n=== MIMIC-IV Demo (clinical) ===")
    for name in ("chartevents", "prescriptions", "transfers"):
        hits = list(config.CLINICAL_DIR.rglob(f"{name}.csv*"))
        if hits:
            cols = list(pd.read_csv(hits[0], nrows=2).columns)
            print(f"  OK  {hits[0].relative_to(config.CLINICAL_DIR)}  colunas: {cols[:8]}...")
        else:
            print(f"  --  {name}.csv.gz não encontrado")

    # --- TORGO ---
    print("\n=== TORGO (audio) ===")
    files = sorted(config.AUDIO_DIR.glob("*.parquet"))
    if not files:
        print("  --  nenhum .parquet")
    for f in files:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(f)
            print(f"  OK  {f.name}: {pf.metadata.num_rows} linhas, colunas {pf.schema_arrow.names}")
        except ImportError:
            print("  pyarrow não instalado (pip install pyarrow)")

    # --- UL-RED ---
    print("\n=== UL-RED (video/motion) ===")
    from .video.amc import index_ulred
    idx = index_ulred()
    if idx.empty:
        print("  --  nenhum .amc")
    else:
        print(f"  OK  {len(idx)} arquivos .amc | sujeitos: {sorted(idx['subject'].unique())}")
        print(f"      sistemas: {idx['system'].value_counts().to_dict()} | movimentos: {idx['motion'].nunique()}")

    # --- vídeos e consultas ---
    print("\n=== Vídeos de fisioterapia (.mp4/.avi/.mov/.mkv) ===")
    vids = [f for f in config.PHYSIO_VIDEO_DIR.iterdir()
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    print(f"  {len(vids)} vídeo(s) em {config.PHYSIO_VIDEO_DIR}")
    print("\n=== Consultas ===")
    txt, wav = list(config.CONSULT_DIR.glob("consult_*.txt")), list(config.CONSULT_DIR.glob("consult_*.wav"))
    print(f"  {len(txt)} roteiros, {len(wav)} gravações .wav")

    # --- Azure ---
    print("\n=== Azure (plano gratuito) ===")
    print(f"  Speech configurado:   {azure_client.speech_enabled()}")
    print(f"  Language configurado: {azure_client.language_enabled()}")
    print(f"  Uso no mês: {azure_client.usage_summary()}")

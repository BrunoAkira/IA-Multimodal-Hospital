"""Inspeção: mostra o que foi encontrado em cada pasta de dados e o estado das chaves do Azure."""
from __future__ import annotations

import pandas as pd

from . import config
from .core import azure_client


def inspect_all():
    print(f"\nPasta de dados: {config.DATASETS}")

    # --- base clínica ---
    print(f"\n=== Base clínica de UTI ({config.CLINICAL_DIR.name}/) ===")
    for name in ("chartevents", "prescriptions", "transfers"):
        hits = list(config.CLINICAL_DIR.rglob(f"{name}.csv*"))
        if hits:
            cols = list(pd.read_csv(hits[0], nrows=2).columns)
            print(f"  OK  {hits[0].relative_to(config.CLINICAL_DIR)}  colunas: {cols[:8]}...")
        else:
            print(f"  --  {name}.csv.gz não encontrado")

    # --- base de fala ---
    print(f"\n=== Base de fala ({config.SPEECH_DIR.name}/) ===")
    files = sorted(config.SPEECH_DIR.glob("*.parquet"))
    if not files:
        print("  --  nenhum .parquet (para baixar a base de fala: python scripts/download_speech_data.py)")
    for f in files:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(f)
            print(f"  OK  {f.name}: {pf.metadata.num_rows} linhas, colunas {pf.schema_arrow.names}")
        except ImportError:
            print("  pyarrow não instalado (pip install pyarrow)")

    # --- captura de movimento ---
    print(f"\n=== Captura de movimento ({config.MOCAP_DIR.name}/) ===")
    from .video.amc import index_motion_capture
    idx = index_motion_capture()
    if idx.empty:
        print("  --  nenhum .amc")
    else:
        print(f"  OK  {len(idx)} arquivos .amc | sujeitos: {sorted(idx['subject'].unique())}")
        print(f"      sistemas: {idx['system'].value_counts().to_dict()} | movimentos: {idx['motion'].nunique()}")

    # --- vídeos e consultas ---
    print("\n=== Vídeos de fisioterapia (.mp4/.avi/.mov/.mkv) ===")
    exts = (".mp4", ".avi", ".mov", ".mkv")
    d = config.PHYSIO_VIDEO_DIR
    vids = [f.name for f in d.iterdir() if f.suffix.lower() in exts] if d.exists() else []
    print(f"  {len(vids)} vídeo(s) em {d}" + (f": {vids}" if vids else ""))

    print("\n=== Vídeos de exercícios com gabarito (exercise_videos/) ===")
    ev = config.EXERCISE_VIDEO_DIR
    seg = next(ev.rglob("Segmentation.csv"), None) if ev.exists() else None
    n_vid = len(list(ev.rglob("*.mp4"))) if ev.exists() else 0
    n_ang = len(list((ev / "angles").glob("*.csv"))) if (ev / "angles").exists() else 0
    print(f"  gabarito: {'OK' if seg else '--'} | vídeos: {n_vid} | ângulos pré-calculados: {n_ang}")

    print("\n=== Consultas ===")
    txt, wav = list(config.CONSULT_DIR.glob("consult_*.txt")), list(config.CONSULT_DIR.glob("consult_*.wav"))
    print(f"  {len(txt)} roteiros, {len(wav)} gravações .wav")

    # --- Azure ---
    print("\n=== Azure (plano gratuito) ===")
    print(f"  Speech configurado:   {azure_client.speech_enabled()}")
    print(f"  Language configurado: {azure_client.language_enabled()}")
    print(f"  Uso no mês: {azure_client.usage_summary()}")

"""
Leitor de arquivos .amc (Acclaim Motion Capture), o formato da captura de movimento.

Um .amc é texto simples:
    :FULLY-SPECIFIED
    :DEGREES
    1                      <- número do quadro
    root 0 900 0 0 0 0     <- articulação seguida dos seus valores (graus de liberdade)
    lfemur -12.3 0.5 2.0
    ...
    2
    ...
Cada linha "articulação valores" vira colunas como lfemur_0, lfemur_1, lfemur_2.
O nome do arquivo segue {Movimento}R{Repetições}S{Sujeito}.amc (ex.: MiniSquatR3S01.amc).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

NAME_RE = re.compile(r"^(?P<motion>[A-Za-z]+?)R(?P<reps>\d+)S(?P<subject>\d+)$")


def parse_amc(path: Path) -> pd.DataFrame:
    """Retorna DataFrame (linhas = quadros, colunas = articulação_grau)."""
    frames, cur = [], None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith((":", "#")):
                continue
            parts = line.split()
            if len(parts) == 1 and parts[0].isdigit():      # início de um novo quadro
                cur = {}
                frames.append(cur)
            elif cur is not None:
                joint = parts[0]
                for i, v in enumerate(parts[1:]):
                    cur[f"{joint}_{i}"] = float(v)
    df = pd.DataFrame(frames)
    # a raiz (root) tem posição + orientação global; para analisar o movimento das
    # articulações usamos só as rotações (exceto a altura do quadril, root_1, que ajuda no agachamento)
    drop = [c for c in df.columns if c.startswith("root_") and c != "root_1"]
    return df.drop(columns=drop).astype(np.float32)


def parse_name(path: Path) -> dict | None:
    m = NAME_RE.match(path.stem)
    if not m:
        return None
    return {"motion": m["motion"], "reps": int(m["reps"]), "subject": f"S{int(m['subject']):02d}"}


def index_motion_capture(folder: Path = config.MOCAP_DIR) -> pd.DataFrame:
    """
    Lista todos os .amc com metadados: sujeito, movimento, nº de repetições e
    sistema de captura (pasta 'Marker' ou 'Markerless'). Preferimos os dados 'Clean'.
    """
    rows = []
    for p in folder.rglob("*.amc"):
        meta = parse_name(p)
        if meta is None:
            continue
        parts = [x.lower() for x in p.parts]
        system = "markerless" if any("markerless" in x or "marker-less" in x for x in parts) else "marker"
        rows.append({**meta, "system": system, "clean": "clean" in parts, "path": str(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # se existir versão "Clean", descartamos a "Raw" do mesmo arquivo
    df = df.sort_values("clean", ascending=False).drop_duplicates(["subject", "motion", "reps", "system"])
    return df.sort_values(["subject", "motion", "reps"]).reset_index(drop=True)

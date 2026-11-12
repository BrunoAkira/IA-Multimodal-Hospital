"""
Leitura do MIMIC-IV Demo e padronização em inglês.

O MIMIC já usa nomes em inglês, mas aqui reduzimos cada tabela às colunas
que o projeto usa e damos nomes claros aos sinais vitais (heart_rate, spo2...).
Tudo é lido em pedaços (chunks) para economizar memória.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config


def find_table(name: str, folder: Path = config.CLINICAL_DIR) -> Path:
    """Procura 'name.csv' ou 'name.csv.gz' em qualquer subpasta do MIMIC."""
    for pattern in (f"{name}.csv.gz", f"{name}.csv"):
        hits = sorted(folder.rglob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Tabela '{name}' não encontrada em {folder}. Extraia o MIMIC-IV Demo em datasets/clinical/."
    )


# ---------------------------------------------------------------------------
# Sinais vitais
# ---------------------------------------------------------------------------
def load_vitals() -> pd.DataFrame:
    """
    Lê icu/chartevents só com os itemids de sinais vitais.
    Saída (formato "longo"): stay_id, subject_id, hadm_id, charttime, vital, value
    """
    # mapa itemid -> nome do vital (ex.: 220045 -> "heart_rate")
    item_to_vital = {i: v for v, ids in config.VITAL_ITEMIDS.items() for i in ids}
    cols = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"]

    parts = []
    reader = pd.read_csv(find_table("chartevents"), usecols=cols, chunksize=config.CSV_CHUNK_ROWS)
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(item_to_vital)]   # descarta o que não é vital
        parts.append(chunk.dropna(subset=["valuenum"]))
    df = pd.concat(parts, ignore_index=True)

    df["vital"] = df["itemid"].map(item_to_vital)
    df["value"] = df["valuenum"].astype(float)

    # Fahrenheit -> Celsius, e unificamos o nome
    is_f = df["vital"] == "temperature_f"
    df.loc[is_f, "value"] = (df.loc[is_f, "value"] - 32) * 5 / 9
    df.loc[is_f, "vital"] = "temperature_c"

    # Remove valores impossíveis (erro de digitação/sensor)
    keep = np.ones(len(df), dtype=bool)
    for vital, (lo, hi) in config.VITAL_PLAUSIBLE.items():
        m = df["vital"] == vital
        keep &= ~(m & ((df["value"] < lo) | (df["value"] > hi)))
    df = df[keep]

    df["charttime"] = pd.to_datetime(df["charttime"])
    return df[["stay_id", "subject_id", "hadm_id", "charttime", "vital", "value"]].sort_values(
        ["stay_id", "charttime"]).reset_index(drop=True)


def hourly_vitals(vitals_long: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega para 1 linha por internação na UTI por hora (mediana de cada vital).
    Isso reduz ruído e volume: é sobre essa série que os detectores rodam.
    """
    v = vitals_long.copy()
    v["hour"] = v["charttime"].dt.floor("h")
    wide = (v.pivot_table(index=["stay_id", "subject_id", "hour"], columns="vital",
                          values="value", aggfunc="median")
              .reset_index()
              .rename(columns={"hour": "charttime"}))
    wide.columns.name = None
    return wide.sort_values(["stay_id", "charttime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prescrições
# ---------------------------------------------------------------------------
def _first_number(x) -> float:
    """dose_val_rx pode vir como '1-2' ou '0.5'. Pegamos o primeiro número."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").split("-")[0].strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_prescriptions() -> pd.DataFrame:
    cols = ["subject_id", "hadm_id", "starttime", "stoptime", "drug",
            "dose_val_rx", "dose_unit_rx", "route", "doses_per_24_hrs"]
    df = pd.read_csv(find_table("prescriptions"), usecols=lambda c: c in cols, low_memory=False)
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    df["stoptime"] = pd.to_datetime(df["stoptime"], errors="coerce")
    df["drug"] = df["drug"].astype(str).str.strip().str.lower()       # nome padronizado
    df["route"] = df["route"].astype(str).str.strip().str.upper()
    df["dose"] = df["dose_val_rx"].map(_first_number)
    # dose diária = dose x vezes por dia (quando o MIMIC informa a frequência)
    freq = pd.to_numeric(df.get("doses_per_24_hrs"), errors="coerce")
    df["daily_dose"] = df["dose"] * freq.fillna(1)
    df = df.dropna(subset=["starttime"])
    return df.sort_values(["hadm_id", "starttime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transferências (movimentação do paciente entre setores)
# ---------------------------------------------------------------------------
def load_transfers() -> pd.DataFrame:
    df = pd.read_csv(find_table("transfers"))
    df = df[df["eventtype"].str.lower() != "discharge"].dropna(subset=["careunit"])
    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    return df.sort_values(["hadm_id", "intime"]).reset_index(drop=True)

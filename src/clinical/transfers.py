"""
Anomalias nos padrões de movimentação do paciente durante a internação.

Usamos a tabela `transfers` (sequência de setores: emergência -> enfermaria -> UTI ...).
  * Cadeia de Markov: aprendemos P(próximo setor | setor atual) com todas as
    internações. Transições muito improváveis viram "unusual_transfer".
  * icu_readmission_48h: saiu da UTI e voltou em até 48 h (sinal clássico de piora).
  * frequent_transfers: 3+ mudanças de setor em 24 h.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ..core.events import Event


def _is_icu(unit: str) -> bool:
    u = str(unit).lower()
    return "icu" in u or "intensive care" in u


def transition_probabilities(tr: pd.DataFrame, alpha: float = 1.0) -> dict:
    """P(destino | origem) com suavização de Laplace (alpha) para não dar zero."""
    counts = defaultdict(Counter)
    units = sorted(tr["careunit"].unique())
    for _, seq in tr.groupby("hadm_id")["careunit"]:
        seq = list(seq)
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a][b] += 1
    probs = {}
    for a in units:
        total = sum(counts[a].values()) + alpha * len(units)
        probs[a] = {b: (counts[a][b] + alpha) / total for b in units}
    return probs


def detect_transfer_events(tr: pd.DataFrame, p_threshold: float = 0.03) -> tuple[list[Event], dict]:
    probs = transition_probabilities(tr)
    events = []
    for hadm, g in tr.groupby("hadm_id"):
        g = g.sort_values("intime").to_dict("records")
        pid = f"mimic:{g[0]['subject_id']}"
        for prev, cur in zip(g[:-1], g[1:]):
            ts = pd.Timestamp(cur["intime"]).isoformat()
            p = probs[prev["careunit"]][cur["careunit"]]
            ev = {"from": prev["careunit"], "to": cur["careunit"], "probability": round(p, 4)}
            if p < p_threshold:
                events.append(Event(pid, ts, "clinical", "transfers", "unusual_transfer",
                                    0.5 + 0.4 * (1 - p / p_threshold),
                                    message=f"Rare transfer {prev['careunit']} -> {cur['careunit']}",
                                    evidence=ev))
        # volta para a UTI em até 48 h
        last_icu_out = None
        for t in g:
            if _is_icu(t["careunit"]):
                if last_icu_out is not None and pd.notna(last_icu_out):
                    gap_h = (t["intime"] - last_icu_out).total_seconds() / 3600
                    if 0 < gap_h <= 48:
                        events.append(Event(pid, pd.Timestamp(t["intime"]).isoformat(), "clinical",
                                            "transfers", "icu_readmission_48h", 0.85,
                                            message=f"Returned to ICU {gap_h:.0f} h after leaving",
                                            evidence={"gap_hours": round(gap_h, 1), "unit": t["careunit"]}))
                last_icu_out = t["outtime"]
        # muitas trocas de setor em 24 h
        times = np.array([t["intime"] for t in g], dtype="datetime64[ns]")
        for i in range(len(times)):
            n = int(((times > times[i] - np.timedelta64(24, "h")) & (times <= times[i])).sum())
            if n >= 4:  # 4 entradas = 3 mudanças
                events.append(Event(pid, pd.Timestamp(times[i]).isoformat(), "clinical", "transfers",
                                    "frequent_transfers", 0.5,
                                    message=f"{n - 1} unit changes in 24 h", evidence={"count_24h": n - 1}))
                break
    return events, probs

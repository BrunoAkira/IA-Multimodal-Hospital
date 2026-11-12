"""
Detecção de anomalias em séries temporais de sinais vitais.

Camadas de detecção:
  1. Limites clínicos fixos (ex.: SpO2 < 90%)        -> regra
  2. NEWS2 (escore de alerta precoce usado em hospitais) -> regra clínica validada
  3. Detector em fluxo (streaming) por paciente:
       z-score robusto = (valor - mediana recente) / MAD recente
       Pega mudanças bruscas em relação ao "normal" daquele paciente,
       mesmo dentro dos limites clínicos.
  4. IsolationForest multivariado: combinações de vitais incomuns
       (ex.: FC alta + PA baixa ao mesmo tempo).

O detector em fluxo processa UM ponto por vez (método `update`),
que é o que permite simular monitoramento em tempo real (ver `replay`).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .. import config
from ..core.events import Event

VITALS = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "resp_rate", "temperature_c"]

# Variação mínima considerada "real" para cada vital (evita z enorme quando o paciente
# está muito estável e a MAD fica perto de zero)
MIN_SPREAD = {"heart_rate": 4, "systolic_bp": 6, "diastolic_bp": 4, "spo2": 1.0,
              "resp_rate": 2, "temperature_c": 0.3}


# ---------------------------------------------------------------------------
# NEWS2 (National Early Warning Score 2) — versão sem consciência e O2 suplementar
# ---------------------------------------------------------------------------
def news2_score(row: pd.Series) -> float:
    s = 0
    rr, spo2, sbp, hr, t = (row.get(k) for k in
                            ("resp_rate", "spo2", "systolic_bp", "heart_rate", "temperature_c"))
    if pd.notna(rr):
        s += 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3
    if pd.notna(spo2):
        s += 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    if pd.notna(sbp):
        s += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3
    if pd.notna(hr):
        s += 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    if pd.notna(t):
        s += 3 if t <= 35 else 1 if t <= 36 else 0 if t <= 38 else 1 if t <= 39 else 2
    return float(s)


# ---------------------------------------------------------------------------
# Detector em fluxo (um ponto por vez)
# ---------------------------------------------------------------------------
class VitalStreamDetector:
    """Guarda as últimas N medições de cada (paciente, vital) e avalia a nova."""

    def __init__(self, window: int = config.ROLLING_WINDOW, z_thr: float = config.ROBUST_Z_THRESHOLD):
        self.window = window
        self.z_thr = z_thr
        self.history = defaultdict(lambda: deque(maxlen=window))

    def update(self, stay_id, vital: str, value: float) -> tuple[float, str, dict]:
        """Retorna (score 0..1, tipo do evento, evidência). score 0 = normal."""
        hist = self.history[(stay_id, vital)]
        score, etype, evidence = 0.0, "", {"value": round(float(value), 2)}

        # 1) limites clínicos fixos
        lo, hi = config.VITAL_HARD_LIMITS[vital]
        span = hi - lo
        if value < lo:
            score, etype = 0.6 + 0.4 * min(1, (lo - value) / (0.2 * span)), f"{vital}_low"
        elif value > hi:
            score, etype = 0.6 + 0.4 * min(1, (value - hi) / (0.2 * span)), f"{vital}_high"

        # 2) mudança brusca em relação ao histórico do próprio paciente
        if len(hist) >= 6:  # precisa de um mínimo de histórico
            arr = np.fromiter(hist, dtype=float)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826       # MAD ~ desvio-padrão robusto
            spread = max(mad, MIN_SPREAD[vital])
            z = (value - med) / spread
            evidence.update(baseline_median=round(float(med), 2), robust_z=round(float(z), 2))
            if abs(z) >= self.z_thr:
                z_score = min(1.0, 0.5 + 0.5 * (abs(z) - self.z_thr) / self.z_thr)
                if z_score > score:
                    score, etype = z_score, f"{vital}_sudden_{'rise' if z > 0 else 'drop'}"

        hist.append(value)
        return score, etype, evidence


def detect_vital_events(hourly: pd.DataFrame, min_score: float = 0.5) -> list[Event]:
    """Roda o detector em fluxo + NEWS2 sobre toda a série horária."""
    det = VitalStreamDetector()
    events: list[Event] = []
    for row in hourly.itertuples(index=False):
        r = row._asdict()
        ts = pd.Timestamp(r["charttime"]).isoformat()
        pid = f"mimic:{r['subject_id']}"
        for vital in VITALS:
            v = r.get(vital)
            if v is None or pd.isna(v):
                continue
            score, etype, ev = det.update(r["stay_id"], vital, float(v))
            if score >= min_score:
                events.append(Event(pid, ts, "clinical", "vitals", etype, score,
                                    message=f"{vital.replace('_', ' ')} = {v:.1f}", evidence=ev))
        # NEWS2 da hora
        n2 = news2_score(pd.Series(r))
        if n2 >= 5:
            score = 0.9 if n2 >= 7 else 0.6
            events.append(Event(pid, ts, "clinical", "vitals", "news2_elevated", score,
                                message=f"NEWS2 = {n2:.0f} (early warning score)",
                                evidence={"news2": n2}))
    return events


# ---------------------------------------------------------------------------
# IsolationForest multivariado
# ---------------------------------------------------------------------------
def detect_multivariate(hourly: pd.DataFrame, contamination: float = 0.01) -> tuple[list[Event], pd.DataFrame]:
    """Procura horas em que a COMBINAÇÃO de vitais é rara na população."""
    cols = ["heart_rate", "systolic_bp", "spo2", "resp_rate"]
    df = hourly.copy()
    # preenche lacunas curtas com o último valor do mesmo paciente (até 3 h)
    df[cols] = df.groupby("stay_id")[cols].ffill(limit=3)
    data = df.dropna(subset=cols)
    if len(data) < 50:
        return [], df
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(data[cols])
    raw = -model.score_samples(data[cols])            # maior = mais anômalo
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    flags = model.predict(data[cols]) == -1
    df.loc[data.index, "if_score"] = norm

    events = []
    for idx in data.index[flags]:
        r = df.loc[idx]
        events.append(Event(f"mimic:{r['subject_id']}", pd.Timestamp(r["charttime"]).isoformat(),
                            "clinical", "vitals", "unusual_vital_pattern",
                            0.5 + 0.5 * float(r["if_score"]),
                            message="Rare combination of vital signs (IsolationForest)",
                            evidence={c: round(float(r[c]), 1) for c in cols}))
    return events, df


# ---------------------------------------------------------------------------
# Avaliação: injeta anomalias artificiais e mede se o detector as encontra
# (a base clínica não tem rótulo de anomalia)
# ---------------------------------------------------------------------------
INJECT_DELTA = {"heart_rate": 45, "systolic_bp": -45, "spo2": -9, "resp_rate": 14}


def evaluate_with_injection(hourly: pd.DataFrame, rate: float = 0.02, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for vital, delta in INJECT_DELTA.items():
        df = hourly[["stay_id", "charttime", vital]].dropna().copy()
        if len(df) < 50:
            continue
        injected = rng.random(len(df)) < rate
        df.loc[injected, vital] = df.loc[injected, vital] + delta
        det = VitalStreamDetector()
        pred = np.array([det.update(s, vital, v)[0] >= 0.5 for s, v in zip(df["stay_id"], df[vital])])
        tp = int((pred & injected).sum())
        fp = int((pred & ~injected).sum())
        fn = int((~pred & injected).sum())
        results[vital] = {
            "injected": int(injected.sum()),
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
            # os "falsos positivos" incluem anomalias reais da base, por isso a precisão é um limite inferior
        }
    return results


# ---------------------------------------------------------------------------
# Simulação de tempo real
# ---------------------------------------------------------------------------
def replay(hourly: pd.DataFrame, speed_seconds: float = 0.2, on_event=print, limit_hours: int | None = None):
    """
    Reproduz as medições em ordem cronológica, como se chegassem do monitor
    à beira do leito. Cada hora simulada leva `speed_seconds` segundos reais.
    """
    det = VitalStreamDetector()
    df = hourly.sort_values("charttime")
    for n, (hour, group) in enumerate(df.groupby("charttime")):
        if limit_hours and n >= limit_hours:
            break
        for r in group.to_dict("records"):
            for vital in VITALS:
                v = r.get(vital)
                if v is None or pd.isna(v):
                    continue
                score, etype, ev = det.update(r["stay_id"], vital, float(v))
                if score >= 0.5:
                    on_event(Event(f"mimic:{r['subject_id']}", pd.Timestamp(hour).isoformat(),
                                   "clinical", "vitals", etype, score,
                                   message=f"{vital.replace('_', ' ')} = {v:.1f}", evidence=ev))
        time.sleep(speed_seconds)

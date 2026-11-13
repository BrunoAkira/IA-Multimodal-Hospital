"""
Fusão multimodal e motor de alertas.

1) Pacientes virtuais
   Os 3 datasets são de pessoas DIFERENTES. Para demonstrar a fusão, cada
   "paciente virtual" (VP001, VP002...) junta: 1 paciente da base clínica + 1 falante da
   base de fala + 1 sujeito da captura de movimento + 1 consulta (+ 1 vídeo de fisioterapia, se houver).
   É uma simulação de integração para demonstração, não um caso clínico real.
   Os eventos de áudio/vídeo/texto (que não têm data) são posicionados no último
   dia da internação do paciente da base clínica.

2) Fusão (por paciente, janela móvel de 24 h)
   Em cada janela pegamos a maior pontuação de cada modalidade e combinamos:
        risco = 1 - Π (1 - peso_modalidade × pontuação_modalidade)
   ("OU probabilístico": vários sinais moderados juntos viram um risco alto)

3) Alertas
   medium (≥0.5), high (≥0.8 e confirmado por mais de um sinal), critical (≥0.9 com 2+ modalidades).
   O mesmo paciente não recebe alerta repetido antes de 12 h, a menos que piore.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config
from .core.events import load_events

LEVEL_ORDER = {"medium": 1, "high": 2, "critical": 3}
ACTIONS = {
    "medium": "Review at next round",
    "high": "Bedside assessment within 1 hour",
    "critical": "Immediate evaluation by the medical team",
}


def build_virtual_patients(events: pd.DataFrame) -> pd.DataFrame:
    def ids(prefix):
        s = events.loc[events["patient_id"].str.startswith(prefix), "patient_id"]
        return sorted(s.unique())

    # pacientes da base clínica ordenados pelo número de eventos
    clinical = (events[events["patient_id"].str.startswith("clinical:")]
             .groupby("patient_id").size().sort_values(ascending=False).index.tolist())
    others = {"speech": ids("speech:"), "motion_capture": ids("mocap:"), "consultation": ids("consult:"),
              "physio_video": ids("physio:"), "exercise_video": ids("exercise:")}
    n = min(len(clinical), max([len(v) for v in others.values()] + [0]))
    rows = []
    for i in range(n):
        row = {"virtual_id": f"VP{i + 1:03d}", "clinical": clinical[i]}
        for k, lst in others.items():
            row[k] = lst[i] if i < len(lst) else None
        rows.append(row)
    return pd.DataFrame(rows)


def apply_virtual_ids(events: pd.DataFrame, vp: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["source_patient_id"] = df["patient_id"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")  # aceita horários com e sem fração de segundo
    epoch = pd.Timestamp(config.RELATIVE_EPOCH)

    # último dia de internação de cada paciente da base clínica (âncora para áudio/vídeo)
    last_seen = df[df["modality"] == "clinical"].groupby("patient_id")["timestamp"].max()
    for r in vp.to_dict("records"):
        anchor = (last_seen.get(r["clinical"], epoch) - pd.Timedelta(hours=24)).normalize()
        for k in ("clinical", "speech", "motion_capture", "consultation", "physio_video", "exercise_video"):
            if not r.get(k):
                continue
            m = df["source_patient_id"] == r[k]
            df.loc[m, "patient_id"] = r["virtual_id"]
            if k != "clinical":   # reposiciona horários relativos
                df.loc[m, "timestamp"] = anchor + (df.loc[m, "timestamp"] - epoch)
    return df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def fuse_and_alert(df: pd.DataFrame) -> pd.DataFrame:
    win = pd.Timedelta(hours=config.FUSION_WINDOW_HOURS)
    cooldown = pd.Timedelta(hours=config.ALERT_COOLDOWN_HOURS)
    alerts = []
    # só eventos com pontuação relevante (medium ou maior) entram na fusão
    df = df[df["score"] >= config.ALERT_MIN_EVENT_SCORE]
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.reset_index(drop=True)
        times = g["timestamp"].to_numpy()
        last_alert_t, last_level = None, 0
        lo = 0
        for i in range(len(g)):
            while times[lo] < times[i] - win.to_timedelta64():
                lo += 1
            w = g.iloc[lo:i + 1]
            per_mod = w.groupby("modality")["score"].max()
            fused = 1 - np.prod([1 - config.MODALITY_WEIGHTS.get(m, 0.5) * s for m, s in per_mod.items()])
            strong = int((per_mod >= 0.5).sum())
            # "high" exige confirmação: dois tipos de sinal diferentes, NEWS2 elevado
            # ou mais de uma modalidade alterada; um sinal isolado fica em "medium"
            confirmed = w["event_type"].nunique() >= 2 or "news2_elevated" in set(w["event_type"]) or strong >= 2
            if fused >= 0.9 and strong >= 2:
                level = "critical"
            elif fused >= 0.8 and confirmed:
                level = "high"
            elif fused >= 0.5:
                level = "medium"
            else:
                continue
            t = g.at[i, "timestamp"]
            if last_alert_t is not None and t - last_alert_t >= cooldown:
                last_level = 0                            # passou o cooldown: "zera" o nível
            worse = LEVEL_ORDER[level] > last_level
            if last_alert_t is not None and t - last_alert_t < cooldown and not worse:
                continue                                  # evita "spam" de alertas
            top = w.sort_values("score", ascending=False).head(3)
            alerts.append({
                "alert_id": f"A{len(alerts) + 1:05d}", "patient_id": pid, "timestamp": t,
                "level": level, "fused_score": round(float(fused), 3),
                "modalities": ",".join(sorted(per_mod.index)), "n_strong_modalities": strong,
                "top_events": " | ".join(f"[{r.modality}] {r.message}" for r in top.itertuples()),
                "recommended_action": ACTIONS[level],
            })
            last_alert_t, last_level = t, LEVEL_ORDER[level]
    return pd.DataFrame(alerts)


def run_fusion() -> dict:
    raw = load_events()
    if not raw:
        raise RuntimeError("Nenhum evento. Rode antes: python main.py clinical / audio / video")
    events = pd.DataFrame(raw)
    vp = build_virtual_patients(events)
    vp.to_csv(config.RESULTS_DIR / "virtual_patients.csv", index=False)
    fused_events = apply_virtual_ids(events, vp)
    fused_events.to_csv(config.RESULTS_DIR / "events_fused.csv", index=False)
    alerts = fuse_and_alert(fused_events)
    alerts.to_csv(config.ALERTS_FILE, index=False)
    summary = {
        "events": len(fused_events), "virtual_patients": len(vp),
        "patients_monitored": int(fused_events["patient_id"].nunique()),
        "alerts": len(alerts),
        "alerts_by_level": alerts["level"].value_counts().to_dict() if len(alerts) else {},
        "multimodal_alerts": int((alerts["n_strong_modalities"] >= 2).sum()) if len(alerts) else 0,
    }
    (config.RESULTS_DIR / "fusion_summary.json").write_text(json.dumps(summary, indent=2))
    print("[fusion]", summary)
    return summary

"""
Anomalias na evolução das prescrições (alterações inesperadas no tratamento).

Para cada internação e cada medicamento, olhamos a sequência de pedidos e
calculamos o que mudou em relação ao pedido anterior:
  * dose_jump            -> dose diária dobrou (ou caiu pela metade) ou mais
  * route_change         -> mudou a via (ex.: PO -> IV)
  * frequent_reorders    -> muitos pedidos do mesmo remédio em 48 h (instabilidade)
  * high_alert_start     -> início de medicamento de alto risco (insulina, heparina...)
  * order_burst          -> muitos pedidos novos em 6 h (piora clínica costuma gerar isso)
Depois um IsolationForest combina essas características e marca as
alterações mais raras da população ("unusual_prescription_change").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .. import config
from ..core.events import Event


def build_change_features(rx: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pedido, com as mudanças em relação ao pedido anterior do mesmo remédio."""
    df = rx.copy()
    df["is_high_alert"] = df["drug"].apply(lambda d: any(h in d for h in config.HIGH_ALERT_DRUGS))
    g = df.groupby(["hadm_id", "drug"], sort=False)

    df["prev_daily_dose"] = g["daily_dose"].shift(1)
    df["prev_route"] = g["route"].shift(1)
    df["order_number"] = g.cumcount()                 # 0 = primeiro pedido desse remédio

    # razão de dose (log2: +1 = dobrou, -1 = caiu pela metade)
    ratio = df["daily_dose"] / df["prev_daily_dose"]
    df["log2_dose_ratio"] = np.log2(ratio.where(ratio > 0)).fillna(0.0)
    df["route_changed"] = (df["prev_route"].notna() & (df["route"] != df["prev_route"])).astype(int)

    # pedidos do mesmo remédio nas 48 h anteriores (inclui o atual)
    df["same_drug_48h"] = 0
    for _, idx in g.groups.items():
        t = df.loc[idx, "starttime"].values
        counts = [(t[: i + 1] >= t[i] - np.timedelta64(48, "h")).sum() for i in range(len(t))]
        df.loc[idx, "same_drug_48h"] = counts

    # total de pedidos novos da internação nas 6 h anteriores
    df["orders_6h"] = 0
    for _, idx in df.groupby("hadm_id").groups.items():
        t = df.loc[idx, "starttime"].values
        lo = np.searchsorted(t, t - np.timedelta64(6, "h"), side="left")
        df.loc[idx, "orders_6h"] = np.arange(len(t)) - lo + 1
    return df


def detect_prescription_events(rx: pd.DataFrame, contamination: float = 0.02) -> tuple[list[Event], pd.DataFrame]:
    df = build_change_features(rx)

    # --- parte ML: IsolationForest sobre as características de mudança ---
    feats = ["log2_dose_ratio", "route_changed", "same_drug_48h", "orders_6h", "is_high_alert"]
    X = df[feats].astype(float).assign(log2_dose_ratio=lambda d: d["log2_dose_ratio"].abs())
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42).fit(X)
    raw = -model.score_samples(X)
    df["ml_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    df["ml_flag"] = model.predict(X) == -1
    burst_thr = max(10, int(df["orders_6h"].quantile(0.99)))

    events = []
    for r in df.to_dict("records"):
        pid, ts = f"clinical:{r['subject_id']}", pd.Timestamp(r["starttime"]).isoformat()
        boost = 0.2 if r["is_high_alert"] else 0.0   # remédio de alto risco pesa mais
        ev = {"drug": r["drug"], "route": r["route"], "daily_dose": r["daily_dose"],
              "prev_daily_dose": r["prev_daily_dose"], "ml_score": round(r["ml_score"], 3)}

        if abs(r["log2_dose_ratio"]) >= 1:
            direction = "increase" if r["log2_dose_ratio"] > 0 else "decrease"
            score = min(1.0, 0.45 + 0.1 * abs(r["log2_dose_ratio"]) + boost)
            events.append(Event(pid, ts, "clinical", "prescriptions", "dose_jump", score,
                                message=f"{r['drug']}: daily dose {direction} x{2 ** abs(r['log2_dose_ratio']):.1f}",
                                evidence=ev))
        if r["route_changed"]:
            events.append(Event(pid, ts, "clinical", "prescriptions", "route_change", 0.4 + boost,
                                message=f"{r['drug']}: route {r['prev_route']} -> {r['route']}", evidence=ev))
        if r["same_drug_48h"] >= 4:
            events.append(Event(pid, ts, "clinical", "prescriptions", "frequent_reorders", 0.5 + boost,
                                message=f"{r['drug']}: {r['same_drug_48h']} orders in 48 h", evidence=ev))
        if r["is_high_alert"] and r["order_number"] == 0:
            events.append(Event(pid, ts, "clinical", "prescriptions", "high_alert_start", 0.35,
                                message=f"High-alert drug started: {r['drug']}", evidence=ev))
        if r["orders_6h"] >= burst_thr:
            events.append(Event(pid, ts, "clinical", "prescriptions", "order_burst", 0.55,
                                message=f"{r['orders_6h']} new orders in 6 h", evidence=ev))
        if r["ml_flag"]:
            events.append(Event(pid, ts, "clinical", "prescriptions", "unusual_prescription_change",
                                0.5 + 0.4 * r["ml_score"],
                                message=f"{r['drug']}: rare change pattern (IsolationForest)", evidence=ev))

    # o order_burst se repete para cada pedido da rajada: guardamos 1 por internação/hora
    seen, unique = set(), []
    for e in events:
        k = (e.patient_id, e.event_type, e.timestamp[:13]) if e.event_type == "order_burst" else id(e)
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return unique, df

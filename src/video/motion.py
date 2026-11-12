"""
Análise de movimento das sessões de fisioterapia (UL-RED, arquivos .amc).

Ideia central: comparar cada repetição com a execução de REFERÊNCIA do mesmo
sujeito (arquivo R1 = uma repetição em ritmo normal).

Para cada repetição calculamos (sempre em relação à referência, então não
depende da taxa de quadros):
  * duration_ratio  -> >1 mais lento, <1 mais rápido
  * rom_ratio       -> amplitude de movimento (range of motion); <1 = amplitude menor
  * peak_vel_ratio  -> velocidade de pico
  * jerk_ratio      -> "trancos" (suavidade); >1 = movimento menos suave
  * dtw_shape       -> diferença de FORMA da curva (Dynamic Time Warping)
  * asymmetry_delta -> aumento da diferença entre lado esquerdo e direito

Depois:
  1) regras clínicas simples (ritmo fora, amplitude reduzida, assimetria)
  2) IsolationForest treinado SÓ com repetições normais de OUTROS sujeitos
     (detecção de novidade, validação "deixa um sujeito de fora")
Os ritmos rápido/lento do UL-RED (repetições 2 e 3 do arquivo R3) servem de
rótulo para medir se o detector acerta.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from ..core.events import Event
from .amc import parse_amc, index_ulred

PACE_LABELS = ["normal", "fast", "slow"]   # ordem das repetições nos arquivos R3 (README do UL-RED)
LEFT_RIGHT = [("lfemur", "rfemur"), ("ltibia", "rtibia"), ("lhumerus", "rhumerus"), ("lradius", "rradius")]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def key_dofs(ref: pd.DataFrame, k: int = 6) -> list[str]:
    """Os k graus de liberdade que mais se movem no exercício (maior amplitude na referência)."""
    rng = ref.max() - ref.min()
    return list(rng.sort_values(ascending=False).head(k).index)


def segment_repetitions(df: pd.DataFrame, dof: str, n_expected: int = 3) -> list[tuple[int, int]]:
    """
    Separa as repetições: medimos o quanto a articulação principal se afasta da
    posição de repouso (início do arquivo) e pegamos os trechos "ativos".
    """
    x = df[dof].to_numpy()
    x = np.convolve(x, np.ones(5) / 5, mode="same")
    rest = np.median(x[: max(5, len(x) // 20)])
    activity = np.abs(x - rest)
    active = activity > 0.25 * activity.max()
    segs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append([start, i]); start = None
    if start is not None:
        segs.append([start, len(x)])
    # junta trechos separados por lacunas muito curtas (ruído)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 0.03 * len(x):
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    merged = sorted(sorted(merged, key=lambda s: s[1] - s[0], reverse=True)[:n_expected])
    return [tuple(s) for s in merged] if len(merged) == n_expected else []


def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.2) -> float:
    """DTW multivariado com banda de Sakoe-Chiba (limita o custo de memória/tempo)."""
    # captura a 100+ Hz gera curvas longas: reduzimos para ~150 pontos (a forma se mantém)
    step = max(1, max(len(a), len(b)) // 150)
    a, b = a[::step], b[::step]
    n, m = len(a), len(b)
    w = max(int(band * max(n, m)), abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        cost = np.linalg.norm(b[lo - 1:hi] - a[i - 1], axis=1)
        for j in range(lo, hi + 1):
            D[i, j] = cost[j - lo] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))   # normaliza pelo tamanho do caminho


def rep_stats(seg: pd.DataFrame, dofs: list[str]) -> dict:
    x = seg[dofs].to_numpy()
    vel = np.abs(np.diff(x, axis=0))
    jerk = np.abs(np.diff(x, n=3, axis=0))
    rom = x.max(0) - x.min(0)
    asym = []
    for l, r in LEFT_RIGHT:
        lc, rc = [c for c in seg.columns if c.startswith(l + "_")], [c for c in seg.columns if c.startswith(r + "_")]
        if lc and rc:
            rl = (seg[lc].max() - seg[lc].min()).max()
            rr = (seg[rc].max() - seg[rc].min()).max()
            if max(rl, rr) > 5:   # só articulações que realmente se movem
                asym.append(abs(rl - rr) / max(rl, rr))
    return {"frames": len(seg), "rom": rom, "peak_vel": vel.max(0) if len(vel) else rom * 0,
            "jerk": jerk.mean(0) if len(jerk) else rom * 0, "asymmetry": float(np.mean(asym)) if asym else 0.0}


# ---------------------------------------------------------------------------
# Extração por sujeito/movimento
# ---------------------------------------------------------------------------
def analyze_pair(r1_path: str, r3_path: str) -> list[dict]:
    """Compara as 3 repetições do arquivo R3 com a repetição única do arquivo R1."""
    ref_full, r3 = parse_amc(Path(r1_path)), parse_amc(Path(r3_path))
    dofs = [d for d in key_dofs(ref_full) if d in r3.columns]
    main = dofs[0]
    ref_seg = segment_repetitions(ref_full, main, n_expected=1)
    reps = segment_repetitions(r3, main, n_expected=3)
    if not ref_seg or not reps:
        return []
    ref = ref_full.iloc[ref_seg[0][0]:ref_seg[0][1]]
    rs = rep_stats(ref, dofs)
    scale = ref[dofs].std().replace(0, 1).to_numpy()       # normaliza cada articulação

    out = []
    for k, (a, b) in enumerate(reps):
        seg = r3.iloc[a:b]
        st = rep_stats(seg, dofs)
        out.append({
            "rep": k + 1,
            "pace_label": PACE_LABELS[k],
            "duration_ratio": st["frames"] / rs["frames"],
            "rom_ratio": float(np.mean(st["rom"] / np.maximum(rs["rom"], 1e-6))),
            "peak_vel_ratio": float(np.mean(st["peak_vel"] / np.maximum(rs["peak_vel"], 1e-6))),
            "jerk_ratio": float(np.mean(st["jerk"] / np.maximum(rs["jerk"], 1e-6))),
            "dtw_shape": dtw_distance(seg[dofs].to_numpy() / scale, ref[dofs].to_numpy() / scale),
            "asymmetry_delta": st["asymmetry"] - rs["asymmetry"],
            "main_dof": main,
            "_ref_curve": ref[main].to_numpy().tolist(),     # usado no relatório
            "_rep_curve": seg[main].to_numpy().tolist(),
        })
    return out


def build_repetition_table(index: pd.DataFrame | None = None, system: str = "marker") -> pd.DataFrame:
    idx = index_ulred() if index is None else index
    if idx.empty:
        raise FileNotFoundError("Nenhum .amc do UL-RED encontrado em datasets/video/")
    idx = idx[idx["system"] == system]
    rows = []
    for (subj, motion), g in idx.groupby(["subject", "motion"]):
        r1, r3 = g[g["reps"] == 1], g[g["reps"] == 3]
        if r1.empty or r3.empty:
            continue
        try:
            for rep in analyze_pair(r1.iloc[0]["path"], r3.iloc[0]["path"]):
                rows.append({"subject": subj, "motion": motion, **rep})
        except Exception as exc:   # arquivo incompleto (o README lista alguns)
            print(f"[video] {subj}/{motion} ignorado: {exc}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------
FEATURES = ["duration_ratio", "rom_ratio", "peak_vel_ratio", "jerk_ratio", "dtw_shape", "asymmetry_delta"]


def score_repetitions(reps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = reps.copy()
    X = df[FEATURES].copy()
    X["duration_ratio"] = np.log(X["duration_ratio"])          # simétrico: x2 e /2 pesam igual
    X["peak_vel_ratio"] = np.log(X["peak_vel_ratio"].clip(lower=1e-3))
    X["jerk_ratio"] = np.log(X["jerk_ratio"].clip(lower=1e-3))
    X["rom_ratio"] = np.log(X["rom_ratio"].clip(lower=1e-3))

    # IsolationForest "deixa um sujeito de fora": treina com repetições normais dos OUTROS
    df["ml_score"] = np.nan
    for subj in df["subject"].unique():
        train = (df["subject"] != subj) & (df["pace_label"] == "normal")
        if train.sum() < 5:
            continue
        model = IsolationForest(n_estimators=200, random_state=42).fit(X[train])
        test = df["subject"] == subj
        df.loc[test, "ml_score"] = -model.score_samples(X[test])
    if df["ml_score"].notna().any():
        s = df["ml_score"]
        df["ml_score"] = (s - s.min()) / (s.max() - s.min() + 1e-9)

    # regras clínicas interpretáveis
    df["pace_deviation"] = np.abs(np.log(df["duration_ratio"])) > np.log(1.35)
    df["reduced_rom"] = df["rom_ratio"] < 0.75
    df["asymmetry"] = df["asymmetry_delta"] > 0.15
    shape_thr = df.loc[df["pace_label"] == "normal", "dtw_shape"].quantile(0.95)
    df["form_deviation"] = df["dtw_shape"] > shape_thr
    rule = df[["pace_deviation", "reduced_rom", "asymmetry", "form_deviation"]].any(axis=1)
    df["anomaly_score"] = np.maximum(df["ml_score"].fillna(0), rule * 0.6)

    # avaliação: ritmo rápido/lento deveria ser detectado como "fora do padrão"
    y = (df["pace_label"] != "normal").astype(int)
    metrics = {"repetitions": len(df), "subjects": int(df["subject"].nunique())}
    if y.nunique() == 2:
        metrics["roc_auc_ml"] = round(roc_auc_score(y, df["ml_score"].fillna(0)), 3)
        metrics["pace_rule_recall"] = round(float(df.loc[y == 1, "pace_deviation"].mean()), 3)
        metrics["pace_rule_false_positive_rate"] = round(float(df.loc[y == 0, "pace_deviation"].mean()), 3)
    return df, metrics


def repetition_events(df: pd.DataFrame, base_ts: str) -> list[Event]:
    events = []
    t0 = pd.Timestamp(base_ts) + pd.Timedelta(hours=14)   # "sessão de fisioterapia às 14h"
    for i, r in enumerate(df.itertuples()):
        ts = (t0 + pd.Timedelta(minutes=i % 120)).isoformat()
        ev = {"motion": r.motion, "rep": r.rep, "duration_ratio": round(r.duration_ratio, 2),
              "rom_ratio": round(r.rom_ratio, 2), "dtw_shape": round(r.dtw_shape, 3)}
        pid = f"ulred:{r.subject}"
        if r.pace_deviation:
            kind = "slower" if r.duration_ratio > 1 else "faster"
            events.append(Event(pid, ts, "video", "physio_motion", "pace_deviation", 0.45,
                                message=f"{r.motion} rep {r.rep}: {kind} than reference (x{r.duration_ratio:.2f})",
                                evidence=ev))
        if r.reduced_rom:
            events.append(Event(pid, ts, "video", "physio_motion", "reduced_range_of_motion",
                                min(1.0, 0.5 + (0.75 - r.rom_ratio)), evidence=ev,
                                message=f"{r.motion} rep {r.rep}: range of motion {r.rom_ratio:.0%} of reference"))
        if r.asymmetry:
            events.append(Event(pid, ts, "video", "physio_motion", "left_right_asymmetry",
                                min(1.0, 0.5 + r.asymmetry_delta), evidence=ev,
                                message=f"{r.motion} rep {r.rep}: left/right asymmetry increased"))
        if r.form_deviation or (r.ml_score >= 0.7 if pd.notna(r.ml_score) else False):
            events.append(Event(pid, ts, "video", "physio_motion", "abnormal_movement_pattern",
                                max(0.5, float(r.ml_score) if pd.notna(r.ml_score) else 0.5), evidence=ev,
                                message=f"{r.motion} rep {r.rep}: movement shape differs from reference"))
    return events

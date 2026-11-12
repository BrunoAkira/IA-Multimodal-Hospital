"""
Avaliação do reconhecimento em vídeo REAL com o gabarito do REHAB24-6.

O REHAB24-6 traz vídeos de 10 pessoas fazendo 6 exercícios e um arquivo
Segmentation.csv dizendo, para cada repetição, o quadro de início/fim e se ela
foi feita CORRETAMENTE (1) ou INCORRETAMENTE (0), segundo um fisioterapeuta.

O que este módulo faz:
  1. Roda a estimativa de pose (OpenPose ou YOLOv8-pose) em cada vídeo escolhido
     (os ângulos ficam em cache: rodar de novo é instantâneo).
  2. Corta a série de ângulos em repetições usando o gabarito.
  3. Mede cada repetição: amplitude, ângulo máximo/mínimo, duração, velocidade,
     inclinação do tronco e movimento do lado que deveria ficar parado.
  4. Mede o quanto cada medida separa correta x incorreta (ROC-AUC) e treina uma
     regressão logística validada "deixa uma pessoa de fora" (nunca testa em quem treinou).
  5. Compara o nº de repetições contadas automaticamente com o gabarito.
  6. Gera relatório HTML e eventos "incorrect_execution" para a fusão.

    python main.py rehab24 --exercises 1,6 --max-videos 4
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from ..core.events import Event, save_events, load_events
from ..core.report import write_report, fig_to_b64
from .vision import analyze_video, analyze_angles

REHAB_DIR = config.DATASETS / "rehab24"
CACHE = config.RESULTS_DIR / "rehab24_angles"

# exercise_id do REHAB24-6 -> nome padronizado (inglês) e articulação principal
EXERCISES = {
    1: {"name": "arm_abduction", "joint": "shoulder", "project_exercise": "arm_raise"},
    2: {"name": "arm_vw", "joint": "shoulder", "project_exercise": None},
    3: {"name": "push_ups", "joint": "elbow", "project_exercise": None},
    4: {"name": "leg_abduction", "joint": "hip", "project_exercise": None},
    5: {"name": "leg_lunge", "joint": "knee", "project_exercise": None},
    6: {"name": "squats", "joint": "knee", "project_exercise": "squat"},
}
FEATURES = ["rom", "max_angle", "min_angle", "duration_s", "mean_speed", "trunk_lean_max",
            "other_side_rom", "pose_coverage"]


# ---------------------------------------------------------------------------
# Leitura do gabarito e dos vídeos
# ---------------------------------------------------------------------------
def load_segmentation() -> pd.DataFrame:
    path = next(REHAB_DIR.rglob("Segmentation.csv"), None)
    if path is None:
        raise FileNotFoundError(f"Segmentation.csv não encontrado em {REHAB_DIR}")
    seg = pd.read_csv(path, sep=None, engine="python")      # o arquivo usa ';'
    seg.columns = [c.strip() for c in seg.columns]
    return seg


def find_videos(exercises: list[int], camera: str = "Camera17", max_per_exercise: int = 4) -> list[dict]:
    """Lista os vídeos de cada exercício (pasta Ex<n>), no máximo N por exercício."""
    out = []
    for ex in exercises:
        files = sorted(REHAB_DIR.rglob(f"Ex{ex}/*{camera}*.mp4"))
        for f in files[:max_per_exercise]:
            m = re.search(r"PM_\d+", f.name)
            if m:
                out.append({"exercise_id": ex, "video_id": m.group(0), "path": f})
    return out


def video_angles(item: dict) -> pd.DataFrame:
    """Ângulos quadro a quadro (com cache em results/rehab24_angles/)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{item['video_id']}_{item['path'].stem.split('-')[1]}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print(f"[rehab24] Pose em {item['path'].name} (só pose, sem YOLO de objetos)...")
    res = analyze_video(item["path"], detect_objects=False, max_seconds=600)
    res["angles"].to_csv(cache, index=False)
    return res["angles"]


# ---------------------------------------------------------------------------
# Medidas por repetição
# ---------------------------------------------------------------------------
def _side(subtype: str, angles: pd.DataFrame, joint: str) -> str:
    """Lado que executa: vem do gabarito ('right arm', 'left leg') ou do que mais se move."""
    s = str(subtype).lower()
    if "left" in s:
        return "l"
    if "right" in s:
        return "r"
    rng = {k: angles[f"{k}_{joint}"].quantile(0.95) - angles[f"{k}_{joint}"].quantile(0.05)
           for k in ("l", "r") if f"{k}_{joint}" in angles}
    return max(rng, key=rng.get) if rng else "r"


def repetition_features(angles: pd.DataFrame, seg_video: pd.DataFrame, joint: str, src_fps: float = 30.0) -> pd.DataFrame:
    rows = []
    for r in seg_video.itertuples():
        side = _side(r.exercise_subtype, angles, joint)
        other = "l" if side == "r" else "r"
        w = angles[(angles["frame"] >= r.first_frame) & (angles["frame"] <= r.last_frame)]
        main_col, other_col = f"{side}_{joint}", f"{other}_{joint}"
        if main_col not in w or w[main_col].notna().sum() < 3:
            continue
        x = w[main_col].interpolate(limit=3)
        t = w["time_s"].to_numpy()
        speed = np.abs(np.diff(x.to_numpy())) / np.maximum(np.diff(t), 1e-6)
        rows.append({
            "video_id": r.video_id, "repetition": r.repetition_number, "person_id": r.person_id,
            "correct": int(r.correctness), "side": side,
            "rom": float(x.quantile(0.95) - x.quantile(0.05)),
            "max_angle": float(x.max()), "min_angle": float(x.min()),
            "duration_s": (r.last_frame - r.first_frame) / src_fps,
            "mean_speed": float(np.nanmean(speed)) if len(speed) else np.nan,
            "trunk_lean_max": float(w["trunk_lean"].max()) if "trunk_lean" in w else np.nan,
            "other_side_rom": float(w[other_col].quantile(0.95) - w[other_col].quantile(0.05))
            if other_col in w and w[other_col].notna().sum() >= 3 else np.nan,
            "pose_coverage": float(w[main_col].notna().mean()),
            "first_frame": r.first_frame, "last_frame": r.last_frame,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def evaluate(reps: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """AUC de cada medida isolada + regressão logística deixa-uma-pessoa-de-fora."""
    res = {"repetitions": len(reps), "correct": int(reps["correct"].sum()),
           "incorrect": int((reps["correct"] == 0).sum()), "videos": int(reps["video_id"].nunique()),
           "persons": int(reps["person_id"].nunique())}
    y = 1 - reps["correct"]      # 1 = incorreta (o que queremos detectar)
    if y.nunique() < 2:
        res["note"] = "só uma classe nas repetições escolhidas"
        return res, reps

    # normaliza cada medida pela mediana do próprio vídeo (pessoas e câmeras diferentes)
    X = reps[FEATURES].copy()
    for f in FEATURES:
        med = reps.groupby("video_id")[f].transform("median")
        X[f] = (reps[f] - med) / (med.abs() + 1e-6)
    X = X.fillna(0)

    # quanto cada medida, sozinha, separa correta x incorreta (0,5 = acaso; 1,0 = perfeito)
    per_feature = {}
    for f in FEATURES:
        if X[f].nunique() > 1:
            auc = roc_auc_score(y, X[f])
            per_feature[f] = {"auc": round(max(auc, 1 - auc), 3),
                              "incorrect_when": "higher" if auc >= 0.5 else "lower"}
    res["single_feature_auc"] = dict(sorted(per_feature.items(), key=lambda kv: -kv[1]["auc"]))

    reps = reps.copy()
    if reps["person_id"].nunique() >= 2:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        proba = cross_val_predict(model, X, y, groups=reps["person_id"], cv=LeaveOneGroupOut(),
                                  method="predict_proba")[:, 1]
        reps["p_incorrect"] = proba
        res["logistic_leave_one_person_out"] = {
            "roc_auc": round(roc_auc_score(y, proba), 3),
            "accuracy": round(accuracy_score(y, proba >= 0.5), 3),
        }
    else:
        res["logistic_leave_one_person_out"] = "precisa de vídeos de 2+ pessoas"
    return res, reps


def _plot_video(angles: pd.DataFrame, reps_v: pd.DataFrame, joint: str, title: str):
    side = reps_v["side"].iloc[0]
    col = f"{side}_{joint}"
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(angles["frame"], angles[col], color="black", lw=1)
    for r in reps_v.itertuples():
        ax.axvspan(r.first_frame, r.last_frame, color="tab:green" if r.correct else "tab:red", alpha=0.18)
    ax.set_title(f"{title} — {col} angle; green = correct, red = incorrect (ground truth)", fontsize=9)
    ax.set_xlabel("frame"); ax.set_ylabel("degrees")
    return fig_to_b64(fig)


def run_rehab24(exercises: list[int], max_per_exercise: int = 4) -> dict:
    seg = load_segmentation()
    videos = find_videos(exercises, max_per_exercise=max_per_exercise)
    if not videos:
        raise FileNotFoundError(f"Nenhum vídeo Camera17 em {REHAB_DIR}/Ex<n>/")
    all_reps, counts, plots = [], [], []
    for item in videos:
        ex = EXERCISES[item["exercise_id"]]
        angles = video_angles(item)
        seg_v = seg[seg["video_id"] == item["video_id"]]
        if seg_v.empty:
            continue
        reps = repetition_features(angles, seg_v, ex["joint"])
        reps["exercise"] = ex["name"]
        all_reps.append(reps)
        # contagem automática (a mesma do relatório de vídeo) x gabarito
        _, summ = analyze_angles(angles, ex["project_exercise"], fps=1 / np.median(np.diff(angles["time_s"])))
        counts.append({"video_id": item["video_id"], "exercise": ex["name"],
                       "ground_truth_reps": len(seg_v),
                       "detected_reps": summ.get(f"{ex['joint']}_repetitions")})
        if len(plots) < 6 and not reps.empty:
            plots.append(_plot_video(angles, reps, ex["joint"], f"{item['video_id']} {ex['name']}"))

    reps = pd.concat(all_reps, ignore_index=True)
    summary = {"exercises": {}, "repetition_counting": counts}
    scored = []
    for name, g in reps.groupby("exercise"):
        res, g2 = evaluate(g)
        summary["exercises"][name] = res
        scored.append(g2)
    scored = pd.concat(scored, ignore_index=True)
    scored.to_csv(config.RESULTS_DIR / "rehab24_repetitions.csv", index=False)
    (config.RESULTS_DIR / "rehab24_evaluation.json").write_text(json.dumps(summary, indent=2, default=str))

    # eventos para a fusão: repetições classificadas como incorretas
    events = []
    t0 = pd.Timestamp(config.RELATIVE_EPOCH) + pd.Timedelta(hours=16)
    for r in scored.itertuples():
        p = getattr(r, "p_incorrect", np.nan)
        if pd.notna(p) and p >= 0.5:
            events.append(Event(f"rehab:{r.video_id}", (t0 + pd.Timedelta(seconds=r.first_frame / 30)).isoformat(),
                                "video", "rehab24_execution", "incorrect_execution", float(0.4 + 0.5 * p),
                                message=f"{r.exercise} rep {r.repetition}: execution flagged as incorrect (p={p:.2f})",
                                evidence={"video_id": r.video_id, "rom": round(r.rom, 1),
                                          "ground_truth_correct": r.correct}))
    kept = [e for e in load_events() if e["source"] != "rehab24_execution"]
    from ..core.events import EVENTS_FILE
    with EVENTS_FILE.open("w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    save_events(events)

    # relatório
    sections = []
    for name, res in summary["exercises"].items():
        feat = pd.DataFrame([{"feature": k, **v} for k, v in res.get("single_feature_auc", {}).items()])
        text = (f"{res['repetitions']} repetitions ({res['correct']} correct, {res['incorrect']} incorrect) "
                f"from {res['videos']} videos / {res['persons']} persons. "
                f"Logistic regression, leave-one-person-out: {res.get('logistic_leave_one_person_out')}")
        sections.append({"heading": f"{name}: correct vs incorrect", "text": text, "table": feat})
    sections.append({"heading": "Repetition counting vs ground truth", "table": pd.DataFrame(counts)})
    sections.append({"heading": "Joint angle with ground-truth repetitions", "images": plots})
    rp = write_report("rehab24_evaluation.html", "REHAB24-6 — real-video evaluation", sections)
    summary["report"] = str(rp)
    summary["events"] = len(events)
    return summary

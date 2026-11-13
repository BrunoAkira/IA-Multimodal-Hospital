"""
Gráficos de resultados para o relatório técnico.

Lê os arquivos que os outros comandos gravam em results/ e salva uma imagem PNG
por gráfico em results/figures/. Nenhum número é digitado aqui: tudo vem dos JSON.

    clinical_summary.json   -> python main.py clinical
    audio_summary.json      -> python main.py audio
    video_summary.json      -> python main.py video
    video_validation.json   -> python main.py video-validation
    fusion_summary.json     -> python main.py fusion

Um gráfico cujo arquivo de origem ainda não existe é pulado com um aviso.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")   # gera imagens sem abrir janela
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np

from . import config

FIG_DIR = config.RESULTS_DIR / "figures"

# paleta: azul, coral, verde-azulado e âmbar (separáveis também para daltonismo)
BLUE, CORAL, TEAL, AMBER, DARK_RED = "#2F6DB5", "#E4572E", "#2A9D8F", "#C58A1A", "#A3241B"
INK, MUTED, GRID = "#1F2933", "#5F6B7A", "#E3E6EA"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})

VITAL_NAMES = {"heart_rate": "Frequência\ncardíaca", "systolic_bp": "PA sistólica", "spo2": "SpO₂",
               "resp_rate": "Frequência\nrespiratória", "temperature": "Temperatura"}
MODEL_NAMES = {"logistic_regression": "Regressão logística", "random_forest": "Random Forest"}
FEATURE_NAMES = {"trunk_lean_max": "Inclinação máx. do tronco", "rom": "Amplitude (ROM)", "min_angle": "Ângulo mínimo",
                 "mean_speed": "Velocidade média", "duration_s": "Duração", "other_side_rom": "Amplitude do outro lado",
                 "max_angle": "Ângulo máximo", "pose_coverage": "Cobertura da pose"}
EXERCISE_NAMES = {"squats": "Agachamento", "arm_abduction": "Abdução de braço"}


def _load(name: str) -> dict | None:
    """Lê um JSON de results/; devolve None se o comando correspondente ainda não rodou."""
    path = config.RESULTS_DIR / name
    if not path.exists():
        print(f"[figures] {name} não encontrado: gráficos dessa etapa pulados")
        return None
    return json.loads(path.read_text())


def _save(fig, name: str, saved: list):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(name)


def _label_bars(ax, bars, fmt="{:.0%}", dy=0.01):
    """Escreve o valor em cima de cada barra."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=8.5, color=INK)


def _fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def architecture(saved):
    """Diagrama do fluxo: fontes -> módulos -> eventos -> fusão e alertas."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4)

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.55, sub, ha="center", va="top", fontsize=7.6, color=MUTED, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))

    rows = [("Dados clínicos de UTI", "sinais vitais, prescrições,\ntransferências", "Módulo clínico", "NEWS2 · z-score robusto\nIsolationForest · Markov"),
            ("Áudio e texto", "falas, consultas", "Módulo de áudio", "features de voz · LogReg/RF\nAzure Speech + Language"),
            ("Vídeo e movimento", "vídeos de exercício,\ncaptura de movimento", "Módulo de vídeo", "OpenPose / YOLOv8-pose\nYOLOv8n · DTW · IsolationForest")]
    for i, (src, src_sub, mod, mod_sub) in enumerate(rows):
        y = 3.9 - i * 1.65
        box(0.1, y, 2.3, 1.25, src, src_sub, "#EEF3FA")
        box(3.0, y, 2.6, 1.25, mod, mod_sub, "#E9F5F3")
        arrow(2.42, y + 0.62, 2.98, y + 0.62)
        arrow(5.62, y + 0.62, 6.25, 2.6)
    box(6.3, 1.95, 1.6, 1.3, "Eventos", "formato único\n(JSONL)", "#FBF3E3")
    arrow(7.92, 2.6, 8.15, 2.6)
    box(8.2, 1.55, 1.75, 2.1, "Fusão e alertas", "janela de 24 h\nmedium · high · critical\n\npainel, relatórios\nHTML, monitor", "#FCEBE6")
    _save(fig, "f01_arquitetura.png", saved)


def clinical_injection(clin, saved):
    """Recall e precisão da avaliação por injeção, por sinal vital."""
    ev = clin.get("injection_evaluation") or {}
    if not ev:
        return
    keys = list(ev)
    rec = [ev[k]["recall"] for k in keys]
    prec = [ev[k]["precision"] for k in keys]
    x, w = np.arange(len(keys)), 0.36
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _label_bars(ax, ax.bar(x - w / 2 - 0.01, rec, w, color=BLUE, label="Recall", zorder=3))
    _label_bars(ax, ax.bar(x + w / 2 + 0.01, prec, w, color=CORAL, label="Precisão", zorder=3))
    ax.set_xticks(x, [VITAL_NAMES.get(k, k) for k in keys])
    ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"O detector encontra de {min(rec):.0%} a {max(rec):.0%} das anomalias injetadas")
    ax.legend(loc="upper center", ncol=2)
    _save(fig, "f02_clinico_injecao.png", saved)


def events_by_source(clin, audio, video, saved):
    """Quantos eventos cada fonte gerou (entrada da fusão)."""
    names = {"vitals": "Sinais vitais (regras e fluxo)", "vitals_multivariate": "Sinais vitais (multivariado)",
             "prescriptions": "Prescrições", "transfers": "Transferências", "audio": "Áudio (voz)",
             "text": "Texto das consultas"}
    counts = {}
    if clin:
        counts.update({names[k]: v for k, v in clin["events"].items()})
    if audio:
        counts.update({names[k]: v for k, v in audio["events"].items()})
    if video:
        counts["Vídeo"] = video.get("events", 0)
    if not counts:
        return
    items = sorted(counts.items(), key=lambda kv: kv[1])
    total = sum(counts.values())
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE, height=0.6, zorder=3)
    top = max(counts.values())
    for b in bars:
        ax.text(b.get_width() + top * 0.01, b.get_y() + b.get_height() / 2, _fmt_int(b.get_width()), va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, top * 1.15); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("eventos gerados"); ax.set_title(f"{_fmt_int(total)} eventos gerados pelos módulos")
    _save(fig, "f03_eventos_fonte.png", saved)


def voice_metrics(audio, saved):
    """Métricas dos classificadores de voz e matrizes de confusão."""
    metrics = (audio.get("classifier") or {}).get("metrics") or {}
    if not metrics:
        return
    models = list(metrics)

    def derived(m):
        # recall e precisão da classe disartria a partir da matriz [[TN, FP], [FN, TP]]
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        return [m["accuracy"], tp / max(tp + fn, 1), tp / max(tp + fp, 1), m["f1"], m["roc_auc"]]

    labels = ["Acurácia", "Recall\n(disartria)", "Precisão\n(disartria)", "F1", "ROC-AUC"]
    x, w = np.arange(len(labels)), 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (name, color) in enumerate(zip(models, [BLUE, TEAL, AMBER])):
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * (w + 0.02), derived(metrics[name]), w, color=color,
                      label=MODEL_NAMES.get(name, name), zorder=3)
        _label_bars(ax, bars, "{:.2f}")
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.2))
    best = audio["classifier"].get("best_model")
    ax.set_title(f"Classificador de disartria (escolhido: {MODEL_NAMES.get(best, best)})")
    ax.legend(loc="upper center", ncol=len(models))
    _save(fig, "f04_voz_metricas.png", saved)

    fig, axs = plt.subplots(1, len(models), figsize=(4 * len(models), 3.4))
    for ax, name in zip(np.atleast_1d(axs), models):
        cm = np.array(metrics[name]["confusion_matrix"])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center", fontsize=10,
                        color="white" if norm[i, j] > 0.55 else INK)
        # destaque nos falsos negativos (disartria classificada como controle)
        ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec=CORAL, lw=2.5))
        ax.set_xticks([0, 1], ["controle", "disartria"]); ax.set_yticks([0, 1], ["controle", "disartria"])
        ax.set_xlabel("previsto"); ax.set_ylabel("real"); ax.set_title(MODEL_NAMES.get(name, name))
        for s in ax.spines.values():
            s.set_visible(False)
    _save(fig, "f05_voz_confusao.png", saved)


def transcription_wer(audio, saved):
    """Erro de palavras da transcrição do Azure por grupo de falantes."""
    wer = audio.get("azure_stt_wer_by_group") or {}
    if not wer or not isinstance(wer, dict) or "control" not in wer:
        return
    groups = [g for g in ("control", "dysarthric") if g in wer]
    names = {"control": "Fala sem alteração", "dysarthric": "Fala com disartria"}
    vals = [wer[g]["mean"] for g in groups]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _label_bars(ax, ax.bar([f"{names[g]}\n({wer[g]['count']} falas)" for g in groups], vals,
                           color=[BLUE, CORAL][:len(groups)], width=0.5, zorder=3))
    ax.set_ylim(0, max(vals) * 1.2); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("taxa de erro de palavras (WER)")
    ax.set_title("Erro da transcrição por grupo")
    _save(fig, "f06_wer.png", saved)


def video_feature_auc(val, saved):
    """Poder de cada característica, sozinha, para separar repetição correta de incorreta."""
    exercises = val.get("exercises") or {}
    if not exercises:
        return
    fig, axs = plt.subplots(1, len(exercises), figsize=(5 * len(exercises), 3.8))
    for ax, (name, color) in zip(np.atleast_1d(axs), zip(exercises, [BLUE, TEAL, AMBER])):
        res = exercises[name]
        auc = {k: v["auc"] for k, v in res["single_feature_auc"].items()}
        keys = sorted(auc, key=auc.get)
        bars = ax.barh([FEATURE_NAMES.get(k, k) for k in keys], [auc[k] for k in keys], color=color, height=0.6, zorder=3)
        for b in bars:
            ax.text(b.get_width() + 0.008, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}", va="center", fontsize=8.5, color=INK)
        ax.axvline(0.5, color=MUTED, ls="--", lw=1)
        ax.text(0.505, len(keys) - 0.55, "acaso", fontsize=7.5, color=MUTED)
        ax.set_xlim(0.4, 1.08); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_xlabel("ROC-AUC de cada característica sozinha")
        ax.set_title(f"{EXERCISE_NAMES.get(name, name)} ({res['persons']} pessoas, {res['repetitions']} rep.)", fontsize=10)
    _save(fig, "f07_auc_features.png", saved)


def repetition_counting(val, saved):
    """Repetições contadas automaticamente x gabarito, por vídeo."""
    counts = [c for c in (val.get("repetition_counting") or []) if c.get("detected_reps") is not None]
    if not counts:
        return
    x, w = np.arange(len(counts)), 0.36
    gt = [c["ground_truth_reps"] for c in counts]
    det = [c["detected_reps"] for c in counts]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(counts)), 3.8))
    _label_bars(ax, ax.bar(x - w / 2 - 0.01, gt, w, color=MUTED, label="Gabarito do fisioterapeuta", zorder=3), "{:.0f}", 0.3)
    _label_bars(ax, ax.bar(x + w / 2 + 0.01, det, w, color=BLUE, label="Contagem automática", zorder=3), "{:.0f}", 0.3)
    ax.set_xticks(x, [f"{c['video_id']}\n{EXERCISE_NAMES.get(c['exercise'], c['exercise']).lower()}" for c in counts], fontsize=8.5)
    ax.set_ylim(0, max(gt + det) * 1.4); ax.set_ylabel("repetições")
    worst = max(abs(a - b) for a, b in zip(gt, det))
    ax.set_title(f"Contagem automática: {sum(det)} repetições para {sum(gt)} do gabarito\n"
                 f"(diferença máxima de {worst} por vídeo)")
    ax.legend(loc="upper center", ncol=2)
    _save(fig, "f08_contagem.png", saved)


def alerts_by_level(fusion, saved):
    """Alertas emitidos por nível de gravidade."""
    levels = fusion.get("alerts_by_level") or {}
    if not levels:
        return
    order = [l for l in ("medium", "high", "critical") if l in levels]
    vals = [levels[l] for l in order]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(order, vals, color=[{"medium": AMBER, "high": CORAL, "critical": DARK_RED}[l] for l in order], width=0.55, zorder=3)
    _label_bars(ax, bars, "{:.0f}", max(vals) * 0.015)
    ax.set_ylim(0, max(vals) * 1.15); ax.set_ylabel("alertas")
    ax.set_title(f"{_fmt_int(fusion['alerts'])} alertas por nível ({fusion.get('multimodal_alerts', 0)} multimodais)")
    _save(fig, "f09_alertas.png", saved)


def azure_usage(audio, saved):
    """Consumo do Azure no mês em relação à cota gratuita."""
    u = audio.get("azure_usage") or {}
    if not u.get("speech_budget_seconds"):
        return
    # o limite do sistema é 80% da cota gratuita; a cota é recuperada a partir dele
    free_speech = u["speech_budget_seconds"] / 0.8
    free_lang = u["language_budget_records"] / 0.8
    labels = [f"Speech to Text\n({u['speech_seconds']:.0f} s de {free_speech / 3600:.0f} h)",
              f"Language\n({u['language_records']} de {_fmt_int(free_lang)} registros)"]
    used = [u["speech_seconds"] / free_speech, u["language_records"] / free_lang]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.barh(labels, [0.8, 0.8], color="#EEF3FA", height=0.55, zorder=2, label="limite do sistema (80%)")
    bars = ax.barh(labels, used, color=BLUE, height=0.55, zorder=3, label="uso no mês")
    for b in bars:
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{b.get_width():.1%}", va="center", fontsize=9, color=INK)
    ax.axvline(1.0, color=CORAL, lw=1.2)
    ax.text(0.985, -0.45, "cota\ngratuita", ha="right", fontsize=7.5, color=CORAL)
    ax.set_xlim(0, 1.05); ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_title(f"Uso do Azure no mês ({u['month']})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    _save(fig, "f10_azure.png", saved)


def run_figures() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    clin, audio = _load("clinical_summary.json"), _load("audio_summary.json")
    video, val, fusion = _load("video_summary.json"), _load("video_validation.json"), _load("fusion_summary.json")
    saved: list[str] = []
    architecture(saved)
    if clin:
        clinical_injection(clin, saved)
    events_by_source(clin, audio, video, saved)
    if audio:
        voice_metrics(audio, saved)
        transcription_wer(audio, saved)
        azure_usage(audio, saved)
    if val:
        video_feature_auc(val, saved)
        repetition_counting(val, saved)
    if fusion:
        alerts_by_level(fusion, saved)
    print(f"[figures] {len(saved)} gráficos em {FIG_DIR}")
    return {"folder": str(FIG_DIR), "figures": sorted(saved)}

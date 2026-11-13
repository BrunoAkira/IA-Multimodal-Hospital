"""
Leitura dos resultados no terminal.

Funções usadas pelos comandos:
    python main.py summary          -> métricas principais de cada módulo
    python main.py alerts           -> fila de alertas (filtros por nível e quantidade)
    python main.py consultations    -> resultado da análise de cada consulta

Tudo é lido dos arquivos que os outros comandos gravam em results/.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config


def _load_json(name: str):
    """Lê um JSON de results/; devolve None se o comando correspondente ainda não rodou."""
    path = config.RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def _line(label: str, text: str):
    print(f"{label:.<22} {text}")


def print_summary():
    """Mostra as principais métricas de cada módulo, a partir dos arquivos de resultado."""
    print(f"Resultados em {config.RESULTS_DIR}\n")
    shown = 0

    clin = _load_json("clinical_summary.json")
    if clin:
        shown += 1
        ev = clin.get("injection_evaluation") or {}
        rec = [v["recall"] for v in ev.values()]
        prec = [v["precision"] for v in ev.values()]
        _line("Clínico", f"{clin['vital_measurements']:,} medições, {clin['patient_hours']:,} horas-paciente")
        if rec:
            _line("", f"recall {min(rec):.2f} a {max(rec):.2f} | precisão {min(prec):.2f} a {max(prec):.2f}")
        _line("", "eventos " + ", ".join(f"{k} {v}" for k, v in clin["events"].items()))

    audio = _load_json("audio_summary.json")
    if audio:
        shown += 1
        clf = audio.get("classifier") or {}
        best = clf.get("best_model")
        _line("Áudio", f"{audio['utterances']} falas de {audio['speakers']} falantes {audio.get('groups', {})}")
        if best:
            m = clf["metrics"][best]
            _line("", f"{best}: ROC-AUC {m['roc_auc']:.2f}, acurácia {m['accuracy']:.2f}, F1 {m['f1']:.2f}")
        wer = audio.get("azure_stt_wer_by_group")
        if isinstance(wer, dict) and all(isinstance(v, dict) for v in wer.values()):
            _line("", "WER " + " | ".join(f"{g} {v['mean']:.2f} ({v['count']} falas)" for g, v in wer.items()))
        else:
            _line("", "WER não calculado (Azure Speech desativado)")
        _line("", f"{audio['consultations']} consultas | eventos {audio['events']}")

    video = _load_json("video_summary.json")
    if video:
        shown += 1
        mc = video.get("motion_capture") or {}
        if mc:
            _line("Captura de movimento", f"{mc.get('repetitions')} repetições, {mc.get('subjects')} participante(s) | "
                                         f"regra de ritmo: recall {mc.get('pace_rule_recall')}, "
                                         f"falso alarme {mc.get('pace_rule_false_positive_rate')}")
        for v in video.get("videos", []):
            _line("Vídeo", f"{v['video']} ({v['exercise']}): pose {v['pose_visible_fraction']:.0%}, {v['findings']} achado(s)")

    val = _load_json("video_validation.json")
    if val:
        shown += 1
        for name, res in val["exercises"].items():
            best = max(res["single_feature_auc"].items(), key=lambda kv: kv[1]["auc"])
            _line("Vídeo x gabarito", f"{name}: {res['repetitions']} repetições, {res['persons']} pessoas | "
                                     f"ROC-AUC {res['logistic_leave_one_person_out']['roc_auc']:.2f} | "
                                     f"melhor sinal {best[0]} ({best[1]['auc']:.2f})")
        counts = [c for c in val.get("repetition_counting", []) if c.get("detected_reps") is not None]
        if counts:
            det = sum(c["detected_reps"] for c in counts)
            gt = sum(c["ground_truth_reps"] for c in counts)
            worst = max(abs(c["detected_reps"] - c["ground_truth_reps"]) for c in counts)
            _line("Contagem", f"{det} contadas para {gt} do gabarito (diferença máxima {worst} por vídeo)")

    fus = _load_json("fusion_summary.json")
    if fus:
        shown += 1
        _line("Fusão", f"{fus['events']:,} eventos, {fus['patients_monitored']} pacientes "
                       f"({fus['virtual_patients']} virtuais)")
        _line("", f"{fus['alerts']:,} alertas {fus['alerts_by_level']} | {fus['multimodal_alerts']} multimodais")

    if audio and audio.get("azure_usage"):
        u = audio["azure_usage"]
        _line("Azure", f"{u['speech_seconds']:.0f} s de áudio e {u['language_records']} registros de texto "
                       f"em {u['month']}")

    figs = sorted((config.RESULTS_DIR / "figures").glob("*.png"))
    _line("Gráficos", f"{len(figs)} arquivo(s) em results/figures")

    if not shown:
        print("\nNenhum resultado encontrado. Rode antes: python main.py all")


def print_alerts(level: str | None = None, limit: int = 10):
    """Mostra a contagem de alertas por nível e os primeiros alertas da fila."""
    if not config.ALERTS_FILE.exists():
        raise SystemExit("Nenhum alerta encontrado. Rode antes: python main.py fusion")
    alerts = pd.read_csv(config.ALERTS_FILE)
    print("Alertas por nível:")
    for lv, n in alerts["level"].value_counts().items():
        print(f"  {lv:9} {n}")
    if level:
        alerts = alerts[alerts["level"] == level]
    # os mais graves primeiro; dentro do nível, maior pontuação primeiro
    order = {"critical": 0, "high": 1, "medium": 2}
    alerts = alerts.assign(_o=alerts["level"].map(order)).sort_values(["_o", "fused_score"], ascending=[True, False])
    print(f"\n{min(limit, len(alerts))} de {len(alerts)} alerta(s){f' {level}' if level else ''}:\n")
    for a in alerts.head(limit).itertuples():
        print(f"[{a.level.upper()}] {a.alert_id}  {a.patient_id}  {str(a.timestamp)[:16]}  risco {a.fused_score:.2f}  ({a.modalities})")
        for ev in str(a.top_events).split(" | "):
            print(f"    - {ev}")
        print(f"    ação: {a.recommended_action}\n")


def print_consultations():
    """Mostra, para cada consulta, a origem da transcrição, o sentimento e os termos encontrados."""
    data = _load_json("consultations.json")
    if not data:
        raise SystemExit("Nenhuma consulta analisada. Rode antes: python main.py audio")
    for c in data:
        print(f"{c['consultation_id']}: transcrição={c['transcript_source']} | sentimento={c['sentiment']} "
              f"({c['sentiment_source']})")
        print(f"    termos críticos: {c['critical_terms'] or '-'}")
        print(f"    negados:         {c['negated_terms'] or '-'}")

"""Orquestra o módulo de áudio: TORGO (voz) + Azure STT (amostra) + consultas (texto)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import joblib
import pandas as pd

from .. import config
from ..core import azure_client
from ..core.events import Event, save_events, clear_events
from . import consultations
from .features import extract_features
from .torgo_loader import iter_torgo
from .voice_model import train_dysarthria_classifier, vocal_indicators


def word_error_rate(ref: str, hyp: str) -> float:
    """WER = (substituições + inserções + remoções) / nº de palavras da referência."""
    r, h = ref.lower().split(), hyp.lower().replace(".", "").replace(",", "").replace("?", "").split()
    if not r:
        return 0.0
    d = list(range(len(h) + 1))                      # distância de Levenshtein por palavra
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def run_audio(torgo_rows: int = config.MAX_TORGO_ROWS) -> dict:
    rows, stt_rows = [], []
    per_group_budget = {"dysarthric": config.AZURE_STT_TORGO_SAMPLES // 2,
                        "control": config.AZURE_STT_TORGO_SAMPLES // 2}
    print(f"[audio] Extraindo características de até {torgo_rows} falas do TORGO...")
    for utt in iter_torgo(max_rows=torgo_rows):
        try:
            feats = extract_features(utt["audio"], utt["sr"])
        except ValueError:
            continue
        rows.append({k: utt[k] for k in ("utterance_id", "speaker_id", "group", "gender", "prompt")} | feats)

        # Azure STT só numa pequena amostra (economiza cota) e só se houver texto de referência
        if azure_client.speech_enabled() and utt["prompt"] and per_group_budget.get(utt["group"], 0) > 0 \
                and feats["duration_s"] < 55:
            res = azure_client.speech_to_text(azure_client.to_wav16k_bytes(utt["audio"], utt["sr"]))
            if res is not None:
                per_group_budget[utt["group"]] -= 1
                stt_rows.append({"utterance_id": utt["utterance_id"], "group": utt["group"],
                                 "prompt": utt["prompt"], "azure_text": res["text"],
                                 "wer": word_error_rate(utt["prompt"], res["text"])})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Nenhuma fala processada. Confira o arquivo Parquet do TORGO.")
    df.to_csv(config.RESULTS_DIR / "voice_features.csv", index=False)
    print(f"[audio] {len(df)} falas de {df['speaker_id'].nunique()} falantes")

    # --- 1) classificador de disartria ---
    clf = train_dysarthria_classifier(df)
    events: list[Event] = []
    base = datetime.fromisoformat(config.RELATIVE_EPOCH)
    if "error" not in clf:
        joblib.dump({"model": clf["model"], "features": clf["feature_columns"]},
                    config.MODELS_DIR / "dysarthria_classifier.joblib")
        per_spk = clf["per_utterance"].groupby("speaker_id")["dysarthria_proba"].mean()
        for spk, p in per_spk.items():
            if p >= 0.5:
                events.append(Event(f"torgo:{spk}", (base + timedelta(hours=9)).isoformat(), "audio",
                                    "speech_classifier", "dysarthria_risk", float(p),
                                    message=f"Speech pattern compatible with dysarthria (p={p:.2f})",
                                    evidence={"model": clf["best_model"]}))
        print(f"[audio] Classificador ({clf['best_model']}):", clf["metrics"][clf["best_model"]])

    # --- 2) indicadores de cansaço e respiração (relativos ao próprio falante) ---
    ind = vocal_indicators(df)
    ind.to_csv(config.RESULTS_DIR / "voice_indicators.csv", index=False)
    for k, r in enumerate(ind.itertuples()):
        ts = (base + timedelta(hours=9, minutes=k % 600)).isoformat()
        for col, etype, label in (("fatigue_index", "fatigue_indicator", "fatigue"),
                                  ("breathing_index", "breathing_difficulty_indicator", "breathing difficulty")):
            val = getattr(r, col)
            if val >= 1.5:
                events.append(Event(f"torgo:{r.speaker_id}", ts, "audio", "speech_features", etype,
                                    min(1.0, 0.35 + 0.15 * val),
                                    message=f"Voice markers of {label} vs. speaker baseline (index {val:.1f})",
                                    evidence={"utterance_id": r.utterance_id, col: round(val, 2)}))

    # --- 3) Azure STT: erro de transcrição por grupo (disartria piora o reconhecimento) ---
    stt_summary = {"status": "not_run (no Azure Speech key)" if not azure_client.speech_enabled()
                   else "not_run (AZURE_STT_TORGO_SAMPLES=0 or no utterances sent)"}
    if stt_rows:
        stt_df = pd.DataFrame(stt_rows)
        stt_df.to_csv(config.RESULTS_DIR / "torgo_azure_stt.csv", index=False)
        stt_summary = stt_df.groupby("group")["wer"].agg(["mean", "count"]).round(3).to_dict("index")

    # --- 4) consultas: STT + Text Analytics ---
    ev_consult, consult_rows = consultations.run_consultations()
    pd.DataFrame(consult_rows).to_json(config.RESULTS_DIR / "consultations.json", orient="records", indent=2)
    print(f"[audio] Consultas: {len(consult_rows)} analisadas -> {len(ev_consult)} eventos de texto")

    clear_events("audio")
    clear_events("text")
    save_events(events + ev_consult)

    summary = {
        "utterances": len(df), "speakers": int(df["speaker_id"].nunique()),
        "groups": df["group"].value_counts().to_dict(),
        "classifier": {k: v for k, v in clf.items() if k in ("best_model", "metrics", "top_features", "error")},
        "azure_stt_wer_by_group": stt_summary,
        "consultations": len(consult_rows),
        "events": {"audio": len(events), "text": len(ev_consult)},
        "azure_usage": azure_client.usage_summary(),
    }
    (config.RESULTS_DIR / "audio_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary

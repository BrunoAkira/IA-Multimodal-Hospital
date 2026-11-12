"""Orquestra o módulo clínico: carrega a base clínica, roda os 3 detectores e salva tudo."""
from __future__ import annotations

import json

from .. import config
from ..core.events import save_events, clear_events
from . import loader, vitals, prescriptions, transfers


def run_clinical(evaluate: bool = True) -> dict:
    print("[clinical] Lendo sinais vitais (chartevents em pedaços)...")
    v_long = loader.load_vitals()
    hourly = loader.hourly_vitals(v_long)
    hourly.to_csv(config.RESULTS_DIR / "vitals_hourly.csv", index=False)  # usado pelo dashboard
    print(f"[clinical] {len(v_long):,} medições -> {len(hourly):,} horas-paciente")

    ev_vitals = vitals.detect_vital_events(hourly)
    ev_multi, _ = vitals.detect_multivariate(hourly)
    print(f"[clinical] Vitais: {len(ev_vitals)} eventos por regra/fluxo, {len(ev_multi)} multivariados")

    print("[clinical] Analisando prescrições...")
    rx = loader.load_prescriptions()
    ev_rx, _ = prescriptions.detect_prescription_events(rx)
    print(f"[clinical] Prescrições: {len(rx):,} pedidos -> {len(ev_rx)} eventos")

    print("[clinical] Analisando transferências...")
    tr = loader.load_transfers()
    ev_tr, _ = transfers.detect_transfer_events(tr)
    print(f"[clinical] Transferências: {len(tr):,} registros -> {len(ev_tr)} eventos")

    clear_events("clinical")   # evita duplicar ao rodar de novo
    save_events(ev_vitals + ev_multi + ev_rx + ev_tr)

    summary = {
        "vital_measurements": len(v_long), "patient_hours": len(hourly),
        "events": {"vitals": len(ev_vitals), "vitals_multivariate": len(ev_multi),
                   "prescriptions": len(ev_rx), "transfers": len(ev_tr)},
    }
    if evaluate:
        summary["injection_evaluation"] = vitals.evaluate_with_injection(hourly)
        print("[clinical] Avaliação por injeção:", summary["injection_evaluation"])
    (config.RESULTS_DIR / "clinical_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

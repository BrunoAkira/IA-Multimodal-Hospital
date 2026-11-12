"""
Análise do texto transcrito: termos clínicos críticos + sentimento.

Caminho principal: Azure AI Language (Text Analytics)
    - analyze_health   -> entidades clínicas (sintomas, remédios...) e se foram NEGADAS
    - analyze_sentiment -> sentimento do paciente (positivo/neutro/negativo)
Proteção de cota (cota do mês esgotada): dicionário de termos + regra de negação
e um léxico simples de sentimento.
"""
from __future__ import annotations

import re

from ..core import azure_client

# Termos críticos (inglês) e peso de gravidade 0..1
CRITICAL_TERMS = {
    "chest pain": 0.9, "shortness of breath": 0.85, "short of breath": 0.85, "can't breathe": 0.95, "cannot breathe": 0.95,
    "difficulty breathing": 0.85, "breathless": 0.75, "wheezing": 0.6, "coughing blood": 0.9,
    "fainted": 0.85, "fainting": 0.85, "passed out": 0.85, "dizzy": 0.5, "dizziness": 0.5,
    "confusion": 0.75, "confused": 0.75, "slurred speech": 0.9, "trouble speaking": 0.85,
    "numbness": 0.7, "numb": 0.7, "weakness": 0.6, "face drooping": 0.95, "palpitations": 0.65,
    "fever": 0.5, "high fever": 0.7, "bleeding": 0.75, "black stools": 0.8, "severe pain": 0.75,
    "fatigue": 0.45, "exhausted": 0.5, "tired all the time": 0.5, "fall": 0.6, "fell": 0.6,
    "swelling": 0.45, "swollen": 0.45, "vomiting": 0.5, "seizure": 0.95, "suicidal": 1.0, "hopeless": 0.7,
}
# Categorias do Text Analytics for health que nos interessam
HEALTH_CATEGORIES = {"SymptomOrSign", "Diagnosis", "MedicationName", "Dosage", "ExaminationName"}

NEGATIONS = r"\b(no|not|denies|denied|without|never|negative for|free of)\b"
NEG_WORDS = {"pain", "worse", "bad", "afraid", "scared", "terrible", "awful", "hurt", "hurts", "tired",
             "weak", "sad", "anxious", "worried", "difficult", "can't", "cannot", "dizzy", "hopeless"}
POS_WORDS = {"better", "good", "fine", "great", "improving", "well", "relieved", "happy", "okay", "thanks"}


def _is_negated(text: str, start: int) -> bool:
    """Negado se uma palavra de negação aparece até 4 palavras antes do termo (na mesma frase)."""
    before = re.split(r"[.!?;]", text[max(0, start - 60):start])[-1].split()[-4:]
    return re.search(NEGATIONS, " ".join(before)) is not None


def local_critical_terms(text: str) -> list[dict]:
    t = text.lower()
    found, taken = [], []
    # termos mais longos primeiro: "high fever" ganha de "fever" no mesmo trecho
    for term in sorted(CRITICAL_TERMS, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", t):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue  # trecho já coberto por um termo maior
            taken.append((m.start(), m.end()))
            found.append({"text": term, "category": "SymptomOrSign", "weight": CRITICAL_TERMS[term],
                          "negated": _is_negated(t, m.start()), "source": "local_lexicon"})
    return found


def local_sentiment(text: str) -> dict:
    words = re.findall(r"[a-z']+", text.lower())
    neg, pos = sum(w in NEG_WORDS for w in words), sum(w in POS_WORDS for w in words)
    total = max(neg + pos, 1)
    label = "negative" if neg > pos * 1.5 else "positive" if pos > neg * 1.5 else "neutral"
    return {"sentiment": label, "scores": {"negative": neg / total, "positive": pos / total,
                                           "neutral": 1 - abs(neg - pos) / total},
            "source": "local_lexicon"}


def analyze_text(text: str) -> dict:
    """Retorna termos críticos e sentimento. Usa Azure quando disponível."""
    health = azure_client.analyze_health(text)
    if health is not None:
        terms = []
        for e in health["entities"]:
            if e["category"] not in HEALTH_CATEGORIES:
                continue
            # peso: termo do nosso dicionário (se bater) senão a confiança do Azure
            w = max((v for k, v in CRITICAL_TERMS.items() if k in e["text"].lower()),
                    default=0.4 * e["confidence"])
            terms.append({"text": e["text"], "category": e["category"], "weight": w,
                          "negated": e.get("assertion", {}).get("certainty", "").startswith("negative"),
                          "source": "azure_text_analytics_health"})
    else:
        terms = local_critical_terms(text)

    sentiment = azure_client.analyze_sentiment(text)
    if sentiment is not None:
        sentiment["source"] = "azure_sentiment"
    else:
        sentiment = local_sentiment(text)
    return {"terms": terms, "sentiment": sentiment}

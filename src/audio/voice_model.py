"""
Modelos sobre as características de voz.

1) Classificador de disartria (supervisionado, rótulos da base de fala)
   - Validação "deixa um falante de fora": o modelo é sempre testado em pessoas
     que ele não ouviu no treino, o que impede que memorize a voz de cada pessoa.
2) Indicadores de alteração vocal (não supervisionado, em relação ao próprio falante)
   - fatigue_indicator: fala mais lenta, mais monótona, mais pausas
   - breathing_indicator: mais pausas longas e voz mais soprosa (HNR menor)
   Comparamos cada fala com a "linha de base" do mesmo falante (mediana/MAD).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META_COLS = ["utterance_id", "speaker_id", "group", "gender", "prompt"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def train_dysarthria_classifier(df: pd.DataFrame) -> dict:
    data = df[df["group"].isin(["dysarthric", "control"])].reset_index(drop=True)
    # cada grupo precisa de 2+ falantes: ao deixar um de fora, o outro continua no treino
    per_group = data.groupby("group")["speaker_id"].nunique()
    if data["group"].nunique() < 2 or per_group.min() < 2:
        return {"error": "Poucos falantes/grupos para validar por falante"}
    X, y = data[feature_columns(data)], (data["group"] == "dysarthric").astype(int)
    groups = data["speaker_id"]
    cv = LeaveOneGroupOut()   # cada rodada testa um falante e treina com todos os outros

    models = {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                              random_state=42, n_jobs=-1)),
    }
    report, best_name, best_auc = {}, None, -1
    for name, model in models.items():
        proba = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y, proba)
        report[name] = {"accuracy": round(accuracy_score(y, pred), 3), "f1": round(f1_score(y, pred), 3),
                        "roc_auc": round(auc, 3), "confusion_matrix": confusion_matrix(y, pred).tolist()}
        if auc > best_auc:
            best_name, best_auc, best_proba = name, auc, proba

    # probabilidade "fora da amostra" de cada fala (o falante não estava no treino)
    data["dysarthria_proba"] = best_proba
    final = models[best_name].fit(X, y)
    importances = None
    if best_name == "random_forest":
        rf = final.named_steps["randomforestclassifier"]
        importances = (pd.Series(rf.feature_importances_, index=X.columns)
                         .sort_values(ascending=False).head(10).round(4).to_dict())
    return {"best_model": best_name, "metrics": report, "top_features": importances,
            "per_utterance": data[["utterance_id", "speaker_id", "group", "dysarthria_proba"]],
            "model": final, "feature_columns": list(X.columns)}


def _robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median() * 1.4826
    return (series - med) / (mad if mad > 1e-9 else (series.std() or 1.0))


def vocal_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Índices de cansaço e respiração relativos à linha de base de cada falante."""
    out = df[["utterance_id", "speaker_id", "group"]].copy()
    z = df.groupby("speaker_id")[["speech_rate_sps", "f0_std_st", "pause_ratio", "long_pauses",
                                  "hnr_db", "rms_cv", "voiced_fraction"]].transform(_robust_z)
    z = z.fillna(0).clip(-6, 6)
    # sinais "para cima" = pior. Ex.: fala mais lenta -> -speech_rate positivo
    out["fatigue_index"] = (-z["speech_rate_sps"] - z["f0_std_st"] + z["pause_ratio"] - z["voiced_fraction"]) / 4
    out["breathing_index"] = (z["long_pauses"] + z["pause_ratio"] - z["hnr_db"]) / 3
    return out

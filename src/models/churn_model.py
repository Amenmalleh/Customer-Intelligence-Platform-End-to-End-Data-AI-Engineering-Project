"""Prediction du churn Olist (Phase 7) : entraine et compare 4 modeles
supervises (LogisticRegression, RandomForest, XGBoost, LightGBM) sur
data/processed/olist_features.csv, optimise le modele gagnant et le
sauvegarde pour l'API (Phase 9).

Chaque etape est une fonction dediee, appelee dans l'ordre par
run_churn_prediction(), suivant le meme style que src/models/segmentation.py.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.feature_report import load_features
from src.ingestion import config
from src.models.segmentation import MODELS_DIR

# FEATURES A UTILISER : la consigne initiale de la Phase 7 incluait
# recency_days. Teste isolement, ce jeu de features fait monter recall /
# precision / f1 / ROC-AUC a 1.0 (ou quasi, ~0.998 pour XGBoost) sur les 4
# modeles : churn est defini comme (recency_days > 180), donc recency_days
# permet de reconstruire churn par une simple regle de seuil au lieu d'un
# vrai signal comportemental appris (le meme risque de fuite deja documente
# dans docs/feature_engineering_report.txt en Phase 5 et docs/segmentation_report.txt
# en Phase 6). Sur decision explicite de l'utilisateur, recency_days est donc
# exclu de X ci-dessous ; sans lui, les scores redeviennent realistes
# (ROC-AUC ~0.54-0.70 selon le modele), un signal faible mais genuine.
FEATURE_COLUMNS = [
    "frequency",
    "monetary_log",
    "avg_review_score",
    "customer_tenure_days",
    "avg_days_between_orders",
    "state_encoded",
    "category_encoded",
]
TARGET_COLUMN = "churn"

CHURN_MODEL_FILE = MODELS_DIR / "churn_model.pkl"
CHURN_MODEL_METADATA_FILE = MODELS_DIR / "churn_model_metadata.json"
ROC_CURVES_PLOT = config.DOCS_DIR / "roc_curves.png"
FEATURE_IMPORTANCE_PLOT = config.DOCS_DIR / "feature_importance.png"
MODEL_COMPARISON_REPORT_FILE = config.DOCS_DIR / "model_comparison_report.txt"

RANDOM_STATE = 42


def prepare_data(df: pd.DataFrame):
    """ETAPE 1 : construit X/y, verifie l'absence de NaN, split stratifie
    80/20, et calcule scale_pos_weight (utilise par XGBoost/LightGBM pour
    compenser le desequilibre de classes sans sur-echantillonner)."""
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN detecte dans les features d'entree : {na_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    n_churn0 = (y_train == 0).sum()
    n_churn1 = (y_train == 1).sum()
    scale_pos_weight = n_churn0 / n_churn1

    print(f"[ETAPE 1] X: {X.shape[1]} features, y: '{TARGET_COLUMN}' | aucune NaN detectee")
    print(f"[ETAPE 1] Split 80/20 stratifie : train={X_train.shape[0]}, test={X_test.shape[0]}")
    print(f"[ETAPE 1] scale_pos_weight (train) = {n_churn0}/{n_churn1} = {scale_pos_weight:.4f}")
    print(f"[ETAPE 1] Distribution churn (train) :\n{(y_train.value_counts(normalize=True) * 100).round(2).to_string()}")
    print(f"[ETAPE 1] Distribution churn (test) :\n{(y_test.value_counts(normalize=True) * 100).round(2).to_string()}")

    return X_train, X_test, y_train, y_test, scale_pos_weight


def run_churn_prediction():
    print(f"=== Churn prediction Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    df = load_features()
    print(f"Chargement : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    X_train, X_test, y_train, y_test, scale_pos_weight = prepare_data(df)

    return X_train, X_test, y_train, y_test, scale_pos_weight


if __name__ == "__main__":
    run_churn_prediction()

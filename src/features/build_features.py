"""Feature engineering Olist (Phase 5) : charge
data/processed/olist_customers_clean.csv, calcule le label churn, les scores
RFM, encode les variables categorielles, normalise pour le clustering, et
sauvegarde les datasets finaux (olist_features.csv / olist_features_scaled.csv).

Chaque etape est une fonction pure (df -> df) appelee dans l'ordre par
run_feature_engineering(), suivant le meme style que src/cleaning/clean.py.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda.analysis import load_customers_clean
from src.ingestion import config

ENCODERS_DIR = config.BASE_DIR / "models" / "encoders"
FEATURES_FILE = config.PROCESSED_DATA_DIR / "olist_features.csv"
FEATURES_SCALED_FILE = config.PROCESSED_DATA_DIR / "olist_features_scaled.csv"

SCALED_COLUMNS = ["recency_scaled", "frequency_scaled", "monetary_scaled", "tenure_scaled", "review_scaled"]


def compute_recency_and_churn(df: pd.DataFrame, churn_threshold_days: int = config.CHURN_THRESHOLD_DAYS) -> pd.DataFrame:
    """ETAPE 1 : calcule recency_days et le label churn.

    date_reference = max(last_order_date), pas datetime.now() : le dataset
    Olist s'arrete en 2018, utiliser la date du jour ferait passer la quasi
    totalite des clients pour churned (cf. Phase 4 EDA). churn_threshold_days
    reutilise config.CHURN_THRESHOLD_DAYS (180) par defaut pour rester
    coherent avec le seuil deja utilise dans src/eda/insights.py.
    """
    df = df.copy()

    date_reference = df["last_order_date"].max()
    df["recency_days"] = (date_reference - df["last_order_date"]).dt.days
    df["churn"] = (df["recency_days"] > churn_threshold_days).astype(int)

    churn_counts = df["churn"].value_counts(normalize=True).mul(100).round(2)
    print(f"[ETAPE 1] date_reference = {date_reference.date()} (seuil churn = {churn_threshold_days} jours)")
    print(f"[ETAPE 1] churn=0 : {churn_counts.get(0, 0.0)}% | churn=1 : {churn_counts.get(1, 0.0)}%")

    return df


def run_feature_engineering() -> pd.DataFrame:
    print(f"=== Feature engineering Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    df = load_customers_clean()
    print(f"Chargement : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    df = compute_recency_and_churn(df)

    return df


if __name__ == "__main__":
    run_feature_engineering()

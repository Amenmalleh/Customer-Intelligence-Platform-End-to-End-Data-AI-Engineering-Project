"""Nettoyage et transformation Olist (Phase 3) : charge
data/processed/olist_merged.csv, applique une sequence fixe de nettoyage
(filtrage, colonnes, types, valeurs manquantes), agrege par client et
sauvegarde data/processed/olist_customers_clean.csv.

Ne calcule aucune feature ML ici (pas de churn, pas de RFM) : c'est l'objet
de la Phase 5.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion import config

CUSTOMERS_CLEAN_FILE = config.PROCESSED_DATA_DIR / "olist_customers_clean.csv"


def load_merged() -> pd.DataFrame:
    if not config.MERGED_OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"{config.MERGED_OUTPUT_FILE} introuvable. Lance d'abord src/ingestion/ingest.py."
        )
    return pd.read_csv(config.MERGED_OUTPUT_FILE, encoding="utf-8-sig")


def filter_orders(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 1 : ne garde que les commandes livrees avec une date d'achat connue.

    Seules les commandes 'delivered' ont un cycle de vie complet exploitable
    pour la segmentation client. Une commande sans order_purchase_timestamp
    est inexploitable pour toute metrique temporelle (RFM en Phase 5).
    """
    before = len(df)
    filtered = df[df["order_status"] == "delivered"].copy()
    filtered = filtered.dropna(subset=["order_purchase_timestamp"])
    after = len(filtered)
    print(f"[ETAPE 1] Filtrage 'delivered' + order_purchase_timestamp non-null : {before} -> {after} lignes")
    return filtered


def run_cleaning() -> pd.DataFrame:
    print(f"=== Nettoyage Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    df = load_merged()
    print(f"Chargement : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    df = filter_orders(df)

    return df


if __name__ == "__main__":
    run_cleaning()

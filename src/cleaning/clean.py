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


UNUSED_COLUMNS = [
    "review_comment_title",
    "review_comment_message",
    "order_approved_at",
    "order_delivered_carrier_date",
]


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 2 : supprime les colonnes trop creuses ou hors scope ML.

    review_comment_title (88% NaN) et review_comment_message (58% NaN) sont
    du texte libre non structure, trop creux pour etre impute sans biais.
    order_approved_at et order_delivered_carrier_date sont des etapes
    intermediaires du cycle de livraison, redondantes avec
    order_delivered_customer_date / order_estimated_delivery_date pour la
    suite du projet (segmentation, churn).
    """
    before_cols = df.shape[1]
    dropped = [c for c in UNUSED_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"[ETAPE 2] Suppression colonnes : {before_cols} -> {df.shape[1]} colonnes | supprimees : {dropped}")
    return df


def run_cleaning() -> pd.DataFrame:
    print(f"=== Nettoyage Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    df = load_merged()
    print(f"Chargement : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    df = filter_orders(df)
    df = drop_unused_columns(df)

    return df


if __name__ == "__main__":
    run_cleaning()

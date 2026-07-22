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


DATETIME_COLUMNS = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "review_creation_date",
]


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 3 : convertit les colonnes de dates en datetime.

    errors='coerce' transforme un format invalide en NaT plutot que de
    lever une exception, dans le meme esprit que le reste du pipeline
    (logguer, ne pas planter).
    """
    df = df.copy()
    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    print(f"[ETAPE 3] Conversion en datetime (errors='coerce') : {DATETIME_COLUMNS}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 4 : traite les valeurs manquantes restantes, dans un ordre precis.

    L'ordre compte : le drop sur price doit precede le fillna de
    payment_value pour garantir qu'on ne comble jamais avec un NaN.

    - review_score NaN -> mediane (resiste mieux qu'une moyenne a la
      distribution asymetrique des scores, cf. docs/data_quality_report.txt).
    - order_delivered_customer_date NaN -> inchange (commande en transit
      legitime, pas une erreur ; imputer casserait le signal de delai de
      livraison).
    - price NaN -> lignes supprimees (un prix manquant sur une ligne
      order_item signale une jointure corrompue, pas une valeur recuperable).
    - payment_value NaN -> comble avec price (des lors que price n'est
      jamais NaN a ce stade).
    """
    df = df.copy()

    review_score_median = df["review_score"].median()
    n_review_score_na = df["review_score"].isna().sum()
    df["review_score"] = df["review_score"].fillna(review_score_median)
    print(f"[ETAPE 4] review_score : {n_review_score_na} NaN combles avec la mediane ({review_score_median})")

    n_delivery_na = df["order_delivered_customer_date"].isna().sum()
    print(f"[ETAPE 4] order_delivered_customer_date : {n_delivery_na} NaN laisses tels quels (commandes en transit)")

    before = len(df)
    df = df.dropna(subset=["price"])
    print(f"[ETAPE 4] price : {before - len(df)} lignes supprimees (prix manquant) -> {len(df)} lignes")

    n_payment_na = df["payment_value"].isna().sum()
    df["payment_value"] = df["payment_value"].fillna(df["price"])
    print(f"[ETAPE 4] payment_value : {n_payment_na} NaN combles avec price")

    return df


def _most_frequent(s: pd.Series):
    """Mode d'une serie, ou NA si le groupe est entierement vide (evite un
    crash sur .mode().iloc[0] quand un client n'a aucune categorie connue)."""
    s = s.dropna()
    if s.empty:
        return pd.NA
    return s.mode().iloc[0]


def aggregate_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 5 : deduplication intelligente -> une ligne par customer_unique_id.

    La table source a du fan-out (plusieurs lignes par order_id a cause des
    joins order_items/order_payments/order_reviews). On agrege donc au
    niveau client plutot que de dedupliquer naivement.
    """
    before = len(df)
    n_customers_before = df["customer_unique_id"].nunique()

    aggregated = df.groupby("customer_unique_id").agg(
        total_orders=("order_id", "nunique"),
        total_spent=("payment_value", "sum"),
        avg_review_score=("review_score", "mean"),
        first_order_date=("order_purchase_timestamp", "min"),
        last_order_date=("order_purchase_timestamp", "max"),
        most_frequent_category=("product_category_name_english", _most_frequent),
        customer_state=("customer_state", "first"),
    ).reset_index()

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(CUSTOMERS_CLEAN_FILE, index=False, encoding="utf-8-sig")

    print(
        f"[ETAPE 5] Agregation par customer_unique_id : {before} lignes "
        f"({n_customers_before} clients) -> {len(aggregated)} clients"
    )
    print(f"[ETAPE 5] Sauvegarde : {CUSTOMERS_CLEAN_FILE}")
    return aggregated


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 6 : detecte (sans supprimer) les outliers sur total_spent et
    total_orders au-dela de Q3 + 3*IQR, un seuil large qui ne signale que les
    valeurs vraiment extremes (gros comptes / achats en volume) plutot que la
    variabilite normale des clients."""
    for col in ("total_spent", "total_orders"):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + 3 * iqr
        outliers = df[df[col] > threshold].sort_values(col, ascending=False)
        print(
            f"[ETAPE 6] {col} : Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} "
            f"seuil={threshold:.2f} -> {len(outliers)} outliers detectes"
        )
        if not outliers.empty:
            print(f"[ETAPE 6] {col} : 20 outliers les plus extremes (sur {len(outliers)}) :")
            print(outliers.head(20).to_string())

    return df


def run_cleaning() -> pd.DataFrame:
    print(f"=== Nettoyage Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    df = load_merged()
    print(f"Chargement : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    df = filter_orders(df)
    df = drop_unused_columns(df)
    df = convert_dtypes(df)
    df = handle_missing_values(df)
    df = aggregate_by_customer(df)
    df = detect_outliers(df)

    return df


if __name__ == "__main__":
    run_cleaning()

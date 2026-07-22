"""Ingestion Olist (Phase 2) : charge les 8 CSV bruts, les merge en un seul
dataset et sauvegarde le resultat dans data/processed/olist_merged.csv.

Aucun nettoyage n'est effectue ici : c'est un merge brut, pense pour etre
observe puis nettoye dans une phase ulterieure.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Permet d'executer ce script directement (`python src/ingestion/ingest.py`)
# tout en gardant des imports absolus stables quand le module est importe
# depuis ailleurs (notebook, tests, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion import config


def load_csv(name: str, filename: str) -> pd.DataFrame:
    """Charge un CSV brut en verifiant son existence et son schema minimal."""
    path = config.RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier attendu introuvable pour la table '{name}' : {path}. "
            f"Verifie que le dataset Olist complet est present dans {config.RAW_DATA_DIR}."
        )

    # encoding="utf-8-sig" : product_category_name_translation.csv contient un
    # BOM en tete de fichier qui, sinon, se retrouverait colle au nom de la
    # premiere colonne ("﻿product_category_name") et casserait le merge
    # sur product_category_name. Ce parametre est sans effet sur les fichiers
    # sans BOM, donc on l'applique uniformement aux 8 fichiers.
    df = pd.read_csv(path, encoding="utf-8-sig")

    missing_critical = [c for c in config.CRITICAL_COLUMNS.get(name, []) if c not in df.columns]
    if missing_critical:
        print(f"  [WARN] '{name}' : colonnes critiques manquantes -> {missing_critical}")

    print(f"[{name}] shape={df.shape} colonnes={list(df.columns)}")
    return df


def load_all_tables() -> dict[str, pd.DataFrame]:
    """Charge les 8 tables brutes definies dans config.CSV_FILES."""
    return {name: load_csv(name, filename) for name, filename in config.CSV_FILES.items()}


def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge LEFT JOIN dans l'ordre impose par la Phase 2.

    orders -> customers (customer_id) -> order_items (order_id)
    -> order_payments (order_id) -> order_reviews (order_id)
    -> products (product_id) -> category_translation (product_category_name)

    Note : 'sellers' est chargee et loggee (elle fait partie des 8 fichiers
    attendus) mais n'est volontairement PAS mergee ici, cet ordre ne le
    prevoyant pas. order_items porte deja seller_id si un merge est
    necessaire dans une phase ulterieure.
    """
    merged = tables["orders"].merge(tables["customers"], on="customer_id", how="left")
    merged = merged.merge(tables["order_items"], on="order_id", how="left")
    merged = merged.merge(tables["order_payments"], on="order_id", how="left")
    merged = merged.merge(tables["order_reviews"], on="order_id", how="left")
    merged = merged.merge(tables["products"], on="product_id", how="left")
    merged = merged.merge(tables["category_translation"], on="product_category_name", how="left")
    return merged


def run_ingestion() -> pd.DataFrame:
    print(f"=== Ingestion Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.MERGED_OUTPUT_FILE.exists():
        print(
            f"[idempotent] {config.MERGED_OUTPUT_FILE} existe deja -> ingestion non relancee. "
            "Supprime ce fichier pour forcer une regeneration."
        )
        return pd.read_csv(config.MERGED_OUTPUT_FILE, encoding="utf-8-sig")

    print("--- Chargement des tables brutes ---")
    tables = load_all_tables()

    print("\n--- Merge (LEFT JOIN) ---")
    merged = merge_tables(tables)
    print(f"Lignes apres merge : {merged.shape[0]} | Colonnes : {merged.shape[1]}")

    merged.to_csv(config.MERGED_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    file_size_mb = config.MERGED_OUTPUT_FILE.stat().st_size / (1024 ** 2)

    print("\n--- Resume ---")
    print(f"Date : {datetime.now().isoformat(timespec='seconds')}")
    print("Lignes par table (brutes) :")
    for name, df in tables.items():
        print(f"  - {name}: {df.shape[0]} lignes")
    print(f"Lignes apres merge : {merged.shape[0]}")
    print(f"Fichier final : {config.MERGED_OUTPUT_FILE} ({file_size_mb:.2f} Mo)")

    return merged


if __name__ == "__main__":
    run_ingestion()

"""Configuration centralisee pour le pipeline d'ingestion Olist (Phase 2).

Toute constante partagee entre ingest.py et validate.py (chemins, noms de
fichiers, colonnes attendues) vit ici pour eviter les valeurs en dur
dupliquees entre les scripts.
"""

from pathlib import Path

# src/ingestion/config.py -> parents[2] = racine du projet
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

MERGED_OUTPUT_FILE = PROCESSED_DATA_DIR / "olist_merged.csv"
DATA_QUALITY_REPORT = DOCS_DIR / "data_quality_report.txt"

# Les 8 fichiers CSV bruts attendus dans data/raw/, mappes a un nom de
# variable utilise comme cle dans le reste du pipeline (ingest.py, validate.py).
CSV_FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Colonnes critiques attendues par table : sert de garde-fou minimal pour
# detecter tot un changement de schema dans le dataset source.
CRITICAL_COLUMNS = {
    "customers": ["customer_id", "customer_unique_id", "customer_city", "customer_state"],
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    "order_items": ["order_id", "product_id", "seller_id", "price"],
    "order_payments": ["order_id", "payment_type", "payment_value"],
    "order_reviews": ["order_id", "review_score"],
    "products": ["product_id", "product_category_name"],
    "sellers": ["seller_id", "seller_city", "seller_state"],
    "category_translation": ["product_category_name", "product_category_name_english"],
}

# Nombre de jours sans nouvelle commande au-dela duquel un client est
# considere churn (utilise dans les phases ulterieures de feature engineering).
CHURN_THRESHOLD_DAYS = 180

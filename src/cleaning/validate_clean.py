"""Validation du dataset client nettoye (Phase 3).

Charge data/processed/olist_customers_clean.csv et verifie les invariants
attendus apres l'agregation par customer_unique_id (src/cleaning/clean.py).
Leve une AssertionError explicite si une verification echoue.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cleaning import clean

CRITICAL_COLUMNS = [
    "customer_unique_id",
    "total_orders",
    "total_spent",
    "last_order_date",
    "customer_state",
]


def load_clean() -> pd.DataFrame:
    if not clean.CUSTOMERS_CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"{clean.CUSTOMERS_CLEAN_FILE} introuvable. Lance d'abord src/cleaning/clean.py."
        )
    return pd.read_csv(
        clean.CUSTOMERS_CLEAN_FILE,
        encoding="utf-8-sig",
        parse_dates=["first_order_date", "last_order_date"],
    )


def check_no_missing_critical(df: pd.DataFrame) -> None:
    for col in CRITICAL_COLUMNS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            raise AssertionError(
                f"Colonne critique '{col}' contient {n_missing} valeur(s) manquante(s), attendu 0."
            )


def check_total_orders_positive(df: pd.DataFrame) -> None:
    invalid = df[df["total_orders"] < 1]
    if not invalid.empty:
        raise AssertionError(
            f"{len(invalid)} ligne(s) ont total_orders < 1 (attendu >= 1 pour toutes les lignes)."
        )


def check_total_spent_positive(df: pd.DataFrame) -> None:
    invalid = df[df["total_spent"] <= 0]
    if not invalid.empty:
        raise AssertionError(
            f"{len(invalid)} ligne(s) ont total_spent <= 0 (attendu > 0 pour toutes les lignes)."
        )


def run_validation() -> None:
    df = load_clean()

    print(f"=== Validation olist_customers_clean.csv ===")
    print(f"Shape : {df.shape}")

    check_no_missing_critical(df)
    check_total_orders_positive(df)
    check_total_spent_positive(df)
    print("Toutes les verifications sont passees.")

    print("\n--- Statistiques descriptives ---")
    print(df.describe(include="all").T)

    print("\n--- Top 10 etats par nombre de clients ---")
    print(df["customer_state"].value_counts().head(10))


if __name__ == "__main__":
    run_validation()

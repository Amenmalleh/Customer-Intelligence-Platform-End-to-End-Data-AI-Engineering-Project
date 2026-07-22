"""Validation qualite des donnees Olist (Phase 2).

Charge data/processed/olist_merged.csv, calcule des indicateurs de qualite
(valeurs manquantes, doublons, types) et ecrit un rapport lisible dans
docs/data_quality_report.txt. Aucune correction/nettoyage n'est effectue ici.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion import config


def load_merged() -> pd.DataFrame:
    if not config.MERGED_OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"{config.MERGED_OUTPUT_FILE} introuvable. Lance d'abord src/ingestion/ingest.py."
        )
    return pd.read_csv(config.MERGED_OUTPUT_FILE, encoding="utf-8-sig")


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre et pourcentage de valeurs manquantes par colonne, tri decroissant."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    report = pd.DataFrame(
        {
            "colonne": missing_count.index,
            "valeurs_manquantes": missing_count.values,
            "pourcentage": missing_pct.values,
        }
    )
    return report.sort_values("pourcentage", ascending=False).reset_index(drop=True)


def count_duplicate_order_ids(df: pd.DataFrame) -> int:
    return int(df["order_id"].duplicated().sum())


def build_report_text(df: pd.DataFrame, missing: pd.DataFrame, dup_count: int) -> str:
    lines = [
        "RAPPORT DE QUALITE DES DONNEES - Olist (dataset merge)",
        f"Date du rapport : {datetime.now().isoformat(timespec='seconds')}",
        f"Fichier analyse : {config.MERGED_OUTPUT_FILE}",
        f"Shape : {df.shape[0]} lignes x {df.shape[1]} colonnes",
        "",
        "Valeurs manquantes par colonne (triees par % decroissant) :",
    ]
    for _, row in missing.iterrows():
        lines.append(f"  - {row['colonne']}: {int(row['valeurs_manquantes'])} ({row['pourcentage']}%)")

    lines += [
        "",
        f"Doublons sur order_id : {dup_count}",
        "",
        "Observations generales :",
        (
            "  - Les doublons sur order_id sont attendus a ce stade : une commande "
            "peut avoir plusieurs articles (order_items), plusieurs versements de "
            "paiement (order_payments) ou plusieurs avis (order_reviews), ce qui "
            "provoque un fan-out lors du merge. Ce n'est pas une anomalie de "
            "qualite en soi, mais un point a garder en tete pour l'agregation "
            "au niveau commande/client dans les phases suivantes."
        ),
    ]

    top_missing = missing[missing["valeurs_manquantes"] > 0].head(5)
    if not top_missing.empty:
        cols = ", ".join(top_missing["colonne"].tolist())
        lines.append(f"  - Colonnes avec le plus de valeurs manquantes : {cols}.")
    else:
        lines.append("  - Aucune valeur manquante detectee sur l'ensemble des colonnes.")

    lines.append(
        "  - Aucune transformation ni nettoyage n'a ete effectue a ce stade "
        "(Phase 2 = ingestion, merge et observation uniquement)."
    )

    return "\n".join(lines)


def run_validation() -> None:
    df = load_merged()

    print(f"=== Validation qualite - olist_merged.csv - {datetime.now().isoformat(timespec='seconds')} ===")
    print(f"Shape : {df.shape}")

    print("\n--- Types de donnees ---")
    print(df.dtypes)

    missing = missing_values_report(df)
    print("\n--- Valeurs manquantes (triees par % decroissant) ---")
    print(missing.to_string(index=False))

    dup_count = count_duplicate_order_ids(df)
    print(f"\nDoublons sur order_id : {dup_count}")

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = build_report_text(df, missing, dup_count)
    config.DATA_QUALITY_REPORT.write_text(report_text, encoding="utf-8")

    print(f"\nRapport ecrit dans {config.DATA_QUALITY_REPORT}")


if __name__ == "__main__":
    run_validation()

"""Rapport de feature engineering Olist (Phase 5).

Charge data/processed/olist_features.csv et ecrit un rapport lisible dans
docs/feature_engineering_report.txt : liste des features, distribution du
churn, statistiques descriptives des features RFM cles, correlation de
chaque feature numerique avec le churn, et une recommandation sur le
desequilibre de classes.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.build_features import FEATURES_FILE
from src.ingestion import config

FEATURE_REPORT_FILE = config.DOCS_DIR / "feature_engineering_report.txt"

# Description manuelle de chaque feature du dataset olist_features.csv :
# le type y est lu depuis le DataFrame au moment du run (cf. build_report_text),
# seule la description business/technique est figee ici.
FEATURE_DESCRIPTIONS = {
    "customer_unique_id": "Identifiant unique du client (cle primaire).",
    "total_orders": "Nombre de commandes distinctes passees par le client.",
    "total_spent": "Somme des payment_value sur toutes les commandes du client.",
    "avg_review_score": "Moyenne des review_score du client (1-5).",
    "first_order_date": "Date de la premiere commande du client.",
    "last_order_date": "Date de la derniere commande du client.",
    "most_frequent_category": "Categorie de produit la plus achetee par le client.",
    "customer_state": "Etat brasilien du client.",
    "recency_days": "Nombre de jours entre last_order_date et la date de reference (max(last_order_date)).",
    "churn": "Label cible : 1 si recency_days > seuil de churn (180j par defaut), sinon 0.",
    "frequency": "Alias RFM de total_orders.",
    "monetary_log": "log(total_spent + 1) : version log-transformee de la depense pour attenuer l'asymetrie.",
    "customer_tenure_days": "Nombre de jours entre first_order_date et last_order_date.",
    "avg_days_between_orders": "customer_tenure_days / (frequency - 1), 0 si une seule commande.",
    "R_score": "Score de recency en quartile (1=pire, 4=meilleur), inverse.",
    "F_score": "Score de frequency en quartile base sur le rang (1=pire, 4=meilleur).",
    "M_score": "Score de monetary_log en quartile (1=pire, 4=meilleur).",
    "rfm_score": "Moyenne des R_score/F_score/M_score.",
    "state_encoded": "Encodage LabelEncoder de customer_state.",
    "category_encoded": "Encodage LabelEncoder de most_frequent_category (NaN -> 'unknown').",
    "recency_scaled": "recency_days apres StandardScaler.",
    "frequency_scaled": "frequency apres StandardScaler.",
    "monetary_scaled": "monetary_log apres StandardScaler.",
    "tenure_scaled": "customer_tenure_days apres StandardScaler.",
    "review_scaled": "avg_review_score apres StandardScaler.",
}

KEY_NUMERIC_FEATURES = ["recency_days", "frequency", "monetary_log", "rfm_score"]


def load_features() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURES_FILE} introuvable. Lance d'abord src/features/build_features.py."
        )
    return pd.read_csv(FEATURES_FILE, encoding="utf-8-sig", parse_dates=["first_order_date", "last_order_date"])


def churn_correlations(df: pd.DataFrame) -> pd.Series:
    numeric_df = df.select_dtypes(include="number")
    correlations = numeric_df.corr()["churn"].drop("churn")
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def build_report_text(df: pd.DataFrame) -> str:
    lines = [
        "RAPPORT DE FEATURE ENGINEERING - Olist (Phase 5)",
        f"Date de generation : {datetime.now().isoformat(timespec='seconds')}",
        f"Fichier analyse : {FEATURES_FILE}",
        f"Shape : {df.shape[0]} lignes x {df.shape[1]} colonnes",
        "",
        "Liste des features :",
    ]
    for col in df.columns:
        description = FEATURE_DESCRIPTIONS.get(col, "(description non renseignee)")
        lines.append(f"  - {col} ({df[col].dtype}) : {description}")

    churn_dist = df["churn"].value_counts(normalize=True).mul(100).round(2)
    lines += [
        "",
        "Distribution du label churn :",
        f"  - churn=0 : {churn_dist.get(0, 0.0)}%",
        f"  - churn=1 : {churn_dist.get(1, 0.0)}%",
        "",
        f"Statistiques descriptives ({', '.join(KEY_NUMERIC_FEATURES)}) :",
        df[KEY_NUMERIC_FEATURES].describe().to_string(),
        "",
        "Correlation de chaque feature numerique avec churn (triee par |correlation| decroissante) :",
    ]
    for feature, corr in churn_correlations(df).items():
        lines.append(f"  - {feature}: {corr:.4f}")

    lines += [
        "",
        "Recommandation sur le desequilibre de classes :",
        (
            f"  - La repartition churn=0 ({churn_dist.get(0, 0.0)}%) / churn=1 "
            f"({churn_dist.get(1, 0.0)}%) est un desequilibre modere (~60/40), pas "
            "un cas extreme (ex. fraude a 1%). SMOTE ou un sur-echantillonnage "
            "agressif ne sont donc pas necessaires a ce stade."
        ),
        (
            "  - Un modele naif qui predirait toujours churn=1 atteindrait deja "
            f"~{churn_dist.get(1, 0.0)}% d'accuracy : l'accuracy seule sera trompeuse "
            "pour evaluer le modele de la Phase 7. Privilegier F1-score, "
            "precision/recall par classe et ROC-AUC."
        ),
        (
            "  - Utiliser un split train/test stratifie sur churn, et envisager "
            "class_weight='balanced' (ou equivalent) dans le modele de "
            "classification plutot qu'un rebalancement des donnees."
        ),
    ]

    return "\n".join(lines)


def run_feature_report() -> None:
    df = load_features()

    print(f"=== Rapport feature engineering - {datetime.now().isoformat(timespec='seconds')} ===")
    print(f"Shape : {df.shape}")

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = build_report_text(df)
    FEATURE_REPORT_FILE.write_text(report_text, encoding="utf-8")

    print(f"\nRapport ecrit dans {FEATURE_REPORT_FILE}")


if __name__ == "__main__":
    run_feature_report()

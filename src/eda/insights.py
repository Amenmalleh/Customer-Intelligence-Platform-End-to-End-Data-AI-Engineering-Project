"""KPI business Olist (Phase 4) : calcule les indicateurs cles a partir de
data/processed/olist_customers_clean.csv et les sauvegarde en texte dans
docs/eda_insights.txt.

Aucune feature ML (churn 0/1, RFM) n'est ecrite dans le dataset ici : ce
module ne fait qu'afficher/sauvegarder des KPI, la variable churn sera
calculee en Phase 5.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda.analysis import load_customers_clean
from src.ingestion import config

DOCS_DIR = config.DOCS_DIR
INSIGHTS_REPORT_FILE = DOCS_DIR / "eda_insights.txt"

SPEND_SEGMENT_BINS = [-float("inf"), 50, 200, 500, float("inf")]
SPEND_SEGMENT_LABELS = ["low", "medium", "high", "very_high"]


def compute_kpis(df: pd.DataFrame) -> dict:
    """Calcule les KPI business a partir du DataFrame client agrege.

    Le seuil de churn (recency > CHURN_THRESHOLD_DAYS, 180 jours) reutilise la
    constante deja definie dans src/ingestion/config.py pour rester coherent
    avec la definition qui sera utilisee en Phase 5 (feature churn 0/1). La
    date de reference est max(last_order_date) plutot que la date du jour,
    car le dataset est historique (derniere commande en 2018) : utiliser
    "aujourd'hui" ferait passer 100% des clients pour churned.
    """
    reference_date = df["last_order_date"].max()
    recency_days = (reference_date - df["last_order_date"]).dt.days
    churned = recency_days > config.CHURN_THRESHOLD_DAYS

    spend_segments = pd.cut(
        df["total_spent"], bins=SPEND_SEGMENT_BINS, labels=SPEND_SEGMENT_LABELS, right=False
    ).value_counts().reindex(SPEND_SEGMENT_LABELS)

    return {
        "reference_date": reference_date,
        "pct_one_order": (df["total_orders"] == 1).mean() * 100,
        "pct_churned": churned.mean() * 100,
        "median_spent": df["total_spent"].median(),
        "avg_review_score": df["avg_review_score"].mean(),
        "top3_categories": df["most_frequent_category"].value_counts().head(3),
        "top3_states": df["customer_state"].value_counts().head(3),
        "spend_segments": spend_segments,
    }


def format_report(kpis: dict) -> str:
    lines = [
        "KPI BUSINESS - Olist (dataset client agrege, Phase 4 EDA)",
        f"Date du rapport : {datetime.now().isoformat(timespec='seconds')}",
        f"Date de reference (max last_order_date) : {kpis['reference_date'].date()}",
        "",
        f"% clients avec exactement 1 commande : {kpis['pct_one_order']:.2f}%",
        (
            f"% clients churned (recency > {config.CHURN_THRESHOLD_DAYS} jours "
            f"depuis la date de reference) : {kpis['pct_churned']:.2f}%"
        ),
        f"Montant median depense (total_spent) : {kpis['median_spent']:.2f}",
        f"Score de satisfaction moyen (avg_review_score) : {kpis['avg_review_score']:.2f}",
        "",
        "Top 3 categories les plus achetees :",
    ]
    for category, count in kpis["top3_categories"].items():
        lines.append(f"  - {category}: {count} clients")

    lines.append("")
    lines.append("Top 3 etats par nombre de clients :")
    for state, count in kpis["top3_states"].items():
        lines.append(f"  - {state}: {count} clients")

    lines.append("")
    lines.append("Nombre de clients par segment de depense :")
    for segment, count in kpis["spend_segments"].items():
        lines.append(f"  - {segment}: {int(count)} clients")

    return "\n".join(lines)


def run_insights() -> None:
    df = load_customers_clean()
    kpis = compute_kpis(df)
    report_text = format_report(kpis)

    print(report_text)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    INSIGHTS_REPORT_FILE.write_text(report_text, encoding="utf-8")
    print(f"\nRapport ecrit dans {INSIGHTS_REPORT_FILE}")


if __name__ == "__main__":
    run_insights()

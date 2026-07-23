"""EDA Olist (Phase 4) : fonctions d'analyse et de visualisation reutilisables
sur data/processed/olist_customers_clean.csv.

Chaque fonction prend le DataFrame client agrege en entree, retourne (fig, ax)
plutot que d'appeler plt.show(), et accepte un axe existant (parametre `ax`)
pour permettre de composer plusieurs graphiques cote a cote dans un notebook.

Aucune feature ML (churn, RFM) n'est calculee ici : c'est l'objet de la
Phase 5. Ce module est un ensemble de fonctions, pas un notebook.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cleaning import clean

DEFAULT_COLOR = "#4C72B0"


def load_customers_clean() -> pd.DataFrame:
    if not clean.CUSTOMERS_CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"{clean.CUSTOMERS_CLEAN_FILE} introuvable. Lance d'abord src/cleaning/clean.py."
        )
    return pd.read_csv(
        clean.CUSTOMERS_CLEAN_FILE,
        encoding="utf-8-sig",
        parse_dates=["first_order_date", "last_order_date"],
    )


def _get_fig_ax(ax=None, figsize=(8, 5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


# --- BLOC 1 : analyse de la retention -------------------------------------

def plot_orders_distribution(df: pd.DataFrame, ax=None):
    """Barplot de la distribution de total_orders, limite a 1-10 pour
    lisibilite (au-dela, le volume de clients est trop faible pour peser sur
    la lecture du graphe)."""
    fig, ax = _get_fig_ax(ax)

    counts = df["total_orders"].value_counts().reindex(range(1, 11), fill_value=0).sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=DEFAULT_COLOR)
    ax.set_title("Distribution du nombre de commandes par client (1-10)")
    ax.set_xlabel("total_orders")
    ax.set_ylabel("Nombre de clients")
    # INSIGHT : l'immense majorite des clients n'a passe qu'une seule commande.
    # C'est le signal retention le plus important du dataset : le business
    # repose structurellement sur l'acquisition plutot que sur la fidelisation.

    return fig, ax


def plot_retention_curve(df: pd.DataFrame, ax=None):
    """Courbe du % de clients ayant N commandes ou plus, pour N de 1 a 10.
    Une courbe de retention cumulative se lit plus facilement qu'un histogramme
    brut pour repondre a la question business "quelle part de la clientele
    est fidele au-dela d'un seuil donne ?"."""
    fig, ax = _get_fig_ax(ax)

    total_customers = len(df)
    thresholds = range(1, 11)
    pct_at_least_n = [
        (df["total_orders"] >= n).sum() / total_customers * 100 for n in thresholds
    ]

    ax.plot(list(thresholds), pct_at_least_n, marker="o", color=DEFAULT_COLOR)
    ax.set_title("% de clients avec N commandes ou plus")
    ax.set_xlabel("N (nombre minimum de commandes)")
    ax.set_ylabel("% de clients")
    ax.set_xticks(list(thresholds))
    # INSIGHT : la courbe chute tres vite entre N=1 et N=2, confirmant que la
    # retention (client qui repasse commande) est le vrai probleme business a
    # adresser, plus que la valeur moyenne par commande.

    return fig, ax

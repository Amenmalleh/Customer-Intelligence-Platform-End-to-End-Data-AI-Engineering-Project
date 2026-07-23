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


# --- BLOC 2 : analyse monetaire --------------------------------------------

def plot_spent_distribution(df: pd.DataFrame, ax=None):
    """Histogramme de total_spent avec echelle log sur l'axe x : la depense
    client est fortement asymetrique (quelques gros comptes, beaucoup de
    petits paniers), une echelle lineaire ecraserait la masse des petites
    valeurs dans une poignee de barres."""
    fig, ax = _get_fig_ax(ax)

    ax.hist(df["total_spent"], bins=50, color=DEFAULT_COLOR)
    ax.set_xscale("log")
    ax.set_title("Distribution de total_spent par client (echelle log)")
    ax.set_xlabel("total_spent (log)")
    ax.set_ylabel("Nombre de clients")
    # INSIGHT : la distribution log-normale typique du e-commerce se confirme
    # ici ; la majorite des clients depense dans une fourchette moderee, avec
    # une longue traine de gros acheteurs qui tirent la moyenne vers le haut.

    return fig, ax


def plot_spent_by_state(df: pd.DataFrame, ax=None):
    """Boxplot de total_spent pour le top 5 des etats par nombre de clients.
    Le boxplot permet de comparer a la fois la mediane et la dispersion entre
    etats, plus informatif qu'une simple moyenne par etat pour une variable
    aussi asymetrique que total_spent."""
    fig, ax = _get_fig_ax(ax, figsize=(9, 5))

    top_states = df["customer_state"].value_counts().head(5).index.tolist()
    data = [df.loc[df["customer_state"] == state, "total_spent"] for state in top_states]

    box = ax.boxplot(data, tick_labels=top_states, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor(DEFAULT_COLOR)
        patch.set_alpha(0.7)
    ax.set_title("total_spent par etat (top 5 etats par nombre de clients)")
    ax.set_xlabel("customer_state")
    ax.set_ylabel("total_spent")
    # INSIGHT : les medianes de depense sont proches d'un etat a l'autre malgre
    # des volumes de clients tres differents, ce qui suggere que la geographie
    # pese surtout sur l'acquisition (nombre de clients), pas sur leur valeur
    # individuelle.

    return fig, ax

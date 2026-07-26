"""Segmentation client Olist (Phase 6) : clustering K-Means sur
data/processed/olist_features_scaled.csv, comparaison avec DBSCAN et
clustering hierarchique, interpretation business des segments.

Chaque etape est une fonction dediee, appelee dans l'ordre par
run_segmentation(), suivant le meme style que src/features/build_features.py.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda.analysis import DEFAULT_COLOR
from src.features.build_features import FEATURES_SCALED_FILE
from src.features.feature_report import load_features
from src.ingestion import config

MODELS_DIR = config.BASE_DIR / "models"
KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_model.pkl"
ELBOW_SILHOUETTE_PLOT = config.DOCS_DIR / "elbow_silhouette.png"
DENDROGRAM_PLOT = config.DOCS_DIR / "dendrogram.png"
SEGMENTS_FILE = config.PROCESSED_DATA_DIR / "olist_segments.csv"
SEGMENTATION_REPORT_FILE = config.DOCS_DIR / "segmentation_report.txt"

CLUSTERING_FEATURES = [
    "recency_scaled",
    "frequency_scaled",
    "monetary_scaled",
    "tenure_scaled",
    "review_scaled",
]

# silhouette_score calcule une matrice de distances par paires : sur 93 358
# clients, la version exacte demanderait de l'ordre de 93358^2 flottants
# (~65 Go), infaisable. sample_size sous-echantillonne pour rendre le calcul
# tractable tout en restant une estimation fiable du score (random_state fixe
# pour la reproductibilite).
SILHOUETTE_SAMPLE_SIZE = 10_000
SILHOUETTE_RANDOM_STATE = 42


def load_features_scaled() -> pd.DataFrame:
    if not FEATURES_SCALED_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURES_SCALED_FILE} introuvable. Lance d'abord src/features/build_features.py."
        )
    return pd.read_csv(FEATURES_SCALED_FILE, encoding="utf-8-sig")


def find_optimal_k(X, k_range=range(2, 10)):
    """ETAPE 1 : calcule l'inertie et le silhouette score pour chaque k,
    trace les courbes elbow + silhouette cote a cote, sauvegarde le graphique
    et retourne le k au meilleur silhouette score (a titre informatif : la
    Phase 6 retient k=4 pour des raisons business, cf. train_kmeans)."""
    inertias = []
    silhouettes = []

    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        inertias.append(model.inertia_)
        silhouettes.append(
            silhouette_score(X, model.labels_, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=SILHOUETTE_RANDOM_STATE)
        )

    k_values = list(k_range)
    best_k = k_values[int(np.argmax(silhouettes))]
    best_score = max(silhouettes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_values, inertias, marker="o", color=DEFAULT_COLOR)
    axes[0].set_title("Methode du coude (elbow) : inertie vs k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertie K-Means")

    axes[1].plot(k_values, silhouettes, marker="o", color=DEFAULT_COLOR)
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"meilleur k = {best_k}")
    axes[1].set_title("Silhouette score vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].legend()

    plt.tight_layout()

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ELBOW_SILHOUETTE_PLOT)

    print(f"[ETAPE 1] Meilleur k selon silhouette score : {best_k} (score={best_score:.4f})")
    print(f"[ETAPE 1] Graphique sauvegarde dans {ELBOW_SILHOUETTE_PLOT}")

    return best_k, fig


def run_segmentation():
    print(f"=== Segmentation client Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    scaled_df = load_features_scaled()
    print(f"Chargement : {scaled_df.shape[0]} lignes x {scaled_df.shape[1]} colonnes")
    X = scaled_df[CLUSTERING_FEATURES].values

    find_optimal_k(X)

    return scaled_df


if __name__ == "__main__":
    run_segmentation()

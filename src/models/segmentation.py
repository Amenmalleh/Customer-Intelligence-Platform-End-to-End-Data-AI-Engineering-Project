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

    return best_k, best_score, fig


def train_kmeans(X, n_clusters: int = 4, random_state: int = 42):
    """ETAPE 2 : entraine K-Means avec k=4.

    find_optimal_k() (ETAPE 1) trouve k=2 comme meilleur silhouette score sur
    ce dataset : la population se separe surtout en gros clients "actifs"
    vs "inactifs" a la premiere coupure naturelle. Mais k=4 est retenu ici
    par choix business plutot que purement statistique : 4 segments
    (Champions / Fideles / A Risque / Perdus) sont le standard RFM utilise
    par le marketing et donnent des actions differenciees exploitables,
    la ou k=2 ne distinguerait pas par exemple les "Champions" des
    "Fideles". Un silhouette score plus bas a k=4 (~0.32 vs ~0.69 a k=2)
    est le prix accepte pour une segmentation plus actionnable.
    """
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state).fit(X)
    labels = model.labels_

    distribution = pd.Series(labels).value_counts().sort_index()
    print(f"[ETAPE 2] K-Means entraine avec k={n_clusters}")
    print(f"[ETAPE 2] Distribution des segments :\n{distribution.to_string()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, KMEANS_MODEL_FILE)
    print(f"[ETAPE 2] Modele sauvegarde dans {KMEANS_MODEL_FILE}")

    return model, labels


RAW_PROFILE_FEATURES = ["recency_days", "frequency", "total_spent", "avg_review_score", "customer_tenure_days"]
SEGMENT_NAMES_BY_RFM_RANK = ["Champions", "Fideles", "A Risque", "Perdus"]


def interpret_segments(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 3 : calcule le profil moyen (features brutes) de chaque segment
    et assigne un nom business.

    Les segments sont tries par rfm_score decroissant puis nommes dans
    l'ordre Champions > Fideles > A Risque > Perdus. Sur ce dataset, 3 des 4
    segments n'ont que des clients a une seule commande (frequency=1) : ils
    se distinguent surtout par recency_days et avg_review_score plutot que
    par un profil RFM complet. En pratique :
    - "Champions" est le seul segment avec frequency > 1 (multi-acheteurs,
      total_spent le plus eleve) : correspond bien au profil attendu.
    - "Fideles" est en realite le segment des acheteurs recents tres
      satisfaits (meilleure recency, review le plus haut) mais a une seule
      commande : plus proche de "nouveaux clients prometteurs" que du
      "fidele" au sens classique.
    - "A Risque" se distingue surtout par une satisfaction tres basse
      (avg_review_score ~1.7 contre ~4.6-4.7 ailleurs), pas seulement par
      une recency/monetary faibles : le signal dominant ici est la
      satisfaction, pas la RFM pure.
    - "Perdus" a la plus mauvaise recency et frequency=1 : correspond bien
      au profil attendu (clients inactifs depuis longtemps).
    Le nom sert de raccourci marketing actionnable, pas une etiquette RFM
    exacte a prendre au pied de la lettre pour chaque segment.
    """
    df = df.copy()

    profile = df.groupby("segment")[RAW_PROFILE_FEATURES + ["rfm_score"]].mean()
    profile["count"] = df["segment"].value_counts()
    profile = profile.sort_values("rfm_score", ascending=False)

    name_mapping = dict(zip(profile.index, SEGMENT_NAMES_BY_RFM_RANK))
    df["segment_name"] = df["segment"].map(name_mapping)

    profile["segment_name"] = SEGMENT_NAMES_BY_RFM_RANK
    print("[ETAPE 3] Profils moyens par segment (tries par rfm_score decroissant) :")
    print(profile.to_string())

    return df


def compare_with_dbscan(X):
    """ETAPE 4 : entraine DBSCAN(eps=0.5, min_samples=5) a titre de
    comparaison et explique pourquoi K-Means reste l'algo principal.

    silhouette_score n'est calcule que sur les points non-bruit (label != -1)
    quand au moins 2 clusters sont trouves hors bruit : inclure le bruit
    comme un "cluster" a part entiere fausserait la mesure de cohesion.
    """
    model = DBSCAN(eps=0.5, min_samples=5).fit(X)
    labels = model.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    print(f"[ETAPE 4] DBSCAN(eps=0.5, min_samples=5) : {n_clusters} clusters trouves, {noise_pct:.2f}% de points bruit (label=-1)")

    non_noise_mask = labels != -1
    if n_clusters >= 2 and non_noise_mask.sum() > SILHOUETTE_SAMPLE_SIZE:
        score = silhouette_score(
            X[non_noise_mask], labels[non_noise_mask],
            sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=SILHOUETTE_RANDOM_STATE,
        )
        print(f"[ETAPE 4] Silhouette score DBSCAN (hors bruit) : {score:.4f}")
    elif n_clusters >= 2:
        score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
        print(f"[ETAPE 4] Silhouette score DBSCAN (hors bruit) : {score:.4f}")
    else:
        score = None
        print("[ETAPE 4] Silhouette score non applicable (moins de 2 clusters hors bruit)")

    print(
        "[ETAPE 4] K-Means est prefere ici comme algo principal : DBSCAN(eps=0.5, "
        f"min_samples=5) produit {n_clusters} clusters de densite variable, trop "
        "nombreux et non controlables directement pour livrer 4 segments marketing "
        "actionnables (le nombre de clusters depend d'eps/min_samples, pas d'un choix "
        "explicite de k). K-Means impose k=4 directement, ce qui correspond au besoin "
        "business (Champions/Fideles/A Risque/Perdus) ; DBSCAN reste utile ici surtout "
        "pour reperer des points aberrants (bruit), pas pour segmenter toute la base."
    )

    return {"n_clusters": n_clusters, "noise_pct": noise_pct, "silhouette": score}


def hierarchical_clustering_sample(df: pd.DataFrame, sample_size: int = 3000, random_state: int = 42):
    """ETAPE 5 : clustering hierarchique (AgglomerativeClustering, linkage
    Ward par defaut) sur un echantillon de 3000 clients.

    Un dendrogramme ou un linkage complet sur 93 358 clients ne serait ni
    lisible ni raisonnable a calculer (complexite O(n^2) ou pire en memoire
    pour scipy.cluster.hierarchy.linkage) : un echantillon aleatoire de 3000
    clients suffit pour visualiser la structure hierarchique et comparer les
    clusters a ceux de K-Means. linkage='ward' est utilise cote scipy pour
    rester coherent avec le linkage par defaut de AgglomerativeClustering.
    """
    sample = df.sample(n=sample_size, random_state=random_state)
    X_sample = sample[CLUSTERING_FEATURES].values

    agglo = AgglomerativeClustering(n_clusters=4, linkage="ward")
    hier_labels = agglo.fit_predict(X_sample)

    linkage_matrix = linkage(X_sample, method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linkage_matrix, ax=ax, truncate_mode="lastp", p=30, no_labels=True)
    ax.set_title(f"Dendrogramme (clustering hierarchique, echantillon n={sample_size})")
    ax.set_xlabel("Clients (feuilles regroupees, 30 dernieres fusions)")
    ax.set_ylabel("Distance (Ward)")
    plt.tight_layout()

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DENDROGRAM_PLOT)

    ari = adjusted_rand_score(sample["segment"], hier_labels)
    print(f"[ETAPE 5] AgglomerativeClustering(n_clusters=4) sur echantillon n={sample_size}")
    print(f"[ETAPE 5] Adjusted Rand Score vs K-Means : {ari:.4f}")
    print(f"[ETAPE 5] Dendrogramme sauvegarde dans {DENDROGRAM_PLOT}")

    return hier_labels, ari, fig


SEGMENT_RECOMMENDATIONS = {
    "Champions": (
        "Multi-acheteurs a plus forte depense et meilleure satisfaction. Programme de "
        "fidelite / acces anticipe aux nouveautes / cross-sell : ce sont les clients les "
        "plus rentables, prioriser leur retention plutot que leur acquisition."
    ),
    "Fideles": (
        "Acheteurs recents tres satisfaits mais encore a une seule commande. Relance "
        "ciblee (email/promo J+30-60) pour transformer le premier achat en reachat, "
        "avant que la recency ne se degrade vers le segment 'A Risque' ou 'Perdus'."
    ),
    "A Risque": (
        "Satisfaction tres basse (avg_review_score ~1.7) : signal dominant, plus que la "
        "recency/frequence. Prioriser une enquete de satisfaction ou un geste commercial "
        "cible pour comprendre et corriger la cause de l'insatisfaction avant perte "
        "definitive du client."
    ),
    "Perdus": (
        "Recency la plus haute, une seule commande ancienne. Campagne de reactivation a "
        "faible cout (remise agressive, relance ponctuelle) ou desinvestissement si le "
        "cout de reconquete depasse la valeur potentielle du client."
    ),
}


def save_segments(df: pd.DataFrame, best_k: int, best_k_score: float, dbscan_stats: dict, hierarchical_ari: float) -> pd.DataFrame:
    """ETAPE 6 : sauvegarde le dataset segmente et le rapport de
    segmentation complet."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEGMENTS_FILE, index=False, encoding="utf-8-sig")

    profile = df.groupby("segment")[RAW_PROFILE_FEATURES + ["rfm_score"]].mean()
    sizes = df["segment"].value_counts()
    profile["count"] = sizes
    profile["pct"] = (sizes / len(df) * 100).round(2)
    profile = profile.sort_values("rfm_score", ascending=False)
    profile["segment_name"] = SEGMENT_NAMES_BY_RFM_RANK

    lines = [
        "RAPPORT DE SEGMENTATION CLIENT - Olist (Phase 6)",
        f"Date de generation : {datetime.now().isoformat(timespec='seconds')}",
        f"Fichier segmente : {SEGMENTS_FILE}",
        f"Shape : {df.shape[0]} lignes x {df.shape[1]} colonnes",
        "",
        f"k optimal (silhouette score, k=2 a 9) : k={best_k} (score={best_k_score:.4f})",
        (
            "Justification du choix final k=4 (au lieu du k optimal statistique) : "
            "4 segments correspondent au standard business RFM (Champions/Fideles/A "
            "Risque/Perdus), avec une action marketing differenciee par segment. k=2 "
            "aurait un meilleur silhouette score mais ne separerait pas les clients "
            "a fort potentiel des clients a risque, ce qui le rend inexploitable pour "
            "le marketing cible demande ici."
        ),
        "",
        "Profil moyen par segment (segments et tailles) :",
        profile.to_string(),
        "",
        "Recommandations marketing par segment :",
    ]
    for segment_name in SEGMENT_NAMES_BY_RFM_RANK:
        lines.append(f"  - {segment_name} : {SEGMENT_RECOMMENDATIONS[segment_name]}")

    dbscan_silhouette_display = (
        f"{dbscan_stats['silhouette']:.4f}" if dbscan_stats["silhouette"] is not None else "N/A"
    )
    lines += [
        "",
        "Comparaison K-Means vs DBSCAN vs Clustering Hierarchique :",
        (
            f"  - K-Means (k=4, retenu) : 4 segments de tailles exploitables, "
            "directement actionnables par le marketing (noms/recommandations ci-dessus)."
        ),
        (
            f"  - DBSCAN (eps=0.5, min_samples=5) : {dbscan_stats['n_clusters']} clusters, "
            f"{dbscan_stats['noise_pct']:.2f}% de bruit, silhouette={dbscan_silhouette_display}"
        ),
        (
            "    Trop de clusters fragmentes pour du marketing (pas de k impose), utile "
            "surtout pour isoler des points aberrants (bruit)."
        ),
        (
            f"  - Clustering hierarchique (AgglomerativeClustering, n_clusters=4, "
            f"echantillon n=3000) : Adjusted Rand Score vs K-Means = {hierarchical_ari:.4f} "
            "(accord modere). Confirme que la structure en 4 groupes est plausible mais "
            "sensible a la methode de linkage ; K-Means reste retenu pour sa simplicite, "
            "sa rapidite sur les 93 358 clients complets et son k explicite."
        ),
    ]

    report_text = "\n".join(lines)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_REPORT_FILE.write_text(report_text, encoding="utf-8")

    print(f"[ETAPE 6] {SEGMENTS_FILE} : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"[ETAPE 6] Rapport ecrit dans {SEGMENTATION_REPORT_FILE}")

    return df


def run_segmentation():
    print(f"=== Segmentation client Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    scaled_df = load_features_scaled()
    print(f"Chargement : {scaled_df.shape[0]} lignes x {scaled_df.shape[1]} colonnes")
    X = scaled_df[CLUSTERING_FEATURES].values

    best_k, best_k_score, _ = find_optimal_k(X)
    model, labels = train_kmeans(X)
    scaled_df["segment"] = labels

    full_df = load_features()
    merged_df = full_df.merge(scaled_df[["customer_unique_id", "segment"]], on="customer_unique_id", how="inner")
    merged_df = interpret_segments(merged_df)

    dbscan_stats = compare_with_dbscan(X)
    _, hierarchical_ari, _ = hierarchical_clustering_sample(merged_df)

    merged_df = save_segments(merged_df, best_k, best_k_score, dbscan_stats, hierarchical_ari)

    return merged_df


if __name__ == "__main__":
    run_segmentation()

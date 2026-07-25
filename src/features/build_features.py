"""Feature engineering Olist (Phase 5) : charge
data/processed/olist_customers_clean.csv, calcule le label churn, les scores
RFM, encode les variables categorielles, normalise pour le clustering, et
sauvegarde les datasets finaux (olist_features.csv / olist_features_scaled.csv).

Chaque etape est une fonction pure (df -> df) appelee dans l'ordre par
run_feature_engineering(), suivant le meme style que src/cleaning/clean.py.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda.analysis import load_customers_clean
from src.ingestion import config

ENCODERS_DIR = config.BASE_DIR / "models" / "encoders"
FEATURES_FILE = config.PROCESSED_DATA_DIR / "olist_features.csv"
FEATURES_SCALED_FILE = config.PROCESSED_DATA_DIR / "olist_features_scaled.csv"

SCALED_COLUMNS = ["recency_scaled", "frequency_scaled", "monetary_scaled", "tenure_scaled", "review_scaled"]


def compute_recency_and_churn(df: pd.DataFrame, churn_threshold_days: int = config.CHURN_THRESHOLD_DAYS) -> pd.DataFrame:
    """ETAPE 1 : calcule recency_days et le label churn.

    date_reference = max(last_order_date), pas datetime.now() : le dataset
    Olist s'arrete en 2018, utiliser la date du jour ferait passer la quasi
    totalite des clients pour churned (cf. Phase 4 EDA). churn_threshold_days
    reutilise config.CHURN_THRESHOLD_DAYS (180) par defaut pour rester
    coherent avec le seuil deja utilise dans src/eda/insights.py.
    """
    df = df.copy()

    date_reference = df["last_order_date"].max()
    df["recency_days"] = (date_reference - df["last_order_date"]).dt.days
    df["churn"] = (df["recency_days"] > churn_threshold_days).astype(int)

    churn_counts = df["churn"].value_counts(normalize=True).mul(100).round(2)
    print(f"[ETAPE 1] date_reference = {date_reference.date()} (seuil churn = {churn_threshold_days} jours)")
    print(f"[ETAPE 1] churn=0 : {churn_counts.get(0, 0.0)}% | churn=1 : {churn_counts.get(1, 0.0)}%")

    return df


def compute_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 2 : calcule les scores RFM (Recency / Frequency / Monetary).

    - frequency : total_orders renomme pour la clarte du vocabulaire RFM.
    - monetary_log = log(total_spent + 1) : total_spent est fortement
      asymetrique (cf. Phase 4 EDA), le log ramene la distribution vers une
      forme plus proche de la normale avant de la quantiler ou de la scaler.
      Le +1 evite log(0) pour un total_spent nul (ne devrait pas arriver
      apres la Phase 3, mais reste sans effet si total_spent > 0).
    - customer_tenure_days : duree entre premiere et derniere commande ;
      avg_days_between_orders : tenure / (frequency - 1) si frequency > 1,
      sinon 0 (un client a une seule commande n'a pas d'intervalle a mesurer).
    - R_score : pd.qcut(recency_days, q=4, labels=[4,3,2,1]) — inverse
      volontairement : une recency elevee (client absent depuis longtemps)
      merite le plus mauvais score (1), une recency faible le meilleur (4).
    - F_score : total_orders est extremement asymetrique (97% des clients
      n'ont qu'une seule commande, cf. Phase 4), donc pd.qcut(frequency, q=4,
      duplicates='drop') echoue : les bornes de quartile se confondent en un
      seul bin (Q1=Q2=Q3=1) et pandas ne peut pas caser 4 labels sur une
      seule categorie. On qcut a la place sur le RANG de frequency
      (`.rank(method='first')`), une technique standard pour scorer une
      variable avec beaucoup d'ex-aequo : elle garantit 4 groupes de taille
      egale et respecte l'ordre (frequency plus elevee => F_score jamais
      inferieur), au prix d'un depart quasi arbitraire entre clients ayant
      exactement 1 commande.
    - M_score : pd.qcut(monetary_log, q=4, labels=[1,2,3,4]), sans probleme
      de doublons ici car monetary_log est une variable continue.
    - rfm_score : moyenne (float) des 3 scores, plus simple qu'une
      concatenation de chiffres pour un usage direct en feature numerique de
      clustering/scoring dans les phases suivantes.
    """
    df = df.copy()

    df["frequency"] = df["total_orders"]
    df["monetary_log"] = np.log(df["total_spent"] + 1)
    df["customer_tenure_days"] = (df["last_order_date"] - df["first_order_date"]).dt.days
    df["avg_days_between_orders"] = np.where(
        df["frequency"] > 1,
        df["customer_tenure_days"] / (df["frequency"] - 1),
        0,
    )

    df["R_score"] = pd.qcut(df["recency_days"], q=4, labels=[4, 3, 2, 1]).astype(int)
    df["F_score"] = pd.qcut(
        df["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]
    ).astype(int)
    df["M_score"] = pd.qcut(df["monetary_log"], q=4, labels=[1, 2, 3, 4]).astype(int)
    df["rfm_score"] = df[["R_score", "F_score", "M_score"]].mean(axis=1)

    print(
        f"[ETAPE 2] RFM : frequency/monetary_log/customer_tenure_days/avg_days_between_orders calcules ; "
        f"R_score, F_score (base sur le rang, cf. docstring), M_score, rfm_score (moyenne={df['rfm_score'].mean():.2f})"
    )

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 3 : encode customer_state et most_frequent_category avec
    sklearn LabelEncoder.

    most_frequent_category contient des NaN (clients dont aucune categorie
    n'a pu etre resolue lors de l'agregation en Phase 3) : LabelEncoder ne
    sait pas encoder un NaN, on le remplace par la categorie explicite
    'unknown' avant l'encodage plutot que de supprimer ces lignes, pour ne
    perdre aucun client. Les deux encodeurs sont sauvegardes pour pouvoir
    appliquer exactement le meme mapping sur de nouvelles donnees (API
    d'inference en Phase 8+).
    """
    df = df.copy()
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)

    state_encoder = LabelEncoder()
    df["state_encoded"] = state_encoder.fit_transform(df["customer_state"])
    joblib.dump(state_encoder, ENCODERS_DIR / "label_encoder_state.pkl")

    category_filled = df["most_frequent_category"].fillna("unknown")
    category_encoder = LabelEncoder()
    df["category_encoded"] = category_encoder.fit_transform(category_filled)
    joblib.dump(category_encoder, ENCODERS_DIR / "label_encoder_category.pkl")

    print(
        f"[ETAPE 3] state_encoded : {len(state_encoder.classes_)} etats distincts | "
        f"category_encoded : {len(category_encoder.classes_)} categories distinctes (NaN -> 'unknown')"
    )
    print(f"[ETAPE 3] Encodeurs sauvegardes dans {ENCODERS_DIR}")

    return df


# Colonne source -> colonne scaled : les noms ne suivent pas tous le meme
# suffixe brut (ex. monetary_log -> monetary_scaled, pas monetary_log_scaled)
# car ce sont les noms courts demandes pour les features de clustering.
SCALE_COLUMN_MAP = {
    "recency_days": "recency_scaled",
    "frequency": "frequency_scaled",
    "monetary_log": "monetary_scaled",
    "customer_tenure_days": "tenure_scaled",
    "avg_review_score": "review_scaled",
}


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """ETAPE 4 : normalise 5 features numeriques avec StandardScaler en vue
    du clustering (Phase 6).

    StandardScaler (moyenne 0, ecart-type 1) plutot qu'un MinMaxScaler : les
    algorithmes de clustering bases sur une distance (K-Means notamment,
    prevu Phase 6) sont sensibles a l'echelle relative des features, et
    monetary_log/recency_days/customer_tenure_days n'ont pas les memes
    unites ni la meme variance. Le scaler est fit sur l'ensemble du dataset
    et sauvegarde pour etre reapplique tel quel sur de nouvelles donnees.
    """
    df = df.copy()
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)

    source_columns = list(SCALE_COLUMN_MAP.keys())
    scaled_columns = list(SCALE_COLUMN_MAP.values())

    scaler = StandardScaler()
    df[scaled_columns] = scaler.fit_transform(df[source_columns])
    joblib.dump(scaler, ENCODERS_DIR / "scaler.pkl")

    print(f"[ETAPE 4] StandardScaler applique sur {source_columns} -> {scaled_columns}")
    print(f"[ETAPE 4] Scaler sauvegarde dans {ENCODERS_DIR / 'scaler.pkl'}")

    return df


def run_feature_engineering() -> pd.DataFrame:
    print(f"=== Feature engineering Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    df = load_customers_clean()
    print(f"Chargement : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    df = compute_recency_and_churn(df)
    df = compute_rfm_scores(df)
    df = encode_categorical(df)
    df = normalize_features(df)

    return df


if __name__ == "__main__":
    run_feature_engineering()

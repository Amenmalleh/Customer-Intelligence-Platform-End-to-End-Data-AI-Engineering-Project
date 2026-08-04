"""Prediction du churn Olist (Phase 7) : entraine et compare 4 modeles
supervises (LogisticRegression, RandomForest, XGBoost, LightGBM) sur
data/processed/olist_features.csv, optimise le modele gagnant et le
sauvegarde pour l'API (Phase 9).

Chaque etape est une fonction dediee, appelee dans l'ordre par
run_churn_prediction(), suivant le meme style que src/models/segmentation.py.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eda.analysis import DEFAULT_COLOR
from src.features.feature_report import load_features
from src.ingestion import config
from src.models.segmentation import MODELS_DIR

# FEATURES A UTILISER : la consigne initiale de la Phase 7 incluait
# recency_days. Teste isolement, ce jeu de features fait monter recall /
# precision / f1 / ROC-AUC a 1.0 (ou quasi, ~0.998 pour XGBoost) sur les 4
# modeles : churn est defini comme (recency_days > 180), donc recency_days
# permet de reconstruire churn par une simple regle de seuil au lieu d'un
# vrai signal comportemental appris (le meme risque de fuite deja documente
# dans docs/feature_engineering_report.txt en Phase 5 et docs/segmentation_report.txt
# en Phase 6). Sur decision explicite de l'utilisateur, recency_days est donc
# exclu de X ci-dessous ; sans lui, les scores redeviennent realistes
# (ROC-AUC ~0.54-0.70 selon le modele), un signal faible mais genuine.
FEATURE_COLUMNS = [
    "frequency",
    "monetary_log",
    "avg_review_score",
    "customer_tenure_days",
    "avg_days_between_orders",
    "state_encoded",
    "category_encoded",
]
TARGET_COLUMN = "churn"

CHURN_MODEL_FILE = MODELS_DIR / "churn_model.pkl"
CHURN_MODEL_METADATA_FILE = MODELS_DIR / "churn_model_metadata.json"
ROC_CURVES_PLOT = config.DOCS_DIR / "roc_curves.png"
FEATURE_IMPORTANCE_PLOT = config.DOCS_DIR / "feature_importance.png"
MODEL_COMPARISON_REPORT_FILE = config.DOCS_DIR / "model_comparison_report.txt"

RANDOM_STATE = 42


def prepare_data(df: pd.DataFrame):
    """ETAPE 1 : construit X/y, verifie l'absence de NaN, split stratifie
    80/20, et calcule scale_pos_weight (utilise par XGBoost/LightGBM pour
    compenser le desequilibre de classes sans sur-echantillonner)."""
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN detecte dans les features d'entree : {na_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    n_churn0 = (y_train == 0).sum()
    n_churn1 = (y_train == 1).sum()
    scale_pos_weight = n_churn0 / n_churn1

    print(f"[ETAPE 1] X: {X.shape[1]} features, y: '{TARGET_COLUMN}' | aucune NaN detectee")
    print(f"[ETAPE 1] Split 80/20 stratifie : train={X_train.shape[0]}, test={X_test.shape[0]}")
    print(f"[ETAPE 1] scale_pos_weight (train) = {n_churn0}/{n_churn1} = {scale_pos_weight:.4f}")
    print(f"[ETAPE 1] Distribution churn (train) :\n{(y_train.value_counts(normalize=True) * 100).round(2).to_string()}")
    print(f"[ETAPE 1] Distribution churn (test) :\n{(y_test.value_counts(normalize=True) * 100).round(2).to_string()}")

    return X_train, X_test, y_train, y_test, scale_pos_weight


def build_models(scale_pos_weight: float) -> dict:
    return {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight, n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=scale_pos_weight, n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_STATE, verbose=-1,
        ),
    }


def train_all_models(X_train, y_train, X_test, y_test, scale_pos_weight: float):
    """ETAPE 2 : entraine les 4 modeles et les compare sur le test set.

    Signature etendue par rapport a la consigne initiale
    (train_all_models(X_train, y_train, scale_pos_weight)) : evaluer recall/
    precision/f1/ROC-AUC sur le test set (demande explicitement juste apres)
    necessite X_test/y_test, donc ils sont passes en parametres plutot que
    recalcules ou charges en global.
    """
    models = build_models(scale_pos_weight)
    trained_models = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n--- {name} ---")
        print(f"recall={recall:.4f} precision={precision:.4f} f1={f1:.4f} roc_auc={roc_auc:.4f}")
        print("Classification report :")
        print(classification_report(y_test, y_pred))
        print(f"Matrice de confusion :\n{cm}")

        trained_models[name] = model
        results[name] = {
            "recall": recall, "precision": precision, "f1": f1, "roc_auc": roc_auc,
            "y_proba": y_proba, "confusion_matrix": cm,
        }

    comparison_df = pd.DataFrame(
        {name: {k: v for k, v in r.items() if k in ("recall", "precision", "f1", "roc_auc")} for name, r in results.items()}
    ).T.sort_values("recall", ascending=False)

    print("\n[ETAPE 2] Tableau comparatif des 4 modeles (trie par recall decroissant) :")
    print(comparison_df.to_string())

    return trained_models, results, comparison_df


# Palette categorielle fixe (identite du modele, pas une magnitude) : meme
# logique que les couleurs par segment du notebook 05_segmentation.
MODEL_COLORS = {
    "LogisticRegression": "#4C72B0",
    "RandomForest": "#55A868",
    "XGBoost": "#DD8452",
    "LightGBM": "#C44E52",
}


def plot_roc_curves(models: dict, X_test, y_test):
    """ETAPE 3 : trace les 4 courbes ROC sur un meme graphique, avec la
    diagonale aleatoire (AUC=0.5) comme reference visuelle."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name), label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatoire (AUC=0.5)")
    ax.set_title("Courbes ROC - comparaison des 4 modeles")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROC_CURVES_PLOT)
    print(f"[ETAPE 3] Courbes ROC sauvegardees dans {ROC_CURVES_PLOT}")

    return fig, ax


XGB_LGBM_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5],
}


def optimize_best_model(X_train, y_train, X_test, y_test, scale_pos_weight, best_model_name, trained_models):
    """ETAPE 4 : identifie le modele gagnant (meilleur recall, deja trie
    dans comparison_df) et l'optimise si c'est XGBoost ou LightGBM.

    Sur ce dataset, RandomForest gagne au recall (cf. ETAPE 2), pas XGBoost
    ni LightGBM : la grille RandomizedSearchCV fournie dans la consigne est
    specifique a ces deux algos (min_child_weight n'a pas de sens pour
    RandomForest/LogisticRegression). Dans ce cas, aucune recherche
    d'hyperparametres n'est definie pour l'algo gagnant : le modele deja
    entraine en ETAPE 2 est conserve tel quel comme modele final, ce qui est
    documente explicitement plutot que force artificiellement vers XGBoost/
    LightGBM pour coller a la grille.
    """
    if best_model_name in ("XGBoost", "LightGBM"):
        base_estimator = (
            XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
            if best_model_name == "XGBoost"
            else LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, verbose=-1)
        )
        search = RandomizedSearchCV(
            base_estimator, param_distributions=XGB_LGBM_PARAM_DISTRIBUTIONS,
            cv=5, scoring="recall", n_iter=20, random_state=RANDOM_STATE, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        print(f"[ETAPE 4] Modele gagnant : {best_model_name} -> RandomizedSearchCV applique")
        print(f"[ETAPE 4] Meilleurs parametres trouves : {best_params}")
    else:
        best_model = trained_models[best_model_name]
        best_params = None
        print(
            f"[ETAPE 4] Modele gagnant : {best_model_name} (ni XGBoost ni LightGBM) -> "
            "grille RandomizedSearchCV non applicable, modele de l'ETAPE 2 conserve tel quel."
        )

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = {
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"[ETAPE 4] Re-evaluation sur le test set : {test_metrics}")

    return best_model, best_params, test_metrics


def plot_feature_importance(best_model, feature_names):
    """ETAPE 5 : extrait feature_importances_ du modele gagnant et trace un
    barplot horizontal trie par importance decroissante."""
    importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importances.index, importances.values, color=DEFAULT_COLOR)
    ax.set_title("Feature importance - modele gagnant")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FEATURE_IMPORTANCE_PLOT)

    top5 = importances.sort_values(ascending=False).head(5)
    print(f"[ETAPE 5] Top 5 features les plus importantes :\n{top5.to_string()}")
    print(f"[ETAPE 5] Graphique sauvegarde dans {FEATURE_IMPORTANCE_PLOT}")

    return fig, ax, top5


def run_churn_prediction():
    print(f"=== Churn prediction Olist - {datetime.now().isoformat(timespec='seconds')} ===\n")

    df = load_features()
    print(f"Chargement : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    X_train, X_test, y_train, y_test, scale_pos_weight = prepare_data(df)
    trained_models, results, comparison_df = train_all_models(X_train, y_train, X_test, y_test, scale_pos_weight)
    plot_roc_curves(trained_models, X_test, y_test)

    best_model_name = comparison_df.index[0]
    best_model, best_params, test_metrics = optimize_best_model(
        X_train, y_train, X_test, y_test, scale_pos_weight, best_model_name, trained_models
    )
    _, _, top5_features = plot_feature_importance(best_model, FEATURE_COLUMNS)

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "scale_pos_weight": scale_pos_weight, "trained_models": trained_models, "results": results,
        "comparison_df": comparison_df, "best_model_name": best_model_name, "best_model": best_model,
        "best_params": best_params, "test_metrics": test_metrics, "top5_features": top5_features,
    }


if __name__ == "__main__":
    run_churn_prediction()

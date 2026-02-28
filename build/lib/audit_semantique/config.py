"""
config.py — Configuration centralisée du projet.
Tous les chemins, hyperparamètres et constantes métier sont ici.
"""
from pathlib import Path

# ─── Racine du projet ────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]

# ─── Chemins de données ───────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"

PATH_LOI_2024  = RAW_DIR / "loi_finances_1_articles_id_titre_chapitre_contenu.json"
PATH_LOI_2025  = RAW_DIR / "loi_finances_2_articles_id_titre_chapitre_contenu.json"

# ─── Chemins de sortie ────────────────────────────────────────────────────────
OUTPUT_DIR     = ROOT_DIR / "outputs"
FIGURES_DIR    = OUTPUT_DIR / "figures"
MODELS_DIR     = OUTPUT_DIR / "models"
REPORTS_DIR    = OUTPUT_DIR / "reports"

# Créer les répertoires s'ils n'existent pas
for _dir in [PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Modèles ──────────────────────────────────────────────────────────────────
CAMEMBERT_MODEL      = "camembert-base"
ZERO_SHOT_MODEL      = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# ─── Hyperparamètres CamemBERT ────────────────────────────────────────────────
CAMEMBERT_PARAMS = {
    "batch_size":  8, # nombre d'article a traiter simultanément
    "max_length":  512, # longueur maximale des séquences
    "lr":          2e-5, # taux d'apprentissage
    "num_epochs":  3, # nombre d'époques
}

# ─── Hyperparamètres Zero-Shot ────────────────────────────────────────────────
ZERO_SHOT_PARAMS = {
    "batch_size":  8, # nombre d'article a traiter simultanément
    "max_length":  256, # longueur maximale des séquences
    "multi_label": True, # classification multi-label (un article peut appartenir à plusieurs piliers)
}

# ─── Piliers SND30 ────────────────────────────────────────────────────────────
# Utiliser pour l'entainement du modèle de classification zero-shot et pour l'évaluation
PILIERS_SND30 = [
    "Transformation structurelle de l'économie pour accélérer la croissance",
    "Développement du capital humain et du bien-être social",
    "Promotion de l'emploi et de l'insertion socio-économique",
    "Gouvernance, décentralisation et gestion stratégique de l'État",
]
## Utiliser pour la visualisation
PILIERS_COURT = [
    "Transformation économique",
    "Capital humain",
    "Emploi et insertion",
    "Gouvernance et décentralisation",
]

# ─── Audit Sémantique ─────────────────────────────────────────────────────────
AUDIT_PARAMS = {
    "seuil_changement": 0.70,
    "top_k_matches":    3,
}

# ─── Topic Modeling (LDA) ─────────────────────────────────────────────────────
LDA_PARAMS = {
    "num_topics":    4,
    "random_state":  42,
    "passes":        50,
    "no_below":      5,
    "no_above":      0.5,
}

# ─── Clustering ───────────────────────────────────────────────────────────────
KMEANS_PARAMS = {
    "n_clusters":   3,
    "random_state": 42,
    "n_init":       10,
}

HDBSCAN_PARAMS = {
    "min_cluster_size": 5,
    "min_samples":      2,
    "metric":           "euclidean",
}

# ─── Réduction dimensionnelle ─────────────────────────────────────────────────
TSNE_PARAMS = {
    "n_components":  2,
    "random_state":  42,
    "perplexity":    30,
}

UMAP_PARAMS = {
    "n_components":  2,
    "n_neighbors":   15,
    "min_dist":      0.1,
    "metric":        "cosine",
    "random_state":  42,
}

# ─── Visualisation ────────────────────────────────────────────────────────────
VIZ_PARAMS = {
    "dpi":         150,
    "figsize_lg":  (18, 8),
    "figsize_md":  (14, 10),
    "figsize_sm":  (10, 6),
    "style":       "seaborn-v0_8-darkgrid",
    "palette":     "husl",
}

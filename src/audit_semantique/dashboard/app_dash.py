"""
Dashboard Dash - Audit Sémantique Loi de Finances Cameroun
Dashboard interactif avec visualisations avancées (nuages de mots, topics, clusters, budget)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import base64
from io import BytesIO
from PIL import Image
import json
from collections import Counter
import re

# Template Plotly global appliqué à tous les graphiques
pio.templates.default = "plotly_white"

from audit_semantique.preprocessing.text_cleaner import TextPreprocessor
from audit_semantique.config import (
    AUDIT_PARAMS,
    FIGURES_DIR,
    MODELS_DIR,
    PILIERS_COURT,
    PILIERS_SND30,
    REPORTS_DIR,
    UMAP_PARAMS,
)

# Cache léger pour éviter de régénérer les nuages de mots à chaque interaction
_BAROMETER_WC_CACHE: dict[tuple[str, str], str] = {}
_CLUSTER_METRICS_CACHE: dict[tuple[int, str], dict] = {}


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convertit en numérique (NaN si impossible)."""
    return pd.to_numeric(series, errors="coerce")


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d (Welch-like pooled SD)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mean_diff = x.mean() - y.mean()
    pooled_std = np.sqrt((x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2)
    return float(mean_diff / pooled_std) if pooled_std > 0 else float("nan")


def _cluster_metrics(X, labels, sample_size: int = 2000, random_state: int = 42) -> dict:
    """Calcule des métriques de clustering de façon robuste et rapide (sampling)."""
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    labels = np.asarray(labels)
    X = np.asarray(X)
    # exclure NaN labels
    valid = np.isfinite(labels)
    labels = labels[valid]
    X = X[valid]

    # HDBSCAN: bruit = -1
    unique_labels = np.unique(labels)
    n_points = int(len(labels))
    n_clusters = int(len([c for c in unique_labels if c != -1]))
    pct_noise = float((labels == -1).mean() * 100.0) if n_points else float("nan")

    # Pour silhouette / DB / CH : besoin >= 2 clusters (hors bruit si présent)
    use_mask = labels != -1 if (-1 in unique_labels) else np.ones_like(labels, dtype=bool)
    X2 = X[use_mask]
    y2 = labels[use_mask]
    uniq2 = np.unique(y2)

    out = {
        "n_points": n_points,
        "n_clusters": n_clusters if (-1 in unique_labels) else int(len(unique_labels)),
        "pct_noise": pct_noise if (-1 in unique_labels) else 0.0,
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }

    if len(uniq2) < 2 or len(X2) < 3:
        return out

    # sampling
    if len(X2) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X2), size=sample_size, replace=False)
        Xs = X2[idx]
        ys = y2[idx]
    else:
        Xs, ys = X2, y2

    try:
        out["silhouette"] = float(silhouette_score(Xs, ys, metric="euclidean"))
    except Exception:
        pass
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(Xs, ys))
    except Exception:
        pass
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xs, ys))
    except Exception:
        pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Charge toutes les données nécessaires pour le dashboard."""
    data = {}

    # Données budgétaires
    budget_file = REPORTS_DIR / "analyse_budgetaire.xlsx"
    if budget_file.exists():
        data['budget_2024'] = pd.read_excel(budget_file, sheet_name="Budget_2024")
        data['budget_2025'] = pd.read_excel(budget_file, sheet_name="Budget_2025")

        piliers_2024 = pd.read_excel(budget_file, sheet_name="Piliers_2024", index_col=0)
        piliers_2025 = pd.read_excel(budget_file, sheet_name="Piliers_2025", index_col=0)
        comparison = pd.read_excel(budget_file, sheet_name="Comparaison_2024_2025", index_col=0)

        piliers_2024.index = piliers_2024.index.map(map_pilier_label)
        piliers_2025.index = piliers_2025.index.map(map_pilier_label)
        comparison.index = comparison.index.map(map_pilier_label)

        data['piliers_2024'] = piliers_2024
        data['piliers_2025'] = piliers_2025
        data['comparison'] = comparison

    # Classifications des objectifs budgétaires (fichiers générés par année)
    for year in [2024, 2025]:
        f = REPORTS_DIR / f"objectifs_classifications_snd30_{year}.xlsx"
        if f.exists():
            df = pd.read_excel(f)
            # Ajouter l'année si absente (rétro‑compatibilité)
            if "annee" not in df.columns:
                df["annee"] = year
            data[f"objectifs_classifications_{year}"] = df

    # Rétro-compatibilité : fusion des deux années si disponibles
    if 'objectifs_classifications_2024' in data and 'objectifs_classifications_2025' in data:
        data['objectifs_classifications'] = pd.concat(
            [data['objectifs_classifications_2024'], data['objectifs_classifications_2025']],
            ignore_index=True,
        )

    # Similarités des embeddings (fichier agrégé)
    emb_all = REPORTS_DIR / "embeddings_similarities.xlsx"
    if emb_all.exists():
        data['embeddings_similarities'] = pd.read_excel(emb_all)
    # Fichiers par année (rétro-compatibilité)
    for year in [2024, 2025]:
        f = REPORTS_DIR / f"embeddings_similarities_{year}.xlsx"
        if f.exists():
            data[f'similarities_{year}'] = pd.read_excel(f)

    # Topics LDA
    for year in [2024, 2025]:
        lda = REPORTS_DIR / f"lda_topics_{year}.xlsx"
        art = REPORTS_DIR / f"articles_{year}_avec_topics.xlsx"
        if lda.exists():
            data[f'lda_topics_{year}'] = pd.read_excel(lda)
        if art.exists():
            data[f'articles_topics_{year}'] = pd.read_excel(art)

    # Clusters
    for year in [2024, 2025]:
        f = REPORTS_DIR / f"articles_{year}_avec_clusters.xlsx"
        if f.exists():
            data[f'articles_clusters_{year}'] = pd.read_excel(f)

    # Tests statistiques
    for fname, key in [
        ("chi2_piliers.xlsx", 'chi2'),
        ("test_mannwhitney.xlsx", 'mannwhitney'),
    ]:
        f = REPORTS_DIR / fname
        if f.exists():
            data[key] = pd.read_excel(f)

    # Embeddings (vecteurs d'articles) + UMAP (projection 2D)
    emb_file_2024 = MODELS_DIR / "embeddings_2024.npy"
    emb_file_2025 = MODELS_DIR / "embeddings_2025.npy"
    if emb_file_2024.exists() and emb_file_2025.exists():
        try:
            emb_2024 = np.load(emb_file_2024)
            emb_2025 = np.load(emb_file_2025)
            data["embeddings_2024"] = emb_2024
            data["embeddings_2025"] = emb_2025

            def _safe_article_meta(year: int) -> pd.DataFrame:
                """Récupère des métadonnées article depuis les exports Excel si disponibles."""
                df = None
                df_topics = data.get(f"articles_topics_{year}")
                df_clusters = data.get(f"articles_clusters_{year}")

                if isinstance(df_topics, pd.DataFrame) and not df_topics.empty:
                    df = df_topics.copy()
                elif isinstance(df_clusters, pd.DataFrame) and not df_clusters.empty:
                    df = df_clusters.copy()
                else:
                    return pd.DataFrame()

                if isinstance(df_clusters, pd.DataFrame) and not df_clusters.empty:
                    if "id" in df.columns and "id" in df_clusters.columns:
                        right = df_clusters.copy()
                        dup = [c for c in right.columns if c in df.columns and c != "id"]
                        if dup:
                            right = right.drop(columns=dup)
                        df = df.merge(right, on="id", how="left")

                return df

            meta_2024 = _safe_article_meta(2024)
            meta_2025 = _safe_article_meta(2025)

            # Projection UMAP 2D (chargée si disponible, sinon générée à la volée
            # à partir des embeddings et sauvegardée dans outputs/models).
            df_umap = None
            try:
                combined = np.vstack([emb_2024, emb_2025])

                umap_coords_file = MODELS_DIR / "umap_coords.npy"
                umap_coords_2024_file = MODELS_DIR / "umap_2024.npy"
                umap_coords_2025_file = MODELS_DIR / "umap_2025.npy"

                coords = None
                if umap_coords_file.exists():
                    coords = np.load(umap_coords_file)
                elif umap_coords_2024_file.exists() and umap_coords_2025_file.exists():
                    coords_24 = np.load(umap_coords_2024_file)
                    coords_25 = np.load(umap_coords_2025_file)
                    coords = np.vstack([coords_24, coords_25])
                else:
                    # Aucune projection UMAP trouvée : la générer automatiquement
                    try:
                        import umap as umap_lib  # type: ignore[import]
                    except ImportError:
                        raise ImportError(
                            "Le package 'umap-learn' n'est pas installé. "
                            "Installez-le pour activer la visualisation UMAP."
                        )

                    reducer = umap_lib.UMAP(**UMAP_PARAMS)
                    coords = reducer.fit_transform(combined)

                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    np.save(umap_coords_file, coords)

                    n24_tmp = len(emb_2024)
                    coords_24 = coords[:n24_tmp]
                    coords_25 = coords[n24_tmp:]
                    np.save(umap_coords_2024_file, coords_24)
                    np.save(umap_coords_2025_file, coords_25)

                n24 = len(emb_2024)
                n25 = len(emb_2025)
                if coords is None or coords.shape[0] != (n24 + n25) or coords.shape[1] < 2:
                    raise ValueError(
                        f"Projection UMAP incohérente: coords={None if coords is None else coords.shape} "
                        f"vs n_total={n24+n25}"
                    )
                df_umap = pd.DataFrame(
                    {
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "annee": (["2024"] * n24) + (["2025"] * n25),
                        "row_idx": list(range(n24 + n25)),
                    }
                )

                def _attach_meta(df_umap_part: pd.DataFrame, meta: pd.DataFrame, n: int) -> pd.DataFrame:
                    if meta is None or meta.empty:
                        df_umap_part["id"] = [f"{df_umap_part['annee'].iloc[0]}_{i+1}" for i in range(n)]
                        return df_umap_part

                    if len(meta) >= n:
                        meta_part = meta.iloc[:n].copy()
                    else:
                        meta_part = meta.copy()
                        missing = n - len(meta_part)
                        if missing > 0:
                            meta_part = pd.concat(
                                [meta_part, pd.DataFrame(index=range(missing))],
                                ignore_index=True,
                            )

                    if "id" in meta_part.columns:
                        meta_part["id"] = meta_part["id"].astype(str)
                    else:
                        meta_part["id"] = [f"{df_umap_part['annee'].iloc[0]}_{i+1}" for i in range(n)]

                    keep_cols = [c for c in [
                        "id", "titre", "chapitre", "dominant_topic", "cluster_kmeans", "cluster_hdbscan"
                    ] if c in meta_part.columns]
                    meta_part = meta_part[keep_cols]

                    meta_part = meta_part.reset_index(drop=True)
                    df_umap_part = df_umap_part.reset_index(drop=True)
                    return pd.concat([df_umap_part, meta_part], axis=1)

                df_24 = df_umap.iloc[:n24].copy()
                df_25 = df_umap.iloc[n24:n24 + n25].copy()
                df_24 = _attach_meta(df_24, meta_2024, n24)
                df_25 = _attach_meta(df_25, meta_2025, n25)
                df_umap = pd.concat([df_24, df_25], ignore_index=True)

                data["umap_df"] = df_umap

            except Exception as e:
                data["umap_error"] = str(e)

        except Exception as e:
            data["embeddings_error"] = str(e)

    # Embeddings des objectifs budgétaires (par année)
    for year in [2024, 2025]:
        emb_obj_path = MODELS_DIR / f"embeddings_objectifs_{year}.npy"
        if emb_obj_path.exists():
            try:
                data[f"emb_objectifs_{year}"] = np.load(emb_obj_path)
            except Exception as e:
                data.setdefault("embeddings_objectifs_error", {})[str(year)] = str(e)

    return data


# Mapping noms longs → noms courts pour les piliers SND30
PILIER_LABEL_MAP = {long: court for long, court in zip(PILIERS_SND30, PILIERS_COURT)}


def map_pilier_label(value):
    """Retourne le nom court du pilier si disponible, sinon la valeur d'origine."""
    return PILIER_LABEL_MAP.get(value, value)


# Conteneur de données pour le dashboard.
#
# IMPORTANT : aucune lecture de fichiers n'est effectuée au moment de
# l'import du package. Les données sont chargées explicitement depuis le
# script d'exécution (scripts/run_dash.py) en appelant `load_all_data()`
# puis en affectant le résultat à cette variable globale.
DATA: dict = {}

# Couleurs des piliers SND30
PILIER_COLORS = {
    'Gouvernance': '#1f77b4',
    'Transformation_Structurelle': '#ff7f0e',
    'Capital_Humain': '#2ca02c',
    'Integration_Regionale': '#d62728'
}


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def create_wordcloud(text, colormap='viridis', max_words=50):
    """Crée un nuage de mots et le retourne en base64."""
    if not text or len(text.strip()) == 0:
        img = Image.new('RGB', (800, 400), color='white')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=max_words,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)

    img = wordcloud.to_image()
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_topic_words(topic_id, df_topics):
    """
    Extrait les mots d'un topic depuis un DataFrame LDA au format long.

    Le DataFrame ``df_topics`` doit avoir les colonnes : topic, word, probability.
    Retourne une chaîne de mots pondérée (répétée selon la probabilité) pour WordCloud.
    """
    if df_topics is None or df_topics.empty:
        return ""
    if "word" in df_topics.columns and "topic" in df_topics.columns:
        subset = df_topics[df_topics["topic"] == topic_id]
        if subset.empty:
            return ""
        words = []
        for _, row in subset.iterrows():
            word = str(row["word"])
            repeats = max(1, int(round(row.get("probability", 0.05) * 100)))
            words.extend([word] * repeats)
        return " ".join(words)
    # Format large legacy (word_0, word_1, ...) — rétro-compatibilité
    words = []
    for i in range(10):
        col = f"word_{i}"
        if col in df_topics.columns:
            word = df_topics[col].iloc[0] if len(df_topics) else None
            if word and pd.notna(word):
                words.append(str(word))
    return " ".join(words * 5)


def get_cluster_texts(df, cluster_id):
    """Récupère un texte agrégé nettoyé pour un cluster donné.

    Priorité :
    1) Colonnes déjà nettoyées (cleaned_title / cleaned_content) produites par le pipeline.
    2) À défaut, nettoyage à la volée de la colonne `content` avec TextPreprocessor.
    """
    cluster_articles = df[df["cluster_kmeans"] == cluster_id]
    if cluster_articles.empty:
        return ""

    texts: list[str] = []

    # 1) Utiliser les colonnes nettoyées si présentes (aligné avec les embeddings)
    has_clean_title = "cleaned_title" in cluster_articles.columns
    has_clean_content = "cleaned_content" in cluster_articles.columns
    if has_clean_title or has_clean_content:
        for _, row in cluster_articles.iterrows():
            parts: list[str] = []
            if has_clean_title:
                parts.append(str(row.get("cleaned_title", "")))
            if has_clean_content:
                parts.append(str(row.get("cleaned_content", "")))
            txt = " ".join(p for p in parts if p).strip()
            if txt:
                texts.append(txt)

        if texts:
            return " ".join(texts)

    # 2) Fallback : nettoyer `content` à la volée si disponible
    if "content" in cluster_articles.columns:
        prep = TextPreprocessor(lower=True, rm_accents=True)
        cleaned_series = (
            cluster_articles["content"]
            .fillna("")
            .astype(str)
            .apply(prep.preprocess)
        )
        return " ".join(cleaned_series.tolist())

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISATION DE L'APPLICATION DASH
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Audit Sémantique - Cameroun",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

# Sidebar
sidebar = html.Div([
    html.Div([
        html.Img(
            src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Flag_of_Cameroon.svg/320px-Flag_of_Cameroon.svg.png",
            style={'width': '100px', 'margin': '20px auto', 'display': 'block'}
        ),
        html.H3("Audit Sémantique", className="text-center text-white"),
        html.P("Loi de Finances du Cameroun", className="text-center text-white-50"),
        html.P("ISE3-DS | ISSEA Yaoundé | 2025-2026", className="text-center text-white-50 small"),
    ], style={'padding': '20px'}, className="sidebar-header"),

    html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)'}),

    dbc.Nav([
        dbc.NavLink([html.I(className="fas fa-home me-2"), "Accueil"],
                    href="/", active="exact", className="text-white sidebar-link"),
        dbc.NavLink([html.I(className="fas fa-chart-line me-2"), "Baromètre"],
                    href="/barometer", active="exact", className="text-white sidebar-link"),
        dbc.NavLink([html.I(className="fas fa-tags me-2"), "Classification SND30"],
                    href="/classification", active="exact", className="text-white sidebar-link"),
        dbc.NavLink([html.I(className="fas fa-brain me-2"), "Topic Modeling"],
                    href="/topics", active="exact", className="text-white sidebar-link"),
        dbc.NavLink([html.I(className="fas fa-project-diagram me-2"), "Clustering"],
                    href="/clustering", active="exact", className="text-white sidebar-link"),
        dbc.NavLink([html.I(className="fas fa-money-bill-wave me-2"), "Analyse Budgétaire"],
                    href="/budget", active="exact", className="text-white sidebar-link"),
        dbc.NavLink([html.I(className="fas fa-chart-bar me-2"), "Tests Statistiques"],
                    href="/stats", active="exact", className="text-white sidebar-link"),
    ], vertical=True, pills=True, class_name="sidebar-nav"),

    html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)', 'marginTop': '20px'}),

    html.Div([
        html.P("Modèles utilisés", className="text-white-50 small mb-1"),
        html.P("• Classification par mots-clés", className="text-white-50 small mb-0"),
        html.P("• Analyse budgétaire AE/CP", className="text-white-50 small mb-0"),
        html.P("• Visualisations Dash/Plotly", className="text-white-50 small mb-0"),
    ], style={'padding': '0 20px'}),
], className="sidebar", style={
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '280px',
    'padding': '0',
    'background': 'linear-gradient(180deg, #1e3a5f 0%, #2c5f8d 100%)',
    'overflowY': 'auto'
})

# Content
content = html.Div(
    id="page-content",
    className="content-area",
    style={
        'marginLeft': '280px',
        'padding': '30px',
        'backgroundColor': '#f8f9fa'
    }
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content
], className="app-layout")


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def create_home_page():
    """Page d'accueil."""
    return html.Div([
        html.H1("Audit Sémantique des Lois de Finances du Cameroun", className="mb-4"),

        dbc.Alert([
            html.H4("Dashboard Interactif d'Analyse Sémantique et Budgétaire", className="alert-heading"),
            html.P("Analyse comparative des lois de finances 2024 et 2025 du Cameroun selon les piliers de la SND30"),
        ], color="info"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("", className="text-center"),
                        html.H5("Baromètre", className="text-center"),
                        html.P("Glissement sémantique et distances", className="text-center small"),
                    ])
                ], className="mb-3 shadow-sm")
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("", className="text-center"),
                        html.H5("Classification SND30", className="text-center"),
                        html.P("Répartition par piliers stratégiques", className="text-center small"),
                    ])
                ], className="mb-3 shadow-sm")
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("", className="text-center"),
                        html.H5("Topic Modeling", className="text-center"),
                        html.P("Thématiques latentes (LDA)", className="text-center small"),
                    ])
                ], className="mb-3 shadow-sm")
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("", className="text-center"),
                        html.H5("Budget AE/CP", className="text-center"),
                        html.P("Analyse des dépenses publiques", className="text-center small"),
                    ])
                ], className="mb-3 shadow-sm")
            ], md=3),
        ]),

        html.Hr(className="my-4"),

        html.H3("À propos du projet", className="mb-3"),
        html.P("""
        Ce dashboard présente une analyse budgétaire et sémantique des lois de finances du Cameroun pour les années
        2024 et 2025. L'objectif est de mesurer l'alignement des dépenses publiques avec les piliers de la
        Stratégie Nationale de Développement (SND30 2020-2030).
        """),

        html.H4("Méthodologie", className="mb-2 mt-4"),
        html.Ul([
            html.Li("Classification des objectifs budgétaires par mots-clés selon les piliers SND30"),
            html.Li("Analyse des Autorisations d'Engagement (AE) et Crédits de Paiement (CP)"),
            html.Li("Visualisations interactives avec Dash et Plotly"),
            html.Li("Nuages de mots pour topics et clusters thématiques"),
            html.Li("Statistiques comparatives 2024 vs 2025"),
        ]),

        html.H4("Piliers SND30", className="mb-2 mt-4"),
        dbc.Row([
            dbc.Col([
                dbc.Badge("Gouvernance", color="primary", className="me-2"),
                html.Span("État de droit, démocratie, institutions"),
            ], md=6, className="mb-2"),
            dbc.Col([
                dbc.Badge("Transformation Structurelle", color="warning", className="me-2"),
                html.Span("Industrie, infrastructures, économie"),
            ], md=6, className="mb-2"),
            dbc.Col([
                dbc.Badge("Capital Humain", color="success", className="me-2"),
                html.Span("Éducation, santé, social"),
            ], md=6, className="mb-2"),
            dbc.Col([
                dbc.Badge("Intégration Régionale", color="danger", className="me-2"),
                html.Span("Coopération, diplomatie, CEMAC"),
            ], md=6, className="mb-2"),
        ]),

        html.Hr(className="my-4"),
        html.P("Développé par ISE3-DS | ISSEA Yaoundé | 2025-2026", className="text-muted text-center"),
    ], className="page-container page-home")


def create_topics_page():
    """Page Topic Modeling avec nuages de mots."""
    if 'lda_topics_2024' not in DATA or 'lda_topics_2025' not in DATA:
        return html.Div([
            dbc.Alert([
                html.H4("Données de topic modeling non disponibles", className="alert-heading"),
                html.P("Les analyses LDA n'ont pas été exécutées. Cette page affiche les topics thématiques issus du Topic Modeling."),
                html.Hr(),
                html.P("Pour générer ces données, exécutez :", className="mb-2"),
                html.Code("poetry run python scripts/run_pipeline.py", className="d-block p-2 bg-dark text-white"),
            ], color="warning")
        ])

    topics_2024 = DATA['lda_topics_2024']
    topics_2025 = DATA['lda_topics_2025']

    return html.Div([
        html.H1("Topic Modeling - Analyse LDA", className="mb-4"),

        dbc.Tabs([
            dbc.Tab(label="2024", tab_id="tab-2024"),
            dbc.Tab(label="2025", tab_id="tab-2025"),
            dbc.Tab(label="Comparaison", tab_id="tab-comparison"),
        ], id="topics-tabs", active_tab="tab-2024", class_name="page-tabs"),

        html.Div(id="topics-content", className="mt-4")
    ], className="page-container page-topics")


def create_clustering_page():
    """Page Clustering avec nuages de mots."""
    if 'articles_clusters_2024' not in DATA or 'articles_clusters_2025' not in DATA:
        return html.Div([
            dbc.Alert([
                html.H4("Données de clustering non disponibles", className="alert-heading"),
                html.P("Les analyses de clustering n'ont pas été exécutées. Cette page affiche les groupes d'articles similaires."),
                html.Hr(),
                html.P("Pour générer ces données, exécutez :", className="mb-2"),
                html.Code("poetry run python scripts/run_pipeline.py", className="d-block p-2 bg-dark text-white"),
            ], color="warning")
        ])

    return html.Div([
        html.H1("Clustering K-Means & HDBSCAN", className="mb-4"),

        dbc.Tabs([
            dbc.Tab(label="2024", tab_id="cluster-2024"),
            dbc.Tab(label="2025", tab_id="cluster-2025"),
        ], id="clustering-tabs", active_tab="cluster-2024", class_name="page-tabs"),

        html.Div(id="clustering-content", className="mt-4")
    ], className="page-container page-clustering")


def create_budget_page():
    """Page Analyse Budgétaire."""
    if 'budget_2024' not in DATA or 'budget_2025' not in DATA:
        return html.Div([
            dbc.Alert("Données budgétaires non disponibles. Exécutez: python scripts/analyze_budget.py", color="warning")
        ])

    budget_2024 = DATA['budget_2024']
    budget_2025 = DATA['budget_2025']
    piliers_2024 = DATA['piliers_2024']
    piliers_2025 = DATA['piliers_2025']
    comparison = DATA['comparison']

    total_ae_2024 = budget_2024['ae'].sum()
    total_cp_2024 = budget_2024['cp'].sum()
    total_ae_2025 = budget_2025['ae'].sum()
    total_cp_2025 = budget_2025['cp'].sum()

    return html.Div([
        html.H1("Analyse Budgétaire AE/CP 2024-2025", className="mb-4"),

        # Métriques globales
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("AE 2024", className="text-muted"),
                        html.H4(f"{total_ae_2024/1e9:.2f} Mds FCFA"),
                        html.Small(f"+{(total_ae_2025/total_ae_2024-1)*100:.1f}% vs 2025", className="text-success"),
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("CP 2024", className="text-muted"),
                        html.H4(f"{total_cp_2024/1e9:.2f} Mds FCFA"),
                        html.Small(f"+{(total_cp_2025/total_cp_2024-1)*100:.1f}% vs 2025", className="text-success"),
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("AE 2025", className="text-muted"),
                        html.H4(f"{total_ae_2025/1e9:.2f} Mds FCFA"),
                        html.Small(f"+{(total_ae_2025-total_ae_2024)/1e9:.2f} Mds", className="text-info"),
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("CP 2025", className="text-muted"),
                        html.H4(f"{total_cp_2025/1e9:.2f} Mds FCFA"),
                        html.Small(f"+{(total_cp_2025-total_cp_2024)/1e9:.2f} Mds", className="text-info"),
                    ])
                ])
            ], md=3),
        ], className="mb-4"),

        html.Hr(),

        # Sélecteur d'année
        dbc.Row([
            dbc.Col([
                html.Label("Sélectionnez l'année :"),
                dcc.RadioItems(
                    id='budget-year-selector',
                    options=[
                        {'label': ' 2024', 'value': '2024'},
                        {'label': ' 2025', 'value': '2025'},
                    ],
                    value='2025',
                    inline=True,
                    className="mb-3"
                )
            ])
        ]),

        # Graphiques
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='budget-pie-chart')
            ], md=6),
            dbc.Col([
                dcc.Graph(id='budget-bar-chart')
            ], md=6),
        ], className="mb-4"),

        html.Hr(),

        html.H3("Top 15 Lignes Budgétaires (AE)", className="mb-3"),
        dcc.Graph(id='budget-top-programs'),

        html.Hr(),

        html.H3("Évolution 2024 → 2025 par Pilier", className="mb-3"),
        dcc.Graph(id='budget-evolution-chart'),
    ], className="page-container page-budget")


def create_classification_page():
    """Page Classification SND30 des objectifs budgétaires."""
    if 'budget_2024' not in DATA or 'budget_2025' not in DATA:
        return html.Div([
            dbc.Alert("Données budgétaires non disponibles", color="warning")
        ])

    return html.Div([
        html.H1("Classification SND30 des Objectifs Budgétaires", className="mb-4"),

        dbc.Alert([
            html.P("Classification des objectifs des lignes budgétaires selon les piliers de la SND30", className="mb-0")
        ], color="info"),

        html.Hr(),

        html.H3("Distribution des Objectifs par Pilier", className="mb-3"),

        dbc.Tabs([
            dbc.Tab(label="2024", tab_id="classif-2024"),
            dbc.Tab(label="2025", tab_id="classif-2025"),
            dbc.Tab(label="Comparaison", tab_id="classif-comp"),
        ], id="classification-tabs", active_tab="classif-2024", class_name="page-tabs"),

        html.Div(id="classification-content", className="mt-4")
    ], className="page-container page-classification")


def create_barometer_page():
    """Page Baromètre avec statistiques budgétaires."""
    if 'budget_2024' not in DATA or 'budget_2025' not in DATA:
        return html.Div([
            dbc.Alert("Données budgétaires non disponibles", color="warning")
        ])

    budget_2024 = DATA['budget_2024']
    budget_2025 = DATA['budget_2025']
    piliers_2024 = DATA['piliers_2024']
    piliers_2025 = DATA['piliers_2025']
    comparison = DATA['comparison']

    total_ae_2024 = budget_2024['ae'].sum()
    total_ae_2025 = budget_2025['ae'].sum()
    croissance_ae = ((total_ae_2025 / total_ae_2024) - 1) * 100

    total_cp_2024 = budget_2024['cp'].sum()
    total_cp_2025 = budget_2025['cp'].sum()
    croissance_cp = ((total_cp_2025 / total_cp_2024) - 1) * 100

    n_lignes_2024 = len(budget_2024)
    n_lignes_2025 = len(budget_2025)

    # ─── Baromètre de glissement sémantique ─────────────────────────────────
    seuil = AUDIT_PARAMS.get("seuil_changement", 0.70)
    sim_df = None

    if 'embeddings_similarities' in DATA:
        sim_df = DATA['embeddings_similarities'].copy()
    elif 'similarities_2024' in DATA or 'similarities_2025' in DATA:
        parts = []
        if 'similarities_2024' in DATA:
            tmp = DATA['similarities_2024'].copy()
            tmp['annee'] = 2024
            parts.append(tmp)
        if 'similarities_2025' in DATA:
            tmp = DATA['similarities_2025'].copy()
            tmp['annee'] = 2025
            parts.append(tmp)
        if parts:
            sim_df = pd.concat(parts, ignore_index=True)

    mean_sim_2024 = mean_sim_2025 = pct_changement_2024 = pct_changement_2025 = None
    fig_glissement = go.Figure()

    if sim_df is not None and not sim_df.empty and 'score_max_similarite' in sim_df.columns:
        if 'annee' not in sim_df.columns:
            sim_df['annee'] = 2024

        sims_2024 = sim_df[sim_df['annee'] == 2024]['score_max_similarite'].dropna()
        sims_2025 = sim_df[sim_df['annee'] == 2025]['score_max_similarite'].dropna()

        if not sims_2024.empty:
            mean_sim_2024 = sims_2024.mean()
            pct_changement_2024 = (sims_2024 < seuil).mean() * 100
        if not sims_2025.empty:
            mean_sim_2025 = sims_2025.mean()
            pct_changement_2025 = (sims_2025 < seuil).mean() * 100

        fig_glissement = px.histogram(
            sim_df,
            x='score_max_similarite',
            color='annee',
            nbins=30,
            barmode='overlay',
            marginal='box',
            title="Distribution des similarités cosinus par année (embeddings)",
        )
        fig_glissement.update_traces(opacity=0.65)
        fig_glissement.add_vline(
            x=seuil,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Seuil changement = {seuil:.2f}",
            annotation_position="top right"
        )

    return html.Div([
        html.H1("Baromètre Budgétaire & Sémantique 2024-2025", className="mb-4"),

        # Métriques principales
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Croissance AE", className="text-muted"),
                        html.H3(f"+{croissance_ae:.1f}%", className="text-success"),
                        html.P(f"{total_ae_2024/1e9:.2f} → {total_ae_2025/1e9:.2f} Mds", className="small text-muted"),
                    ])
                ], className="shadow-sm")
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Croissance CP", className="text-muted"),
                        html.H3(f"+{croissance_cp:.1f}%", className="text-success"),
                        html.P(f"{total_cp_2024/1e9:.2f} → {total_cp_2025/1e9:.2f} Mds", className="small text-muted"),
                    ])
                ], className="shadow-sm")
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Lignes Budgétaires", className="text-muted"),
                        html.H3(f"{n_lignes_2024} / {n_lignes_2025}"),
                        html.P("2024 / 2025", className="small text-muted"),
                    ])
                ], className="shadow-sm")
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Piliers SND30", className="text-muted"),
                        html.H3("4"),
                        html.P("Axes stratégiques", className="small text-muted"),
                    ])
                ], className="shadow-sm")
            ], md=3),
        ], className="mb-4"),

        html.Hr(),

        dbc.Tabs([
            # ── Onglet Budget & Piliers ───────────────────────────────────────
            dbc.Tab(
                label="Budget & Piliers",
                children=[
                    html.Br(),
                    html.H4("Évolution par Pilier SND30", className="mb-3"),
                    dcc.Graph(id='barometer-evolution-piliers'),
                    html.Hr(),
                    html.H4("Concentration Budgétaire", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.H5("2024"),
                            dcc.Graph(id='barometer-concentration-2024')
                        ], md=6),
                        dbc.Col([
                            html.H5("2025"),
                            dcc.Graph(id='barometer-concentration-2025')
                        ], md=6),
                    ]),
                    html.Hr(),
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                title="Baromètre des objectifs (SND30) — Wordclouds & Répartition",
                                children=[
                                    dbc.Alert(
                                        "Nuages de mots des objectifs budgétaires par pilier (stopwords supprimés via `TextPreprocessor`).",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Année :"),
                                            dcc.RadioItems(
                                                id="barometer-objectifs-year",
                                                options=[
                                                    {"label": "2024", "value": "2024"},
                                                    {"label": "2025", "value": "2025"},
                                                    {"label": "Toutes", "value": "all"},
                                                ],
                                                value="all",
                                                inline=True,
                                            ),
                                        ], md=12),
                                    ], className="mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.H5("Répartition des objectifs par pilier", className="mb-2"),
                                            dcc.Graph(id="barometer-pilier-pie"),
                                        ], md=6),
                                        dbc.Col([
                                            html.H5("Nombre d'objectifs par pilier", className="mb-2"),
                                            dcc.Graph(id="barometer-pilier-count-bar"),
                                        ], md=6),
                                    ], className="mb-4"),
                                    html.H5("Nuages de mots par pilier (objectifs)", className="mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardHeader("Transformation économique"),
                                                dbc.CardBody(
                                                    html.Img(id="barometer-wc-pilier-0", style={"width": "100%"})
                                                ),
                                            ], className="shadow-sm"),
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardHeader("Capital humain"),
                                                dbc.CardBody(
                                                    html.Img(id="barometer-wc-pilier-1", style={"width": "100%"})
                                                ),
                                            ], className="shadow-sm"),
                                        ], md=6),
                                    ], className="mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardHeader("Emploi et insertion"),
                                                dbc.CardBody(
                                                    html.Img(id="barometer-wc-pilier-2", style={"width": "100%"})
                                                ),
                                            ], className="shadow-sm"),
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardHeader("Gouvernance et décentralisation"),
                                                dbc.CardBody(
                                                    html.Img(id="barometer-wc-pilier-3", style={"width": "100%"})
                                                ),
                                            ], className="shadow-sm"),
                                        ], md=6),
                                    ]),
                                ],
                            )
                        ],
                        start_collapsed=True,
                        className="mt-2",
                    ),
                ],
            ),

            # ── Onglet Glissement sémantique ─────────────────────────────────
            dbc.Tab(
                label="Glissement sémantique",
                children=[
                    html.Br(),
                    html.H4("Baromètre de glissement sémantique (embeddings)", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Similarité moyenne 2024", className="text-muted"),
                                    html.H3(
                                        f"{mean_sim_2024:.2f}" if mean_sim_2024 is not None else "—",
                                        className="text-primary"
                                    ),
                                    html.P(
                                        f"{pct_changement_2024:.1f}% d'articles sous le seuil {seuil:.2f}"
                                        if pct_changement_2024 is not None else
                                        "Données de similarité non disponibles pour 2024",
                                        className="small text-muted"
                                    ),
                                ])
                            ], className="shadow-sm")
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Similarité moyenne 2025", className="text-muted"),
                                    html.H3(
                                        f"{mean_sim_2025:.2f}" if mean_sim_2025 is not None else "—",
                                        className="text-primary"
                                    ),
                                    html.P(
                                        f"{pct_changement_2025:.1f}% d'articles sous le seuil {seuil:.2f}"
                                        if pct_changement_2025 is not None else
                                        "Données de similarité non disponibles pour 2025",
                                        className="small text-muted"
                                    ),
                                ])
                            ], className="shadow-sm")
                        ], md=6),
                    ], className="mb-4"),
                    dcc.Graph(figure=fig_glissement),
                ],
            ),

            # ── Onglet Correspondances objectifs ↔ articles ────────────────
            dbc.Tab(
                label="🔗 Objectifs ↔ Articles",
                children=[
                    html.Br(),
                    html.H4("🔗 Baromètre objectifs ↔ articles", className="mb-3"),
                    dbc.Alert(
                        "Sélectionnez un objectif budgétaire pour voir les 3 objectifs les plus similaires "
                        "(avec leurs piliers) et les 2 articles de loi les plus proches dans l'espace des embeddings.",
                        color="info",
                        className="mb-3",
                    ),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Année des objectifs :"),
                            dcc.RadioItems(
                                id="obj-art-year",
                                options=[
                                    {"label": "2024", "value": "2024"},
                                    {"label": "2025", "value": "2025"},
                                ],
                                value="2025",
                                inline=True,
                            ),
                        ], md=3),
                        dbc.Col([
                            html.Label("Objectif budgétaire :"),
                            dcc.Dropdown(
                                id="obj-art-select",
                                placeholder="Choisissez un objectif…",
                                options=[],
                                value=None,
                            ),
                        ], md=9),
                    ], className="mb-4"),

                    dbc.Row([
                        dbc.Col([
                            html.H5("Objectif sélectionné", className="mb-2"),
                            html.Div(id="obj-art-summary"),
                        ], md=12),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            html.H5("Top 3 objectifs similaires", className="mb-2"),
                            html.Div(id="obj-art-top-objectifs"),
                        ], md=6),
                        dbc.Col([
                            html.H5("Top 2 articles similaires", className="mb-2"),
                            html.Div(id="obj-art-top-articles"),
                        ], md=6),
                    ]),
                ],
            ),

            # ── Onglet Embeddings UMAP ────────────────────────────────────────
            dbc.Tab(
                label="Embeddings (UMAP)",
                children=[
                    html.Br(),
                    html.H4("Visualisation des embeddings (UMAP)", className="mb-3"),
                    dbc.Alert(
                        "Projection 2D des vecteurs d'embedding (sentence-transformers). Survolez un point pour voir l'ID/titre (si disponible).",
                        color="secondary",
                        className="mb-3",
                    ),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Filtrer par année :"),
                            dcc.Checklist(
                                id="barometer-umap-year-filter",
                                options=[
                                    {"label": "2024", "value": "2024"},
                                    {"label": "2025", "value": "2025"},
                                ],
                                value=["2024", "2025"],
                                inline=True,
                            ),
                        ], md=4),
                        dbc.Col([
                            html.Label("Filtrer par cluster K-Means :"),
                            dcc.Dropdown(
                                id="barometer-umap-cluster-filter",
                                multi=True,
                                placeholder="Tous les clusters",
                            ),
                        ], md=4),
                        dbc.Col([
                            html.Label("Filtrer par titre :"),
                            dcc.Dropdown(
                                id="barometer-umap-title-filter",
                                multi=True,
                                placeholder="Tous les titres",
                            ),
                        ], md=4),
                    ], className="mb-3"),
                    dcc.Graph(id="barometer-umap-graph"),
                    html.Br(),
                    html.H5("Distribution des clusters K-Means par année", className="mb-3"),
                    dcc.Graph(id="barometer-umap-cluster-bar"),
                ],
            ),

            # ── Onglet Observations clés ──────────────────────────────────────
            dbc.Tab(
                label="Observations clés",
                children=[
                    html.Br(),
                    html.H4("Observations Clés", className="mb-3"),
                    dbc.Alert([
                        html.H5("Points saillants", className="alert-heading"),
                        html.Ul([
                            html.Li(f"Le budget total AE augmente de {croissance_ae:.1f}% entre 2024 et 2025"),
                            html.Li(
                                f"Le pilier {piliers_2025['pct_ae'].idxmax()} concentre "
                                f"{piliers_2025['pct_ae'].max():.1f}% des AE en 2025"
                            ) if 'pct_ae' in piliers_2025.columns else html.Li("Données de piliers non disponibles"),
                            html.Li(
                                f"Plus forte évolution: {comparison['evolution'].idxmax()} "
                                f"(+{comparison['evolution'].max():.1f}%)"
                            ) if 'evolution' in comparison.columns else html.Li(""),
                            html.Li(f"Écart AE-CP 2025: {(total_ae_2025 - total_cp_2025)/1e9:.2f} Mds FCFA"),
                        ])
                    ], color="info"),
                ],
            ),
        ]),
    ])


def create_stats_page():
    """Page Tests Statistiques."""
    if 'budget_2024' not in DATA or 'budget_2025' not in DATA:
        return html.Div([
            dbc.Alert("Données non disponibles", color="warning")
        ])

    return html.Div([
        html.H1("Tests Statistiques et Analyses", className="mb-4"),

        dbc.Tabs([
            dbc.Tab(label="Statistiques Descriptives", tab_id="stats-desc"),
            dbc.Tab(label="Tests Comparatifs", tab_id="stats-tests"),
            dbc.Tab(label="Distributions", tab_id="stats-dist"),
        ], id="stats-tabs", active_tab="stats-desc", class_name="page-tabs"),

        html.Div(id="stats-content", className="mt-4")
    ], className="page-container page-stats")


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Router principal pour les pages."""
    if pathname == '/topics':
        return create_topics_page()
    elif pathname == '/clustering':
        return create_clustering_page()
    elif pathname == '/budget':
        return create_budget_page()
    elif pathname == '/classification':
        return create_classification_page()
    elif pathname == '/barometer':
        return create_barometer_page()
    elif pathname == '/stats':
        return create_stats_page()
    else:
        return create_home_page()


@app.callback(
    Output('topics-content', 'children'),
    Input('topics-tabs', 'active_tab')
)
def render_topics_content(active_tab):
    """Affiche le contenu de la page Topics selon l'onglet sélectionné."""
    if active_tab == 'tab-2024':
        return create_topics_year_content(2024)
    elif active_tab == 'tab-2025':
        return create_topics_year_content(2025)
    elif active_tab == 'tab-comparison':
        return create_topics_comparison_content()


@app.callback(
    [
        Output("obj-art-select", "options"),
        Output("obj-art-select", "value"),
    ],
    Input("obj-art-year", "value"),
)
def update_obj_art_options(year_value: str):
    """Met à jour la liste des objectifs disponibles pour l'année choisie."""
    try:
        year_int = int(year_value)
    except (TypeError, ValueError):
        return [], None

    key = f"objectifs_classifications_{year_int}"
    df = DATA.get(key)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return [], None

    if "id_objectif" not in df.columns:
        # Rétro‑compatibilité : index comme identifiant
        df = df.reset_index().rename(columns={"index": "id_objectif"})
        DATA[key] = df

    options = []
    for _, row in df.iterrows():
        obj_txt = str(row.get("objectif", "")).strip()
        prog_txt = str(row.get("programme", "")).strip()
        label = prog_txt if prog_txt else obj_txt
        if obj_txt:
            label = f"{prog_txt} — {obj_txt}" if prog_txt else obj_txt
        # Tronquer pour éviter des labels trop longs
        if len(label) > 160:
            label = label[:157] + "…"
        options.append({"label": label, "value": int(row["id_objectif"])})

    default_value = options[0]["value"] if options else None
    return options, default_value


@app.callback(
    [
        Output("obj-art-summary", "children"),
        Output("obj-art-top-objectifs", "children"),
        Output("obj-art-top-articles", "children"),
    ],
    [
        Input("obj-art-year", "value"),
        Input("obj-art-select", "value"),
    ],
)
def update_obj_art_barometer(year_value: str, objectif_id: int):
    """Calcule les top 3 objectifs et top 2 articles similaires pour un objectif donné."""
    if not year_value or objectif_id is None:
        msg = dbc.Alert("Sélectionnez une année et un objectif.", color="secondary")
        return msg, html.Div(), html.Div()

    try:
        year_int = int(year_value)
    except (TypeError, ValueError):
        msg = dbc.Alert("Année invalide.", color="danger")
        return msg, html.Div(), html.Div()

    key_obj = f"objectifs_classifications_{year_int}"
    emb_obj_key = f"emb_objectifs_{year_int}"
    emb_art_key = f"embeddings_{year_int}"
    art_key = f"articles_clusters_{year_int}"

    df_obj = DATA.get(key_obj)
    emb_obj = DATA.get(emb_obj_key)
    emb_art = DATA.get(emb_art_key)
    df_art = DATA.get(art_key)

    if not isinstance(df_obj, pd.DataFrame) or df_obj.empty:
        msg = dbc.Alert("Données d'objectifs non disponibles pour cette année.", color="warning")
        return msg, html.Div(), html.Div()
    if not isinstance(emb_obj, np.ndarray) or emb_obj.ndim != 2:
        msg = dbc.Alert("Embeddings des objectifs non disponibles pour cette année.", color="warning")
        return msg, html.Div(), html.Div()

    # S'assurer de la présence d'un identifiant aligné avec les embeddings
    if "id_objectif" not in df_obj.columns:
        df_obj = df_obj.reset_index().rename(columns={"index": "id_objectif"})
        DATA[key_obj] = df_obj

    # Récupérer la ligne de l'objectif sélectionné
    sub = df_obj[df_obj["id_objectif"].astype(int) == int(objectif_id)]
    if sub.empty:
        msg = dbc.Alert("Objectif sélectionné introuvable dans les données.", color="danger")
        return msg, html.Div(), html.Div()

    row = sub.iloc[0]
    idx = int(row["id_objectif"])
    if idx < 0 or idx >= emb_obj.shape[0]:
        msg = dbc.Alert("Incohérence entre les embeddings et les objectifs (index hors bornes).", color="danger")
        return msg, html.Div(), html.Div()

    v = emb_obj[idx]

    # Top 3 objectifs similaires (même année)
    sims_obj = emb_obj @ v
    sims_obj[idx] = -1.0  # exclure l'objectif lui‑même
    order_obj = np.argsort(-sims_obj)
    top_idx_obj = [int(i) for i in order_obj[:3] if 0 <= i < emb_obj.shape[0]]

    df_top_obj = df_obj.iloc[top_idx_obj].copy() if top_idx_obj else pd.DataFrame()
    if not df_top_obj.empty:
        df_top_obj["similarite"] = [float(sims_obj[i]) for i in top_idx_obj]

    # Top 2 articles similaires (si embeddings + métadonnées disponibles)
    df_top_art = pd.DataFrame()
    if isinstance(emb_art, np.ndarray) and emb_art.ndim == 2 and isinstance(df_art, pd.DataFrame):
        try:
            sims_art = emb_art @ v
            order_art = np.argsort(-sims_art)
            top_idx_art = [int(i) for i in order_art[:2] if 0 <= i < emb_art.shape[0]]
            df_top_art = df_art.iloc[top_idx_art].copy()
            df_top_art["similarite"] = [float(sims_art[i]) for i in top_idx_art]
        except Exception:
            df_top_art = pd.DataFrame()

    # Résumé de l'objectif sélectionné
    pilier_dom = row.get("pilier_dominant", "N/A")
    score_dom = row.get("score_pilier_dominant", None)
    # Construction du résumé de l'objectif sélectionné sans concaténer directement
    # des composants Dash avec des chaînes de caractères.
    if isinstance(score_dom, (int, float)):
        score_children = [html.B("Score pilier dominant : "), f"{float(score_dom):.3f}"]
    else:
        score_children = [html.B("Score pilier dominant : "), html.Span("N/A")]

    summary = dbc.Card([
        dbc.CardBody([
            html.P(f"Année : {year_int}", className="mb-1"),
            html.P([html.B("Programme : "), str(row.get("programme", "N/A"))]),
            html.P([html.B("Objectif : "), str(row.get("objectif", "N/A"))]),
            html.P([html.B("Pilier dominant : "), str(pilier_dom)]),
            html.P(score_children),
        ])
    ], className="mb-3")

    # Tableau Top 3 objectifs
    if df_top_obj.empty:
        top_obj_div = dbc.Alert("Pas d'autres objectifs comparables trouvés.", color="secondary")
    else:
        cols = [
            "id_objectif", "programme", "objectif", "pilier_dominant",
            "score_pilier_dominant", "similarite",
        ]
        display_cols = [c for c in cols if c in df_top_obj.columns]
        df_disp = df_top_obj[display_cols].copy()
        if "score_pilier_dominant" in df_disp.columns:
            df_disp["score_pilier_dominant"] = df_disp["score_pilier_dominant"].astype(float).round(3)
        if "similarite" in df_disp.columns:
            df_disp["similarite"] = df_disp["similarite"].astype(float).round(3)
        top_obj_div = dbc.Table.from_dataframe(df_disp, striped=True, bordered=True, hover=True)

    # Tableau Top 2 articles
    if df_top_art.empty:
        top_art_div = dbc.Alert("Embeddings / métadonnées articles non disponibles pour calculer les similarités.", color="secondary")
    else:
        cols_art = ["id", "titre", "chapitre", "cluster_kmeans", "similarite"]
        display_cols_art = [c for c in cols_art if c in df_top_art.columns]
        df_art_disp = df_top_art[display_cols_art].copy()
        if "similarite" in df_art_disp.columns:
            df_art_disp["similarite"] = df_art_disp["similarite"].astype(float).round(3)
        top_art_div = dbc.Table.from_dataframe(df_art_disp, striped=True, bordered=True, hover=True)

    return summary, top_obj_div, top_art_div


def create_topics_year_content(year):
    """Crée le contenu pour une année donnée (topics)."""
    topics_df = DATA[f'lda_topics_{year}']

    if "topic" in topics_df.columns:
        topic_ids = sorted(topics_df["topic"].unique().tolist())
    else:
        topic_ids = list(range(4))

    topic_selector = dbc.Row([
        dbc.Col([
            html.Label(f"Sélectionnez un topic {year} :"),
            dcc.Dropdown(
                id=f'topic-selector-{year}',
                options=[{'label': f'Topic {i}', 'value': i} for i in topic_ids],
                value=topic_ids[0] if topic_ids else 0,
                clearable=False
            )
        ], md=6)
    ], className="mb-4")

    return html.Div([
        topic_selector,
        html.Div(id=f'topic-details-{year}')
    ])


@app.callback(
    Output('topic-details-2024', 'children'),
    Input('topic-selector-2024', 'value')
)
def update_topic_2024(topic_id):
    """Met à jour les détails du topic 2024."""
    return render_topic_details(2024, topic_id)


@app.callback(
    Output('topic-details-2025', 'children'),
    Input('topic-selector-2025', 'value')
)
def update_topic_2025(topic_id):
    """Met à jour les détails du topic 2025."""
    return render_topic_details(2025, topic_id)


def render_topic_details(year, topic_id):
    """Affiche les détails d'un topic avec nuage de mots."""
    topics_df = DATA[f'lda_topics_{year}']
    articles_df = DATA.get(f'articles_topics_{year}')

    words = []
    if "word" in topics_df.columns and "topic" in topics_df.columns:
        subset = topics_df[topics_df["topic"] == topic_id]
        words = subset["word"].dropna().astype(str).tolist()

    topic_text = extract_topic_words(topic_id, topics_df)
    wordcloud_img = create_wordcloud(topic_text, colormap='viridis')

    if articles_df is not None and "dominant_topic" in articles_df.columns:
        n_articles = int((articles_df["dominant_topic"] == topic_id).sum())
    else:
        n_articles = 0

    fig_words = go.Figure()
    if "word" in topics_df.columns and "topic" in topics_df.columns:
        sub = topics_df[topics_df["topic"] == topic_id].copy()
        if "probability" in sub.columns:
            sub["probability"] = _safe_numeric(sub["probability"]).fillna(0.0)
            sub = sub.sort_values("probability", ascending=False).head(15)
            fig_words = px.bar(
                sub.sort_values("probability", ascending=True),
                x="probability",
                y="word",
                orientation="h",
                title="Top mots (pondérés) du topic",
                labels={"probability": "Probabilité", "word": "Mot"},
            )
            fig_words.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        else:
            counts = pd.Series(words).value_counts().head(15).reset_index()
            counts.columns = ["word", "count"]
            fig_words = px.bar(
                counts.sort_values("count", ascending=True),
                x="count",
                y="word",
                orientation="h",
                title="Top mots du topic (non pondéré)",
                labels={"count": "Fréquence", "word": "Mot"},
            )
            fig_words.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))

    fig_prev = go.Figure()
    if articles_df is not None and "dominant_topic" in articles_df.columns:
        vc = (
            articles_df["dominant_topic"]
            .fillna(-1)
            .astype(int)
            .value_counts()
            .sort_index()
            .reset_index()
        )
        vc.columns = ["topic", "nb_articles"]
        vc["topic"] = vc["topic"].astype(int)
        vc["selected"] = vc["topic"].apply(lambda t: "Sélectionné" if int(t) == int(topic_id) else "Autres")
        fig_prev = px.bar(
            vc,
            x="topic",
            y="nb_articles",
            color="selected",
            title="Prévalence des topics (topic dominant)",
            color_discrete_map={"Sélectionné": "#d62728", "Autres": "#1f77b4"},
        )
        fig_prev.update_layout(height=320, xaxis_title="Topic", yaxis_title="Nb articles", showlegend=False)

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4(f"Topic {topic_id} - {year}")),
                    dbc.CardBody([
                        html.H6("Mots-clés principaux :"),
                        html.P(', '.join(words), className="text-primary"),
                        html.Hr(),
                        html.H6("Statistiques :"),
                        html.P(f"{n_articles} articles associés (score > 0.3)"),
                    ])
                ])
            ], md=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Nuage de Mots")),
                    dbc.CardBody([
                        html.Img(
                            src=f'data:image/png;base64,{wordcloud_img}',
                            style={'width': '100%', 'height': 'auto'}
                        )
                    ])
                ])
            ], md=6),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_prev)], md=5),
            dbc.Col([dcc.Graph(figure=fig_words)], md=7),
        ]),
    ])


def create_topics_comparison_content():
    """Crée le contenu de comparaison des topics 2024 vs 2025."""
    topics_2024 = DATA['lda_topics_2024']
    topics_2025 = DATA['lda_topics_2025']
    articles_2024 = DATA.get("articles_topics_2024")
    articles_2025 = DATA.get("articles_topics_2025")

    def _top_words(df, topic_id, n=5):
        if "word" not in df.columns or "topic" not in df.columns:
            return []
        sub = df[df["topic"] == topic_id].nlargest(n, "probability") if "probability" in df.columns \
            else df[df["topic"] == topic_id].head(n)
        return sub["word"].dropna().astype(str).tolist()

    topic_ids_2024 = sorted(topics_2024["topic"].unique()) if "topic" in topics_2024.columns else range(4)
    topic_ids_2025 = sorted(topics_2025["topic"].unique()) if "topic" in topics_2025.columns else range(4)

    fig_prev = go.Figure()
    if (
        isinstance(articles_2024, pd.DataFrame)
        and isinstance(articles_2025, pd.DataFrame)
        and "dominant_topic" in articles_2024.columns
        and "dominant_topic" in articles_2025.columns
    ):
        c24 = articles_2024["dominant_topic"].fillna(-1).astype(int).value_counts().sort_index()
        c25 = articles_2025["dominant_topic"].fillna(-1).astype(int).value_counts().sort_index()
        idx = sorted(set(c24.index.tolist()) | set(c25.index.tolist()))
        df_prev = pd.DataFrame(
            {
                "topic": idx,
                "2024": [int(c24.get(i, 0)) for i in idx],
                "2025": [int(c25.get(i, 0)) for i in idx],
            }
        ).melt(id_vars=["topic"], var_name="annee", value_name="nb_articles")
        fig_prev = px.bar(
            df_prev,
            x="topic",
            y="nb_articles",
            color="annee",
            barmode="group",
            title="Prévalence des topics (topic dominant) : 2024 vs 2025",
        )
        fig_prev.update_layout(height=420, xaxis_title="Topic", yaxis_title="Nb articles")

    def _prepare_top_words(df: pd.DataFrame, year_label: str) -> pd.DataFrame:
        if df is None or df.empty or "topic" not in df.columns or "word" not in df.columns:
            return pd.DataFrame()
        d = df.copy()
        d["annee"] = year_label
        if "probability" in d.columns:
            d["probability"] = _safe_numeric(d["probability"]).fillna(0.0)
            d = d.sort_values(["topic", "probability"], ascending=[True, False])
            d = d.groupby("topic").head(8)
        else:
            d["probability"] = 1.0
            d = d.groupby("topic").head(8)
        return d[["annee", "topic", "word", "probability"]]

    dfw = pd.concat(
        [
            _prepare_top_words(topics_2024, "2024"),
            _prepare_top_words(topics_2025, "2025"),
        ],
        ignore_index=True,
    )

    fig_words = go.Figure()
    if not dfw.empty:
        fig_words = px.bar(
            dfw,
            x="probability",
            y="word",
            color="annee",
            facet_col="topic",
            orientation="h",
            title="Top mots par topic (pondération LDA) — comparaison 2024 vs 2025",
        )
        fig_words.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        fig_words.for_each_annotation(lambda a: a.update(text=a.text.replace("topic=", "Topic ")))

    return html.Div([
        html.H3("Comparaison des Topics 2024 vs 2025", className="mb-4"),
        dbc.Alert("Comparaison des thématiques : prévalence (topic dominant) et mots-clés (LDA).", color="info"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_prev)], md=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_words)], md=12),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H5("Topics 2024 (top mots)"),
                html.Ul([html.Li(f"Topic {i}: {', '.join(_top_words(topics_2024, i))}") for i in topic_ids_2024])
            ], md=6),
            dbc.Col([
                html.H5("Topics 2025 (top mots)"),
                html.Ul([html.Li(f"Topic {i}: {', '.join(_top_words(topics_2025, i))}") for i in topic_ids_2025])
            ], md=6),
        ]),
    ])


@app.callback(
    Output('clustering-content', 'children'),
    Input('clustering-tabs', 'active_tab')
)
def render_clustering_content(active_tab):
    """Affiche le contenu de la page Clustering selon l'onglet."""
    year = int(active_tab.split('-')[1])
    return create_clustering_year_content(year)


def create_clustering_year_content(year):
    """Crée le contenu clustering pour une année donnée."""
    clusters_df = DATA[f'articles_clusters_{year}']

    if 'cluster_kmeans' not in clusters_df.columns:
        return dbc.Alert("Colonne cluster_kmeans non trouvée", color="warning")
    n_clusters = clusters_df['cluster_kmeans'].nunique()

    # ── Répartition des articles par cluster (sans UMAP) ──────────────────
    # K-Means : nombre d'articles dans chaque cluster
    km_counts = clusters_df['cluster_kmeans'].value_counts().sort_index()
    fig_kmeans = px.bar(
        x=km_counts.index.astype(str),
        y=km_counts.values,
        labels={"x": "Cluster K-Means", "y": "Nombre d'articles"},
        title=f"Répartition des articles par cluster K-Means ({year})",
    )

    # HDBSCAN : nombre d'articles par cluster (incluant le bruit -1)
    fig_hdbscan = go.Figure()
    if 'cluster_hdbscan' in clusters_df.columns:
        hdb_counts = clusters_df['cluster_hdbscan'].value_counts().sort_index()
        fig_hdbscan = px.bar(
            x=hdb_counts.index.astype(str),
            y=hdb_counts.values,
            labels={"x": "Cluster HDBSCAN (-1 = bruit)", "y": "Nombre d'articles"},
            title=f"Répartition des articles par cluster HDBSCAN ({year})",
        )
    else:
        fig_hdbscan.add_annotation(
            text="Labels HDBSCAN non disponibles pour cette année.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig_hdbscan.update_xaxes(visible=False)
        fig_hdbscan.update_yaxes(visible=False)

    # ── Courbe du coude (inertie K-Means) ─────────────────────────────────
    from sklearn.cluster import KMeans  # import local pour éviter le coût global

    fig_elbow = go.Figure()
    emb_key = f"embeddings_{year}"
    X = DATA.get(emb_key)
    if isinstance(X, np.ndarray) and X.shape[0] >= 3:
        max_k = max(2, min(10, X.shape[0] - 1))
        ks: list[int] = []
        inertias: list[float] = []
        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                ks.append(k)
                inertias.append(float(km.inertia_))
            except Exception:
                continue
        if ks:
            fig_elbow.add_trace(
                go.Scatter(
                    x=ks,
                    y=inertias,
                    mode="lines+markers",
                    name="Inertie intra-cluster",
                )
            )
            fig_elbow.update_layout(
                title=f"Règle du coude — K-Means ({year})",
                xaxis_title="Nombre de clusters k",
                yaxis_title="Inertie",
            )
        else:
            fig_elbow.add_annotation(
                text="Impossible de calculer l'inertie (données insuffisantes).",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            fig_elbow.update_xaxes(visible=False)
            fig_elbow.update_yaxes(visible=False)
    else:
        fig_elbow.add_annotation(
            text="Embeddings non disponibles pour cette année.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig_elbow.update_xaxes(visible=False)
        fig_elbow.update_yaxes(visible=False)

    # ── Métriques de performance des clusters ─────────────────────────────

    def _get_metrics(year_: int, algo: str) -> dict:
        """Récupère (avec cache) les métriques de clustering pour une année / algo."""
        cache_key = (year_, algo)
        if cache_key in _CLUSTER_METRICS_CACHE:
            return _CLUSTER_METRICS_CACHE[cache_key]

        emb_key = f"embeddings_{year_}"
        X = DATA.get(emb_key)
        if X is None:
            return {}

        col = f"cluster_{algo}"
        if col not in clusters_df.columns:
            return {}

        labels = clusters_df[col].to_numpy()
        n = min(len(X), len(labels))
        if n < 3:
            return {}

        metrics = _cluster_metrics(X[:n], labels[:n])
        _CLUSTER_METRICS_CACHE[cache_key] = metrics
        return metrics

    def _metrics_card(title: str, metrics: dict, color: str) -> dbc.Card:
        def fmt(x):
            if x is None:
                return "N/A"
            try:
                if isinstance(x, (int, float)) and not np.isfinite(x):
                    return "N/A"
                return f"{x:.3f}" if isinstance(x, float) else str(x)
            except Exception:
                return str(x)

        n_points = metrics.get("n_points")
        n_clusters = metrics.get("n_clusters")
        pct_noise = metrics.get("pct_noise")
        sil = metrics.get("silhouette")
        db = metrics.get("davies_bouldin")
        ch = metrics.get("calinski_harabasz")

        body = [
            html.P(f"Nombre de points : {fmt(n_points)}"),
            html.P(f"Nombre de clusters : {fmt(n_clusters)}"),
        ]
        if pct_noise is not None:
            body.append(html.P(f"Bruit: {fmt(pct_noise)}%"))
        body.extend([
            html.Hr(),
            html.P(f"Silhouette : {fmt(sil)}"),
            html.P(f"Davies-Bouldin : {fmt(db)}"),
            html.P(f"Calinski-Harabasz : {fmt(ch)}"),
        ])

        return dbc.Card([
            dbc.CardHeader(html.H5(title), className=f"bg-{color} text-white"),
            dbc.CardBody(body),
        ])

    metrics_kmeans = _get_metrics(year, "kmeans")
    metrics_hdbscan = _get_metrics(year, "hdbscan")

    metrics_row = dbc.Row([
        dbc.Col([
            _metrics_card("K-Means", metrics_kmeans, "primary")
            if metrics_kmeans else
            dbc.Alert("Métriques K-Means non disponibles.", color="warning"),
        ], md=6),
        dbc.Col([
            _metrics_card("HDBSCAN", metrics_hdbscan, "info")
            if metrics_hdbscan else
            dbc.Alert("Métriques HDBSCAN non disponibles.", color="warning"),
        ], md=6),
    ], className="mb-4")

    # ── Sélecteur de cluster + détails (K-Means) ───────────────────────────
    cluster_selector = dbc.Row([
        dbc.Col([
            html.Label(f"Sélectionnez un cluster K-Means {year} :"),
            dcc.Dropdown(
                id=f'cluster-selector-{year}',
                options=[{'label': f'Cluster {i}', 'value': i} for i in range(n_clusters)],
                value=0,
                clearable=False
            ),
        ], md=6),
    ], className="mb-4")

    return html.Div([
        html.H3(f"Clustering des articles - {year}", className="mb-3"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_kmeans)], md=6),
            dbc.Col([dcc.Graph(figure=fig_hdbscan)], md=6),
        ], className="mb-4"),
        html.H4("Règle du coude (inertie K-Means)", className="mb-3"),
        dcc.Graph(figure=fig_elbow),
        html.H4("Métriques de performance des clusters", className="mb-3 mt-4"),
        metrics_row,
        html.Hr(),
        cluster_selector,
        html.Div(id=f'cluster-details-{year}')
    ])


@app.callback(
    Output('cluster-details-2024', 'children'),
    Input('cluster-selector-2024', 'value')
)
def update_cluster_2024(cluster_id):
    """Met à jour les détails du cluster 2024."""
    return render_cluster_details(2024, cluster_id)


@app.callback(
    Output('cluster-details-2025', 'children'),
    Input('cluster-selector-2025', 'value')
)
def update_cluster_2025(cluster_id):
    """Met à jour les détails du cluster 2025."""
    return render_cluster_details(2025, cluster_id)


def render_cluster_details(year, cluster_id):
    """Affiche les détails d'un cluster avec nuage de mots."""
    clusters_df = DATA[f'articles_clusters_{year}']

    cluster_articles = clusters_df[clusters_df['cluster_kmeans'] == cluster_id]
    n_articles = len(cluster_articles)

    cluster_text = get_cluster_texts(clusters_df, cluster_id)
    wordcloud_img = create_wordcloud(cluster_text, colormap='plasma', max_words=50)

    pilier_dominant: str | None = None
    if 'pilier_dominant' in cluster_articles.columns:
        pilier_counts = cluster_articles['pilier_dominant'].dropna().value_counts()
        if len(pilier_counts) > 0:
            pilier_dominant = str(pilier_counts.index[0])

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4(f"Cluster {cluster_id} - {year}")),
                    dbc.CardBody([
                        html.H6("Statistiques :"),
                        html.P(f"{n_articles} articles"),
                        *([
                            html.P(f"Pilier dominant : {pilier_dominant}")
                        ] if pilier_dominant else []),
                        html.Hr(),
                        html.H6("Exemples d'articles :"),
                        html.Ul([
                            html.Li(row['content'][:100] + "...")
                            for _, row in cluster_articles.head(3).iterrows()
                        ] if 'content' in cluster_articles.columns else [html.Li("Pas de contenu disponible")])
                    ])
                ])
            ], md=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Nuage de Mots du Cluster")),
                    dbc.CardBody([
                        html.Img(
                            src=f'data:image/png;base64,{wordcloud_img}',
                            style={'width': '100%', 'height': 'auto'}
                        )
                    ])
                ])
            ], md=6),
        ])
    ])


# Callbacks Budget
@app.callback(
    [Output('budget-pie-chart', 'figure'),
     Output('budget-bar-chart', 'figure'),
     Output('budget-top-programs', 'figure')],
    Input('budget-year-selector', 'value')
)
def update_budget_charts(year):
    """Met à jour les graphiques budgétaires selon l'année sélectionnée."""
    try:
        if f'budget_{year}' not in DATA or f'piliers_{year}' not in DATA:
            return go.Figure(), go.Figure(), go.Figure()

        budget_df = DATA[f'budget_{year}']
        piliers_df = DATA[f'piliers_{year}']

        fig_pie = px.pie(
            piliers_df,
            values='ae',
            names=piliers_df.index,
            title=f"Répartition AE {year} par Pilier SND30",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='AE',
            x=piliers_df.index,
            y=piliers_df['ae'] / 1e9,
            marker_color='steelblue'
        ))
        fig_bar.add_trace(go.Bar(
            name='CP',
            x=piliers_df.index,
            y=piliers_df['cp'] / 1e9,
            marker_color='lightcoral'
        ))
        fig_bar.update_layout(
            title=f"AE vs CP {year} par Pilier (Milliards FCFA)",
            barmode='group',
            xaxis_title="Pilier SND30",
            yaxis_title="Montant (Milliards FCFA)"
        )

        top_15 = budget_df.nlargest(15, 'ae').copy()

        if 'pilier_dominant' in top_15.columns:
            top_15['pilier_label'] = top_15['pilier_dominant'].map(map_pilier_label)
            fig_top = px.bar(
                top_15,
                y='programme',
                x='ae',
                color='pilier_label',
                orientation='h',
                title=f"Top 15 Programmes {year} par AE (FCFA)",
                labels={'ae': 'AE (FCFA)', 'programme': 'Programme', 'pilier_label': 'Pilier SND30'},
                height=600
            )
        else:
            fig_top = px.bar(
                top_15,
                y='programme',
                x='ae',
                orientation='h',
                title=f"Top 15 Programmes {year} par AE (FCFA)",
                labels={'ae': 'AE (FCFA)', 'programme': 'Programme'},
                height=600,
                color_discrete_sequence=['steelblue']
            )

        fig_top.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Montant AE (FCFA)",
            yaxis_title="Programme"
        )

        return fig_pie, fig_bar, fig_top
    except Exception as e:
        print(f"Erreur dans update_budget_charts: {e}")
        return go.Figure(), go.Figure(), go.Figure()


@app.callback(
    Output('budget-evolution-chart', 'figure'),
    Input('budget-year-selector', 'value')
)
def update_budget_evolution(year):
    """Graphique d'évolution budgétaire 2024 → 2025."""
    try:
        if 'comparison' not in DATA or DATA['comparison'] is None:
            return go.Figure()

        comparison = DATA['comparison']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='2024',
            x=comparison.index,
            y=comparison['ae_2024'] / 1e9,
            marker_color='lightblue',
            text=[f"{v/1e9:.2f}" for v in comparison['ae_2024']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='2025',
            x=comparison.index,
            y=comparison['ae_2025'] / 1e9,
            marker_color='darkblue',
            text=[f"{v/1e9:.2f}" for v in comparison['ae_2025']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Évolution des AE par Pilier 2024 → 2025 (Milliards FCFA)",
            barmode='group',
            xaxis_title="Pilier SND30",
            yaxis_title="Montant AE (Milliards FCFA)",
            height=500
        )

        return fig
    except Exception as e:
        print(f"Erreur dans update_budget_evolution: {e}")
        return go.Figure()


# Callbacks pour Classification
@app.callback(
    Output('classification-content', 'children'),
    Input('classification-tabs', 'active_tab')
)
def render_classification_content(active_tab):
    """Affiche le contenu de la page Classification selon l'onglet."""
    try:
        if active_tab == 'classif-2024':
            return create_classification_year_content(2024)
        elif active_tab == 'classif-2025':
            return create_classification_year_content(2025)
        elif active_tab == 'classif-comp':
            return create_classification_comparison_content()
        else:
            return dbc.Alert("Onglet non reconnu", color="warning")
    except Exception as e:
        print(f"Erreur dans render_classification_content: {e}")
        return dbc.Alert(f"Erreur lors du chargement: {str(e)}", color="danger")


def create_classification_year_content(year):
    """Crée le contenu classification pour une année donnée."""
    try:
        if f'budget_{year}' not in DATA or f'piliers_{year}' not in DATA:
            return dbc.Alert(f"Données {year} non disponibles", color="warning")

        budget_df = DATA[f'budget_{year}']
        piliers_df = DATA[f'piliers_{year}']

        piliers_reset = piliers_df.reset_index()
        piliers_reset.columns = ['pilier'] + list(piliers_reset.columns[1:])
        fig_bar = px.bar(
            piliers_reset,
            x='pilier',
            y=['ae', 'cp'],
            title=f"Montants AE/CP par Pilier - {year}",
            labels={'pilier': 'Pilier SND30', 'value': 'Montant (FCFA)', 'variable': 'Type'},
            barmode='group',
            color_discrete_map={'ae': 'steelblue', 'cp': 'lightcoral'}
        )
        fig_bar.update_layout(height=500)

        score_cols = [col for col in budget_df.columns if col.startswith('score_')]

        if score_cols:
            score_data = []
            for col in score_cols:
                pilier = col.replace('score_', '')
                score_data.append({
                    'Pilier': pilier,
                    'Scores': budget_df[col].tolist()
                })

            fig_box = go.Figure()
            for item in score_data:
                fig_box.add_trace(go.Box(
                    y=item['Scores'],
                    name=item['Pilier'],
                    boxmean='sd'
                ))
            fig_box.update_layout(
                title=f"Distribution des Scores de Classification - {year}",
                yaxis_title="Score de probabilité",
                height=500
            )
            boxplot_section = dcc.Graph(figure=fig_box)
        else:
            boxplot_section = dbc.Alert("Scores de classification non disponibles", color="warning")

        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5(f"Statistiques {year}")),
                        dbc.CardBody([
                            html.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("Pilier"),
                                        html.Th("Lignes"),
                                        html.Th("AE (%)"),
                                        html.Th("CP (%)"),
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(pilier),
                                        html.Td(f"{int(row['nb_lignes'])}"),
                                        html.Td(f"{row['pct_ae']:.1f}%"),
                                        html.Td(f"{row['pct_cp']:.1f}%"),
                                    ]) for pilier, row in piliers_df.iterrows()
                                ])
                            ], className="table table-striped table-hover")
                        ])
                    ])
                ], md=4),

                dbc.Col([
                    dcc.Graph(
                        figure=px.pie(
                            piliers_reset,
                            values='ae',
                            names='pilier',
                            title=f"Répartition AE par Pilier - {year}",
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                    )
                ], md=8),
            ], className="mb-4"),

            dcc.Graph(figure=fig_bar),
            html.Hr(),
            boxplot_section,
        ])
    except Exception as e:
        print(f"Erreur dans create_classification_year_content: {e}")
        return dbc.Alert(f"Erreur lors de la création du contenu: {str(e)}", color="danger")


def create_classification_comparison_content():
    """Comparaison des classifications 2024 vs 2025."""
    try:
        if 'comparison' not in DATA:
            return dbc.Alert("Données de comparaison non disponibles", color="warning")

        comparison = DATA['comparison']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='2024',
            x=comparison.index,
            y=comparison['ae_2024'] / 1e9,
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='2025',
            x=comparison.index,
            y=comparison['ae_2025'] / 1e9,
            marker_color='darkblue'
        ))
        fig.update_layout(
            title="Comparaison AE 2024 vs 2025 par Pilier (Milliards FCFA)",
            barmode='group',
            xaxis_title="Pilier SND30",
            yaxis_title="Montant AE (Milliards FCFA)",
            height=500
        )

        return html.Div([
            dcc.Graph(figure=fig),
            html.Hr(),
            html.H4("Tableau Comparatif", className="mb-3"),
            dbc.Table.from_dataframe(
                comparison.reset_index().round(2),
                striped=True,
                bordered=True,
                hover=True
            )
        ])
    except Exception as e:
        print(f"Erreur dans create_classification_comparison_content: {e}")
        return dbc.Alert(f"Erreur lors de la comparaison: {str(e)}", color="danger")


# Callbacks pour Baromètre
@app.callback(
    [Output('barometer-evolution-piliers', 'figure'),
     Output('barometer-concentration-2024', 'figure'),
     Output('barometer-concentration-2025', 'figure')],
    Input('url', 'pathname')
)
def update_barometer_charts(pathname):
    """Met à jour les graphiques du baromètre."""
    try:
        if ('comparison' not in DATA or 'piliers_2024' not in DATA or
                'piliers_2025' not in DATA):
            return go.Figure(), go.Figure(), go.Figure()

        comparison = DATA['comparison']
        piliers_2024 = DATA['piliers_2024']
        piliers_2025 = DATA['piliers_2025']

        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(
            x=comparison.index,
            y=comparison['ae_2024'] / 1e9,
            mode='lines+markers',
            name='2024',
            line=dict(color='lightblue', width=3),
            marker=dict(size=10)
        ))
        fig_evol.add_trace(go.Scatter(
            x=comparison.index,
            y=comparison['ae_2025'] / 1e9,
            mode='lines+markers',
            name='2025',
            line=dict(color='darkblue', width=3),
            marker=dict(size=10)
        ))
        fig_evol.update_layout(
            title="Évolution des AE par Pilier (Milliards FCFA)",
            xaxis_title="Pilier SND30",
            yaxis_title="Montant AE (Milliards FCFA)",
            height=400
        )

        _p24 = piliers_2024.reset_index()
        _path_col_24 = _p24.columns[0]
        fig_conc_2024 = px.treemap(
            _p24,
            path=[_path_col_24],
            values='ae',
            title="Concentration AE 2024",
            color='ae',
            color_continuous_scale='Blues'
        )
        fig_conc_2024.update_layout(height=400)

        _p25 = piliers_2025.reset_index()
        _path_col_25 = _p25.columns[0]
        fig_conc_2025 = px.treemap(
            _p25,
            path=[_path_col_25],
            values='ae',
            title="Concentration AE 2025",
            color='ae',
            color_continuous_scale='Blues'
        )
        fig_conc_2025.update_layout(height=400)

        return fig_evol, fig_conc_2024, fig_conc_2025

    except Exception as e:
        print(f"Erreur dans update_barometer_charts: {e}")
        return go.Figure(), go.Figure(), go.Figure()


@app.callback(
    [
        Output("barometer-umap-graph", "figure"),
        Output("barometer-umap-cluster-filter", "options"),
        Output("barometer-umap-title-filter", "options"),
        Output("barometer-umap-cluster-bar", "figure"),
    ],
    [
        Input("barometer-umap-year-filter", "value"),
        Input("barometer-umap-cluster-filter", "value"),
        Input("barometer-umap-title-filter", "value"),
    ],
)
def update_barometer_umap(selected_years, selected_clusters, selected_titles):
    """Met à jour la vue UMAP + les options de filtres en fonction des données disponibles."""
    fig = go.Figure()
    fig_clusters = go.Figure()

    def _empty_fig(msg: str) -> go.Figure:
        f = go.Figure()
        f.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                         showarrow=False, font=dict(size=14))
        f.update_xaxes(visible=False)
        f.update_yaxes(visible=False)
        return f

    if "umap_df" not in DATA or not isinstance(DATA["umap_df"], pd.DataFrame) or DATA["umap_df"].empty:
        msg = "Embeddings/UMAP non disponibles (vérifiez `outputs/models/embeddings_2024.npy` et `embeddings_2025.npy`)."
        if "umap_error" in DATA:
            msg += f" Détail: {DATA['umap_error']}"
        return (
            _empty_fig(msg),
            [],
            [],
            _empty_fig("Clusters K-Means non disponibles (pas de données UMAP)."),
        )

    df = DATA["umap_df"].copy()

    cluster_options = []
    if "cluster_kmeans" in df.columns:
        cluster_options = [
            {"label": f"Cluster {int(c)}", "value": int(c)}
            for c in sorted(df["cluster_kmeans"].dropna().unique())
        ]

    title_options = []
    if "titre" in df.columns:
        title_options = [
            {"label": str(t), "value": str(t)}
            for t in sorted(df["titre"].dropna().unique())
        ]

    if selected_years:
        df = df[df["annee"].isin(selected_years)]
    if selected_clusters and "cluster_kmeans" in df.columns:
        df = df[df["cluster_kmeans"].isin(selected_clusters)]
    if selected_titles and "titre" in df.columns:
        df = df[df["titre"].isin(selected_titles)]

    if df.empty:
        return (
            _empty_fig("Aucun point ne correspond aux filtres sélectionnés."),
            cluster_options,
            title_options,
            _empty_fig("Aucune donnée de cluster pour les filtres sélectionnés."),
        )

    hover_cols = [c for c in ["id", "titre", "chapitre", "dominant_topic", "cluster_kmeans"] if c in df.columns]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="annee",
        hover_data=hover_cols if hover_cols else None,
        title="UMAP des embeddings (articles 2024 vs 2025)",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.65))
    fig.update_layout(height=650, legend_title_text="Année")

    if "cluster_kmeans" in df.columns:
        cluster_summary = (
            df.dropna(subset=["cluster_kmeans"])
              .groupby(["annee", "cluster_kmeans"])
              .size()
              .reset_index(name="nb_articles")
        )
        if not cluster_summary.empty:
            cluster_summary["cluster_kmeans"] = cluster_summary["cluster_kmeans"].astype(int)
            cluster_summary["cluster_label"] = cluster_summary["cluster_kmeans"].apply(
                lambda c: f"Cluster {c}"
            )
            fig_clusters = px.bar(
                cluster_summary,
                x="cluster_label",
                y="nb_articles",
                color="annee",
                barmode="group",
                title="Nombre d'articles par cluster K-Means et par année",
            )
            fig_clusters.update_layout(
                xaxis_title="Cluster K-Means",
                yaxis_title="Nombre d'articles",
                height=450,
            )
        else:
            fig_clusters = _empty_fig("Aucune donnée de cluster après filtrage.")
    else:
        fig_clusters = _empty_fig("Colonnes de clusters K-Means non disponibles dans les données UMAP.")

    return fig, cluster_options, title_options, fig_clusters


@app.callback(
    [
        Output("barometer-pilier-pie", "figure"),
        Output("barometer-pilier-count-bar", "figure"),
        Output("barometer-wc-pilier-0", "src"),
        Output("barometer-wc-pilier-1", "src"),
        Output("barometer-wc-pilier-2", "src"),
        Output("barometer-wc-pilier-3", "src"),
    ],
    Input("barometer-objectifs-year", "value"),
)
def update_barometer_objectifs(year_value: str):
    """Baromètre objectifs: wordclouds par pilier + répartition (pie/bar)."""

    def _msg_fig(text: str) -> go.Figure:
        f = go.Figure()
        f.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5,
                         showarrow=False, font=dict(size=14))
        f.update_xaxes(visible=False)
        f.update_yaxes(visible=False)
        return f

    # Récupérer la source de données
    df_parts = []
    if year_value in ("2024", "2025"):
        key = f"objectifs_classifications_{int(year_value)}"
        if key in DATA and isinstance(DATA[key], pd.DataFrame):
            df_parts = [DATA[key]]
    else:
        for y in (2024, 2025):
            key = f"objectifs_classifications_{y}"
            if key in DATA and isinstance(DATA[key], pd.DataFrame):
                df_parts.append(DATA[key])

    empty_img = f"data:image/png;base64,{create_wordcloud('')}"

    if not df_parts:
        return (
            _msg_fig("Données objectifs non disponibles. Exécutez le pipeline pour générer `objectifs_classifications_snd30_*.xlsx`."),
            _msg_fig("Données objectifs non disponibles."),
            empty_img, empty_img, empty_img, empty_img,
        )

    df = pd.concat(df_parts, ignore_index=True).copy()

    objectif_col = "objectif" if "objectif" in df.columns else None
    pilier_col = "pilier_dominant" if "pilier_dominant" in df.columns else None
    if not objectif_col or not pilier_col:
        return (
            _msg_fig("Colonnes manquantes dans les données objectifs (attendu: `objectif`, `pilier_dominant`)."),
            _msg_fig("Colonnes manquantes dans les données objectifs."),
            empty_img, empty_img, empty_img, empty_img,
        )

    prep = TextPreprocessor(lower=True, rm_accents=True)
    df["_objectif_clean"] = df[objectif_col].fillna("").astype(str).apply(prep.preprocess)

    counts = (
        df[pilier_col]
        .fillna("Inconnu")
        .astype(str)
        .value_counts()
        .reset_index()
    )
    counts.columns = ["pilier", "nb_objectifs"]
    counts["pilier_court"] = counts["pilier"].map(map_pilier_label)

    fig_pie = px.pie(
        counts,
        names="pilier_court",
        values="nb_objectifs",
        title="Répartition des objectifs par pilier SND30",
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=450, legend_title_text="Pilier")

    fig_bar = px.bar(
        counts.sort_values("nb_objectifs", ascending=False),
        x="pilier_court",
        y="nb_objectifs",
        title="Nombre d'objectifs par pilier SND30",
        text="nb_objectifs",
    )
    fig_bar.update_layout(height=450, xaxis_title="Pilier", yaxis_title="Nombre d'objectifs")
    fig_bar.update_traces(textposition="outside")

    wc_srcs: list[str] = []
    for pilier in PILIERS_SND30:
        cache_key = (str(year_value), str(pilier))
        if cache_key in _BAROMETER_WC_CACHE:
            img_b64 = _BAROMETER_WC_CACHE[cache_key]
        else:
            sub = df[df[pilier_col].astype(str) == str(pilier)]
            text = " ".join(sub["_objectif_clean"].dropna().astype(str).tolist())
            img_b64 = create_wordcloud(text, max_words=80)
            _BAROMETER_WC_CACHE[cache_key] = img_b64
        wc_srcs.append(f"data:image/png;base64,{img_b64}")

    while len(wc_srcs) < 4:
        wc_srcs.append(empty_img)

    return fig_pie, fig_bar, wc_srcs[0], wc_srcs[1], wc_srcs[2], wc_srcs[3]


# Callbacks pour Stats
@app.callback(
    Output('stats-content', 'children'),
    Input('stats-tabs', 'active_tab')
)
def render_stats_content(active_tab):
    """Affiche le contenu de la page Stats selon l'onglet."""
    if active_tab == 'stats-desc':
        return create_stats_descriptives()
    elif active_tab == 'stats-tests':
        return create_stats_tests()
    elif active_tab == 'stats-dist':
        return create_stats_distributions()


def create_stats_descriptives():
    """Statistiques descriptives."""
    budget_2024 = DATA['budget_2024']
    budget_2025 = DATA['budget_2025']

    stats_2024 = budget_2024[['ae', 'cp']].describe()
    stats_2025 = budget_2025[['ae', 'cp']].describe()

    return html.Div([
        html.H3("Statistiques Descriptives", className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H5("2024", className="text-center"),
                dbc.Table.from_dataframe(
                    stats_2024.reset_index().round(2),
                    striped=True, bordered=True, hover=True
                )
            ], md=6),

            dbc.Col([
                html.H5("2025", className="text-center"),
                dbc.Table.from_dataframe(
                    stats_2025.reset_index().round(2),
                    striped=True, bordered=True, hover=True
                )
            ], md=6),
        ]),

        html.Hr(),

        html.H4("Résumé", className="mb-3"),
        dbc.Alert([
            html.Ul([
                html.Li(f"Moyenne AE 2024: {budget_2024['ae'].mean()/1e6:.2f} M FCFA"),
                html.Li(f"Moyenne AE 2025: {budget_2025['ae'].mean()/1e6:.2f} M FCFA"),
                html.Li(f"Médiane AE 2024: {budget_2024['ae'].median()/1e6:.2f} M FCFA"),
                html.Li(f"Médiane AE 2025: {budget_2025['ae'].median()/1e6:.2f} M FCFA"),
            ])
        ], color="info")
    ])


def create_stats_tests():
    """Tests statistiques comparatifs."""
    budget_2024 = DATA['budget_2024']
    budget_2025 = DATA['budget_2025']

    mean_diff_ae = budget_2025['ae'].mean() - budget_2024['ae'].mean()
    mean_diff_cp = budget_2025['cp'].mean() - budget_2024['cp'].mean()

    chi2_df = DATA.get('chi2')
    mw_df = DATA.get('mannwhitney')

    chi2_section = html.Div()
    if chi2_df is not None and not chi2_df.empty:
        row = chi2_df.iloc[0]
        chi2_section = dbc.Card([
            dbc.CardHeader(html.H5("Test du Chi² sur la distribution des topics/piliers")),
            dbc.CardBody([
                html.P(f"Statistique χ² : {row.get('chi2', float('nan')):.2f}"),
                html.P(f"p-value : {row.get('p_value', float('nan')):.4f}"),
                html.P(f"Degrés de liberté : {int(row.get('dof', 0))}"),
                html.Hr(),
                html.P(
                    "Résultat : "
                    + (
                        "différence significative entre 2024 et 2025 (au seuil 5%)."
                        if bool(row.get('significatif', False))
                        else "aucune différence significative détectée au seuil 5%."
                    ),
                    className="text-muted"
                ),
            ])
        ], className="mb-4")

    mw_section = html.Div()
    if mw_df is not None and not mw_df.empty:
        mw_section = html.Div([
            html.H5("Test de Mann-Whitney sur les distributions de probabilités de topics", className="mb-3"),
            dbc.Table.from_dataframe(
                mw_df.round(4),
                striped=True, bordered=True, hover=True
            )
        ])

    return html.Div([
        html.H3("Tests Comparatifs 2024 vs 2025", className="mb-4"),

        dbc.Card([
            dbc.CardHeader(html.H5("Comparaison des Moyennes AE/CP")),
            dbc.CardBody([
                html.P(f"Différence moyenne AE: {mean_diff_ae/1e6:+.2f} M FCFA"),
                html.P(f"Différence moyenne CP: {mean_diff_cp/1e6:+.2f} M FCFA"),
                html.Hr(),
                html.P(
                    "Les montants moyens des lignes budgétaires augmentent entre 2024 et 2025, "
                    "reflétant une hausse générale des enveloppes par programme.",
                    className="text-muted"
                )
            ])
        ], className="mb-4"),

        chi2_section,
        mw_section,
    ])


def create_stats_distributions():
    """Visualisation des distributions."""
    budget_2024 = DATA['budget_2024']
    budget_2025 = DATA['budget_2025']

    fig_hist = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution AE 2024", "Distribution AE 2025")
    )

    fig_hist.add_trace(
        go.Histogram(x=budget_2024['ae']/1e6, name='2024', marker_color='lightblue'),
        row=1, col=1
    )
    fig_hist.add_trace(
        go.Histogram(x=budget_2025['ae']/1e6, name='2025', marker_color='darkblue'),
        row=1, col=2
    )

    fig_hist.update_xaxes(title_text="AE (Millions FCFA)", row=1, col=1)
    fig_hist.update_xaxes(title_text="AE (Millions FCFA)", row=1, col=2)
    fig_hist.update_yaxes(title_text="Fréquence", row=1, col=1)
    fig_hist.update_layout(height=500, showlegend=False)

    return html.Div([
        html.H3("Distributions des Montants", className="mb-4"),
        dcc.Graph(figure=fig_hist),

        html.Hr(),

        html.H4("Observations", className="mb-3"),
        dbc.Alert([
            html.P(
                "Les distributions montrent une concentration importante des lignes budgétaires "
                "sur des montants faibles, avec quelques lignes à très fort montant (dette, infrastructures).",
                className="mb-0"
            )
        ], color="info")
    ])


# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT DE L'APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
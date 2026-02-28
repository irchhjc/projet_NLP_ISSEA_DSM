"""
visualization/plots.py
Toutes les fonctions de visualisation du projet (Matplotlib, Seaborn, Plotly).
"""
from __future__ import annotations # pour 

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

from audit_semantique.config import (
    AUDIT_PARAMS,
    FIGURES_DIR,
    PILIERS_COURT,
    PILIERS_SND30,
    TSNE_PARAMS,
    UMAP_PARAMS,
    VIZ_PARAMS,
)

# ── Style global ──────────────────────────────────────────────────────────────
plt.style.use(VIZ_PARAMS["style"])
sns.set_palette(VIZ_PARAMS["palette"])
_DPI = VIZ_PARAMS["dpi"]


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 1. AUDIT SÉMANTIQUE
# ══════════════════════════════════════════════════════════════════════════════

def plot_similarity_matrix(
    sim_matrix: np.ndarray,
    ids_2024: List[str],
    ids_2025: List[str],
    best_scores: np.ndarray,
    avg_sim: float,
    seuil: float = AUDIT_PARAMS["seuil_changement"],
    n_display: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Heatmap de similarité cosinus + distribution des scores."""
    n = min(n_display, sim_matrix.shape[0], sim_matrix.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=VIZ_PARAMS["figsize_lg"])

    # Heatmap
    sns.heatmap(
        sim_matrix[:n, :n],
        xticklabels=ids_2025[:n],
        yticklabels=ids_2024[:n],
        cmap="RdYlGn",
        center=0.7,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Similarité Cosinus"},
        ax=axes[0],
        square=True,
    )
    axes[0].set_title(
        f"Matrice de Similarité Cosinus\n2024 vs 2025 (échantillon {n} articles)",
        fontweight="bold",
    )
    axes[0].set_xlabel("Articles 2024-2025")
    axes[0].set_ylabel("Articles 2023-2024")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=8)

    # Distribution
    axes[1].hist(best_scores, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].axvline(avg_sim, color="red", linestyle="--", lw=2,
                    label=f"Moyenne : {avg_sim:.3f}")
    axes[1].axvline(seuil, color="orange", linestyle=":", lw=2,
                    label=f"Seuil : {seuil}")
    axes[1].set_title("Distribution des Scores de Similarité Maximale", fontweight="bold")
    axes[1].set_xlabel("Score de Similarité")
    axes[1].set_ylabel("Nombre d'articles")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        _save(fig, "similarity_matrix")
    return fig


def plot_tsne(
    embeddings_2024: np.ndarray,
    embeddings_2025: np.ndarray,
    loi_2024: pd.DataFrame,
    loi_2025: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Visualisation t-SNE colorée par titre et année."""
    combined = np.vstack([embeddings_2024, embeddings_2025])
    labels   = (["2023-2024"] * len(embeddings_2024)
                + ["2024-2025"] * len(embeddings_2025))
    titres   = loi_2024["titre"].tolist() + loi_2025["titre"].tolist()

    tsne     = TSNE(**TSNE_PARAMS)
    coords   = tsne.fit_transform(combined)

    df_viz = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "annee": labels, "titre": titres}
    )

    fig, ax = plt.subplots(figsize=VIZ_PARAMS["figsize_md"])
    titres_uniques = df_viz["titre"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(titres_uniques)))
    color_map = dict(zip(titres_uniques, colors))

    for titre in titres_uniques:
        sub = df_viz[df_viz["titre"] == titre]
        for annee, marker in [("2023-2024", "o"), ("2024-2025", "^")]:
            pts = sub[sub["annee"] == annee]
            if len(pts):
                ax.scatter(pts["x"], pts["y"], c=[color_map[titre]],
                           alpha=0.7, s=80, marker=marker, edgecolors="black", lw=0.4)

    ax.set_title("t-SNE des Articles de Loi par Titre\n(○ = 2023-24, △ = 2024-25)",
                 fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save:
        _save(fig, "tsne_articles")
    return fig


def plot_umap(
    embeddings_2024: np.ndarray,
    embeddings_2025: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Visualisation UMAP colorée par année."""
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("Installez umap-learn : poetry add umap-learn")

    combined = np.vstack([embeddings_2024, embeddings_2025])
    labels   = (["2023-2024"] * len(embeddings_2024)
                + ["2024-2025"] * len(embeddings_2025))

    reducer = umap_lib.UMAP(**UMAP_PARAMS)
    coords  = reducer.fit_transform(combined)

    fig, ax = plt.subplots(figsize=VIZ_PARAMS["figsize_md"])
    df_viz = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "annee": labels})

    for annee, color in [("2023-2024", "blue"), ("2024-2025", "red")]:
        pts = df_viz[df_viz["annee"] == annee]
        ax.scatter(pts["x"], pts["y"], c=color, alpha=0.6, s=80,
                   label=annee, edgecolors=f"dark{color}", lw=0.5)

    ax.set_title("UMAP des Articles de Loi\n(2023-2024 vs 2024-2025)", fontweight="bold")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save:
        _save(fig, "umap_articles")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION SND30
# ══════════════════════════════════════════════════════════════════════════════

def plot_snd30_scores(
    loi_2024: pd.DataFrame,
    loi_2025: pd.DataFrame,
    save: bool = True,
) -> go.Figure:
    """Boxplots Plotly des scores de classification par pilier SND30."""
    df_full = pd.concat([loi_2024, loi_2025], ignore_index=True)
    df_full["annee"] = df_full["annee"].astype(str)

    fig = make_subplots(rows=2, cols=2, subplot_titles=PILIERS_COURT)
    for i, pilier in enumerate(PILIERS_SND30):
        row, col = i // 2 + 1, i % 2 + 1
        for annee, color in [("2024", "#2196F3"), ("2025", "#FF5722")]:
            sub = df_full[df_full["annee"] == annee]
            fig.add_trace(
                go.Box(y=sub[f"score_{pilier}"], name=annee, marker_color=color,
                       showlegend=(i == 0)),
                row=row, col=col,
            )

    fig.update_layout(
        height=700, width=900,
        title_text="Scores de Classification des Articles par Pilier SND30",
        boxmode="group",
    )
    path = FIGURES_DIR / "snd30_scores.html"
    fig.write_html(str(path))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOPIC MODELING
# ══════════════════════════════════════════════════════════════════════════════

def plot_topic_words(
    lda_model,
    num_topics: int = 4,
    num_words: int = 10,
    title_prefix: str = "",
    save: bool = True,
    save_name: str = "topic_words",
) -> plt.Figure:
    """Barplots horizontaux des top mots par topic."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{title_prefix} — Top {num_words} mots par topic", fontsize=16)

    for t in range(num_topics):
        ax     = axes[t // 2, t % 2]
        terms  = lda_model.show_topic(t, topn=num_words)
        words  = [w for w, _ in terms][::-1]
        probs  = [p for _, p in terms][::-1]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))

        bars = ax.barh(range(len(words)), probs, alpha=0.8)
        for bar, c in zip(bars, colors[::-1]):
            bar.set_color(c)

        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel("Probabilité")
        ax.set_title(f"Topic {t}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for i, (w, p) in enumerate(zip(words, probs)):
            ax.text(p, i, f" {p:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    if save:
        _save(fig, save_name)
    return fig


def plot_topic_comparison(
    lda_2024, lda_2025, num_words: int = 10, save: bool = True
) -> plt.Figure:
    """Comparaison côte-à-côte des topics 2024 et 2025."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle("Comparaison des Topics LDA : 2024 vs 2025", fontsize=18)

    for t in range(4):
        for col_idx, (model, color, label) in enumerate(
            [(lda_2024, "steelblue", "2024"), (lda_2025, "coral", "2025")]
        ):
            ax    = axes[t, col_idx]
            terms = model.show_topic(t, topn=num_words)
            words = [w for w, _ in terms][::-1]
            probs = [p for _, p in terms][::-1]
            ax.barh(range(len(words)), probs, alpha=0.8, color=color)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel("Probabilité")
            ax.set_title(f"Topic {t} — {label}", fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save:
        _save(fig, "topic_comparison")
    return fig


def plot_topic_distributions(
    dist_2024: np.ndarray, dist_2025: np.ndarray, save: bool = True
) -> plt.Figure:
    """Histogrammes comparatifs de distribution de topics + boxplots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribution des probabilités par topic (2024 vs 2025)", fontsize=16)

    for t in range(4):
        ax = axes[t // 2, t % 2]
        bp = ax.boxplot(
            [dist_2024[:, t], dist_2025[:, t]],
            labels=["2024", "2025"],
            patch_artist=True,
        )
        for patch, c in zip(bp["boxes"], ["lightblue", "lightcoral"]):
            patch.set_facecolor(c)
        ax.set_title(f"Topic {t}")
        ax.set_ylabel("Probabilité")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        _save(fig, "topic_distributions")
    return fig


def plot_wordcloud(
    lda_model,
    num_topics: int = 4,
    title_prefix: str = "",
    save: bool = True,
    save_name: str = "wordcloud",
) -> plt.Figure:
    """Nuages de mots pour chaque topic."""
    from wordcloud import WordCloud

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{title_prefix} — Nuages de mots par topic", fontsize=18)

    for t in range(num_topics):
        ax        = axes[t // 2, t % 2]
        terms     = lda_model.show_topic(t, topn=50)
        word_freq = {w: p for w, p in terms}
        wc = WordCloud(
            width=800, height=600, background_color="white",
            colormap="viridis", relative_scaling=0.5, min_font_size=10,
        ).generate_from_frequencies(word_freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {t}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save:
        _save(fig, save_name)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def plot_clustering(
    pca_coords: np.ndarray,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
    pca_variance: np.ndarray,
    kmeans_centers: np.ndarray,
    labels_year: List[str],
    save: bool = True,
) -> plt.Figure:
    """Visualisation PCA des résultats de clustering."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # K-Means
    sc1 = axes[0].scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=kmeans_labels, cmap="viridis", alpha=0.6, s=80,
    )
    axes[0].scatter(
        kmeans_centers[:, 0], kmeans_centers[:, 1],
        c="red", marker="X", s=300, edgecolors="black", lw=2, label="Centroïdes",
    )
    axes[0].set_title("K-Means", fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({pca_variance[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca_variance[1]:.1%})")
    axes[0].legend()
    plt.colorbar(sc1, ax=axes[0], label="Cluster")

    # HDBSCAN
    sc2 = axes[1].scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=hdbscan_labels, cmap="viridis", alpha=0.6, s=80,
    )
    axes[1].set_title("HDBSCAN (-1 = bruit)", fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({pca_variance[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca_variance[1]:.1%})")
    plt.colorbar(sc2, ax=axes[1], label="Cluster")

    # Par année
    colors = ["blue" if y == "2024" else "red" for y in labels_year]
    axes[2].scatter(pca_coords[:, 0], pca_coords[:, 1], c=colors, alpha=0.6, s=80)
    from matplotlib.patches import Patch
    axes[2].legend(handles=[
        Patch(facecolor="blue", label="2024"),
        Patch(facecolor="red", label="2025"),
    ])
    axes[2].set_title("Distribution par année", fontweight="bold")
    axes[2].set_xlabel(f"PC1 ({pca_variance[0]:.1%})")

    plt.tight_layout()
    if save:
        _save(fig, "clustering")
    return fig


def plot_elbow(inertia_dict: Dict[int, float], save: bool = True) -> plt.Figure:
    """Courbe du coude pour K-Means."""
    fig, ax = plt.subplots(figsize=VIZ_PARAMS["figsize_sm"])
    ax.plot(list(inertia_dict.keys()), list(inertia_dict.values()), "bo-")
    ax.set_xlabel("Nombre de clusters (k)")
    ax.set_ylabel("Inertie")
    ax.set_title("Méthode du Coude — K-Means", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        _save(fig, "elbow_kmeans")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. STATISTIQUES
# ══════════════════════════════════════════════════════════════════════════════

def plot_mannwhitney_results(df_results: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Barplot des p-valeurs des tests de Mann-Whitney par topic."""
    fig, ax = plt.subplots(figsize=VIZ_PARAMS["figsize_sm"])
    colors = ["green" if sig else "salmon" for sig in df_results["significatif"]]
    ax.barh(
        [f"Topic {t}" for t in df_results["topic"]],
        df_results["p_value"],
        color=colors, alpha=0.8,
    )
    ax.axvline(0.05, color="red", linestyle="--", lw=2, label="α = 0.05")
    ax.set_xlabel("p-value")
    ax.set_title("Tests de Mann-Whitney U par Topic\n(vert = significatif)", fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        _save(fig, "mannwhitney_tests")
    return fig

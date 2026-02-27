"""
clustering/clusterer.py
Clustering des articles de loi (K-Means + HDBSCAN) sur les distributions de topics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import hdbscan
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from audit_semantique.config import HDBSCAN_PARAMS, KMEANS_PARAMS


@dataclass
class ClusteringResults:
    """Résultats consolidés des deux algorithmes de clustering."""
    kmeans_labels:   np.ndarray
    hdbscan_labels:  np.ndarray
    pca_coords:      np.ndarray          # Coordonnées 2D après PCA
    pca_variance:    np.ndarray          # Variance expliquée par composante
    kmeans_centers:  np.ndarray          # Centroïdes K-Means
    metrics:         Dict = field(default_factory=dict)


class DocumentClusterer:
    """
    Applique K-Means et HDBSCAN aux distributions de topics des documents.

    Parameters
    ----------
    n_clusters : int
        Nombre de clusters K-Means.
    kmeans_seed : int
        Graine aléatoire K-Means.
    hdbscan_min_cluster_size : int
        Taille minimale d'un cluster HDBSCAN.
    hdbscan_min_samples : int
        Paramètre min_samples de HDBSCAN.
    """

    def __init__(
        self,
        n_clusters:               int = KMEANS_PARAMS["n_clusters"],
        kmeans_seed:              int = KMEANS_PARAMS["random_state"],
        hdbscan_min_cluster_size: int = HDBSCAN_PARAMS["min_cluster_size"],
        hdbscan_min_samples:      int = HDBSCAN_PARAMS["min_samples"],
    ) -> None:
        self.n_clusters               = n_clusters
        self.kmeans_seed              = kmeans_seed
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples      = hdbscan_min_samples

    def fit(self, X: np.ndarray) -> ClusteringResults:
        """
        Applique K-Means, HDBSCAN et PCA sur la matrice ``X``.

        Parameters
        ----------
        X : np.ndarray de forme ``(n_docs, n_topics)``.

        Returns
        -------
        ClusteringResults
        """
        logger.info(f"📐 Clustering sur {X.shape[0]} documents, {X.shape[1]} features...")

        # ── K-Means ──────────────────────────────────────────────────────────
        logger.info(f"  → K-Means (k={self.n_clusters})...")
        km = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.kmeans_seed,
            n_init=KMEANS_PARAMS["n_init"],
        )
        km_labels = km.fit_predict(X)

        # ── HDBSCAN ──────────────────────────────────────────────────────────
        logger.info("  → HDBSCAN...")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric=HDBSCAN_PARAMS["metric"],
        )
        hdb_labels = hdb.fit_predict(X)
        n_hdb_clusters = len(set(hdb_labels) - {-1})
        n_noise        = int((hdb_labels == -1).sum())
        logger.info(
            f"     HDBSCAN → {n_hdb_clusters} clusters, {n_noise} points de bruit"
        )

        # ── PCA 2D ───────────────────────────────────────────────────────────
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(X)
        centers_2d = pca.transform(km.cluster_centers_)

        # ── Métriques ────────────────────────────────────────────────────────
        metrics: Dict = {}
        metrics["kmeans"] = self._evaluate(X, km_labels, label="K-Means")

        mask_no_noise = hdb_labels != -1
        if n_hdb_clusters > 1 and mask_no_noise.sum() > 0:
            metrics["hdbscan"] = self._evaluate(
                X[mask_no_noise], hdb_labels[mask_no_noise], label="HDBSCAN"
            )
        else:
            metrics["hdbscan"] = {"warning": "Pas assez de clusters HDBSCAN."}

        results = ClusteringResults(
            kmeans_labels  = km_labels,
            hdbscan_labels = hdb_labels,
            pca_coords     = coords_2d,
            pca_variance   = pca.explained_variance_ratio_,
            kmeans_centers = centers_2d,
            metrics        = metrics,
        )
        logger.info("✅ Clustering terminé.")
        return results

    @staticmethod
    def _evaluate(X: np.ndarray, labels: np.ndarray, label: str) -> Dict:
        """Calcule les métriques de qualité de clustering."""
        try:
            sil = silhouette_score(X, labels)
            db  = davies_bouldin_score(X, labels)
            ch  = calinski_harabasz_score(X, labels)
            logger.info(
                f"  [{label}] Silhouette={sil:.4f} | "
                f"Davies-Bouldin={db:.4f} | "
                f"Calinski-Harabasz={ch:.4f}"
            )
            return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
        except Exception as e:
            logger.warning(f"Évaluation {label} impossible : {e}")
            return {"error": str(e)}

    def elbow_inertia(self, X: np.ndarray, k_range: range = range(2, 11)) -> dict:
        """
        Calcule l'inertie K-Means pour plusieurs valeurs de k (méthode du coude).

        Returns
        -------
        dict : {k: inertia}
        """
        logger.info("📉 Calcul de la courbe du coude...")
        result = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.kmeans_seed, n_init=10)
            km.fit(X)
            result[k] = km.inertia_
        return result

"""
stats/tests.py
Tests de significativité statistique sur les distributions de topics.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr


class StatisticalAnalyzer:
    """
    Analyse statistique de l'évolution des thèmes entre deux lois de finances.

    Méthodes disponibles :
    - Test de Mann-Whitney U (non paramétrique, par topic)
    - Test du Chi² sur les piliers dominants
    - Corrélation de Spearman entre fréquences thématiques et dotations budgétaires
    """

    # ── Tests par topic ───────────────────────────────────────────────────────

    @staticmethod
    def test_mannwhitney_topics(
        dist_2024: np.ndarray,
        dist_2025: np.ndarray,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Teste la différence de distribution de chaque topic entre 2024 et 2025.

        Parameters
        ----------
        dist_2024 : np.ndarray (n_docs_2024, n_topics)
        dist_2025 : np.ndarray (n_docs_2025, n_topics)
        alpha     : float — Seuil de significativité.

        Returns
        -------
        pd.DataFrame avec colonnes : topic, U_stat, p_value, significatif, delta_moyen.
        """
        n_topics = dist_2024.shape[1]
        rows = []

        logger.info("Tests de Mann-Whitney U par topic...")
        logger.info(
            "  H0 : les distributions de probabilités sont identiques entre 2024 et 2025"
        )

        for t in range(n_topics):
            x = dist_2024[:, t]
            y = dist_2025[:, t]
            stat, pval = mannwhitneyu(x, y, alternative="two-sided")
            delta = float(y.mean() - x.mean())
            sig   = pval < alpha

            rows.append(
                {
                    "topic":       t,
                    "U_stat":      round(stat, 4),
                    "p_value":     round(pval, 4),
                    "significatif": sig,
                    "delta_moyen": round(delta, 4),
                    "direction":   "↑ 2025" if delta > 0 else "↓ 2025",
                }
            )
            logger.info(
                f"  Topic {t}: U={stat:.2f}, p={pval:.4f} "
                f"→ {'Significatif' if sig else 'Non significatif'}"
            )

        return pd.DataFrame(rows)

    # ── Test du Chi² sur les piliers dominants ────────────────────────────────

    @staticmethod
    def test_chi2_piliers(
        df_2024: pd.DataFrame,
        df_2025: pd.DataFrame,
        col: str = "pilier_dominant",
        alpha: float = 0.05,
    ) -> Dict:
        """
        Test du Chi² sur la distribution des piliers dominants entre 2024 et 2025.

        Returns
        -------
        dict avec : chi2, p_value, dof, contingency_table, significatif.
        """
        logger.info("Test du Chi² sur les piliers SND30...")

        ct = pd.crosstab(
            pd.Series(["2024"] * len(df_2024) + ["2025"] * len(df_2025), name="annee"),
            pd.concat([df_2024[col], df_2025[col]], ignore_index=True).rename("pilier"),
        )

        chi2, pval, dof, expected = chi2_contingency(ct)
        sig = pval < alpha

        logger.info(
            f"  Chi²={chi2:.4f}, ddl={dof}, p={pval:.4f} "
            f"→ {'Significatif' if sig else 'Non significatif'}"
        )

        return {
            "chi2":               round(chi2, 4),
            "p_value":            round(pval, 4),
            "dof":                dof,
            "contingency_table":  ct,
            "significatif":       sig,
        }

    # ── Corrélation Spearman fréquences ↔ dotations ───────────────────────────

    @staticmethod
    def correlation_spearman(
        freq_thematiques: np.ndarray,
        dotations_budgetaires: np.ndarray,
        labels: List[str] | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Corrèle les fréquences thématiques extraites par LDA/zero-shot
        avec les montants financiers réels (BIP).

        Parameters
        ----------
        freq_thematiques : np.ndarray (n_piliers,) — proportions par pilier.
        dotations_budgetaires : np.ndarray (n_piliers,) — montants (FCFA ou rang).
        labels : List[str] | None — noms des piliers.
        alpha : float — seuil de significativité.

        Returns
        -------
        pd.DataFrame avec : rho, p_value, significatif.
        """
        rho, pval = spearmanr(freq_thematiques, dotations_budgetaires)
        sig = pval < alpha

        logger.info(f"Corrélation de Spearman : ρ={rho:.4f}, p={pval:.4f}")

        df = pd.DataFrame(
            {
                "pilier":    labels if labels else range(len(freq_thematiques)),
                "freq":      freq_thematiques,
                "dotation":  dotations_budgetaires,
            }
        )
        df.attrs["rho"]          = rho
        df.attrs["p_value"]      = pval
        df.attrs["significatif"] = sig
        return df

    # ── Résumé des statistiques descriptives des topics ───────────────────────

    @staticmethod
    def describe_topic_distributions(
        dist_2024: np.ndarray, dist_2025: np.ndarray
    ) -> pd.DataFrame:
        """
        Retourne un tableau comparatif des statistiques descriptives par topic.
        """
        rows = []
        n_topics = dist_2024.shape[1]
        for t in range(n_topics):
            rows.append(
                {
                    "topic":         t,
                    "mean_2024":     dist_2024[:, t].mean(),
                    "mean_2025":     dist_2025[:, t].mean(),
                    "median_2024":   np.median(dist_2024[:, t]),
                    "median_2025":   np.median(dist_2025[:, t]),
                    "std_2024":      dist_2024[:, t].std(),
                    "std_2025":      dist_2025[:, t].std(),
                }
            )
        return pd.DataFrame(rows).round(4)

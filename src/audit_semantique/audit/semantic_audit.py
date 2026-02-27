"""
audit/semantic_audit.py
Auditeur sémantique : calcul du glissement sémantique entre deux lois de finances.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from audit_semantique.config import AUDIT_PARAMS


class AuditeurSemantique:
    """
    Mesure le glissement sémantique entre deux corpus (lois de finances).

    Parameters
    ----------
    embeddings_ref : np.ndarray
        Matrice d'embeddings de l'année de référence (n_ref, d).
    embeddings_comp : np.ndarray
        Matrice d'embeddings de l'année à comparer (n_comp, d).
    df_ref : pd.DataFrame
        DataFrame associé à l'année de référence.
    df_comp : pd.DataFrame
        DataFrame associé à l'année de comparaison.
    """

    def __init__( 
        self,
        embeddings_ref: np.ndarray,
        embeddings_comp: np.ndarray,
        df_ref: pd.DataFrame,
        df_comp: pd.DataFrame,
    ) -> None:
        self.embeddings_ref  = embeddings_ref
        self.embeddings_comp = embeddings_comp
        self.df_ref          = df_ref.reset_index(drop=True)
        self.df_comp         = df_comp.reset_index(drop=True)
        self._sim_matrix: np.ndarray | None = None

    # ── Matrice de similarité ─────────────────────────────────────────────────

    def calculer_matrice_similarite(self) -> np.ndarray:
        """
        Calcule la matrice de similarité cosinus entre les deux corpus.

        Returns
        -------
        np.ndarray de forme ``(n_ref, n_comp)``.
        """
        logger.info("🔢 Calcul de la matrice de similarité cosinus...")
        self._sim_matrix = cosine_similarity(self.embeddings_ref, self.embeddings_comp)
        logger.info(f"✅ Matrice calculée : {self._sim_matrix.shape}")
        return self._sim_matrix

    @property
    def sim_matrix(self) -> np.ndarray:
        if self._sim_matrix is None:
            self.calculer_matrice_similarite()
        return self._sim_matrix  # type: ignore

    # ── Meilleurs correspondances ─────────────────────────────────────────────

    def trouver_meilleurs_matches(self, top_k: int = AUDIT_PARAMS["top_k_matches"]) -> pd.DataFrame:
        """
        Trouve les ``top_k`` meilleures correspondances pour chaque article de référence.

        Returns
        -------
        pd.DataFrame avec colonnes :
        article_ref_id, article_comp_id, similarite, rang, texte_ref, texte_comp.
        """
        matrix = self.sim_matrix
        logger.info(f"🎯 Recherche des {top_k} meilleures correspondances...")
        resultats = []

        for i in range(len(self.df_ref)):
            scores     = matrix[i]
            top_idx    = np.argsort(scores)[-top_k:][::-1]
            top_scores = scores[top_idx]
            row_ref    = self.df_ref.iloc[i]

            for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
                row_comp = self.df_comp.iloc[idx]
                resultats.append(
                    {
                        "article_ref_id":    row_ref.get("id", i),
                        "article_ref_annee": row_ref.get("annee", ""),
                        "article_comp_id":   row_comp.get("id", idx),
                        "article_comp_annee":row_comp.get("annee", ""),
                        "similarite":        float(score),
                        "rang":              rank,
                        "texte_ref":         str(row_ref.get("cleaned_content", ""))[:200] + "...",
                        "texte_comp":        str(row_comp.get("cleaned_content", ""))[:200] + "...",
                    }
                )

        df_matches = pd.DataFrame(resultats)
        logger.info(f"✅ {len(df_matches)} correspondances trouvées.")
        return df_matches

    # ── Analyse du glissement ─────────────────────────────────────────────────

    def analyser_glissement(
        self, seuil: float = AUDIT_PARAMS["seuil_changement"]
    ) -> Dict:
        """
        Analyse le glissement sémantique global.

        Parameters
        ----------
        seuil : float
            Seuil de similarité en dessous duquel un article est considéré
            comme « changé » (défaut : 0.70).

        Returns
        -------
        dict avec les métriques : moyenne, médiane, std, score_glissement,
        nb/pct changements, indices des articles les plus modifiés.
        """
        matrix       = self.sim_matrix
        best_scores  = matrix.max(axis=1)

        moyenne      = float(best_scores.mean())
        mediane      = float(np.median(best_scores))
        std          = float(best_scores.std())
        score_gliss  = 1.0 - moyenne

        mask_change  = best_scores < seuil
        nb_chgt      = int(mask_change.sum())
        pct_chgt     = nb_chgt / len(best_scores) * 100.0
        indices_chgt = np.argsort(best_scores)[:10].tolist()

        interpretation = (
            "Faible" if score_gliss < 0.2
            else "Modéré" if score_gliss < 0.4
            else "Élevé"
        )

        logger.info("=" * 60)
        logger.info("📊 ANALYSE DU GLISSEMENT SÉMANTIQUE")
        logger.info(f"  • Similarité moyenne : {moyenne:.3f}")
        logger.info(f"  • Score de glissement : {score_gliss:.3f} ({interpretation})")
        logger.info(f"  • Articles avec changements significatifs : {nb_chgt} ({pct_chgt:.1f}%)")
        logger.info("=" * 60)

        return {
            "best_scores":         best_scores,
            "moyenne_similarite":  moyenne,
            "mediane_similarite":  mediane,
            "std_similarite":      std,
            "score_glissement":    score_gliss,
            "interpretation":      interpretation,
            "nb_changements":      nb_chgt,
            "pct_changements":     pct_chgt,
            "indices_changements": indices_chgt,
        }

"""
modeling/zero_shot.py
Classification zero-shot des articles de loi selon les 4 piliers SND30.
Modèle : MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingue).
"""
from __future__ import annotations

from typing import List

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
from transformers import pipeline

from audit_semantique.config import PILIERS_SND30, ZERO_SHOT_MODEL, ZERO_SHOT_PARAMS


class ZeroShotClassifier:
    """
    Classifie des textes dans les 4 piliers SND30 en zero-shot.

    Parameters
    ----------
    model_name : str
        Identifiant HuggingFace du modèle (défaut : mDeBERTa-v3-mnli-xnli).
    piliers : List[str]
        Labels cibles. Par défaut, les 4 piliers SND30.
    device : int
        ``0`` pour GPU, ``-1`` pour CPU.
    """

    def __init__(
        self,
        model_name: str = ZERO_SHOT_MODEL,
        piliers: List[str] = PILIERS_SND30,
        device: int = -1,
    ) -> None:
        self.model_name = model_name
        self.piliers = piliers
        self.device = device
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        logger.info(f"📥 Chargement du classifieur zero-shot : {self.model_name}...")
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
        )
        logger.info("✅ Classifieur prêt.")

    # ── API publique ─────────────────────────────────────────────────────────

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "cleaned_content",
        batch_size: int = ZERO_SHOT_PARAMS["batch_size"],
        max_length: int = ZERO_SHOT_PARAMS["max_length"],
        multi_label: bool = ZERO_SHOT_PARAMS["multi_label"],
    ) -> pd.DataFrame:
        """
        Ajoute une colonne ``score_<pilier>`` et ``pilier_dominant`` au DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contenant la colonne texte.
        text_col : str
            Nom de la colonne texte.
        batch_size : int
            Taille des micro-batches.
        max_length : int
            Longueur max (tokens) avant troncature.
        multi_label : bool
            Si True, les scores ne sont pas normalisés à 1.

        Returns
        -------
        pd.DataFrame enrichi.
        """
        self._load()
        df = df.copy()

        # Pré-créer les colonnes de scores
        for pilier in self.piliers:
            df[f"score_{pilier}"] = 0.0

        textes = df[text_col].fillna("").astype(str).tolist()
        logger.info(f"🔍 Classification zero-shot SND30 sur {len(textes)} articles...")

        for start in tqdm(range(0, len(textes), batch_size), desc="Batches ZS", unit="batch"):
            batch = textes[start : start + batch_size]
            results = self._pipeline(
                batch,
                candidate_labels=self.piliers,
                multi_label=multi_label,
                truncation=True,
                max_length=max_length,
            )

            for j, res in enumerate(results):
                idx = start + j
                for pilier, score in zip(res["labels"], res["scores"]):
                    df.loc[idx, f"score_{pilier}"] = float(score)

        # Pilier dominant = argmax des scores
        score_cols = [f"score_{p}" for p in self.piliers]
        df["pilier_dominant"] = df[score_cols].idxmax(axis=1).str.replace(
            "score_", "", regex=False
        )
        logger.info("✅ Classification SND30 terminée.")
        return df

    def export_excel(self, df: pd.DataFrame, path: str | None = None) -> str:
        """Exporte le DataFrame classifié en Excel."""
        from audit_semantique.config import REPORTS_DIR

        out = path or str(REPORTS_DIR / "classification_snd30.xlsx")
        df.to_excel(out, index=False)
        logger.info(f"💾 Résultats exportés → {out}")
        return out

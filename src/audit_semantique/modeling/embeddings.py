"""
modeling/embeddings.py
Encodage des textes en vecteurs d'embeddings (sentence-transformers).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from audit_semantique.config import (
    SENTENCE_TRANSFORMER_MODEL,
    EMBEDDING_PARAMS,
    MODELS_DIR,
)


class SentenceTransformerEncoder:
    """Encodeur basé sur sentence-transformers, optimisé pour similarités.

    Utilise par défaut le modèle multilingue
    ``sentence-transformers/paraphrase-multilingual-mpnet-base-v2``
    adapté au français.
    """

    def __init__(
        self,
        model_name: str = SENTENCE_TRANSFORMER_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Sentence-Transformers device : {self.device}")
        self._model: Optional[SentenceTransformer] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Chargement du modèle sentence-transformers {self.model_name}...")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Modèle sentence-transformers prêt.")

    def encode(
        self,
        texts: List[str],
        batch_size: int = EMBEDDING_PARAMS["batch_size"],
        max_length: int = EMBEDDING_PARAMS["max_length"],
    ) -> np.ndarray:
        """Encode une liste de textes en embeddings normalisés.

        Retourne un ``np.ndarray`` de forme ``(n_textes, d)`` (d = 768).
        Les vecteurs sont L2-normalisés, optimisés pour similarité cosinus.
        """
        self._load()
        assert self._model is not None

        logger.info(f"Encodage (sentence-transformers) de {len(texts)} textes...")

        # Limite de longueur gérée au niveau du modèle
        self._model.max_seq_length = max_length

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info(f"Encodage sentence-transformers terminé : shape {embeddings.shape}")
        return embeddings

    # ── Persistance ──────────────────────────────────────────────────────────

    def save_embeddings(self, embeddings: np.ndarray, name: str) -> Path:
        """Sauvegarde les embeddings en .npy dans ``outputs/models/``."""
        path = MODELS_DIR / f"embeddings_{name}.npy"
        np.save(path, embeddings)
        logger.info(f"💾 Embeddings sauvegardés → {path}")
        return path

    def load_embeddings(self, name: str) -> np.ndarray:
        """Charge des embeddings préalablement sauvegardés."""
        path = MODELS_DIR / f"embeddings_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Embeddings introuvables : {path}")
        emb = np.load(path)
        logger.info(f"Embeddings chargés depuis {path} : shape {emb.shape}")
        return emb

    # Méthodes de persistance réutilisables pour SentenceTransformerEncoder

    @staticmethod
    def save(embeddings: np.ndarray, name: str) -> Path:
        path = MODELS_DIR / f"embeddings_{name}.npy"
        np.save(path, embeddings)
        logger.info(f"💾 Embeddings (ST) sauvegardés → {path}")
        return path

    @staticmethod
    def load(name: str) -> np.ndarray:
        path = MODELS_DIR / f"embeddings_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Embeddings ST introuvables : {path}")
        emb = np.load(path)
        logger.info(f"Embeddings (ST) chargés depuis {path} : shape {emb.shape}")
        return emb

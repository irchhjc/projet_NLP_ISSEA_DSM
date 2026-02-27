"""
modeling/embeddings.py
Encodage des textes en vecteurs d'embeddings via CamemBERT.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from transformers import CamembertModel, CamembertTokenizer

from audit_semantique.config import CAMEMBERT_MODEL, CAMEMBERT_PARAMS, MODELS_DIR


class CamembertEncoder:
    """
    Encode des textes en vecteurs denses (768 dims) avec CamemBERT.

    Le token [CLS] de la dernière couche cachée est utilisé comme
    représentation de document.

    Parameters
    ----------
    model_name : str
        Identifiant HuggingFace du modèle (défaut : ``camembert-base``).
    device : str | None
        ``"cuda"`` ou ``"cpu"`` — détecté automatiquement si None.
    """

    def __init__(
        self,
        model_name: str = CAMEMBERT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"🖥️  Device détecté : {self.device}")
        self._tokenizer: Optional[CamembertTokenizer] = None
        self._model: Optional[CamembertModel] = None

    # ── Chargement paresseux ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"📥 Chargement du modèle {self.model_name}...")
        self._tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
        self._model = CamembertModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        logger.info("✅ Modèle prêt.")


    @property
    def tokenizer(self) -> CamembertTokenizer:
        self._load()
        return self._tokenizer  # type: ignore

    @property
    def model(self) -> CamembertModel:
        self._load()
        return self._model  # type: ignore

    # ── Encodage ─────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: List[str],
        batch_size: int = CAMEMBERT_PARAMS["batch_size"],
        max_length: int = CAMEMBERT_PARAMS["max_length"],
    ) -> np.ndarray:
        """
        Encode une liste de textes en embeddings.

        Parameters
        ----------
        texts : List[str]
            Textes à encoder (déjà nettoyés).
        batch_size : int
            Taille des micro-batches.
        max_length : int
            Longueur max de séquence (tokens).

        Returns
        -------
        np.ndarray de forme ``(n_textes, 768)``.
        """
        self._load()
        embeddings: list[np.ndarray] = []

        logger.info(f"🔄 Encodage de {len(texts)} textes...")

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self._model(**encoded)
                # [CLS] token — première position de la dernière couche cachée
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)

                done = min(start + batch_size, len(texts))
                if done % (batch_size * 5) == 0 or done == len(texts):
                    logger.info(f"  ✓ {done}/{len(texts)} textes traités")

        result = np.vstack(embeddings)
        logger.info(f"✅ Encodage terminé : shape {result.shape}")
        return result

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
        logger.info(f"📂 Embeddings chargés depuis {path} : shape {emb.shape}")
        return emb

"""
preprocessing/data_loader.py
Chargement et validation des fichiers JSON des lois de finances.
"""
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from audit_semantique.config import PATH_LOI_2024, PATH_LOI_2025, MODELS_DIR
from audit_semantique.preprocessing.text_cleaner import TextPreprocessor


# Mapping des colonnes selon l'année (les JSONs peuvent varier légèrement)
_COLUMN_ALIASES = {
    "contenu": "content",
    "texte":   "content",
    "body":    "content",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise les noms de colonnes entre les deux fichiers JSON."""
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})
    return df


def load_json(path: Union[str, Path], annee: int) -> pd.DataFrame:
    """
    Charge un fichier JSON de loi de finances et retourne un DataFrame.

    Parameters
    ----------
    path : str | Path
        Chemin vers le fichier JSON.
    annee : int
        Année de la loi (ex. 2024).

    Returns
    -------
    pd.DataFrame avec colonnes : id, titre, chapitre, content, annee.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Placez vos JSON dans data/raw/ (voir README)."
        )

    logger.info(f"Chargement de {path.name}...")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df = _normalize_columns(df)

    # Garantir les colonnes minimales
    for col in ("id", "titre", "content"):
        if col not in df.columns:
            logger.warning(f"Colonne '{col}' absente — création d'une colonne vide.")
            df[col] = ""

    df["annee"] = annee
    logger.info(f"  → {len(df)} articles chargés (année {annee}).")
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les deux lois de finances depuis les chemins configurés.

    Returns
    -------
    (loi_2024, loi_2025) : tuple de DataFrames.
    """
    loi_2024 = load_json(PATH_LOI_2024, annee=2024)
    loi_2025 = load_json(PATH_LOI_2025, annee=2025)
    return loi_2024, loi_2025


def generate_and_save_embeddings(
    df: pd.DataFrame,
    annee: int,
    encoder=None,
    force: bool = False,
) -> np.ndarray:
    """
    Génère et sauvegarde les embeddings pour une loi de finances.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant au minimum la colonne 'content'.
    annee : int
        Année de la loi (ex. 2024).
    encoder : SentenceTransformerEncoder | None
        Encoder à utiliser. Si None, un nouvel encoder sera créé.
    force : bool
        Si True, régénère les embeddings même s'ils existent déjà.

    Returns
    -------
    np.ndarray de forme (n_articles, 768) contenant les embeddings.
    """
    # Import local pour éviter les dépendances circulaires
    from audit_semantique.modeling.embeddings import SentenceTransformerEncoder

    embedding_name = f"loi_{annee}"
    embedding_path = MODELS_DIR / f"embeddings_{embedding_name}.npy"

    # Vérifier si les embeddings existent déjà
    if embedding_path.exists() and not force:
        logger.info(f"Embeddings déjà existants pour {annee} : {embedding_path}")
        embeddings = np.load(embedding_path)
        logger.info(f"  → Shape : {embeddings.shape}")
        return embeddings

    # Créer l'encoder si nécessaire
    if encoder is None:
        logger.info("🔧 Création d'un nouvel encoder sentence-transformers...")
        encoder = SentenceTransformerEncoder()

    # Générer les embeddings à partir de titre + contenu nettoyés
    logger.info(f"⚙️  Génération des embeddings pour la loi de finances {annee} (titre + contenu)...")

    prep = TextPreprocessor(lower=True, rm_accents=True)

    if "cleaned_content" not in df.columns:
        df["cleaned_content"] = prep.preprocess_series(df["content"])
    if "cleaned_title" not in df.columns and "titre" in df.columns:
        df["cleaned_title"] = prep.preprocess_series(df["titre"])

    titles = df.get("cleaned_title", df.get("titre", "")).fillna("")
    contents = df["cleaned_content"].fillna("")
    texts = (titles + " [SEP] " + contents).str.strip().tolist()

    embeddings = encoder.encode(texts)

    # Sauvegarder
    encoder.save_embeddings(embeddings, embedding_name)
    
    return embeddings


def load_saved_embeddings(annee: int) -> np.ndarray:
    """
    Charge les embeddings préalablement sauvegardés pour une année donnée.

    Parameters
    ----------
    annee : int
        Année de la loi (ex. 2024).

    Returns
    -------
    np.ndarray de forme (n_articles, 768) contenant les embeddings.
    """
    embedding_name = f"loi_{annee}"
    embedding_path = MODELS_DIR / f"embeddings_{embedding_name}.npy"
    
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embeddings introuvables pour l'année {annee} : {embedding_path}\n"
            "Utilisez generate_and_save_embeddings() pour les créer."
        )
    
    embeddings = np.load(embedding_path)
    logger.info(f"Embeddings chargés pour {annee} : shape {embeddings.shape}")
    return embeddings


def generate_all_embeddings(force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère et sauvegarde les embeddings pour les deux lois de finances (2024 et 2025).

    Parameters
    ----------
    force : bool
        Si True, régénère les embeddings même s'ils existent déjà.

    Returns
    -------
    (embeddings_2024, embeddings_2025) : tuple de np.ndarray.
    """
    # Import local pour éviter les dépendances circulaires
    from audit_semantique.modeling.embeddings import SentenceTransformerEncoder

    # Charger les données
    loi_2024, loi_2025 = load_all()
    
    # Créer un seul encoder pour les deux années (pour réutiliser le modèle chargé)
    logger.info("🔧 Initialisation de l'encoder sentence-transformers...")
    encoder = SentenceTransformerEncoder()
    
    # Générer les embeddings pour chaque année
    logger.info("\n" + "="*60)
    logger.info("Génération des embeddings pour 2024")
    logger.info("="*60)
    embeddings_2024 = generate_and_save_embeddings(loi_2024, 2024, encoder, force)
    
    logger.info("\n" + "="*60)
    logger.info("Génération des embeddings pour 2025")
    logger.info("="*60)
    embeddings_2025 = generate_and_save_embeddings(loi_2025, 2025, encoder, force)
    
    logger.info("\n" + "="*60)
    logger.info("Génération des embeddings terminée !")
    logger.info(f"  • 2024 : {embeddings_2024.shape}")
    logger.info(f"  • 2025 : {embeddings_2025.shape}")
    logger.info("="*60)
    
    return embeddings_2024, embeddings_2025

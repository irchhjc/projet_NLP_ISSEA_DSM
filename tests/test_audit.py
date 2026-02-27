"""
tests/test_audit.py
Tests unitaires pour le module d'audit sémantique.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_semantique.audit.semantic_audit import AuditeurSemantique


@pytest.fixture
def sample_data():
    """Crée des données d'exemple pour les tests."""
    np.random.seed(42)
    emb_2024 = np.random.rand(10, 768).astype(np.float32)
    emb_2025 = np.random.rand(8, 768).astype(np.float32)

    df_2024 = pd.DataFrame({
        "id": [f"Art.{i}" for i in range(10)],
        "annee": 2024,
        "cleaned_content": [f"texte article {i}" for i in range(10)],
    })
    df_2025 = pd.DataFrame({
        "id": [f"Art.{i}" for i in range(8)],
        "annee": 2025,
        "cleaned_content": [f"texte article {i} nouveau" for i in range(8)],
    })
    return emb_2024, emb_2025, df_2024, df_2025


def test_calculer_matrice_similarite(sample_data):
    emb_2024, emb_2025, df_2024, df_2025 = sample_data
    auditeur = AuditeurSemantique(emb_2024, emb_2025, df_2024, df_2025)
    matrix = auditeur.calculer_matrice_similarite()
    assert matrix.shape == (10, 8)
    assert matrix.min() >= -1.0
    assert matrix.max() <= 1.0


def test_trouver_meilleurs_matches(sample_data):
    emb_2024, emb_2025, df_2024, df_2025 = sample_data
    auditeur = AuditeurSemantique(emb_2024, emb_2025, df_2024, df_2025)
    df_matches = auditeur.trouver_meilleurs_matches(top_k=3)
    assert len(df_matches) == 10 * 3  # 10 articles ref × top_k=3
    assert "similarite" in df_matches.columns
    assert "rang" in df_matches.columns


def test_analyser_glissement(sample_data):
    emb_2024, emb_2025, df_2024, df_2025 = sample_data
    auditeur = AuditeurSemantique(emb_2024, emb_2025, df_2024, df_2025)
    resultats = auditeur.analyser_glissement(seuil=0.7)
    assert 0.0 <= resultats["score_glissement"] <= 1.0
    assert resultats["interpretation"] in ("Faible", "Modéré", "Élevé")
    assert "best_scores" in resultats
    assert len(resultats["best_scores"]) == 10

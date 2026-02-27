"""
tests/test_preprocessing.py
Tests unitaires pour le module de prétraitement.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_semantique.preprocessing.text_cleaner import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor(lower=True, rm_accents=True)


def test_preprocess_basic(preprocessor):
    text = "Le budget général de l'État pour 2024 est de 5 000 milliards FCFA."
    result = preprocessor.preprocess(text)
    assert isinstance(result, str)
    assert len(result) > 0
    # Les nombres doivent être remplacés
    assert "5" not in result.split()


def test_preprocess_empty(preprocessor):
    assert preprocessor.preprocess("") == ""
    assert preprocessor.preprocess("   ") == ""


def test_preprocess_removes_stopwords(preprocessor):
    text = "le la les du de des"
    result = preprocessor.preprocess(text)
    # Tous ces mots sont des stopwords -> résultat vide ou très court
    assert result.strip() == "" or len(result.split()) == 0


def test_tokenize_for_lda(preprocessor):
    text = "transformation structurelle économie développement capital humain"
    tokens = preprocessor.tokenize_for_lda(text, min_len=3)
    assert isinstance(tokens, list)
    assert all(len(t) >= 3 for t in tokens)


def test_preprocess_series(preprocessor):
    import pandas as pd
    series = pd.Series(["Le budget", "La loi de finances", None, ""])
    result = preprocessor.preprocess_series(series)
    assert len(result) == 4
    assert result[2] == ""
    assert result[3] == ""

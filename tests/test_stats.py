"""
tests/test_stats.py
Tests unitaires pour le module de statistiques.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_semantique.stats.tests import StatisticalAnalyzer


@pytest.fixture
def topic_distributions():
    np.random.seed(42)
    dist_2024 = np.random.dirichlet([1, 1, 1, 1], size=50)  # (50, 4)
    dist_2025 = np.random.dirichlet([2, 1, 1, 0.5], size=40)  # distributions légèrement différentes
    return dist_2024, dist_2025


def test_mannwhitney_topics(topic_distributions):
    dist_2024, dist_2025 = topic_distributions
    df = StatisticalAnalyzer.test_mannwhitney_topics(dist_2024, dist_2025)
    assert len(df) == 4  # 4 topics
    assert "p_value" in df.columns
    assert "significatif" in df.columns
    assert df["p_value"].between(0, 1).all()


def test_describe_topic_distributions(topic_distributions):
    dist_2024, dist_2025 = topic_distributions
    df = StatisticalAnalyzer.describe_topic_distributions(dist_2024, dist_2025)
    assert len(df) == 4
    assert "mean_2024" in df.columns
    assert "mean_2025" in df.columns


def test_chi2_piliers():
    df_2024 = pd.DataFrame({
        "pilier_dominant": ["A", "B", "A", "C", "D", "A", "B", "C"]
    })
    df_2025 = pd.DataFrame({
        "pilier_dominant": ["A", "A", "B", "D", "D", "A", "C", "D"]
    })
    result = StatisticalAnalyzer.test_chi2_piliers(df_2024, df_2025)
    assert "chi2" in result
    assert "p_value" in result
    assert "contingency_table" in result

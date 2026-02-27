# Audit Sémantique des Finances Publiques du Cameroun

> Utilisation de l'IA/NLP pour mesurer mathématiquement l'évolution des priorités budgétaires de l'État camerounais entre les Lois de Finances 2024 et 2025 — ISE3-DS, ISSEA Yaoundé.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-Dependency%20Manager-blueviolet.svg)](https://python-poetry.org/)
[![Dash](https://img.shields.io/badge/Dash-Interactive%20Dashboard-green.svg)](https://dash.plotly.com/)

---

## Contexte

Problématique posée dans le cadre de l'ISE3-DS (ISSEA, Yaoundé) :

> *Comment l'IA, à travers le NLP, peut-elle mesurer mathématiquement l'évolution des priorités de l'État camerounais entre la Loi de Finances 2024 et 2025 ?*

---

## Installation

**Prérequis** : Python 3.11+, [Poetry](https://python-poetry.org/docs/#installation)

```powershell
git clone <repository-url>
cd audit_semantique_cameroun
poetry install
```

> **GPU (optionnel)** : pour accélérer CamemBERT
> ```powershell
> poetry run pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Utilisation

### 1. Lancer le pipeline complet

```powershell
poetry run python scripts/main_pipeline.py
```

Durée ~10 minutes. Le script exécute en séquence :

| Étape | Description | Fichiers générés |
|-------|-------------|------------------|
| 1 | Chargement des données JSON | — |
| 2 | Prétraitement des textes | — |
| 3 | Embeddings CamemBERT (articles des lois de finances) | `outputs/models/embeddings_*.npy` |
| 4 | Similarités cosinus + visualisations | `outputs/reports/embeddings_similarities_*.xlsx` |
| 5 | Topic Modeling LDA | `outputs/reports/lda_topics_*.xlsx` |
| 6 | Clustering K-Means + HDBSCAN | `outputs/reports/articles_*_avec_clusters.xlsx` |
| 7 | Classification Zero-Shot SND30 (objectifs AE/CP) | `outputs/reports/objectifs_classifications_snd30_*.xlsx` |
| 8 | Tests statistiques (Chi², Mann-Whitney) | `outputs/reports/chi2_piliers.xlsx` |
| 9 | Analyse budgétaire AE/CP | `outputs/reports/analyse_budgetaire.xlsx` |

### 2. Lancer le dashboard

```powershell
poetry run python scripts/run_dash.py
```

Dashboard accessible sur **http://127.0.0.1:8050**

---

## Structure du Projet

```
audit_semantique_cameroun/
│
├── data/raw/                        # Données d'entrée (JSON)
│   ├── loi_finances_1_*.json        # Articles Loi de Finances 2024
│   ├── loi_finances_2_*.json        # Articles Loi de Finances 2025
│   ├── ae_pb2024.json               # Budget AE/CP 2024
│   └── ae_pb2025.json               # Budget AE/CP 2025
│
├── src/audit_semantique/            # Package principal
│   ├── config.py                    # ⭐ Configuration centrale (chemins, hyperparamètres)
│   ├── preprocessing/               # Chargement et nettoyage des textes
│   ├── modeling/                    # CamemBERT (embeddings) + mDeBERTa (zero-shot)
│   ├── audit/                       # Similarité cosinus et glissement sémantique
│   ├── topic_modeling/              # LDA Gensim
│   ├── clustering/                  # K-Means + HDBSCAN
│   ├── stats/                       # Tests Mann-Whitney, Chi², Spearman
│   └── visualization/               # Graphiques Matplotlib / Plotly
│
├── scripts/
│   ├── main_pipeline.py             # ⭐ Pipeline complet
│   └── run_dash.py                  # Lancement dashboard Dash
│
├── dashboard/
│   └── app_dash.py                  # Application Dash interactive
│
├── outputs/                         # Artefacts générés (recréés par main_pipeline.py)
│   ├── figures/                     # Graphiques .png et .html
│   ├── models/                      # Embeddings .npy
│   └── reports/                     # Rapports .xlsx
│
├── notebooks/audit_financier.ipynb  # Analyses exploratoires
├── tests/                           # Tests unitaires
└── pyproject.toml                   # Dépendances Poetry
```

---

## Données d'Entrée

### Format Lois de Finances (`loi_finances_*.json`)

```json
[
  {
    "id": 1,
    "titre": "TITRE I - BUDGET GENERAL",
    "chapitre": "Chapitre 1",
    "content": "Texte de l'article..."
  }
]
```

### Format Budget AE/CP (`ae_pb*.json`)

```json
[
  {
    "programme": "Nom du programme",
    "objectifs": ["Objectif 1", "Objectif 2"],
    "ae": 1000000000,
    "cp": 950000000
  }
]
```

---

## Configuration

Tous les paramètres sont centralisés dans `src/audit_semantique/config.py`.
Modifier ce seul fichier suffit à adapter tout le projet.

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `CAMEMBERT_MODEL` | `camembert-base` | Modèle d'embeddings HuggingFace |
| `ZERO_SHOT_MODEL` | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Modèle de classification |
| `LDA_PARAMS["num_topics"]` | `4` | Nombre de topics LDA |
| `KMEANS_PARAMS["n_clusters"]` | `3` | Nombre de clusters K-Means |
| `AUDIT_PARAMS["seuil_changement"]` | `0.70` | Seuil de similarité cosinus |
| `CAMEMBERT_PARAMS["batch_size"]` | `8` | Taille de batch (réduire si peu de RAM) |

---

## Piliers SND30

Les articles sont classifiés selon les 4 piliers de la Stratégie Nationale de Développement 2030 :

1. Transformation structurelle de l'économie pour accélérer la croissance
2. Développement du capital humain et du bien-être social
3. Promotion de l'emploi et de l'insertion socio-économique
4. Gouvernance, décentralisation et gestion stratégique de l'État

---

## Méthodologie

```
Données JSON
     │
     ▼
Prétraitement (nettoyage, tokenisation)
     │
     ├──→ CamemBERT → Embeddings des articles (768D)
     │         ├──→ Similarité cosinus  → Glissement sémantique 2024→2025
     │         ├──→ t-SNE               → Visualisation
     │         └──→ K-Means + HDBSCAN   → Clustering
     │
     ├──→ LDA Gensim → Topics thématiques
     │
     └──→ mDeBERTa Zero-Shot → Classification SND30
                │
                └──→ Tests Chi², Mann-Whitney → Significativité statistique
```

---

## Références

- Martin et al. (2020). *CamemBERT: a Tasty French Language Model*
- He et al. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*
- République du Cameroun. *Stratégie Nationale de Développement SND30, 2020-2030*

---

**ISE3-DS — ISSEA Yaoundé | 2025-2026**

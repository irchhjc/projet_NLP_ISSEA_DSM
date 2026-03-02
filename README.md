# Audit Sémantique Cameroun

Outil complet d'audit sémantique de la Loi de Finances du Cameroun :

- embeddings de textes avec sentence-transformers
- classification zero-shot des objectifs sur les piliers du SND30
- topic modeling (LDA), clustering et indicateurs statistiques
- dashboard interactif Dash/Plotly (nuages de mots, UMAP, baromètre, budget)

---

## 1. Prérequis & installation

### Python

- Version supportée : `>= 3.11, < 3.15`

### Installation pour le développement (avec Poetry)

Depuis la racine du projet :

```bash
poetry install

# (recommandé) télécharger le modèle spaCy FR utilisé pour les stopwords
poetry run python -m spacy download fr_core_news_sm
```

Cette méthode crée un environnement virtuel géré par Poetry et installe le package `audit_semantique` en mode développement.

### Installation du package avec pip (utilisation comme bibliothèque)

Si vous souhaitez simplement utiliser le package `audit_semantique` dans un autre projet Python sans tout l’environnement Poetry, vous pouvez l’installer avec `pip` à partir des sources :

```bash
# Depuis la racine du dépôt cloné
pip install .
```

ou bien, en construisant d’abord les artefacts avec Poetry :

```bash
# 1) Construire la distribution
poetry build

# 2) Installer le wheel généré
pip install dist/audit_semantique_cameroun-*.whl
```

Une fois installé, le package peut être importé dans n’importe quel script Python :

```python
from audit_semantique.preprocessing.data_loader import load_raw_articles
from audit_semantique.modeling.embeddings import ArticleEmbedder
```

Vous pouvez aussi ajouter ce projet comme dépendance locale dans un autre projet géré par Poetry :

```bash
poetry add ../chemin/vers/audit_semantique_cameroun
```

---

## 2. Données d'entrée

Placer les fichiers bruts dans `data/raw` :

- `loi_finances_1_articles_id_titre_chapitre_contenu.json` → Loi de finances 2024
- `loi_finances_2_articles_id_titre_chapitre_contenu.json` → Loi de finances 2025

Les sorties sont générées automatiquement dans :

- `outputs/models` : embeddings (`embeddings_2024.npy`, `embeddings_2025.npy`, objectifs…)
- `outputs/reports` : fichiers Excel (audit sémantique, topics, clusters, budget, baromètre)
- `outputs/figures` : figures statiques (UMAP, distributions, etc.)

Les chemins et hyperparamètres sont centralisés dans `src/audit_semantique/config.py`.

---

## 3. Exécuter le pipeline complet

Commande unique pour lancer tout le pipeline d’audit :

```bash
poetry run python scripts/main_pipeline.py
```

Ce script ([scripts/main_pipeline.py](scripts/main_pipeline.py)) enchaîne :

1. Chargement des lois 2024/2025 depuis `data/raw`.
2. Prétraitement des textes via `TextPreprocessor` :
   - minuscules, normalisation, suppression URLs/nombres/ponctuation
   - stopwords français spaCy + stopwords métiers
3. Construction du texte d'entrée pour les embeddings articles :
   - `cleaned_title + " [SEP] " + cleaned_content`
4. Embeddings des articles avec sentence-transformers :
   - modèle : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
   - longueur max tronquée à 512 tokens (voir `EMBEDDING_PARAMS`)
5. Embeddings des objectifs budgétaires :
   - `cleaned_programme + " [SEP] " + cleaned_objectif`
6. Topic modeling (LDA) + topic dominant par article.
7. Clustering (K-Means, HDBSCAN) sur les embeddings d’articles.
8. Classification zero-shot des objectifs sur les piliers SND30 :
   - modèle : `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
9. Calcul des indicateurs statistiques et génération des rapports Excel.

Toutes les étapes loggent dans `logs/main_pipeline.log`.

---

## 4. Lancer le dashboard interactif

Une fois le pipeline exécuté (données et rapports disponibles), lancer l’application Dash :

```bash
poetry run python scripts/run_dash.py
```

Cela démarre le serveur Dash défini dans [src/audit_semantique/dashboard/app_dash.py](src/audit_semantique/dashboard/app_dash.py).

Par défaut, le dashboard est accessible à l’adresse :

- http://127.0.0.1:8050/

Le dashboard propose notamment :

- baromètre budgétaire & sémantique (AE/CP, piliers SND30)
- glissement sémantique (similarités 2024 vs 2025)
- topics LDA par année et comparaison
- clustering des articles (K-Means / HDBSCAN + métriques)
- correspondances objectifs ↔ articles via les embeddings
- tests statistiques et distributions budgétaires

### WordClouds & texte nettoyé

Les nuages de mots utilisent systématiquement le texte prétraité :

- objectifs : texte passé par `TextPreprocessor` avant agrégation
- topics : mots issus directement des sorties LDA pondérées
- clusters : agrégation de `cleaned_title` / `cleaned_content`

---

## 5. Structure du code

Le package `audit_semantique` (sous `src/`) est organisé en sous-modules :

- `audit` : audit sémantique principal (similarité inter‑annuelle, rapprochement d’articles 2024/2025)
- `clustering` : algorithmes et wrappers de clustering (K-Means, HDBSCAN, métriques)
- `modeling` : embeddings sentence-transformers & classification zero-shot SND30
- `preprocessing` : chargement et nettoyage des données brutes (articles, objectifs, budget)
- `stats` : statistiques descriptives et tests (KS, Mann-Whitney, Chi², Kruskal, etc.)
- `topic_modeling` : LDA et gestion des topics (topics dominants, distributions doc×topic)
- `visualization` : figures Matplotlib/Seaborn/Plotly pour le notebook et les rapports
- `dashboard` : application Dash/Plotly et callbacks (baromètre, topics, clustering, budget, stats)

Scripts principaux (dossier `scripts/`) :

- `main_pipeline.py` : exécute l’ensemble du pipeline d’audit (prétraitement → modèles → rapports)
- `run_dash.py` : démarre le dashboard interactif
- d’autres scripts utilitaires peuvent être ajoutés pour des analyses spécifiques.

---

## 6. Exécution rapide (récap)

```bash
# 1) Installer dépendances
poetry install
poetry run python -m spacy download fr_core_news_sm

# 2) Vérifier/placer les fichiers bruts dans data/raw

# 3) Lancer le pipeline (génère embeddings, topics, clusters, rapports Excel…)
poetry run python scripts/main_pipeline.py

# 4) Lancer le dashboard
poetry run python scripts/run_dash.py
```

---

## 7. Équipe projet

Ce projet réalisé par :

- **NGOULOU NGOUBILI Irch Defluviaire**
- **NOFOZO Sylvain**
- **AZONFACK Mirriam**
- **NGONO MVOGO Frank**

Superviseur : **MBIA NDI Marie Thérèse**

Promotion ISEL5‑DSM, ISSEA‑CEMAC.


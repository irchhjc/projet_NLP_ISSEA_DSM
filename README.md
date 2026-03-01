# Audit Sémantique Cameroun

Outil complet d'audit sémantique de la Loi de Finances du Cameroun :

- embeddings de textes avec sentence-transformers
- classification zero-shot des objectifs sur les piliers du SND30
- topic modeling (LDA), clustering et indicateurs statistiques
- dashboard interactif Dash/Plotly (nuages de mots, UMAP, baromètre, budget)

---

## 1. Installation

Le projet est géré avec [Poetry](https://python-poetry.org/).

```bash
poetry install

# (optionnel) téléchargement des ressources spaCy FR
poetry run python -m spacy download fr_core_news_sm || echo "spaCy FR déjà installé ou optionnel"
```

> Remarque : un environnement virtuel `.audit/` peut être utilisé pour l'exécution locale.

---

## 2. Données attendues

Les fichiers bruts doivent être placés dans `data/raw` :

- `loi_finances_1_articles_id_titre_chapitre_contenu.json` (année 2024)
- `loi_finances_2_articles_id_titre_chapitre_contenu.json` (année 2025)

Les sorties sont générées dans :

- `outputs/models` : embeddings (`embeddings_2024.npy`, `embeddings_2025.npy`…)
- `outputs/reports` : fichiers Excel d'analyse (audit, topics, clusters, budget, baromètre)
- `outputs/figures` : figures statiques (heatmaps, t‑SNE/UMAP, etc.)

La configuration centrale se trouve dans `src/audit_semantique/config.py`.

---

## 3. Pipeline principal

Le pipeline de bout en bout est orchestré par :

```bash
poetry run python scripts/main_pipeline.py
```

Ce script effectue notamment :

1. Chargement des lois 2024/2025 depuis `data/raw`.
2. Prétraitement des textes (titres et contenus) via `TextPreprocessor` :
	- passage en minuscules, normalisation
	- suppression des URLs, nombres, ponctuation
	- stopwords français de spaCy + stopwords métiers personnalisés
3. Construction du texte d'entrée pour les embeddings des articles :
	- `cleaned_title + " [SEP] " + cleaned_content` pour chaque article.
4. Génération des embeddings avec sentence-transformers :
	- modèle : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
	- paramètres contrôlés par `EMBEDDING_PARAMS` dans `config.py`
	- les embeddings sont **recalculés à chaque exécution** et sauvegardés dans `outputs/models`.
5. Génération des embeddings pour les objectifs budgétaires :
	- texte d'entrée : `cleaned_programme + " [SEP] " + cleaned_objectif` (même prétraitement que les articles)
	- fichiers produits : `embeddings_objectifs_2024.npy`, `embeddings_objectifs_2025.npy`.
6. Topic modeling (LDA) et affectation d'un topic dominant par article.
7. Clustering (K-Means, HDBSCAN) sur les embeddings d'articles.
8. Classification zero-shot des objectifs budgétaires sur les piliers du SND30 :
	- modèle : `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
	- paramètres contrôlés par `ZERO_SHOT_PARAMS`.
9. Calcul d'indicateurs statistiques (tests, distributions) et génération des rapports Excel.

Tous les hyperparamètres (embeddings, zero-shot, LDA, clustering, UMAP, audit) sont centralisés dans `config.py`.

---

## 4. Dashboard interactif

Le dashboard Dash consomme les sorties du pipeline et offre plusieurs vues :

- glissement sémantique entre les lois 2024 et 2025
- topics LDA par année et comparaison 2024/2025
- clustering des articles (K-Means / HDBSCAN, métriques internes)
- baromètre SND30 (répartition des objectifs, scores, budget AE/CP)
- correspondances objectifs ↔ articles (top 3 objectifs similaires et top 2 articles de loi proches pour un objectif donné)
- tests statistiques et analyses budgétaires

Lancer le dashboard après avoir exécuté le pipeline :

```bash
poetry run python scripts/run_dash.py
```

Le serveur Dash est alors accessible sur `http://127.0.0.1:8050/` (par défaut).

### Nuages de mots (WordClouds)

Tous les nuages de mots sont construits **à partir de texte nettoyé** :

- pour le baromètre des objectifs, les objectifs sont prétraités avec `TextPreprocessor` avant agrégation
- pour les topics, les mots proviennent directement des sorties LDA pondérées
- pour les clusters, les textes agrégés utilisent `cleaned_title` / `cleaned_content` (ou, à défaut, un nettoyage à la volée)

Cela garantit que les WordClouds reflètent le même texte prétraité que celui utilisé pour les embeddings et les analyses.

---

## 5. Structure du package

Le code source est organisé en modules :

- `audit` : logique d'audit sémantique principal (similarité inter‑annuelle)
- `clustering` : algorithmes et wrappers de clustering
- `modeling` : embeddings (sentence-transformers) et classification zero-shot
- `preprocessing` : chargement des données et nettoyage de texte
- `stats` : statistiques descriptives et tests
- `topic_modeling` : LDA, sujets et distributions de mots
- `visualization` : figures Matplotlib/Seaborn/Plotly
- `dashboard` : application Dash et callbacks

---

## 6. Auteur et licence

- Auteur : Irch Defluviaire, ISEL5-DSM ISSEA-CEMAC
- Licence : MIT

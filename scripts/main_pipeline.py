"""
Script principal pour exécuter le pipeline complet d'audit sémantique.
Pipeline: Prétraitement → Embeddings → Visualisations → Topic Modeling → Clustering
        → Classification Zero-Shot (objectifs budgétaires uniquement)
        → Génération des indicateurs pour le dashboard

Usage:
    poetry run python scripts/main_pipeline.py
"""
import sys
from pathlib import Path
from tqdm import tqdm
import time
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
import pandas as pd
import numpy as np
from loguru import logger

from audit_semantique.preprocessing.data_loader import load_all
from audit_semantique.preprocessing.text_cleaner import TextPreprocessor
from audit_semantique.modeling.embeddings import SentenceTransformerEncoder
from audit_semantique.audit.semantic_audit import AuditeurSemantique
from audit_semantique.topic_modeling.lda_model import LDATopicModeler
from audit_semantique.clustering.clusterer import DocumentClusterer
from audit_semantique.stats.tests import StatisticalAnalyzer
from audit_semantique.modeling.zero_shot import ZeroShotClassifier
from audit_semantique.visualization import plots
from audit_semantique.config import (
    RAW_DIR, OUTPUT_DIR, REPORTS_DIR, MODELS_DIR, FIGURES_DIR,
    LDA_PARAMS, KMEANS_PARAMS, AUDIT_PARAMS,
)

logger.add("logs/main_pipeline.log", rotation="10 MB")


_ILLEGAL_XLSX_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


def _clean_illegal_excel_chars(value):
    """Supprime les caractères de contrôle illégaux pour Excel."""
    if isinstance(value, str):
        return _ILLEGAL_XLSX_CHARS.sub("", value)
    return value


def _safe_to_excel(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """
    Sauvegarde un DataFrame en Excel en gérant les fichiers verrouillés par Excel
    sous Windows (PermissionError / fichier ~$*.xlsx ouvert).

    Si le fichier cible est verrouillé, propose une alternative horodatée et
    continue le pipeline sans interruption.
    """
    path = Path(path)
    lock_file = path.parent / f"~${path.name}"

    # Supprimer le fichier verrou résiduel si Excel est fermé mais a laissé le ~$
    if lock_file.exists():
        try:
            lock_file.unlink()
            logger.debug(f"🗑️  Fichier verrou supprimé : {lock_file.name}")
        except PermissionError:
            pass  # Excel a vraiment le fichier ouvert

    # Nettoyer les caractères illégaux dans les chaînes (erreur openpyxl.IllegalCharacterError)
    df = df.applymap(_clean_illegal_excel_chars)

    retries = 3
    for attempt in range(1, retries + 1):
        try:
            df.to_excel(path, **kwargs)
            if attempt > 1:
                logger.info(f"  ✅ Sauvegardé après {attempt} tentatives : {path.name}")
            return
        except PermissionError:
            if attempt < retries:
                logger.warning(
                    f"  ⚠️  {path.name} est ouvert dans Excel "
                    f"(tentative {attempt}/{retries}) — attente 3 s…"
                )
                time.sleep(3)
            else:
                # Dernier recours : nom de fichier alternatif horodaté
                from datetime import datetime
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                alt = path.with_stem(f"{path.stem}_{stamp}")
                try:
                    df.to_excel(alt, **kwargs)
                    logger.warning(
                        f"  ⚠️  Impossible d'écrire {path.name} (fichier ouvert dans Excel).\n"
                        f"       Fichier sauvegardé sous : {alt.name}\n"
                        f"       → Fermez Excel puis relancez le pipeline pour écraser l'original."
                    )
                except Exception as e2:
                    logger.error(f"  ❌ Échec de la sauvegarde de secours : {e2}")


def print_header(text: str):
    """Affiche un en-tête formaté."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    """Pipeline principal avec barre de progression."""
    
    print_header("🚀 PIPELINE AUDIT SÉMANTIQUE - LOI DE FINANCES CAMEROUN")
    
    # Définir les étapes du pipeline
    steps = [
        "Chargement des données",
        "Prétraitement des textes",
        "Génération des embeddings",
        "Visualisation des embeddings",
        "Topic Modeling (LDA)",
        "Clustering des documents",
        "Classification Zero-Shot (objectifs budgétaires)",
        "Tests statistiques",
        "Génération des indicateurs dashboard"
    ]
    
    with tqdm(total=len(steps), desc="Pipeline", unit="étape") as pbar:
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 1 : CHARGEMENT DES DONNÉES
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("📂 Chargement des données")
        logger.info("\n📂 ÉTAPE 1 - Chargement des données")
        
        loi_2024, loi_2025 = load_all()
        logger.info(f"  📄 2024 : {len(loi_2024)} articles")
        logger.info(f"  📄 2025 : {len(loi_2025)} articles")
        
        pbar.update(1)
        time.sleep(0.5)
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 2 : PRÉTRAITEMENT
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("🧹 Prétraitement des textes")
        logger.info("\n🧹 ÉTAPE 2 - Prétraitement des textes")
        
        prep = TextPreprocessor(lower=True, rm_accents=True)
        # Nettoyage du contenu
        loi_2024["cleaned_content"] = prep.preprocess_series(loi_2024["content"])
        loi_2025["cleaned_content"] = prep.preprocess_series(loi_2025["content"])
        # Nettoyage des titres
        loi_2024["cleaned_title"] = prep.preprocess_series(loi_2024["titre"])
        loi_2025["cleaned_title"] = prep.preprocess_series(loi_2025["titre"])

        logger.info("Textes et titres nettoyés")
        pbar.update(1)
        time.sleep(0.5)
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 3 : GÉNÉRATION DES EMBEDDINGS
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("🤖 Génération des embeddings")
        logger.info("\n🤖 ÉTAPE 3 - Génération des embeddings (sentence-transformers)")

        encoder = SentenceTransformerEncoder()

        # Toujours recalculer les embeddings et écraser les anciens fichiers
        emb_file_2024 = MODELS_DIR / "embeddings_2024.npy"
        emb_file_2025 = MODELS_DIR / "embeddings_2025.npy"

        logger.info(
            "  🔄 Calcul des embeddings sur titre + contenu nettoyés (les fichiers existants seront écrasés)..."
        )
        texts_2024 = (
            loi_2024["cleaned_title"].fillna("")
            + " [SEP] "
            + loi_2024["cleaned_content"].fillna("")
        ).str.strip()
        texts_2025 = (
            loi_2025["cleaned_title"].fillna("")
            + " [SEP] "
            + loi_2025["cleaned_content"].fillna("")
        ).str.strip()

        emb_2024 = encoder.encode(texts_2024.tolist())  # articles loi 2024
        emb_2025 = encoder.encode(texts_2025.tolist())  # articles loi 2025

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(emb_file_2024, emb_2024)
        np.save(emb_file_2025, emb_2025)
        
        logger.info(f"  ✅ Embeddings : 2024 {emb_2024.shape}, 2025 {emb_2025.shape}")
        pbar.update(1)
        time.sleep(0.5)
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 4 : VISUALISATION DES EMBEDDINGS
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("📊 Visualisation des embeddings")
        logger.info("\n📊 ÉTAPE 4 - Visualisation et analyse des similarités")
        
        # Audit sémantique : similarité cosinus + export
        auditeur = AuditeurSemantique(emb_2024, emb_2025, loi_2024, loi_2025)
        glissement = auditeur.analyser_glissement()
        matches_df = auditeur.trouver_meilleurs_matches()
        _safe_to_excel(matches_df, REPORTS_DIR / "matches_articles.xlsx", index=False)

        sim_2024 = pd.DataFrame({
            "id": loi_2024["id"].values,
            "score_max_similarite": auditeur.sim_matrix.max(axis=1),
        })
        sim_2025 = pd.DataFrame({
            "id": loi_2025["id"].values,
            "score_max_similarite": auditeur.sim_matrix.max(axis=0),
        })
        # Fichiers par année (rétro-compatibilité)
        _safe_to_excel(sim_2024, REPORTS_DIR / "embeddings_similarities_2024.xlsx", index=False)
        _safe_to_excel(sim_2025, REPORTS_DIR / "embeddings_similarities_2025.xlsx", index=False)

        # Fichier agrégé pour le dashboard (baromètre de glissement sémantique)
        sim_2024["annee"] = 2024
        sim_2025["annee"] = 2025
        sim_all = pd.concat([sim_2024, sim_2025], ignore_index=True)
        _safe_to_excel(sim_all, REPORTS_DIR / "embeddings_similarities.xlsx", index=False)

        # Visualisations
        plots.plot_tsne(emb_2024, emb_2025, loi_2024, loi_2025)
        plots.plot_similarity_matrix(
            auditeur.sim_matrix,
            loi_2024["id"].tolist(), loi_2025["id"].tolist(),
            auditeur.sim_matrix.max(axis=1),
            glissement["moyenne_similarite"],
            seuil=AUDIT_PARAMS["seuil_changement"],
        )
        
        logger.info("  ✅ Visualisations et similarités générées")
        pbar.update(1)
        time.sleep(0.5)
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 5 : TOPIC MODELING (LDA)
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("📚 Topic Modeling (LDA)")
        logger.info("\n📚 ÉTAPE 5 - Topic Modeling (LDA)")
        
        lda_model = LDATopicModeler(
            num_topics=LDA_PARAMS["num_topics"],
            random_state=LDA_PARAMS["random_state"],
        )
        
        # Tokeniser les textes pour LDA
        logger.info("  🔄 Tokenisation des textes...")
        tokens_2024 = [prep.tokenize_for_lda(t) for t in loi_2024["cleaned_content"].fillna("")]
        tokens_2025 = [prep.tokenize_for_lda(t) for t in loi_2025["cleaned_content"].fillna("")]
        
        # 2024
        logger.info("  📊 Entraînement LDA 2024...")
        lda_model.fit(tokens_2024)
        topics_2024 = lda_model.get_topics_dataframe()
        dist_2024 = lda_model.get_doc_topic_matrix()
        loi_2024_topics = loi_2024.copy()
        loi_2024_topics["dominant_topic"] = dist_2024.argmax(axis=1)
        
        _safe_to_excel(topics_2024, REPORTS_DIR / "lda_topics_2024.xlsx", index=False)
        _safe_to_excel(loi_2024_topics, REPORTS_DIR / "articles_2024_avec_topics.xlsx", index=False)

        # 2025
        logger.info("  📊 Entraînement LDA 2025...")
        lda_model.fit(tokens_2025)
        topics_2025 = lda_model.get_topics_dataframe()
        dist_2025 = lda_model.get_doc_topic_matrix()
        loi_2025_topics = loi_2025.copy()
        loi_2025_topics["dominant_topic"] = dist_2025.argmax(axis=1)

        _safe_to_excel(topics_2025, REPORTS_DIR / "lda_topics_2025.xlsx", index=False)
        _safe_to_excel(loi_2025_topics, REPORTS_DIR / "articles_2025_avec_topics.xlsx", index=False)
        
        logger.info("  ✅ Topic Modeling terminé")
        pbar.update(1)
        time.sleep(0.5)
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 6 : CLUSTERING
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("🎯 Clustering des documents")
        logger.info("\n🎯 ÉTAPE 6 - Clustering des documents")
        
        clusterer = DocumentClusterer(
            n_clusters=KMEANS_PARAMS["n_clusters"],
            kmeans_seed=KMEANS_PARAMS["random_state"],
        )
        
        # 2024
        logger.info("  📊 Clustering 2024...")
        results_2024 = clusterer.fit(emb_2024)
        loi_2024_clusters = loi_2024.copy()
        loi_2024_clusters["cluster_kmeans"] = results_2024.kmeans_labels
        loi_2024_clusters["cluster_hdbscan"] = results_2024.hdbscan_labels
        _safe_to_excel(loi_2024_clusters, REPORTS_DIR / "articles_2024_avec_clusters.xlsx", index=False)

        # 2025
        logger.info("  📊 Clustering 2025...")
        results_2025 = clusterer.fit(emb_2025)
        loi_2025_clusters = loi_2025.copy()
        loi_2025_clusters["cluster_kmeans"] = results_2025.kmeans_labels
        loi_2025_clusters["cluster_hdbscan"] = results_2025.hdbscan_labels
        _safe_to_excel(loi_2025_clusters, REPORTS_DIR / "articles_2025_avec_clusters.xlsx", index=False)
        
        logger.info("  ✅ Clustering terminé")
        pbar.update(1)
        time.sleep(0.5)
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 7 : CLASSIFICATION ZERO-SHOT (OBJECTIFS BUDGÉTAIRES UNIQUEMENT)
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("🏷️ Classification objectifs budgétaires")
        logger.info("\n🏷️ ÉTAPE 7 - Classification Zero-Shot (objectifs budgétaires)")
        
        # Classification Zero-Shot des OBJECTIFS BUDGÉTAIRES (ae_pb) par pilier SND30
        import json as _json
        ae_2024_path = RAW_DIR / "ae_pb2024.json"
        ae_2025_path = RAW_DIR / "ae_pb2025.json"

        classifier = ZeroShotClassifier()

        for ae_path, annee in [(ae_2024_path, 2024), (ae_2025_path, 2025)]:
            if not ae_path.exists():
                logger.warning(f"  ⚠️  {ae_path.name} introuvable — classification ignorée")
                continue

            ae_data = _json.load(open(ae_path, encoding="utf-8"))
            df_ae = pd.DataFrame(ae_data)

            # Extraire les objectifs (colonne liste → une ligne par objectif)
            if "objectifs" in df_ae.columns:
                rows = []
                for _, row in df_ae.iterrows():
                    objectifs = row["objectifs"] if isinstance(row["objectifs"], list) else [row["objectifs"]]
                    for obj in objectifs:
                        entry = row.drop("objectifs").to_dict()
                        entry["objectif"] = str(obj)
                        rows.append(entry)
                df_objectifs = pd.DataFrame(rows)
            else:
                logger.warning(f"  ⚠️  Colonne 'objectifs' absente dans {ae_path.name}")
                continue

            logger.info(f"  🔍 Classification Zero-Shot {annee} — {len(df_objectifs)} objectifs...")
            df_classified = classifier.classify_dataframe(df_objectifs, text_col="objectif")
            _safe_to_excel(
                df_classified,
                REPORTS_DIR / f"objectifs_classifications_snd30_{annee}.xlsx",
                index=False,
            )
            logger.info(f"  ✅ objectifs_classifications_snd30_{annee}.xlsx exporté")
        
        logger.info("  ✅ Classification des objectifs terminée")
        pbar.update(1)
        time.sleep(0.5)
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 8 : TESTS STATISTIQUES
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("📈 Tests statistiques")
        logger.info("\n📈 ÉTAPE 8 - Tests statistiques")
        
        analyzer = StatisticalAnalyzer()

        # Chi² sur les topics dominants
        chi2_dict = analyzer.test_chi2_piliers(
            loi_2024_topics, loi_2025_topics, col="dominant_topic"
        )
        chi2_results = pd.DataFrame([{
            "chi2":        chi2_dict["chi2"],
            "p_value":     chi2_dict["p_value"],
            "dof":         chi2_dict["dof"],
            "significatif": chi2_dict["significatif"],
        }])
        _safe_to_excel(chi2_results, REPORTS_DIR / "chi2_piliers.xlsx", index=False)

        # Mann-Whitney U (distributions de probabilités par topic)
        mw_results = analyzer.test_mannwhitney_topics(dist_2024, dist_2025)
        _safe_to_excel(mw_results, REPORTS_DIR / "test_mannwhitney.xlsx", index=False)
        
        logger.info("  ✅ Tests statistiques terminés")
        pbar.update(1)
        time.sleep(0.5)
        
        # ═════════════════════════════════════════════════════════════════
        # ÉTAPE 9 : GÉNÉRATION DES INDICATEURS DASHBOARD
        # ═════════════════════════════════════════════════════════════════
        pbar.set_description("📊 Génération indicateurs dashboard")
        logger.info("\n📊 ÉTAPE 9 - Génération des indicateurs pour le dashboard")
        
        # Générer analyse_budgetaire.xlsx avec tous les onglets nécessaires au dashboard
        import json as _json
        ae_2024_path = RAW_DIR / "ae_pb2024.json"
        ae_2025_path = RAW_DIR / "ae_pb2025.json"

        if ae_2024_path.exists() and ae_2025_path.exists():
            df_ae_2024 = pd.DataFrame(_json.load(open(ae_2024_path, encoding="utf-8")))
            df_ae_2025 = pd.DataFrame(_json.load(open(ae_2025_path, encoding="utf-8")))

            # Agréger par pilier depuis les résultats de classification zero-shot
            classif_2024_path = REPORTS_DIR / "objectifs_classifications_snd30_2024.xlsx"
            classif_2025_path = REPORTS_DIR / "objectifs_classifications_snd30_2025.xlsx"

            def _aggregate_by_pilier(classif_path: Path) -> pd.DataFrame:
                """Agrège AE, CP et nb_lignes par pilier dominant, avec pourcentages."""
                _empty = pd.DataFrame(
                    columns=["pilier", "ae", "cp", "nb_lignes", "pct_ae", "pct_cp"]
                )
                if not classif_path.exists():
                    return _empty
                df_c = pd.read_excel(classif_path)
                if "pilier_dominant" not in df_c.columns:
                    return _empty

                grp = df_c.groupby("pilier_dominant")
                result = pd.DataFrame({"nb_lignes": grp.size()}).reset_index()
                result = result.rename(columns={"pilier_dominant": "pilier"})

                for col in ("ae", "cp"):
                    if col in df_c.columns:
                        result[col] = grp[col].sum().values
                    else:
                        result[col] = 0

                total_ae = result["ae"].sum() or 1
                total_cp = result["cp"].sum() or 1
                result["pct_ae"] = (result["ae"] / total_ae * 100).round(2)
                result["pct_cp"] = (result["cp"] / total_cp * 100).round(2)
                return result

            piliers_2024 = _aggregate_by_pilier(classif_2024_path)
            piliers_2025 = _aggregate_by_pilier(classif_2025_path)

            # Tableau comparatif par pilier avec colonne evolution
            if not piliers_2024.empty and not piliers_2025.empty:
                p24 = piliers_2024.set_index("pilier")[["ae", "cp"]].rename(
                    columns={"ae": "ae_2024", "cp": "cp_2024"}
                )
                p25 = piliers_2025.set_index("pilier")[["ae", "cp"]].rename(
                    columns={"ae": "ae_2025", "cp": "cp_2025"}
                )
                comparison = p24.join(p25, how="outer").fillna(0)
                comparison["evolution"] = (
                    ((comparison["ae_2025"] - comparison["ae_2024"])
                     / comparison["ae_2024"].replace(0, np.nan) * 100)
                    .fillna(0).round(2)
                )
            else:
                comparison = pd.DataFrame()

            budget_path = REPORTS_DIR / "analyse_budgetaire.xlsx"
            lock_file   = budget_path.parent / f"~${budget_path.name}"
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except PermissionError:
                    logger.warning(
                        "  ⚠️  analyse_budgetaire.xlsx est ouvert dans Excel. "
                        "Fermez-le puis relancez l'étape 9."
                    )
            try:
                with pd.ExcelWriter(budget_path) as writer:
                    df_ae_2024.to_excel(writer, sheet_name="Budget_2024", index=False)
                    df_ae_2025.to_excel(writer, sheet_name="Budget_2025", index=False)
                    piliers_2024.to_excel(writer, sheet_name="Piliers_2024", index=False)
                    piliers_2025.to_excel(writer, sheet_name="Piliers_2025", index=False)
                    if not comparison.empty:
                        comparison.to_excel(writer, sheet_name="Comparaison_2024_2025")
                logger.info("  ✅ analyse_budgetaire.xlsx généré avec tous les onglets")
            except PermissionError:
                from datetime import datetime
                alt = budget_path.with_stem(f"analyse_budgetaire_{datetime.now().strftime('%H%M%S')}")
                with pd.ExcelWriter(alt) as writer:
                    df_ae_2024.to_excel(writer, sheet_name="Budget_2024", index=False)
                    df_ae_2025.to_excel(writer, sheet_name="Budget_2025", index=False)
                    piliers_2024.to_excel(writer, sheet_name="Piliers_2024", index=False)
                    piliers_2025.to_excel(writer, sheet_name="Piliers_2025", index=False)
                    if not comparison.empty:
                        comparison.to_excel(writer, sheet_name="Comparaison_2024_2025")
                logger.warning(
                    f"  ⚠️  analyse_budgetaire.xlsx est verrouillé → sauvegardé sous {alt.name}\n"
                    "       Fermez Excel et relancez pour écraser l'original."
                )
        else:
            logger.warning("  ⚠️ Fichiers ae_pb2024.json / ae_pb2025.json introuvables")

        logger.info("  ✅ Tous les indicateurs sont prêts")
        pbar.update(1)
        time.sleep(0.5)
    
    # ═════════════════════════════════════════════════════════════════════
    # RÉSUMÉ FINAL
    # ═════════════════════════════════════════════════════════════════════
    print_header("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    
    print("\n📁 Fichiers générés dans outputs/reports/ :")
    print(" analyse_budgetaire.xlsx")
    print(" lda_topics_2024.xlsx, lda_topics_2025.xlsx")
    print(" articles_2024_avec_topics.xlsx, articles_2025_avec_topics.xlsx")
    print(" articles_2024_avec_clusters.xlsx, articles_2025_avec_clusters.xlsx")
    print(" embeddings_similarities_2024.xlsx, embeddings_similarities_2025.xlsx")
    print(" embeddings_similarities.xlsx (agrégé pour le dashboard)")
    print(" chi2_piliers.xlsx, test_mannwhitney.xlsx")
    
    print("\n🚀 Lancer le dashboard :")
    print("  poetry run python scripts/run_dash.py")
    print()


if __name__ == "__main__":
    main()

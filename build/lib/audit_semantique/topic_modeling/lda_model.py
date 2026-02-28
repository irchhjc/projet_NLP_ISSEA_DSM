"""
topic_modeling/lda_model.py
Modélisation thématique LDA sur les lois de finances (Gensim + pyLDAvis).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from loguru import logger

from audit_semantique.config import LDA_PARAMS, MODELS_DIR, REPORTS_DIR


class LDATopicModeler:
    """
    Entraîne et analyse des modèles LDA sur un corpus de textes.

    Parameters
    ----------
    num_topics : int
        Nombre de topics (défaut : 4, calé sur les piliers SND30).
    random_state : int
        Graine pour la reproductibilité.
    passes : int
        Nombre de passes d'entraînement LDA.
    no_below : int
        Seuil minimum de documents pour conserver un terme.
    no_above : float
        Proportion maximale de documents autorisée pour un terme.
    """

    def __init__(
        self,
        num_topics:   int   = LDA_PARAMS["num_topics"],
        random_state: int   = LDA_PARAMS["random_state"],
        passes:       int   = LDA_PARAMS["passes"],
        no_below:     int   = LDA_PARAMS["no_below"],
        no_above:     float = LDA_PARAMS["no_above"],
    ) -> None:
        self.num_topics   = num_topics
        self.random_state = random_state
        self.passes       = passes
        self.no_below     = no_below
        self.no_above     = no_above

        self.dictionary: Optional[Dictionary] = None
        self.corpus:     Optional[list]       = None
        self.model:      Optional[LdaModel]   = None

    # ── Préparation du corpus ─────────────────────────────────────────────────

    def build_corpus(self, tokenized_docs: List[List[str]]) -> None:
        """
        Construit le dictionnaire et le corpus BOW à partir de tokens.

        Parameters
        ----------
        tokenized_docs : List[List[str]]
            Liste de documents tokenisés.
        """
        logger.info("📚 Construction du dictionnaire LDA...")
        self.dictionary = Dictionary(tokenized_docs)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        logger.info(
            f"  → {len(self.dictionary)} termes uniques | "
            f"{len(self.corpus)} documents"
        )

    # ── Entraînement ─────────────────────────────────────────────────────────

    def fit(self, tokenized_docs: List[List[str]]) -> "LDATopicModeler":
        """Construit le corpus et entraîne le modèle LDA."""
        self.build_corpus(tokenized_docs)
        logger.info(
            f"🔧 Entraînement LDA ({self.num_topics} topics, {self.passes} passes)..."
        )
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            passes=self.passes,
        )
        logger.info("✅ Modèle LDA entraîné.")
        return self

    # ── Distribution par document ─────────────────────────────────────────────

    def get_doc_topic_matrix(self, tokenized_docs: Optional[List[List[str]]] = None) -> np.ndarray:
        """
        Retourne la matrice ``(n_docs, num_topics)`` des probabilités.

        Parameters
        ----------
        tokenized_docs : List[List[str]] | None
            Documents tokenisés. Si None, utilise le corpus entraîné.

        Returns
        -------
        np.ndarray
        """
        if self.model is None:
            raise RuntimeError("Modèle non entraîné. Appelez `.fit()` ou `.load()`.")
        
        # Si des documents sont fournis, créer un nouveau corpus
        if tokenized_docs is not None:
            if self.dictionary is None:
                raise RuntimeError("Dictionnaire non disponible.")
            corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        elif self.corpus is not None:
            corpus = self.corpus
        else:
            raise RuntimeError(
                "Aucun corpus disponible. Passez tokenized_docs ou appelez `.fit()` d'abord."
            )

        dist = []
        for bow in corpus:
            topic_probs = self.model.get_document_topics(bow, minimum_probability=0)
            vec = [prob for _, prob in sorted(topic_probs, key=lambda x: x[0])]
            dist.append(vec)
        return np.array(dist)

    def get_topics_dataframe(self, num_words: int = 10) -> pd.DataFrame:
        """Retourne un DataFrame résumant les top mots par topic."""
        if self.model is None:
            raise RuntimeError("Modèle non entraîné.")
        rows = []
        for t in range(self.num_topics):
            terms = self.model.show_topic(t, topn=num_words)
            for word, prob in terms:
                rows.append({"topic": t, "word": word, "probability": prob})
        return pd.DataFrame(rows)

    # ── Persistance ──────────────────────────────────────────────────────────

    def save(self, name: str) -> None:
        """Sauvegarde le modèle, le dictionnaire et le corpus dans ``outputs/models/``."""
        if self.model is None:
            raise RuntimeError("Aucun modèle à sauvegarder.")
        base = MODELS_DIR / name
        self.model.save(str(base) + ".lda")
        self.dictionary.save(str(base) + ".dict")
        
        # Sauvegarder aussi le corpus pour pouvoir utiliser get_doc_topic_matrix
        if self.corpus is not None:
            import pickle
            with open(str(base) + ".corpus", "wb") as f:
                pickle.dump(self.corpus, f)
        
        logger.info(f"💾 Modèle LDA sauvegardé → {base}.lda")

    @classmethod
    def load(cls, name: str) -> "LDATopicModeler":
        """Charge un modèle, son dictionnaire et son corpus précédemment sauvegardés."""
        base = MODELS_DIR / name
        obj = cls()
        obj.model = LdaModel.load(str(base) + ".lda")
        obj.dictionary = Dictionary.load(str(base) + ".dict")
        
        # Charger aussi le corpus si disponible
        corpus_path = str(base) + ".corpus"
        if Path(corpus_path).exists():
            import pickle
            with open(corpus_path, "rb") as f:
                obj.corpus = pickle.load(f)
            logger.info(f"📂 Modèle LDA + corpus chargé depuis {base}.lda")
        else:
            logger.warning(f"⚠️  Corpus non trouvé. Utilisez get_doc_topic_matrix(tokenized_docs=...)")
            logger.info(f"📂 Modèle LDA chargé depuis {base}.lda")
        
        return obj

    # ── Export pyLDAvis ───────────────────────────────────────────────────────

    def export_ldavis(self, name: str) -> str:
        """
        Génère une visualisation interactive pyLDAvis et la sauvegarde en HTML.

        Returns
        -------
        str : chemin vers le fichier HTML généré.
        """
        if self.model is None or self.corpus is None or self.dictionary is None:
            raise RuntimeError("Modèle non entraîné ou corpus manquant.")

        try:
            import pyLDAvis
            import pyLDAvis.gensim_models as gensimvis
        except ImportError as e:
            raise ImportError("Installez pyldavis : poetry add pyldavis") from e

        out_path = str(REPORTS_DIR / f"lda_{name}.html")
        logger.info(f"🎨 Génération pyLDAvis → {out_path}...")
        vis = gensimvis.prepare(self.model, self.corpus, self.dictionary)
        pyLDAvis.save_html(vis, out_path)
        logger.info("✅ Visualisation LDA sauvegardée.")
        return out_path

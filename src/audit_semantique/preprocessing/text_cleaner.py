"""
preprocessing/text_cleaner.py
Nettoyage et normalisation des textes des lois de finances.
"""
import re
import unicodedata
from typing import Optional

from loguru import logger


# ─── Patterns de nettoyage ────────────────────────────────────────────────────
_URL_RE     = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"#\w+", flags=re.UNICODE)
_NUMBER_RE  = re.compile(r"\b\d+[\d.,/]*\b")
_PUNCT_RE   = re.compile(
    r"[\u2000-\u206F\u2E00-\u2E7F\'!\"#$%&()*+,\-./:;<=>?@\\^_`{|}~]",
    flags=re.UNICODE,
)

# ─── Stopwords français enrichis ─────────────────────────────────────────────
_STOPWORDS_FR: set[str] = set(
    """a ai avons avez ont suis es est sommes êtes sont je tu il elle on nous vous
    ils elles me te se y en du de des un une le la les au aux ce cet cette ces
    mon ton son ma ta sa mes tes ses notre votre leur nos vos leurs d l c j n qu
    ne pas plus pour et ou mais donc or ni car comme avec sans sous sur dans par
    vers entre chez que qui quoi dont où quand comment pourquoi très aussi encore
    bien mal trop moins même si très tout tous toutes puis parce afin alors fcfa num
    sav promo commande livraison article www http ... """.split()
)

_STOPWORDS_FR.update(
    {
        "premier", "deuxieme", "troisieme", "quatrieme", "cinquieme",
        "sixieme", "septieme", "huitieme", "neuvieme", "dixieme",
        "onzieme", "douzieme", "treizieme", "quatorzieme", "quinzieme",
        "seizieme", "dix-septieme", "dix-huitieme", "dix-neuvieme",
        "vingtieme", "un", "deux", "trois", "quatre", "cinq", "six",
        "sept", "huit", "neuf", "vingt", "cent", "mille",
        "millions", "milliard", "milliards", "million",
    }
)
# Lettres singletons
_STOPWORDS_FR.update({chr(i) for i in range(ord("a"), ord("z") + 1)})


class TextPreprocessor:
    """
    Préprocesseur de textes pour les lois de finances camerounaises.

    Parameters
    ----------
    lower : bool
        Mettre le texte en minuscules (défaut : True).
    rm_accents : bool
        Supprimer les accents (défaut : True).
    extra_stopwords : set[str] | None
        Stopwords additionnels à fusionner avec le set par défaut.
    """

    def __init__(
        self,
        lower: bool = True,
        rm_accents: bool = True,
        extra_stopwords: Optional[set] = None,
    ) -> None:
        self.lower = lower
        self.rm_accents = rm_accents
        self.stopwords = _STOPWORDS_FR.copy()
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)

    # ── Méthodes internes ────────────────────────────────────────────────────

    @staticmethod
    def _strip_accents(text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def _normalize(self, text: str) -> str:
        if self.lower:
            text = text.lower()
        text = _URL_RE.sub("<URL>", text)
        text = _MENTION_RE.sub("<MENTION>", text)
        text = _HASHTAG_RE.sub("<HASHTAG>", text)
        text = _NUMBER_RE.sub("<NUM>", text)
        text = _PUNCT_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if self.rm_accents:
            text = self._strip_accents(text)
        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+(?:'\w+)?", text.lower())

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self.stopwords]

    # ── API publique ─────────────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        """Nettoie un texte et retourne une chaîne prête pour les embeddings."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = self._normalize(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        return " ".join(tokens)

    def preprocess_series(self, series) -> "pd.Series":
        """Applique le préprocessing à une Series pandas."""
        import pandas as pd
        logger.info(f"Préprocessing de {len(series)} textes...")
        cleaned = series.fillna("").apply(self.preprocess)
        n_empty = (cleaned == "").sum()
        if n_empty:
            logger.warning(f"{n_empty} textes vides après nettoyage.")
        logger.info("✅ Préprocessing terminé.")
        return cleaned

    def tokenize_for_lda(self, text: str, min_len: int = 3) -> list[str]:
        """
        Tokenisation allégée pour LDA Gensim
        (lettres uniquement, longueur minimale, stopwords retirés).
        """
        text = text.lower()
        text = re.sub(r"[^a-zàâçéèêëîïôùûüÿñæœ\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        return [
            t for t in tokens
            if len(t) >= min_len and t not in self.stopwords
        ]

"""
nlp_utils.py
------------
All the NLP steps from the lab notebook, packaged as plain functions:

    tokenization, sentence splitting, language detection, stemming,
    lemmatization, POS tagging, named entities, n-grams

Both train_model.py and app.py import from here, so text is cleaned
in exactly the same way during training and during prediction.
"""

import re
from collections import Counter
from pathlib import Path

import nltk

# --------------------------------------------------------------------------
# NLTK data: bundled in ./nltk_data, downloaded on demand if anything is missing
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
NLTK_DIR = BASE_DIR / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DIR))

_REQUIRED = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab": "chunkers/maxent_ne_chunker_tab",
    "words": "corpora/words",
}


def ensure_nltk_data():
    """Make sure every NLTK resource exists; download the missing ones."""
    for package, path in _REQUIRED.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[nltk] downloading {package} ...")
            nltk.download(package, download_dir=str(NLTK_DIR), quiet=True)


ensure_nltk_data()

from nltk.corpus import stopwords  # noqa: E402  (needs data path set first)
from nltk.stem import PorterStemmer, WordNetLemmatizer  # noqa: E402
from nltk.tokenize import sent_tokenize, word_tokenize  # noqa: E402
from nltk.util import ngrams  # noqa: E402

try:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0  # make language detection deterministic
    _HAS_LANGDETECT = True
except ImportError:  # pragma: no cover
    _HAS_LANGDETECT = False

STOP_WORDS = set(stopwords.words("english"))
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()

MAX_CHARS = 5000  # reviews longer than this are truncated for the demo UI

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "es": "Spanish",
    "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "ar": "Arabic", "ja": "Japanese",
    "ko": "Korean", "zh-cn": "Chinese", "tr": "Turkish", "bn": "Bengali",
    "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}

POS_NAMES = {
    "NN": "noun", "NNS": "noun (plural)", "NNP": "proper noun", "NNPS": "proper noun (plural)",
    "VB": "verb", "VBD": "verb (past)", "VBG": "verb (-ing)", "VBN": "verb (past participle)",
    "VBP": "verb (present)", "VBZ": "verb (3rd person)",
    "JJ": "adjective", "JJR": "adjective (comparative)", "JJS": "adjective (superlative)",
    "RB": "adverb", "RBR": "adverb (comparative)", "RBS": "adverb (superlative)",
    "PRP": "pronoun", "PRP$": "possessive pronoun", "DT": "determiner", "IN": "preposition",
    "CC": "conjunction", "CD": "number", "MD": "modal verb", "TO": "to", "WDT": "wh-determiner",
    "WP": "wh-pronoun", "WRB": "wh-adverb", "EX": "existential there", "UH": "interjection",
    "POS": "possessive 's", "RP": "particle", "PDT": "predeterminer", "FW": "foreign word",
}


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t\r\f\v]+")


def clean_text(text) -> str:
    """Remove the HTML line-breaks that the IMDB dataset contains and tidy whitespace."""
    text = _BR.sub(" ", str(text))
    text = _TAGS.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------
# Individual analysis steps
# --------------------------------------------------------------------------
def detect_language(text: str) -> dict:
    if not _HAS_LANGDETECT:
        return {"code": "en", "name": "English", "confidence": None}
    try:
        best = detect_langs(text)[0]
        return {
            "code": best.lang,
            "name": LANGUAGE_NAMES.get(best.lang, best.lang),
            "confidence": round(best.prob, 3),
        }
    except Exception:
        return {"code": "unknown", "name": "Unknown", "confidence": None}


def _wordnet_pos(tag: str) -> str:
    return {"J": "a", "V": "v", "R": "r"}.get(tag[:1], "n")


def text_statistics(text: str, words: list, sentences: list) -> dict:
    alpha = [w.lower() for w in words if w.isalpha()]
    unique = set(alpha)
    n_words = len(alpha)
    return {
        "characters": len(text),
        "words": n_words,
        "sentences": len(sentences),
        "unique_words": len(unique),
        "avg_word_length": round(sum(len(w) for w in alpha) / n_words, 2) if n_words else 0,
        "avg_sentence_length": round(n_words / len(sentences), 1) if sentences else 0,
        "lexical_diversity": round(len(unique) / n_words, 3) if n_words else 0,
        "stopword_share": round(sum(w in STOP_WORDS for w in alpha) / n_words, 3) if n_words else 0,
        "reading_minutes": round(n_words / 230, 1),
    }


def stem_and_lemmatize(tagged: list, limit: int = 30) -> list:
    """Compare Porter stems with POS-aware WordNet lemmas (notebook experiment 2)."""
    rows, seen = [], set()
    for word, tag in tagged:
        lower = word.lower()
        if not word.isalpha() or lower in STOP_WORDS or lower in seen:
            continue
        seen.add(lower)
        rows.append({
            "word": word,
            "stem": _stemmer.stem(lower),
            "lemma": _lemmatizer.lemmatize(lower, _wordnet_pos(tag)),
        })
        if len(rows) >= limit:
            break
    return rows


def pos_summary(tagged: list) -> tuple:
    """Return (first 40 tagged tokens, distribution of coarse tags)."""
    sample = [
        {"word": w, "tag": t, "meaning": POS_NAMES.get(t, "")}
        for w, t in tagged[:40]
    ]
    groups = Counter()
    for _, tag in tagged:
        if tag.startswith("NN"):
            groups["Nouns"] += 1
        elif tag.startswith("VB") or tag == "MD":
            groups["Verbs"] += 1
        elif tag.startswith("JJ"):
            groups["Adjectives"] += 1
        elif tag.startswith("RB"):
            groups["Adverbs"] += 1
        elif tag.startswith("PRP") or tag.startswith("WP"):
            groups["Pronouns"] += 1
        elif tag in {"IN", "DT", "CC", "TO", "PDT", "WDT", "EX", "RP"}:
            groups["Function words"] += 1
    distribution = [{"label": k, "count": v} for k, v in groups.most_common()]
    return sample, distribution


def named_entities(tagged: list) -> list:
    """NLTK's NE chunker (notebook experiment 4)."""
    tree = nltk.ne_chunk(tagged)
    found = Counter()
    for subtree in tree:
        if hasattr(subtree, "label"):
            name = " ".join(leaf[0] for leaf in subtree.leaves())
            found[(name, subtree.label())] += 1
    return [
        {"text": name, "label": label, "count": count}
        for (name, label), count in found.most_common(25)
    ]


def top_ngrams(words: list, n: int, k: int = 8) -> list:
    """Most common n-grams, ignoring ones made only of stop-words (notebook experiment 7)."""
    grams = ngrams(words, n)
    counts = Counter(g for g in grams if not all(w in STOP_WORDS for w in g))
    return [{"gram": " ".join(g), "count": c} for g, c in counts.most_common(k)]


# --------------------------------------------------------------------------
# One call that runs everything
# --------------------------------------------------------------------------
def analyze_text(text: str) -> dict:
    """Run the full NLP toolkit on one review and return JSON-friendly data."""
    text = clean_text(text)[:MAX_CHARS]
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words[:400])  # cap the slow steps for very long reviews
    alpha_lower = [w.lower() for w in words if w.isalpha()]
    pos_sample, pos_distribution = pos_summary(tagged)

    return {
        "text": text,
        "language": detect_language(text) if len(text) >= 3 else {"code": "unknown", "name": "Unknown", "confidence": None},
        "stats": text_statistics(text, words, sentences),
        "tokens": words[:60],
        "sentences": sentences[:6],
        "roots": stem_and_lemmatize(tagged),
        "pos": pos_sample,
        "pos_distribution": pos_distribution,
        "entities": named_entities(tagged),
        "ngrams": {
            "unigrams": top_ngrams([w for w in alpha_lower if w not in STOP_WORDS], 1),
            "bigrams": top_ngrams(alpha_lower, 2),
            "trigrams": top_ngrams(alpha_lower, 3),
        },
    }

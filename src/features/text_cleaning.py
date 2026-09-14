"""Text normalisation and lightweight feature extraction.

Deliberately conservative. The dataset is short English social-media messages
that are already lowercase, so aggressive cleaning destroys more signal than it
removes noise.

What this module does NOT do, on purpose:

* No stopword removal. "not", "no", "but" carry emotion. Data Modeling will decide that
* No stemming or lemmatising, for the same reason
* Nothing irreversible. The original text is always kept alongside the cleaned
  version so the transformer track can use raw input.
"""

from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
HTML_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|#\d+);")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s']")
ELONGATION_RE = re.compile(r"(.)\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")

NEGATIONS = {
    "not", "no", "never", "none", "nothing", "nobody", "nowhere",
    "cant", "cannot", "wont", "dont", "didnt", "doesnt", "isnt",
    "arent", "wasnt", "werent", "shouldnt", "wouldnt", "couldnt", "aint",
}


def clean_text(text: str) -> str:
    """Normalise one message. Returns lowercase text with punctuation removed.

    >>> clean_text("I'm SOOOO happy!!! @friend http://x.co #blessed")
    "i'm soo happy blessed"
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    text = HTML_ENTITY_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)       # keep the word, drop the '#'

    text = NON_TEXT_RE.sub(" ", text)
    text = ELONGATION_RE.sub(r"\1\1", text)  # "sooooo" -> "soo"
    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def word_count(text: str) -> int:
    return len(str(text).split())


def char_count(text: str) -> int:
    return len(str(text))


def has_negation(text: str) -> int:
    """1 if the message contains a negation token, else 0.

    Negation flips sentiment and is a common source of classifier errors, so it
    is worth having as an explicit column for error analysis.
    """
    return int(bool(NEGATIONS & set(str(text).split())))


def is_usable(text: str, min_words: int = 1) -> bool:
    """False for rows that are empty after cleaning."""
    return word_count(text) >= min_words

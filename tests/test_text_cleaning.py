"""Unit tests for text cleaning.

Run from the project root::

    pytest -q
"""

from src.features.text_cleaning import (
    char_count,
    clean_text,
    has_negation,
    is_usable,
    word_count,
)


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("I Feel GREAT") == "i feel great"

    def test_strips_urls(self):
        assert "http" not in clean_text("look https://example.com/x now")

    def test_strips_mentions_but_keeps_hashtag_words(self):
        out = clean_text("@someone i feel #blessed")
        assert "someone" not in out
        assert "blessed" in out

    def test_collapses_elongation(self):
        assert clean_text("sooooo happy") == "soo happy"

    def test_removes_punctuation_but_keeps_apostrophes(self):
        assert clean_text("i'm happy!!! really?") == "i'm happy really"

    def test_collapses_whitespace(self):
        assert clean_text("  i   feel    sad  ") == "i feel sad"

    def test_decodes_html_entities(self):
        assert "amp" not in clean_text("me &amp; you")

    def test_handles_empty_and_none(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_is_idempotent(self):
        once = clean_text("I'm SOOO hapyy!! @x")
        assert clean_text(once) == once


class TestFeatures:
    def test_word_count(self):
        assert word_count("i feel sad") == 3

    def test_char_count(self):
        assert char_count("abc") == 3

    def test_detects_negation(self):
        assert has_negation("i do not feel happy") == 1
        assert has_negation("i dont feel happy") == 1

    def test_no_false_negation(self):
        assert has_negation("i feel happy") == 0

    def test_is_usable(self):
        assert is_usable("word")
        assert not is_usable("")


class TestSchemaContract:
    def test_label_order_is_official(self):
        from src.data.schema import LABEL_NAMES

        assert LABEL_NAMES == [
            "sadness", "joy", "love", "anger", "fear", "surprise",
        ]

    def test_label_maps_are_consistent(self):
        from src.data.schema import ID2LABEL, LABEL2ID

        for i, name in ID2LABEL.items():
            assert LABEL2ID[name] == i

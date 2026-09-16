"""Tests for the project's shared data contract."""

from src.data.schema import (
    EXPECTED_ROWS,
    ID2LABEL,
    LABEL2ID,
    LABEL_NAMES,
    N_CLASSES,
    SPLITS,
)


def test_label_contract_has_six_emotions():
    assert N_CLASSES == 6
    assert LABEL_NAMES == [
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    ]


def test_label_mappings_are_reversible():
    for name, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == name


def test_expected_splits_are_defined():
    assert SPLITS == ["train", "validation", "test"]
    assert EXPECTED_ROWS == {
        "train": 16000,
        "validation": 2000,
        "test": 2000,
    }


def test_raw_schema_is_stable():
    from src.data.schema import RAW_COLUMNS

    assert RAW_COLUMNS == ["text", "label"]

"""Dataset contract: label encoding and expected shape.

This is the single source of truth for the label mapping. Members 2, 3 and 4
must import from here rather than re-typing the list, otherwise a silent
re-ordering will make every confusion matrix wrong.

The order below is the official ClassLabel order of `dair-ai/emotion` and must
not be changed.
"""

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

N_CLASSES = len(LABEL_NAMES)

# Official row counts for the "split" configuration, used as a data-quality gate.
EXPECTED_ROWS = {"train": 16000, "validation": 2000, "test": 2000}

SPLITS = list(EXPECTED_ROWS)

RAW_COLUMNS = ["text", "label"]

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_split_hashes_and_counts():
    manifest = json.loads((ROOT / "data/splits/split_manifest.json").read_text(encoding="utf-8"))
    for split, info in manifest["splits"].items():
        path = ROOT / "data/splits" / info["file"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"]
        df = pd.read_csv(path)
        assert len(df) == info["num_samples"]
        assert set(df.columns) >= {"text", "label", "label_name"}


def test_test_class_distribution():
    df = pd.read_csv(ROOT / "data/splits/test.csv")
    counts = df["label_name"].value_counts().to_dict()
    assert counts == {"CLEAN": 5548, "HATE": 688, "OFFENSIVE": 444}
    assert df["text"].nunique() == 6571
    assert len(df) == 6680


def test_lexicon_entries():
    original = pd.read_csv(ROOT / "data/processed/lexicon/verified_lexicon.csv")
    leakfree = pd.read_csv(ROOT / "data/processed/lexicon/leakage_free/verified_lexicon.csv")
    assert len(original) == 668
    assert len(leakfree) == 671
    assert set(leakfree["term"]) - set(original["term"]) == {"bá dơ", "mà đéo", "quái thai"}
    changed = original.set_index("term")["lexical_labels"].to_dict()
    new = leakfree.set_index("term")["lexical_labels"].to_dict()
    assert changed["chịch"] == "OFFENSIVE"
    assert new["chịch"] == "HATE"

import json
from pathlib import Path

import pandas as pd

from app.features import MODEL_FEATURES
from ml.generate_data import SEED, generate_synthetic_bookings


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def test_synthetic_generation_is_reproducible() -> None:
    first = generate_synthetic_bookings(row_count=100, seed=SEED)
    second = generate_synthetic_bookings(row_count=100, seed=SEED)

    pd.testing.assert_frame_equal(first, second)


def test_model_features_exclude_time_and_labels() -> None:
    assert "event_day" not in MODEL_FEATURES
    assert all(not feature.endswith("_label") for feature in MODEL_FEATURES)
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))


def test_saved_metadata_matches_artifacts() -> None:
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["seed"] == SEED
    assert metadata["row_count"] == 15_000
    assert metadata["features"] == MODEL_FEATURES
    assert (ARTIFACTS / "payment_fraud_model.json").stat().st_size > 0
    assert (ARTIFACTS / "inventory_abuse_model.json").stat().st_size > 0


def test_reported_metrics_are_bounded_and_confusion_counts_match_test_size() -> None:
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    test_size = metadata["split_counts"]["test"]

    for metrics in metadata["test_metrics"].values():
        for key in ("precision", "recall", "pr_auc", "roc_auc", "false_positive_rate"):
            assert 0 <= metrics[key] <= 1
        assert sum(metrics["confusion_matrix"].values()) == test_size
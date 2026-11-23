import pandas as pd
from data.generate_synthetic import generate_dataset
from data.feature_schema import FEATURE_COLUMNS, CATEGORY_LABELS


def test_dataset_row_count():
    from data.generate_synthetic import _REAL_EXAMPLES
    df = generate_dataset(n_samples=100, seed=0)
    assert len(df) == 100 + len(_REAL_EXAMPLES)


def test_dataset_has_all_feature_columns():
    df = generate_dataset(n_samples=50, seed=0)
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_labels_present():
    df = generate_dataset(n_samples=50, seed=0)
    assert "label_importance" in df.columns
    assert "label_urgency" in df.columns
    assert "label_category" in df.columns


def test_importance_range():
    df = generate_dataset(n_samples=200, seed=1)
    assert df["label_importance"].between(1, 10).all()


def test_urgency_range():
    df = generate_dataset(n_samples=200, seed=1)
    assert df["label_urgency"].between(1, 10).all()


def test_category_labels_valid():
    df = generate_dataset(n_samples=200, seed=1)
    assert set(df["label_category"].unique()).issubset(set(CATEGORY_LABELS))


def test_cold_start_fraction():
    df = generate_dataset(n_samples=500, seed=2)
    cold = df["sender_open_rate"].isna().mean()
    assert 0.10 <= cold <= 0.20, f"Cold-start fraction {cold:.2f} out of 10-20% range"


def test_category_distribution_rough():
    df = generate_dataset(n_samples=1000, seed=3)
    msg_frac = (df["label_category"] == "message").mean()
    assert 0.20 <= msg_frac <= 0.40, f"Message fraction {msg_frac:.2f} out of 20-40%"


def test_sleep_hours_urgency_penalized():
    df = generate_dataset(n_samples=500, seed=4)
    sleep_promo = df[(df["is_sleep_hours"] == True) & (df["label_category"] == "promo")]
    if len(sleep_promo) >= 5:
        assert sleep_promo["label_urgency"].mean() <= 5.0


def test_tier1_importance_high():
    df = generate_dataset(n_samples=500, seed=5)
    tier1 = df[df["sender_open_rate"] >= 0.90]
    if len(tier1) >= 10:
        assert tier1["label_importance"].mean() >= 7.0


def test_cold_start_importance_conservative():
    df = generate_dataset(n_samples=500, seed=6)
    cold = df[df["sender_open_rate"].isna()]
    if len(cold) >= 10:
        mean = cold["label_importance"].mean()
        assert 3.5 <= mean <= 6.5, f"Cold-start mean importance {mean:.2f} not in 3.5-6.5"

"""Tests for PersonalizationLayer."""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from training.personalization import PersonalizationLayer, APP_CATEGORIES, _CLIP


# ── helpers ────────────────────────────────────────────────────────────────

def _default_kwargs(app_cat="social"):
    return dict(base_imp=5.0, base_urg=5.0, app_cat=app_cat,
                is_group=False, sender_open_rate=0.3, hour_of_day=14)


def _combined(imp, urg):
    return imp * 0.6 + urg * 0.4


# ── test 1: zero weights → identity ────────────────────────────────────────

def test_personalization_zero_weights_is_identity():
    layer = PersonalizationLayer()
    for cat in APP_CATEGORIES:
        for base_imp in (1.0, 5.0, 9.5):
            for base_urg in (1.0, 5.0, 9.5):
                adj_imp, adj_urg = layer.adjust(
                    base_imp, base_urg, cat, False, 0.3, 12)
                assert adj_imp == pytest.approx(base_imp, abs=1e-9)
                assert adj_urg == pytest.approx(base_urg, abs=1e-9)


# ── test 2: converges on dismissals ────────────────────────────────────────

def test_personalization_converges_on_dismissals():
    layer = PersonalizationLayer()
    kwargs = _default_kwargs("social")
    for _ in range(30):
        layer.update(**kwargs, outcome=-1.0)
    adj_imp, adj_urg = layer.adjust(**kwargs)
    combined = _combined(adj_imp, adj_urg)
    assert combined < 3.5, f"Expected combined < 3.5 after 30 dismissals, got {combined:.3f}"


# ── test 3: clip prevents override (security alert safety) ─────────────────

def test_personalization_clip_prevents_override():
    layer = PersonalizationLayer()
    # Simulate 100 dismiss signals on a high-scoring notification (e.g. security alert)
    for _ in range(100):
        layer.update(base_imp=9.0, base_urg=9.0, app_cat="social",
                     is_group=False, sender_open_rate=0.02, hour_of_day=22,
                     outcome=-1.0)
    adj_imp, adj_urg = layer.adjust(9.0, 9.0, "social", False, 0.02, 22)
    # Each delta is clipped to ±CLIP, so adj ≥ base - CLIP
    assert adj_imp >= 9.0 - _CLIP - 1e-9
    assert adj_urg >= 9.0 - _CLIP - 1e-9
    combined = _combined(adj_imp, adj_urg)
    # Combined still well above suppress threshold (3.5)
    assert combined > 5.0, f"Security alert score dropped too low: {combined:.3f}"


# ── test 4: save/load roundtrip ────────────────────────────────────────────

def test_personalization_save_load_roundtrip():
    layer = PersonalizationLayer()
    # Apply some updates to create non-zero weights
    for _ in range(10):
        layer.update(base_imp=4.0, base_urg=5.0, app_cat="shopping",
                     is_group=False, sender_open_rate=0.05, hour_of_day=10,
                     outcome=-1.0)
    for _ in range(5):
        layer.update(base_imp=7.0, base_urg=6.0, app_cat="messaging",
                     is_group=True, sender_open_rate=0.8, hour_of_day=9,
                     outcome=1.0)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)

    try:
        layer.save(tmp)
        layer2 = PersonalizationLayer()
        layer2.load(tmp)

        for cat in APP_CATEGORIES:
            np.testing.assert_array_almost_equal(layer.w_imp[cat], layer2.w_imp[cat])
            np.testing.assert_array_almost_equal(layer.w_urg[cat], layer2.w_urg[cat])
    finally:
        tmp.unlink(missing_ok=True)


# ── test 5: category isolation ─────────────────────────────────────────────

def test_personalization_category_isolation():
    """Dismissals in shopping must not affect messaging weights."""
    layer = PersonalizationLayer()
    for _ in range(20):
        layer.update(base_imp=3.5, base_urg=4.5, app_cat="shopping",
                     is_group=False, sender_open_rate=0.02, hour_of_day=14,
                     outcome=-1.0)
    # Messaging scores must be unchanged (zero weights still)
    adj_imp, adj_urg = layer.adjust(5.0, 5.0, "messaging", False, 0.5, 10)
    assert adj_imp == pytest.approx(5.0, abs=1e-9)
    assert adj_urg == pytest.approx(5.0, abs=1e-9)


# ── test 6: positive feedback boosts score ─────────────────────────────────

def test_personalization_converges_on_taps():
    layer = PersonalizationLayer()
    for _ in range(30):
        layer.update(base_imp=5.0, base_urg=5.0, app_cat="messaging",
                     is_group=False, sender_open_rate=0.9, hour_of_day=9,
                     outcome=1.0)
    adj_imp, adj_urg = layer.adjust(5.0, 5.0, "messaging", False, 0.9, 9)
    combined = _combined(adj_imp, adj_urg)
    # base=5.0, CLIP=2.0 → max adj=7.0 → expect combined > 6.5 after convergence
    assert combined > 6.5, f"Expected combined > 6.5 after 30 taps, got {combined:.3f}"


# ── test 7: NaN sender_open_rate handled ────────────────────────────────────

def test_personalization_nan_sender_open_rate():
    layer = PersonalizationLayer()
    # Should not raise; NaN replaced with 0.5 fallback
    adj_imp, adj_urg = layer.adjust(5.0, 5.0, "social", False, float("nan"), 12)
    assert not math.isnan(adj_imp)
    assert not math.isnan(adj_urg)

import numpy as np
import pandas as pd
from data.feature_engineering import BertTinyEncoder, build_feature_matrix
from data.feature_schema import FEATURE_COLUMNS


def test_encoder_output_shape():
    enc = BertTinyEncoder()
    result = enc.encode(["Hello world", "Are you okay?"])
    assert result.shape == (2, 128)


def test_encoder_deterministic():
    enc = BertTinyEncoder()
    a = enc.encode(["test"])
    b = enc.encode(["test"])
    np.testing.assert_array_equal(a, b)


def test_encoder_different_texts_differ():
    enc = BertTinyEncoder()
    a = enc.encode(["50% off sale promo"])
    b = enc.encode(["Are you okay? I'm worried"])
    assert not np.allclose(a, b, atol=0.01)


def _make_sample(title, text, app_category, sender_open_rate=0.95):
    return {
        "title": title,
        "text": text,
        "sender_open_rate": sender_open_rate,
        "sender_avg_response_time_sec": 45.0,
        "sender_interaction_frequency": 8.2,
        "sender_recency_days": 0.1,
        "sender_conversation_ratio": 0.85,
        "app_open_rate": 0.78,
        "app_daily_volume": 42,
        "hour_of_day": 19,
        "is_weekend": True,
        "is_sleep_hours": False,
        "minutes_since_last_from_sender": 120.0,
        "is_group_conversation": False,
        "notification_has_reply_action": True,
        "notification_importance_raw": 4,
        "is_messaging_style": True,
        "flag_foreground_service": False,
        "app_category": app_category,
    }


def test_build_feature_matrix_shape():
    enc = BertTinyEncoder()
    df = pd.DataFrame([_make_sample("Mom", "Are you coming home?", "messaging")])
    mat = build_feature_matrix(df, enc)
    assert mat.shape == (1, 151)


def test_feature_matrix_column_order():
    enc = BertTinyEncoder()
    df = pd.DataFrame([_make_sample("Deal", "50% off", "shopping", sender_open_rate=float("nan"))])
    mat = build_feature_matrix(df, enc)
    # app_cat_shopping is index 130 (128 embed + 2 = index 130)
    assert mat[0, 130] == 1.0
    assert mat[0, 128] == 0.0  # app_cat_messaging

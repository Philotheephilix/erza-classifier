from data.sender_resolution import resolve_sender_key
from data.feature_schema import FEATURE_COLUMNS, CATEGORY_LABELS


def test_whatsapp_uses_title():
    assert resolve_sender_key("com.whatsapp", "Mom") == "com.whatsapp:Mom"


def test_amazon_uses_package():
    assert resolve_sender_key("in.amazon.mShop.android.shopping", "Deal Alert") == "in.amazon.mShop.android.shopping"


def test_null_title_messaging_falls_back_to_package():
    assert resolve_sender_key("com.whatsapp", None) == "com.whatsapp"


def test_gmail_uses_title():
    assert resolve_sender_key("com.google.android.gm", "Priya Sharma") == "com.google.android.gm:Priya Sharma"


def test_discord_uses_title():
    assert resolve_sender_key("com.discord", "Freddy") == "com.discord:Freddy"


def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 151


def test_feature_columns_no_duplicates():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_category_labels_count():
    assert len(CATEGORY_LABELS) == 8
    assert set(CATEGORY_LABELS) == {"message", "email", "social", "promo", "transaction", "system", "delivery", "reminder"}

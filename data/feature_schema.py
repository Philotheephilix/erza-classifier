FEATURE_COLUMNS = (
    [f"embed_{i}" for i in range(128)]
    + [
        "app_cat_messaging",
        "app_cat_social",
        "app_cat_shopping",
        "app_cat_finance",
        "app_cat_productivity",
        "app_cat_media",
        "app_cat_system",
    ]
    + [
        "sender_open_rate",
        "sender_avg_response_time_sec",
        "sender_interaction_frequency",
        "sender_recency_days",
        "sender_conversation_ratio",
        "app_open_rate",
        "app_daily_volume",
        "hour_of_day",
        "is_weekend",
        "is_sleep_hours",
        "minutes_since_last_from_sender",
        "is_group_conversation",
        "notification_has_reply_action",
        "notification_importance_raw",
        "is_messaging_style",
        "flag_foreground_service",
    ]
)

CATEGORY_LABELS = [
    "message",
    "email",
    "social",
    "promo",
    "transaction",
    "system",
    "delivery",
    "reminder",
]

APP_CATEGORY_MAP = {
    # messaging
    "com.whatsapp": "messaging",
    "org.telegram.messenger": "messaging",
    "com.facebook.orca": "messaging",
    "com.instagram.android": "messaging",
    "com.discord": "messaging",
    "com.google.android.apps.messaging": "messaging",
    "org.thoughtcrime.securesms": "messaging",
    # social
    "com.instagram.android.feed": "social",
    "com.twitter.android": "social",
    "com.linkedin.android": "social",
    "com.reddit.frontpage": "social",
    "com.snapchat.android": "social",
    "com.zhiliaoapp.musically": "social",
    # shopping
    "in.amazon.mShop.android.shopping": "shopping",
    "app.zomato": "shopping",
    "in.swiggy.android": "shopping",
    "com.flipkart.android": "shopping",
    "com.myntra.android": "shopping",
    # finance
    "com.phonepe.app": "finance",
    "com.google.android.apps.nbu.paisa.user": "finance",
    "net.one97.paytm": "finance",
    "com.jupiter.app": "finance",
    "com.bankofbaroda.mconnect": "finance",
    "com.dreamplug.androidapp": "finance",
    # productivity
    "com.google.android.gm": "productivity",
    "com.microsoft.office.outlook": "productivity",
    "com.slack": "productivity",
    "com.google.android.apps.meetings": "productivity",
    "com.microsoft.teams": "productivity",
    "com.google.android.calendar": "productivity",
    "com.todoist": "productivity",
    # media
    "com.google.android.youtube": "media",
    "com.spotify.music": "media",
    "tv.twitch.android.app": "media",
    # system
    "com.android.systemui": "system",
    "com.google.android.gms": "system",
    "android": "system",
}

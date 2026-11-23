import numpy as np
import pandas as pd
from data.feature_engineering import BertTinyEncoder, build_feature_matrix
from training.evaluate import predict, BYPASS_THRESHOLD, SUPPRESS_THRESHOLD

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = BertTinyEncoder()
    return _encoder


def _make_row(title, text, app_category, sender_open_rate, sender_avg_response_time_sec,
              sender_interaction_frequency, sender_recency_days, sender_conversation_ratio,
              app_open_rate, app_daily_volume, hour_of_day, is_weekend, is_sleep_hours,
              minutes_since_last_from_sender, is_group_conversation,
              notification_has_reply_action, notification_importance_raw,
              is_messaging_style, flag_foreground_service):
    return {
        "title": title, "text": text, "app_category": app_category,
        "sender_open_rate": sender_open_rate,
        "sender_avg_response_time_sec": sender_avg_response_time_sec,
        "sender_interaction_frequency": sender_interaction_frequency,
        "sender_recency_days": sender_recency_days,
        "sender_conversation_ratio": sender_conversation_ratio,
        "app_open_rate": app_open_rate,
        "app_daily_volume": app_daily_volume,
        "hour_of_day": float(hour_of_day),
        "is_weekend": float(is_weekend),
        "is_sleep_hours": float(is_sleep_hours),
        "minutes_since_last_from_sender": minutes_since_last_from_sender,
        "is_group_conversation": float(is_group_conversation),
        "notification_has_reply_action": float(notification_has_reply_action),
        "notification_importance_raw": float(notification_importance_raw),
        "is_messaging_style": float(is_messaging_style),
        "flag_foreground_service": float(flag_foreground_service),
    }


def _tier1_urgent_texts():
    return [
        ("Mom", "Are you okay? Please call me right now"),
        ("Dad", "Emergency! Call immediately"),
        ("Partner", "Where are you? I'm worried"),
        ("Mom", "It's really important, please respond"),
        ("Best Friend", "Are you safe? There's been an accident nearby"),
    ]


def _tier5_promo_texts():
    return [
        ("Zomato", "50% off your next order! Use SAVE50"),
        ("Amazon", "Flash sale ends at midnight!"),
        ("Flipkart", "Big Billion Days deals you can't miss"),
        ("Swiggy", "Free delivery for the next 2 hours"),
        ("MakeMyTrip", "Special offer on flights this weekend"),
        ("Myntra", "End of season sale: flat 60% off on top brands"),
        ("Nykaa", "Buy 2 get 1 free on skincare — today only"),
        ("BookMyShow", "Early bird offers for upcoming movies"),
        ("Hotstar", "New show alert: streaming now"),
        ("BigBasket", "Fresh morning deals: 20% off on vegetables"),
    ]


def run_condition_tests() -> dict:
    enc = _get_encoder()
    results = {}

    # ── 1. Tier1 urgent: family/partner, high open_rate, urgent text → should bypass DND
    rows = []
    for title, text in _tier1_urgent_texts():
        rows.append(_make_row(
            title=title, text=text, app_category="messaging",
            sender_open_rate=0.95, sender_avg_response_time_sec=30.0,
            sender_interaction_frequency=10.0, sender_recency_days=0.1,
            sender_conversation_ratio=0.9, app_open_rate=0.8, app_daily_volume=20,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=60.0, is_group_conversation=False,
            notification_has_reply_action=True, notification_importance_raw=4,
            is_messaging_style=True, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    bypass_rate = float((combined >= BYPASS_THRESHOLD).mean())
    results["tier1_urgent_bypass_rate"] = bypass_rate

    # ── 2. Tier5 promo: low open_rate, promotional text → should suppress
    rows = []
    for title, text in _tier5_promo_texts():
        rows.append(_make_row(
            title=title, text=text, app_category="shopping",
            sender_open_rate=0.03, sender_avg_response_time_sec=86400.0,
            sender_interaction_frequency=5.0, sender_recency_days=0.5,
            sender_conversation_ratio=0.0, app_open_rate=0.05, app_daily_volume=8,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=float("nan"), is_group_conversation=False,
            notification_has_reply_action=False, notification_importance_raw=2,
            is_messaging_style=False, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    suppress_rate = float((combined <= SUPPRESS_THRESHOLD).mean())
    results["tier5_suppress_rate"] = suppress_rate

    # ── 3. Cold start WhatsApp DM — all NaN behavioral features → must NOT suppress
    rows = []
    cold_texts = [
        ("Mom", "Hey what's up"),
        ("Friend", "Are you free tonight?"),
        ("Partner", "Missing you"),
        ("Sibling", "Can you call when free?"),
        ("Close Friend", "Check this out!"),
    ]
    for title, text in cold_texts:
        rows.append(_make_row(
            title=title, text=text, app_category="messaging",
            sender_open_rate=float("nan"), sender_avg_response_time_sec=float("nan"),
            sender_interaction_frequency=float("nan"), sender_recency_days=float("nan"),
            sender_conversation_ratio=float("nan"), app_open_rate=float("nan"),
            app_daily_volume=float("nan"), hour_of_day=14, is_weekend=False,
            is_sleep_hours=False, minutes_since_last_from_sender=float("nan"),
            is_group_conversation=False, notification_has_reply_action=True,
            notification_importance_raw=4, is_messaging_style=True, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    suppress_rate = float((combined <= SUPPRESS_THRESHOLD).mean())
    results["cold_start_message_suppress_rate"] = suppress_rate

    # ── 4. Sleep hours promo → must never bypass DND
    rows = []
    sleep_promos = [
        ("Zomato", "Flash sale tonight only — 60% off!"),
        ("Amazon", "Night deals: hurry, limited stock"),
        ("Swiggy", "Midnight munchies? Order now!"),
    ]
    for title, text in sleep_promos:
        rows.append(_make_row(
            title=title, text=text, app_category="shopping",
            sender_open_rate=0.04, sender_avg_response_time_sec=86400.0,
            sender_interaction_frequency=5.0, sender_recency_days=0.2,
            sender_conversation_ratio=0.0, app_open_rate=0.04, app_daily_volume=6,
            hour_of_day=2, is_weekend=False, is_sleep_hours=True,
            minutes_since_last_from_sender=float("nan"), is_group_conversation=False,
            notification_has_reply_action=False, notification_importance_raw=2,
            is_messaging_style=False, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    bypass_rate = float((combined >= BYPASS_THRESHOLD).mean())
    results["sleep_promo_bypass_rate"] = bypass_rate

    # ── 5. Group vs direct: same tier2 sender, group should have lower importance
    group_rows = []
    direct_rows = []
    texts = [
        ("Priya", "Hey what's up"),
        ("Rahul", "Are you coming tomorrow?"),
        ("Team", "Meeting notes are ready"),
    ]
    for title, text in texts:
        base = dict(
            title=title, text=text, app_category="messaging",
            sender_open_rate=0.8, sender_avg_response_time_sec=300.0,
            sender_interaction_frequency=3.0, sender_recency_days=1.0,
            sender_conversation_ratio=0.4, app_open_rate=0.6, app_daily_volume=15,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=200.0,
            notification_has_reply_action=True, notification_importance_raw=3,
            is_messaging_style=True, flag_foreground_service=False,
        )
        group_rows.append(_make_row(**{**base, "is_group_conversation": True}))
        direct_rows.append(_make_row(**{**base, "is_group_conversation": False}))

    df_g = pd.DataFrame(group_rows)
    df_d = pd.DataFrame(direct_rows)
    imp_g, _, _ = predict(build_feature_matrix(df_g, enc))
    imp_d, _, _ = predict(build_feature_matrix(df_d, enc))
    results["group_avg_importance"] = float(imp_g.mean())
    results["direct_avg_importance"] = float(imp_d.mean())

    # ── 6. Work hours transaction urgency
    rows = []
    transaction_texts = [
        ("PhonePe", "You paid ₹500 to Priya via UPI"),
        ("Paytm", "₹1,200 debited from your account"),
        ("HDFC Bank", "OTP: 482931 for transaction of ₹2,500"),
        ("GPay", "Payment of ₹350 received from Rahul"),
    ]
    for title, text in transaction_texts:
        rows.append(_make_row(
            title=title, text=text, app_category="finance",
            sender_open_rate=0.4, sender_avg_response_time_sec=3600.0,
            sender_interaction_frequency=1.0, sender_recency_days=2.0,
            sender_conversation_ratio=0.0, app_open_rate=0.35, app_daily_volume=3,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=float("nan"), is_group_conversation=False,
            notification_has_reply_action=False, notification_importance_raw=4,
            is_messaging_style=False, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    _, urg, _ = predict(X)
    results["work_hours_transaction_avg_urgency"] = float(urg.mean())

    # ── 7. High frequency sender (50+ msgs/day) should not all bypass DND
    rows = []
    freq_texts = [
        ("News Bot", "Breaking: market update"),
        ("Alert Bot", "Server CPU at 45%"),
        ("News Bot", "Top stories of the hour"),
        ("Channel", "New post from your subscription"),
        ("Bot", "Daily reminder: drink water"),
        ("Channel", "Live score: 45-32 at halftime"),
        ("Alert Bot", "Disk usage 60%"),
        ("News Bot", "Tech roundup"),
        ("Bot", "Weekly digest is ready"),
        ("Channel", "Your weekly report"),
    ]
    for title, text in freq_texts:
        rows.append(_make_row(
            title=title, text=text, app_category="social",
            sender_open_rate=0.1, sender_avg_response_time_sec=7200.0,
            sender_interaction_frequency=55.0, sender_recency_days=0.0,
            sender_conversation_ratio=0.0, app_open_rate=0.1, app_daily_volume=60,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=2.0, is_group_conversation=False,
            notification_has_reply_action=False, notification_importance_raw=2,
            is_messaging_style=False, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    bypass_rate = float((combined >= BYPASS_THRESHOLD).mean())
    results["high_freq_bypass_rate"] = bypass_rate

    # ── 8. T5 spam messages: cold callers, insurance, MLM → should suppress
    rows = []
    spam_texts = [
        ("Insurance Agent", "Hello sir/ma'am, I'm calling about a special insurance plan for you"),
        ("Marketing", "Hi, we offer loans up to Rs.10 lakhs. Interested?"),
        ("Recruiter", "Exciting opportunity! We have a position that matches your profile"),
        ("Sales", "Limited offer: invest in our scheme and get 30% returns"),
        ("Job Portal", "10 new jobs match your profile. Apply now"),
        ("Unknown", "Congratulations! You have been selected for a prize. Click to claim"),
    ]
    for title, text in spam_texts:
        rows.append(_make_row(
            title=title, text=text, app_category="messaging",
            sender_open_rate=0.03, sender_avg_response_time_sec=86400.0,
            sender_interaction_frequency=2.0, sender_recency_days=0.2,
            sender_conversation_ratio=0.0, app_open_rate=0.05, app_daily_volume=8,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=float("nan"), is_group_conversation=False,
            notification_has_reply_action=False, notification_importance_raw=2,
            is_messaging_style=True, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    suppress_rate = float((combined <= SUPPRESS_THRESHOLD).mean())
    results["t5_spam_suppress_rate"] = suppress_rate

    # ── 9. Delivery partner arrived: must bypass DND (person has to step outside)
    rows = []
    arrived_texts = [
        ("Swiggy", "Your delivery partner Suresh has arrived! Step outside to collect your order."),
        ("Zomato", "Your delivery partner has arrived! Come collect your food now"),
        ("Uber", "Pradeep has arrived and is waiting outside for you."),
        ("Blinkit", "Your delivery partner has arrived. Come collect ASAP!"),
        ("Dunzo", "Your delivery partner is at your door — come out now!"),
    ]
    for title, text in arrived_texts:
        rows.append(_make_row(
            title=title, text=text, app_category="shopping",
            sender_open_rate=0.05, sender_avg_response_time_sec=7200.0,
            sender_interaction_frequency=3.0, sender_recency_days=0.1,
            sender_conversation_ratio=0.0, app_open_rate=0.2, app_daily_volume=10,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=5.0, is_group_conversation=False,
            notification_has_reply_action=False, notification_importance_raw=4,
            is_messaging_style=False, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    bypass_rate = float((combined >= BYPASS_THRESHOLD).mean())
    results["delivery_arrived_bypass_rate"] = bypass_rate

    # ── 10. Critical transactions: OTP and payment failures → must bypass DND
    rows = []
    critical_tx_texts = [
        ("PhonePe", "OTP: 847291. Valid for 10 minutes. DO NOT share with anyone."),
        ("HDFC Bank", "OTP: 382910. Valid for 5 minutes. Do NOT share with anyone."),
        ("PhonePe", "Payment of Rs.799 to Swiggy FAILED. Try again."),
        ("Google Pay", "Transaction declined. Please check your bank account."),
        ("ICICI Bank", "Your UPI transaction of Rs.5000 has FAILED"),
    ]
    for title, text in critical_tx_texts:
        rows.append(_make_row(
            title=title, text=text, app_category="finance",
            sender_open_rate=0.4, sender_avg_response_time_sec=3600.0,
            sender_interaction_frequency=1.0, sender_recency_days=1.0,
            sender_conversation_ratio=0.0, app_open_rate=0.25, app_daily_volume=5,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=float("nan"), is_group_conversation=False,
            notification_has_reply_action=False, notification_importance_raw=4,
            is_messaging_style=False, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    bypass_rate = float((combined >= BYPASS_THRESHOLD).mean())
    results["critical_transaction_bypass_rate"] = bypass_rate

    # ── 11. T1 emergency at 2AM: should bypass even during sleep hours
    rows = []
    t1_night_texts = [
        ("Mom", "Emergency! Are you okay? Call me RIGHT NOW"),
        ("Dad", "Please call immediately, it's urgent"),
        ("Partner", "Are you safe? Something happened, call me now!"),
        ("Best Friend", "Emergency! I need help right now. Call me!"),
    ]
    for title, text in t1_night_texts:
        rows.append(_make_row(
            title=title, text=text, app_category="messaging",
            sender_open_rate=0.95, sender_avg_response_time_sec=30.0,
            sender_interaction_frequency=10.0, sender_recency_days=0.1,
            sender_conversation_ratio=0.9, app_open_rate=0.8, app_daily_volume=20,
            hour_of_day=2, is_weekend=False, is_sleep_hours=True,
            minutes_since_last_from_sender=60.0, is_group_conversation=False,
            notification_has_reply_action=True, notification_importance_raw=4,
            is_messaging_style=True, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    bypass_rate = float((combined >= BYPASS_THRESHOLD).mean())
    results["t1_night_emergency_bypass_rate"] = bypass_rate

    # ── 12. Social-app security alerts: must bypass DND regardless of low open_rate ──
    rows = []
    security_alert_texts = [
        ("Instagram", "A login from an unrecognized device was detected.", "social"),
        ("Facebook", "We noticed an unusual login attempt on your account.", "social"),
        ("Twitter", "Your account may have been compromised. Change your password now.", "social"),
        ("Google", "A new sign-in on Windows. Was this you?", "social"),
        ("WhatsApp", "Someone tried to access your account. Verify your identity.", "messaging"),
    ]
    for title, text, app_cat in security_alert_texts:
        rows.append(_make_row(
            title=title, text=text, app_category=app_cat,
            sender_open_rate=0.05, sender_avg_response_time_sec=99999.0,
            sender_interaction_frequency=0.5, sender_recency_days=0.1,
            sender_conversation_ratio=0.0, app_open_rate=0.1, app_daily_volume=5,
            hour_of_day=14, is_weekend=False, is_sleep_hours=False,
            minutes_since_last_from_sender=1440.0, is_group_conversation=False,
            notification_has_reply_action=False, notification_importance_raw=4,
            is_messaging_style=False, flag_foreground_service=False,
        ))
    df = pd.DataFrame(rows)
    X = build_feature_matrix(df, enc)
    imp, urg, _ = predict(X)
    combined = imp * 0.6 + urg * 0.4
    bypass_rate = float((combined >= BYPASS_THRESHOLD).mean())
    results["security_alert_bypass_rate"] = bypass_rate

    return results


def predict_with_personalization(
    title: str,
    text: str,
    app_category: str,
    sender_open_rate: float,
    sender_avg_response_time_sec: float,
    sender_interaction_frequency: float,
    sender_recency_days: float,
    sender_conversation_ratio: float,
    app_open_rate: float,
    app_daily_volume: float,
    hour_of_day: int,
    is_weekend: bool,
    is_sleep_hours: bool,
    minutes_since_last_from_sender: float,
    is_group_conversation: bool,
    notification_has_reply_action: bool,
    notification_importance_raw: int,
    is_messaging_style: bool,
    flag_foreground_service: bool,
    personalization_layer,
) -> tuple:
    """Run base ONNX scorer + personalization layer. Returns (combined, imp, urg)."""
    enc = _get_encoder()
    row = _make_row(
        title=title, text=text, app_category=app_category,
        sender_open_rate=sender_open_rate,
        sender_avg_response_time_sec=sender_avg_response_time_sec,
        sender_interaction_frequency=sender_interaction_frequency,
        sender_recency_days=sender_recency_days,
        sender_conversation_ratio=sender_conversation_ratio,
        app_open_rate=app_open_rate, app_daily_volume=app_daily_volume,
        hour_of_day=hour_of_day, is_weekend=is_weekend,
        is_sleep_hours=is_sleep_hours,
        minutes_since_last_from_sender=minutes_since_last_from_sender,
        is_group_conversation=is_group_conversation,
        notification_has_reply_action=notification_has_reply_action,
        notification_importance_raw=notification_importance_raw,
        is_messaging_style=is_messaging_style,
        flag_foreground_service=flag_foreground_service,
    )
    df = pd.DataFrame([row])
    X = build_feature_matrix(df, enc)
    imp_arr, urg_arr, _ = predict(X)
    base_imp = float(imp_arr[0])
    base_urg = float(urg_arr[0])

    adj_imp, adj_urg = personalization_layer.adjust(
        base_imp=base_imp,
        base_urg=base_urg,
        app_cat=app_category,
        is_group=bool(is_group_conversation),
        sender_open_rate=sender_open_rate,
        hour_of_day=hour_of_day,
    )
    combined = adj_imp * 0.6 + adj_urg * 0.4
    return combined, adj_imp, adj_urg


if __name__ == "__main__":
    r = run_condition_tests()
    print("=== Condition Test Results ===")
    for k, v in r.items():
        print(f"  {k}: {v:.3f}")

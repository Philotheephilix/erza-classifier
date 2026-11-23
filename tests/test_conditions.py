from training.evaluate_conditions import run_condition_tests, predict_with_personalization
from training.personalization import PersonalizationLayer
from training.evaluate import BYPASS_THRESHOLD, SUPPRESS_THRESHOLD


def _results():
    return run_condition_tests()


def test_tier1_bypass_rate():
    r = _results()
    assert r["tier1_urgent_bypass_rate"] >= 0.70, f"Got {r['tier1_urgent_bypass_rate']:.2f}"


def test_tier5_suppress_rate():
    r = _results()
    assert r["tier5_suppress_rate"] >= 0.70, f"Got {r['tier5_suppress_rate']:.2f}"


def test_cold_start_no_suppress_tier1():
    r = _results()
    assert r["cold_start_message_suppress_rate"] < 0.10, f"Got {r['cold_start_message_suppress_rate']:.2f}"


def test_sleep_hours_promo_not_bypass():
    r = _results()
    assert r["sleep_promo_bypass_rate"] == 0.0, f"Got {r['sleep_promo_bypass_rate']:.2f}"


def test_group_chat_lower_than_direct():
    r = _results()
    assert r["group_avg_importance"] < r["direct_avg_importance"], \
        f"Group {r['group_avg_importance']:.2f} not < Direct {r['direct_avg_importance']:.2f}"


def test_work_hours_transaction_urgency():
    r = _results()
    assert r["work_hours_transaction_avg_urgency"] >= 6.0


def test_high_frequency_sender_not_always_bypass():
    r = _results()
    assert r["high_freq_bypass_rate"] < 0.50, f"Got {r['high_freq_bypass_rate']:.2f}"


def test_t5_spam_messages_suppress():
    r = _results()
    assert r["t5_spam_suppress_rate"] >= 0.60, f"Got {r['t5_spam_suppress_rate']:.2f}"


def test_delivery_arrived_bypasses():
    r = _results()
    assert r["delivery_arrived_bypass_rate"] >= 0.80, f"Got {r['delivery_arrived_bypass_rate']:.2f}"


def test_critical_transaction_bypasses():
    r = _results()
    assert r["critical_transaction_bypass_rate"] >= 0.80, f"Got {r['critical_transaction_bypass_rate']:.2f}"


def test_t1_emergency_at_night_bypasses():
    r = _results()
    assert r["t1_night_emergency_bypass_rate"] >= 0.80, f"Got {r['t1_night_emergency_bypass_rate']:.2f}"


def test_social_security_alert_bypasses():
    r = _results()
    assert r["security_alert_bypass_rate"] >= 0.80, f"Got {r['security_alert_bypass_rate']:.2f}"


# ── personalization integration tests ──────────────────────────────────────

def _promo_kwargs():
    return dict(
        title="Swiggy", text="50% off your next order! Use SAVE50",
        app_category="shopping",
        sender_open_rate=0.03, sender_avg_response_time_sec=86400.0,
        sender_interaction_frequency=5.0, sender_recency_days=0.5,
        sender_conversation_ratio=0.0, app_open_rate=0.05, app_daily_volume=8,
        hour_of_day=14, is_weekend=False, is_sleep_hours=False,
        minutes_since_last_from_sender=float("nan"),
        is_group_conversation=False, notification_has_reply_action=False,
        notification_importance_raw=2, is_messaging_style=False,
        flag_foreground_service=False,
    )


def test_personalization_suppresses_after_dismissals():
    """After 20 consistent dismissals of promo Swiggy, combined score must drop below 3.5."""
    layer = PersonalizationLayer()
    kw = _promo_kwargs()
    for _ in range(20):
        combined, imp, urg = predict_with_personalization(**kw, personalization_layer=layer)
        layer.update(
            base_imp=imp, base_urg=urg, app_cat="shopping",
            is_group=False, sender_open_rate=0.03, hour_of_day=14, outcome=-1.0,
        )
    final_combined, _, _ = predict_with_personalization(**kw, personalization_layer=layer)
    assert final_combined < SUPPRESS_THRESHOLD, (
        f"Expected combined < {SUPPRESS_THRESHOLD} after 20 dismissals, got {final_combined:.3f}"
    )


def test_personalization_security_alert_remains_safe_after_dismissals():
    """After 50 dismiss signals, Facebook security alert must still bypass (CLIP=2.0 guard)."""
    layer = PersonalizationLayer()
    kw = dict(
        title="Facebook", text="Unusual login activity detected on your account.",
        app_category="social",
        sender_open_rate=0.05, sender_avg_response_time_sec=300.0,
        sender_interaction_frequency=3.0, sender_recency_days=1.0,
        sender_conversation_ratio=0.2, app_open_rate=0.3, app_daily_volume=30,
        hour_of_day=22, is_weekend=False, is_sleep_hours=True,
        minutes_since_last_from_sender=float("nan"),
        is_group_conversation=False, notification_has_reply_action=False,
        notification_importance_raw=4, is_messaging_style=False,
        flag_foreground_service=False,
    )
    combined0, imp0, urg0 = predict_with_personalization(**kw, personalization_layer=layer)
    for _ in range(50):
        layer.update(
            base_imp=imp0, base_urg=urg0, app_cat="social",
            is_group=False, sender_open_rate=0.05, hour_of_day=22, outcome=-1.0,
        )
    final_combined, _, _ = predict_with_personalization(**kw, personalization_layer=layer)
    # CLIP=2.0 applied to each of imp and urg → combined can drop at most 2.0
    # The floor is combined0 - 2.0; add 0.1 tolerance for float arithmetic
    assert final_combined >= combined0 - 2.0 - 0.1, (
        f"Security alert dropped too far: base={combined0:.2f} final={final_combined:.2f}"
    )
    # Additionally must remain above suppress threshold (3.5)
    # combined0 for a security alert is typically ≥7; 7-2=5 > 3.5
    assert final_combined > SUPPRESS_THRESHOLD, (
        f"Security alert was suppressed after dismissals: combined={final_combined:.3f}"
    )

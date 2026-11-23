# Notification Classifier — Design Document

> **Status:** Approved for implementation
> **Date:** 2026-10-18

## Intent

Replace the on-device LLM (Qwen3 0.6B, ~500MB, ~200ms/inference) in the Erza Android app with a lightweight XGBoost-based classifier that scores and categorizes incoming notifications in under 50ms. The classifier is personalized: it learns who and what matters to the user from their notification interaction history, updated daily.

## Why

The current LLM approach works but is heavy — 500MB model in the APK, 200ms+ inference per notification, high RAM/CPU usage, and fragile JSON parsing of free-text LLM output. A structured ML classifier is 10-100x smaller, 10-50x faster, produces deterministic typed output, and the feature engineering makes the decision process transparent and debuggable.

## Goals

1. Classify each notification with:
   - **Importance score** (1-10): how much this sender/app matters to the user personally
   - **Time-sensitivity score** (1-10): how urgent this notification is right now
   - **Category label**: message, email, social, promo, transaction, system, delivery, reminder
   - **Priority action**: suppress, pass_through, or bypass_dnd (derived from scores)
2. Total inference latency under 50ms on a mid-range Android device (Snapdragon 7-series or equivalent)
3. Model size under 20MB total (XGBoost model + text encoder)
4. Personalized via daily behavioral feature refresh — no on-device retraining needed
5. Ship initially with a synthetic-trained model; architecture supports swapping in real-user-trained models later

## Non-Goals

- On-device model retraining or fine-tuning (future work)
- Cloud inference or any network dependency
- Replacing the NotificationToolExecutor or DND bypass logic (those stay as-is)
- Supporting iOS or any platform beyond Android

## Anti-Goals

- A model that requires >100MB storage — defeats the purpose of replacing the LLM
- A model that needs internet to classify — must work fully offline
- A black-box that can't explain why it scored high — feature contributions should be inspectable

## Evaluation Criteria

The model must meet these thresholds before shipping:

| Metric | Target | Rationale |
|--------|--------|-----------|
| bypass_dnd false positive rate | < 2% | Don't wake people up at 3am for spam |
| suppress false positive rate | < 5% | Don't hide important messages |
| Category accuracy | > 80% | Reasonable for 8-class problem |
| Importance MAE | < 1.5 | Within 1-2 points of ground truth on 1-10 scale |
| Urgency MAE | < 1.5 | Same |
| Inference latency (p95) | < 50ms | On Snapdragon 7-series or better |

## Architecture

### Two-Stage Pipeline

```
Notification arrives
        │
        ▼
┌──────────────────────┐
│  1. Text Encoder      │  prajjwal1/bert-tiny (ONNX, INT8)
│     (text → 128-dim)  │  ~4MB, ~5-10ms
└──────────┬───────────┘
           │ embedding vector
           ▼
┌──────────────────────────────────────┐
│  2. XGBoost (3 models via ONNX)      │
│     Input: embedding + features      │  ~3-5ms sequential
│     Output: importance, urgency,     │
│             category                 │
└──────────────────────────────────────┘
           │
           ▼
    Priority Decision Logic
    (score thresholds → action)
```

### Text Encoder: `prajjwal1/bert-tiny`

- 2 layers, 128 hidden size, 2 attention heads
- ~4.4M params → ~4MB INT8 quantized ONNX
- Native 128-dim output — no projection layer needed
- Trained on English; sufficient for notification text (short, informal, often English or transliterated)
- No TF-IDF fallback — single encoder, single feature space, deterministic behavior

### Feature Groups

#### A. Text Features (from encoder)
- 128-dimensional embedding of notification text (title + " " + body concatenated)
- Captures semantic content: "are you okay?" vs "lol nice" vs "50% off"

#### B. Sender/Relationship Features (from daily behavioral snapshot)

Sender identity resolution is **app-aware**:
- **Messaging apps** (WhatsApp, Telegram, Signal, SMS, Instagram DM, Discord DM): sender = `android.title` (contact name or group name)
- **Email apps** (Gmail, Outlook): sender = parsed from `android.title` (typically "Sender Name" or "Sender Name - Subject")
- **All other apps** (Uber, Zomato, Amazon, etc.): sender = package name (the app itself is the "sender")

Features per resolved sender:
- `sender_open_rate` (float 0-1): fraction of notifications from this sender the user engaged with (not dismissed)
- `sender_avg_response_time_sec` (float): average time between notification arrival and dismissal/removal
- `sender_interaction_frequency` (float): notifications per day from this sender (7-day rolling average)
- `sender_recency_days` (float): days since last notification from this sender
- `sender_conversation_ratio` (float): ratio of two-way vs one-way interactions (messaging apps only, 0.0 for non-messaging)

**Cold start (day 1, no history):** All sender features are set to NaN. XGBoost handles missing values natively — it learns default split directions during training. The synthetic training data includes ~15% of samples with all-NaN behavioral features to teach the model to fall back on text + context + app features when behavioral data is absent.

#### C. App Features (from daily snapshot)
- `app_open_rate` (float 0-1): fraction of notifications from this app the user engages with
- `app_daily_volume` (int): average daily notification count from this app
- `app_category` (one-hot encoded, 7 categories): messaging, social, shopping, finance, productivity, media, system

#### D. Context Features (computed at classification time)
- `hour_of_day` (int 0-23): device local timezone
- `day_of_week` (int 0-6)
- `is_weekend` (bool)
- `is_sleep_hours` (bool): 11pm-7am device local timezone, configurable
- `minutes_since_last_from_sender` (float): tracked in-memory map, reset to NaN on service restart
- `is_group_conversation` (bool): from `android.isGroupConversation` extra
- `notification_has_reply_action` (bool)
- `notification_importance_raw` (int 0-5): Android's own channel importance level

#### E. Notification Metadata Features
- `has_actions` (bool)
- `is_messaging_style` (bool)
- `flag_auto_cancel` (bool)
- `flag_ongoing` (bool)
- `flag_foreground_service` (bool)

### Total feature count: 128 (embedding) + 7 (app_category one-hot) + 16 (scalar) = 151 features

### XGBoost: 3 Separate Models

XGBoost does not support multi-output natively. Three independent models run sequentially through ONNX Runtime, sharing the same `OrtSession` environment (one environment, three sessions):

| Model | Type | Output |
|-------|------|--------|
| `importance.onnx` | Regression | float 1.0-10.0, rounded to int |
| `urgency.onnx` | Regression | float 1.0-10.0, rounded to int |
| `category.onnx` | Multi-class (8) | argmax → category label |

All three share the same 151-feature input vector. Sequential execution: ~1-2ms each = ~3-6ms total.

### Priority Action Derivation

```kotlin
data class ClassificationOutput(
    val importanceScore: Int,        // 1-10
    val timeSensitivityScore: Int,   // 1-10
    val category: String,            // one of 8 labels
    val action: AgentToolCall,       // derived
    val combinedScore: Float,        // for logging/debugging
)

fun deriveAction(importance: Int, urgency: Int, id: String): AgentToolCall {
    val combined = (importance * 0.6f + urgency * 0.4f)
    return when {
        combined >= 8.0f -> AgentToolCall.ReemitBypassDnd(id, "importance=$importance urgency=$urgency", combined.toInt())
        combined <= 2.5f -> AgentToolCall.SuppressNotification(id, "importance=$importance urgency=$urgency")
        else -> AgentToolCall.PassThrough(id)
    }
}
```

Thresholds are configurable, tuned during evaluation.

## Interaction Tracking (On-Device)

### V1: Dismissal-based (no new permissions)

`onNotificationRemoved(sbn, rankingMap, reason)` provides a `reason` int:
- `REASON_CLICK` (1): user tapped the notification → opened
- `REASON_CANCEL` (2): app cancelled it
- `REASON_CANCEL_ALL` (3): user cleared all
- `REASON_LISTENER_CANCEL` (4): a listener cancelled it (our own suppress action)
- `REASON_GROUP_SUMMARY_CANCELED` (12): group summary removed

We track `REASON_CLICK` as "user engaged" and `REASON_CANCEL_ALL` / swipe-dismiss as "user ignored." This gives us open rate and response time without any new permissions.

### Database

```
┌─────────────────────────────────┐
│  NotificationHistoryDB (Room)    │
│  ────────────────────────────── │
│  id (auto, PK)                   │
│  package_name (String, indexed)  │
│  sender_key (String, indexed)    │  ← app-aware resolved sender
│  timestamp_ms (Long)             │
│  removal_reason (Int)            │  ← REASON_CLICK, REASON_CANCEL, etc.
│  removal_delay_ms (Long)         │  ← time between post and removal
│  category_raw (String, nullable) │
│  is_group (Boolean)              │
└─────────────────────┬───────────┘
                      │ daily aggregation (WorkManager, midnight)
                      ▼
┌─────────────────────────────────┐
│  SenderStatsCache (SharedPrefs)  │
│  ────────────────────────────── │
│  key: "sender:{sender_key}"     │
│  value: JSON {                   │
│    open_rate,                    │
│    avg_response_time,            │
│    frequency,                    │
│    recency,                      │
│    conversation_ratio            │
│  }                               │
│                                  │
│  key: "app:{package_name}"      │
│  value: JSON {                   │
│    open_rate,                    │
│    daily_volume,                 │
│    category                      │
│  }                               │
└─────────────────────────────────┘
```

Retention: 30 days of raw history, then pruned by the daily worker.

## Synthetic Training Data

### 5000 samples across 5 user personas

| Persona | Apps | Relationship spread |
|---------|------|---------------------|
| College student | WhatsApp, Instagram, Snapchat, Discord, Zomato, Swiggy, YouTube | Heavy tier 2-3, many group chats |
| Working professional | Gmail, Slack, WhatsApp, LinkedIn, PhonePe, Uber, Calendar | Heavy tier 3, work-hours bias |
| Parent | WhatsApp, SMS, Amazon, School app, PhonePe, YouTube Kids | Heavy tier 1, family groups |
| Gamer | Discord, Twitch, YouTube, Reddit, Steam | Heavy tier 2, server notifications |
| Mixed/general | All 15 apps | Balanced across all tiers |

### Per sample, the generator produces:

```json
{
  "title": "Mom",
  "text": "Are you coming home this weekend?",
  "package_name": "com.whatsapp",
  "app_category": "messaging",
  "sender_key": "whatsapp:Mom",
  "sender_open_rate": 0.95,
  "sender_avg_response_time_sec": 45.0,
  "sender_interaction_frequency": 8.2,
  "sender_recency_days": 0.1,
  "sender_conversation_ratio": 0.85,
  "app_open_rate": 0.78,
  "app_daily_volume": 42,
  "hour_of_day": 19,
  "day_of_week": 5,
  "is_weekend": true,
  "is_sleep_hours": false,
  "minutes_since_last_from_sender": 120.0,
  "is_group_conversation": false,
  "notification_has_reply_action": true,
  "notification_importance_raw": 4,
  "has_actions": true,
  "is_messaging_style": true,
  "flag_auto_cancel": true,
  "flag_ongoing": false,
  "flag_foreground_service": false,
  "label_importance": 9,
  "label_urgency": 7,
  "label_category": "message"
}
```

### Cold-start simulation

~15% of training samples have all sender/app behavioral features set to NaN, simulating a fresh install. Labels for these samples are derived purely from text content + notification metadata + app type, at slightly more conservative scores (biased toward mid-range 4-6 importance).

### Category distribution in training data

| Category | % of samples | Examples |
|----------|-------------|----------|
| message | 30% | WhatsApp, SMS, Telegram, Discord DM |
| social | 15% | Instagram likes, Twitter mentions, LinkedIn |
| promo | 15% | Zomato deals, Amazon sales, app marketing |
| transaction | 10% | PhonePe payments, bank SMS, Jupiter |
| email | 10% | Gmail threads |
| delivery | 8% | Swiggy order tracking, Amazon shipment |
| system | 7% | Android system, battery, USB |
| reminder | 5% | Calendar events, to-do apps |

## Model Output & Data Classes

### Updated ClassificationResult

```kotlin
data class ClassificationResult(
    val id: String,
    val importanceScore: Int,          // 1-10
    val timeSensitivityScore: Int,     // 1-10
    val category: String,              // one of 8 labels
    val combinedScore: Float,          // importance * 0.6 + urgency * 0.4
    val action: String,                // "suppress" | "pass_through" | "bypass_dnd"
    val dndBypass: Boolean,            // combinedScore >= 8.0
    val suppress: Boolean,             // combinedScore <= 2.5
)
```

This replaces the old `ClassificationResult` which had only `id, score, dndBypass, suppress, reason`. The `score` field is replaced by `importanceScore` + `timeSensitivityScore` + `combinedScore`. Downstream code (`NotificationToolExecutor`, `MainActivity` test UI) reads `dndBypass` and `suppress` booleans, which remain compatible.

### InferenceMetrics (unchanged)

The `ClassifierPipeline` returns `InferenceMetrics` with the same fields (`elapsedMs`, `ramUsedMb`, `cpuPercent`, `gpuMemMb`). The metrics track the full pipeline (text encoding + 3x XGBoost inference).

## File Structure

```
erza-classifier/                          # Training & data pipeline (Python)
├── data/
│   ├── generate_synthetic.py             # generates 5000 training samples
│   ├── feature_engineering.py            # computes features from raw data
│   └── sender_resolution.py             # app-aware sender key extraction
├── training/
│   ├── train_importance.py               # XGBoost importance regressor
│   ├── train_urgency.py                  # XGBoost urgency regressor
│   ├── train_category.py                 # XGBoost category classifier
│   ├── export_onnx.py                    # converts all 3 models + bert-tiny to ONNX
│   └── evaluate.py                       # metrics, confusion matrices, threshold tuning
├── models/                               # exported ONNX models
│   ├── importance.onnx
│   ├── urgency.onnx
│   ├── category.onnx
│   └── text_encoder.onnx
├── notebooks/                            # exploratory analysis
│   └── data_exploration.ipynb
├── docs/superpowers/specs/
│   └── 2026-10-18-notification-classifier-design.md
├── notifications_combined.json           # existing real + synthetic data
└── requirements.txt

erza/                                     # Android app (Kotlin)
├── app/src/main/kotlin/com/erza/agent/
│   ├── classifier/                       # NEW: replaces llm/
│   │   ├── NotificationFeatureExtractor.kt
│   │   ├── TextEncoder.kt
│   │   ├── XGBoostClassifier.kt
│   │   ├── ClassifierPipeline.kt
│   │   └── SenderKeyResolver.kt
│   ├── db/                               # NEW: interaction history
│   │   ├── ErzaDatabase.kt
│   │   ├── NotificationHistoryDao.kt
│   │   ├── NotificationHistoryEntity.kt
│   │   └── SenderStatsCache.kt
│   ├── worker/                           # NEW: daily aggregation
│   │   └── DailyStatsWorker.kt
│   ├── model/
│   │   ├── NotificationData.kt           # modified: add sender_key field
│   │   └── ClassificationResult.kt       # modified: new shape (see above)
│   ├── service/
│   │   └── ErzaNotificationService.kt    # modified: use ClassifierPipeline, track removals
│   ├── tools/
│   │   └── NotificationToolExecutor.kt   # unchanged
│   └── llm/                              # REMOVED entirely
│       ├── LlmEngine.kt                  # deleted
│       ├── LlamaJni.kt                   # deleted
│       ├── PromptBuilder.kt              # deleted
│       ├── NotificationClassifier.kt     # deleted
│       ├── InferenceMetrics.kt           # moved to classifier/ or kept as shared
│       └── GpuProbeActivity.kt           # deleted
├── app/src/main/assets/
│   ├── models/                           # NEW: ONNX models (~12MB total)
│   │   ├── importance.onnx
│   │   ├── urgency.onnx
│   │   ├── category.onnx
│   │   └── text_encoder.onnx
│   └── qwen3/                            # REMOVED: ~500MB model
└── app/build.gradle.kts                  # modified: remove CMake/JNI, add ONNX Runtime + Room deps
```

## Dependencies

### Python (training)
```
xgboost>=2.0
scikit-learn>=1.3
onnxmltools>=1.12
onnxruntime>=1.16
transformers>=4.35
torch>=2.1
numpy
pandas
```

### Android (app)
```gradle
// ADD
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")  // ~8MB AAR
implementation("androidx.room:room-runtime:2.6.1")
implementation("androidx.room:room-ktx:2.6.1")
kapt("androidx.room:room-compiler:2.6.1")
implementation("androidx.work:work-runtime-ktx:2.9.0")

// REMOVE
// CMake/externalNativeBuild block (llama.cpp JNI)
// noCompress "gguf" entry
// qwen3 model from assets
```

The existing CMake native build for LlamaJni is removed entirely. ONNX Runtime ships its own native libs inside the AAR — no STL conflict since we no longer have our own native code.

## Constraints

- **Android minSdk 28** (already set in Erza)
- **ONNX Runtime Android** requires arm64-v8a (already the only ABI filter)
- **Model files ship inside the APK** in assets/ — total ~12MB vs current ~500MB
- **No network calls** during classification — fully offline
- **No new permissions** — interaction tracking uses only `onNotificationRemoved` reason codes, already available to NotificationListenerService
- **Python 3.9+** for training pipeline

## Migration Path

1. Build and evaluate the classifier in `erza-classifier/` (Python)
2. Export ONNX models to `erza/app/src/main/assets/models/`
3. Implement `classifier/` package in the Android app
4. Implement `db/` and `worker/` for interaction tracking
5. Update `NotificationData` and `ClassificationResult` data classes
6. Swap `ErzaNotificationService` to use `ClassifierPipeline`
7. Remove `llm/` package, CMake config, Qwen3 model asset, and LlamaJni native code
8. Update `MainActivity` test UI to show new output format
9. Update `build.gradle.kts` dependencies

The swap is a clean cut — no gradual migration or feature flags needed. The `NotificationToolExecutor` and its channel/DND bypass logic remain untouched.

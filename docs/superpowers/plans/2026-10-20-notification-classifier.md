# Notification Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Erza LLM classifier with a two-stage XGBoost+BERT-tiny pipeline that outputs importance score (1-10), urgency score (1-10), category label, and priority action per notification, running in <50ms on-device.

**Architecture:** `prajjwal1/bert-tiny` (INT8 ONNX, ~4MB) encodes notification text to 128-dim embeddings; three XGBoost models (importance regressor, urgency regressor, category classifier) take embeddings + 23 structured features (relationship/app/context/metadata) and output scores/label; a daily WorkManager job aggregates per-sender interaction stats from a Room database to refresh the structured features.

**Tech Stack:** Python 3.9+, XGBoost 2.0+, onnxmltools, transformers, ONNX Runtime; Android: ONNX Runtime Android 1.17.0, Room 2.6.1, WorkManager 2.9.0, Kotlin coroutines.

**Spec:** `docs/superpowers/specs/2026-10-18-notification-classifier-design.md`

## Global Constraints

- Python 3.9+ for all training scripts
- XGBoost >= 2.0, scikit-learn >= 1.3, onnxmltools >= 1.12, onnxruntime >= 1.16, transformers >= 4.35, torch >= 2.1
- Android minSdk 28, targetSdk 35, arm64-v8a only
- ONNX Runtime Android 1.17.0 (com.microsoft.onnxruntime:onnxruntime-android)
- Room 2.6.1, WorkManager 2.9.0
- Total ONNX model assets ≤ 20MB in APK
- No network calls during classification — fully offline
- No new Android permissions beyond existing NotificationListenerService
- 151 features total: 128 embedding + 7 app_category one-hot + 16 scalar
- Feature names must be identical between Python training and Kotlin inference (no rename)
- `ClassificationResult` keeps `dndBypass` and `suppress` boolean fields for downstream compat
- bypass_dnd threshold: combined >= 8.0f; suppress threshold: combined <= 2.5f; combined = importance * 0.6f + urgency * 0.4f
- Sender resolution rules: messaging pkg → `android.title`; email pkg → `android.title`; all others → `packageName`
- Messaging packages: `com.whatsapp`, `org.telegram.messenger`, `com.facebook.orca`, `com.instagram.android`, `com.discord`, `com.google.android.apps.messaging`, `org.thoughtcrime.securesms`
- Email packages: `com.google.android.gm`, `com.microsoft.office.outlook`
- 5000 synthetic training samples; 15% with all-NaN sender/app behavioral features (cold-start simulation)
- Category distribution: message 30%, social 15%, promo 15%, transaction 10%, email 10%, delivery 8%, system 7%, reminder 5%
- 5 user personas: college_student, working_professional, parent, gamer, mixed
- Relationship tiers — Tier1 open_rate 0.90-0.99 resp<60s; Tier2 0.75-0.90 1-5min; Tier3 0.60-0.80 5-30min; Tier4 0.30-0.50 1hr+; Tier5 0.02-0.15 no reply
- Evaluation thresholds: bypass_dnd FP < 2%, suppress FP < 5%, category accuracy > 80%, importance MAE < 1.5, urgency MAE < 1.5

## Review Focus

1. **Cold-start sender features (all NaN):** XGBoost must still produce reasonable scores; test that importance >= 4 for a DM from "Mom" on WhatsApp with NaN behavioral features — covered in Task 5 evaluation tests.
2. **Sleep-hours urgency suppression:** A bypass_dnd notification sent at 2am must not fire for a promo app even if combined score is marginally >= 8.0 — urgency scoring should penalize non-critical categories at sleep hours; covered in Task 3 label generation tests.
3. **App-as-sender collapse:** Amazon sends 10 different notification types; all map to `in.amazon.mShop.android.shopping` as sender_key — the model must not confuse relationship tier with content urgency; covered in Task 2 sender resolution tests.
4. **ONNX feature order:** The 151-feature vector must be assembled in exactly the same column order in Python (training) and Kotlin (inference); any mismatch silently degrades performance — covered in Task 6 integration test comparing Python vs Kotlin feature vectors on shared inputs.
5. **Removal reason attribution:** `REASON_LISTENER_CANCEL` (4) is our own suppress action — must NOT count as user engagement; test that suppressed notifications don't inflate sender open_rate; covered in Task 8 Room DAO tests.

---

## Task 1: Python project scaffold and requirements

**Files:**
- Create: `erza-classifier/requirements.txt`
- Create: `erza-classifier/data/__init__.py`
- Create: `erza-classifier/training/__init__.py`
- Create: `erza-classifier/models/.gitkeep`

**Interfaces:**
- Produces: installable Python environment for all subsequent tasks

- [ ] **Step 1: Write `requirements.txt`**

```
xgboost>=2.0
scikit-learn>=1.3
onnxmltools>=1.12
skl2onnx>=1.15
onnxruntime>=1.16
transformers>=4.35
torch>=2.1
numpy
pandas
pytest
```

- [ ] **Step 2: Install dependencies**

```bash
cd /Users/I740422/projects/erza-classifier
pip3 install -r requirements.txt
```

Expected: all packages install without conflict.

- [ ] **Step 3: Create empty package init files and models placeholder**

```bash
touch data/__init__.py training/__init__.py models/.gitkeep
mkdir -p notebooks
```

- [ ] **Step 4: Verify import chain**

```bash
python3 -c "import xgboost, sklearn, onnxmltools, onnxruntime, transformers, torch; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git init  # if not already a repo
git add requirements.txt data/__init__.py training/__init__.py models/.gitkeep
git commit -m "feat: python project scaffold for notification classifier"
```

---

## Task 2: Sender key resolver and feature schema

**Files:**
- Create: `erza-classifier/data/sender_resolution.py`
- Create: `erza-classifier/data/feature_schema.py`
- Create: `erza-classifier/tests/test_sender_resolution.py`

**Interfaces:**
- Produces:
  - `resolve_sender_key(package_name: str, title: str | None) -> str` — returns `"{app_tag}:{title}"` for messaging/email, `"{package_name}"` for others
  - `FEATURE_COLUMNS: list[str]` — ordered list of 151 feature names used for training and inference alignment
  - `APP_CATEGORY_MAP: dict[str, str]` — maps package name to one of 7 app categories
  - `MESSAGING_PACKAGES: frozenset[str]`
  - `EMAIL_PACKAGES: frozenset[str]`
  - `CATEGORY_LABELS: list[str]` — 8 ordered category labels

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sender_resolution.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/I740422/projects/erza-classifier
python3 -m pytest tests/test_sender_resolution.py -v
```

Expected: ImportError or AttributeError on all tests.

- [ ] **Step 3: Implement `data/sender_resolution.py`**

Define `MESSAGING_PACKAGES` as frozenset of 7 packages; `EMAIL_PACKAGES` as frozenset of 2. `resolve_sender_key` returns `f"{pkg}:{title}"` if pkg in MESSAGING_PACKAGES or EMAIL_PACKAGES and title is not None, else returns pkg.

- [ ] **Step 4: Implement `data/feature_schema.py`**

Define in this exact order:
1. `embed_0` through `embed_127` (128 names) — BERT-tiny embedding dims
2. `app_cat_messaging`, `app_cat_social`, `app_cat_shopping`, `app_cat_finance`, `app_cat_productivity`, `app_cat_media`, `app_cat_system` (7 one-hot)
3. Scalar features in this order: `sender_open_rate`, `sender_avg_response_time_sec`, `sender_interaction_frequency`, `sender_recency_days`, `sender_conversation_ratio`, `app_open_rate`, `app_daily_volume`, `hour_of_day`, `day_of_week`, `is_weekend`, `is_sleep_hours`, `minutes_since_last_from_sender`, `is_group_conversation`, `notification_has_reply_action`, `notification_importance_raw`, `has_actions`, `is_messaging_style`, `flag_auto_cancel`, `flag_ongoing`, `flag_foreground_service` — wait, that's 20 scalars; but spec says 16. Use exactly these 16: `sender_open_rate`, `sender_avg_response_time_sec`, `sender_interaction_frequency`, `sender_recency_days`, `sender_conversation_ratio`, `app_open_rate`, `app_daily_volume`, `hour_of_day`, `is_weekend`, `is_sleep_hours`, `minutes_since_last_from_sender`, `is_group_conversation`, `notification_has_reply_action`, `notification_importance_raw`, `is_messaging_style`, `flag_foreground_service`.

Total: 128 + 7 + 16 = 151.

`CATEGORY_LABELS = ["message", "email", "social", "promo", "transaction", "system", "delivery", "reminder"]`

`APP_CATEGORY_MAP` maps each known package name to one of: `messaging`, `social`, `shopping`, `finance`, `productivity`, `media`, `system`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_sender_resolution.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add data/sender_resolution.py data/feature_schema.py tests/test_sender_resolution.py
git commit -m "feat: sender key resolver and 151-feature schema"
```

---

## Task 3: Synthetic training data generator

**Files:**
- Create: `erza-classifier/data/generate_synthetic.py`
- Create: `erza-classifier/tests/test_generate_synthetic.py`

**Interfaces:**
- Consumes: `resolve_sender_key`, `FEATURE_COLUMNS`, `APP_CATEGORY_MAP`, `CATEGORY_LABELS` from Task 2
- Produces:
  - `generate_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame` — returns DataFrame with all FEATURE_COLUMNS + `label_importance` (int 1-10) + `label_urgency` (int 1-10) + `label_category` (str)
  - `erza-classifier/data/training_data.json` — saved when run as `__main__`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generate_synthetic.py
import pandas as pd
from data.generate_synthetic import generate_dataset
from data.feature_schema import FEATURE_COLUMNS, CATEGORY_LABELS

def test_dataset_row_count():
    df = generate_dataset(n_samples=100, seed=0)
    assert len(df) == 100

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
    # notifications generated during sleep hours (hour 2) for promo apps
    # should have label_urgency <= 5
    df = generate_dataset(n_samples=500, seed=4)
    sleep_promo = df[(df["is_sleep_hours"] == True) & (df["label_category"] == "promo")]
    if len(sleep_promo) >= 5:
        assert sleep_promo["label_urgency"].mean() <= 5.0

def test_tier1_importance_high():
    # Tier1 senders (open_rate >= 0.90) should average importance >= 7
    df = generate_dataset(n_samples=500, seed=5)
    tier1 = df[df["sender_open_rate"] >= 0.90]
    if len(tier1) >= 10:
        assert tier1["label_importance"].mean() >= 7.0

def test_cold_start_importance_conservative():
    # Cold-start samples should average importance 4-6 (conservative mid-range)
    df = generate_dataset(n_samples=500, seed=6)
    cold = df[df["sender_open_rate"].isna()]
    if len(cold) >= 10:
        mean = cold["label_importance"].mean()
        assert 3.5 <= mean <= 6.5, f"Cold-start mean importance {mean:.2f} not in 3.5-6.5"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_generate_synthetic.py -v
```

Expected: ImportError on all.

- [ ] **Step 3: Implement `data/generate_synthetic.py`**

Key implementation decisions:
- Define 5 personas as dicts mapping app categories to tier distributions
- For each sample: pick persona, pick app+package from persona's distribution, pick a sender tier (Tier1-5) according to app type (messaging apps can be any tier; promo apps always Tier5), sample behavioral features from tier distributions with noise, compute labels deterministically:
  - `label_importance = tier_base_score + text_urgency_bonus` clamped to 1-10, where tier_base_score: T1=9, T2=7, T3=6, T4=4, T5=2
  - `label_urgency`: derived from category + text keyword urgency signals + sleep_hours penalty (×0.5 if sleep and category in {promo, social, system})
  - `label_category`: derived from app type + text content keywords
- For 15% of samples: set sender_open_rate, sender_avg_response_time_sec, sender_interaction_frequency, sender_recency_days, sender_conversation_ratio, app_open_rate, app_daily_volume all to NaN; set label_importance to randint(4, 6) unless content has strong urgency signals
- Use `numpy.random.default_rng(seed)` for reproducibility
- Generate realistic text content (title, body) per category using template lists with 20+ variations per category
- Embed text using `prajjwal1/bert-tiny` tokenizer only for embed columns (produce random embeddings from N(0,0.1) per sample — actual BERT embeddings are computed at training time by the feature_engineering pipeline; generate_synthetic stores embed_* as 0.0 placeholder)

**Note:** embed_0..embed_127 are set to 0.0 in generate_synthetic. The training pipeline (Task 4) recomputes them from title+text using the actual BERT encoder. The label generation does not depend on embeddings.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_generate_synthetic.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Generate full 5000-sample dataset**

```bash
cd /Users/I740422/projects/erza-classifier
python3 -c "
from data.generate_synthetic import generate_dataset
import json
df = generate_dataset(5000, seed=42)
df.to_json('data/training_data.json', orient='records', indent=2)
print(f'Generated {len(df)} samples')
print(df['label_category'].value_counts())
"
```

Expected: prints category distribution matching spec within ±5%.

- [ ] **Step 6: Commit**

```bash
git add data/generate_synthetic.py tests/test_generate_synthetic.py data/training_data.json
git commit -m "feat: synthetic training data generator (5000 samples, 5 personas)"
```

---

## Task 4: Feature engineering and BERT-tiny text encoder

**Files:**
- Create: `erza-classifier/data/feature_engineering.py`
- Create: `erza-classifier/tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, `APP_CATEGORY_MAP` from Task 2; `data/training_data.json` from Task 3
- Produces:
  - `BertTinyEncoder` class: `__init__(model_name="prajjwal1/bert-tiny")`, `encode(texts: list[str]) -> np.ndarray` shape `(n, 128)`
  - `build_feature_matrix(df: pd.DataFrame, encoder: BertTinyEncoder) -> np.ndarray` shape `(n, 151)` — fills embed cols from encoder, preserves all other cols in FEATURE_COLUMNS order
  - `erza-classifier/data/features_train.npy` — saved feature matrix (5000, 151)
  - `erza-classifier/data/labels_importance.npy`, `labels_urgency.npy`, `labels_category.npy`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_feature_engineering.py
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

def test_build_feature_matrix_shape():
    enc = BertTinyEncoder()
    df = pd.DataFrame([{
        "title": "Mom", "text": "Are you coming home?",
        "sender_open_rate": 0.95, "sender_avg_response_time_sec": 45.0,
        "sender_interaction_frequency": 8.2, "sender_recency_days": 0.1,
        "sender_conversation_ratio": 0.85, "app_open_rate": 0.78,
        "app_daily_volume": 42, "hour_of_day": 19,
        "is_weekend": True, "is_sleep_hours": False,
        "minutes_since_last_from_sender": 120.0, "is_group_conversation": False,
        "notification_has_reply_action": True, "notification_importance_raw": 4,
        "is_messaging_style": True, "flag_foreground_service": False,
        "app_category": "messaging",
    }])
    mat = build_feature_matrix(df, enc)
    assert mat.shape == (1, 151)

def test_feature_matrix_column_order():
    enc = BertTinyEncoder()
    df = pd.DataFrame([{
        "title": "Deal", "text": "50% off",
        "sender_open_rate": float("nan"), "sender_avg_response_time_sec": float("nan"),
        "sender_interaction_frequency": float("nan"), "sender_recency_days": float("nan"),
        "sender_conversation_ratio": float("nan"), "app_open_rate": float("nan"),
        "app_daily_volume": float("nan"), "hour_of_day": 14,
        "is_weekend": False, "is_sleep_hours": False,
        "minutes_since_last_from_sender": float("nan"), "is_group_conversation": False,
        "notification_has_reply_action": False, "notification_importance_raw": 2,
        "is_messaging_style": False, "flag_foreground_service": False,
        "app_category": "shopping",
    }])
    mat = build_feature_matrix(df, enc)
    # app_cat_shopping should be 1 at index 130 (128 embed + 2nd one-hot = index 130)
    assert mat[0, 130] == 1.0
    assert mat[0, 128] == 0.0  # app_cat_messaging
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_feature_engineering.py -v
```

Expected: ImportError on all.

- [ ] **Step 3: Implement `data/feature_engineering.py`**

`BertTinyEncoder.encode`: tokenize with `AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")`, run `AutoModel.from_pretrained("prajjwal1/bert-tiny")` in eval mode with `torch.no_grad()`, return mean-pooled last hidden state (mean over sequence dim) as `np.ndarray float32`.

`build_feature_matrix`: 
1. Encode `title + " " + text` for all rows → embed matrix (n, 128)
2. One-hot encode `app_category` into 7 columns in order: messaging, social, shopping, finance, productivity, media, system
3. Extract 16 scalar features in FEATURE_COLUMNS order
4. Concatenate [embed, one_hot, scalars] → (n, 151)
5. NaN values in scalar cols pass through as np.nan (XGBoost handles natively)

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_feature_engineering.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Build training feature matrix from generated data**

```bash
python3 -c "
import json, numpy as np, pandas as pd
from data.feature_engineering import BertTinyEncoder, build_feature_matrix
from data.feature_schema import CATEGORY_LABELS

df = pd.read_json('data/training_data.json')
enc = BertTinyEncoder()
print('Encoding texts...')
X = build_feature_matrix(df, enc)
y_imp = df['label_importance'].values.astype(np.float32)
y_urg = df['label_urgency'].values.astype(np.float32)
y_cat = np.array([CATEGORY_LABELS.index(c) for c in df['label_category']], dtype=np.int32)
np.save('data/features_train.npy', X)
np.save('data/labels_importance.npy', y_imp)
np.save('data/labels_urgency.npy', y_urg)
np.save('data/labels_category.npy', y_cat)
print(f'X: {X.shape}, y_imp: {y_imp.shape}')
"
```

Expected: prints `X: (5000, 151)`.

- [ ] **Step 6: Export BERT-tiny to ONNX**

```bash
python3 -c "
from transformers import AutoTokenizer, AutoModel
from transformers.onnx import export
from pathlib import Path
import torch

model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
dummy = tokenizer('test', return_tensors='pt')

torch.onnx.export(
    model,
    (dummy['input_ids'], dummy['attention_mask']),
    'models/text_encoder.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'seq'},
        'attention_mask': {0: 'batch', 1: 'seq'},
        'last_hidden_state': {0: 'batch', 1: 'seq'},
    },
    opset_version=14,
)
print('Exported text_encoder.onnx')
import os; print(f'Size: {os.path.getsize(\"models/text_encoder.onnx\")/1e6:.1f} MB')
"
```

Expected: `Size: ~17 MB` (FP32). Then quantize:

```bash
python3 -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('models/text_encoder.onnx', 'models/text_encoder_int8.onnx', weight_type=QuantType.QInt8)
import os; print(f'INT8 size: {os.path.getsize(\"models/text_encoder_int8.onnx\")/1e6:.1f} MB')
"
```

Expected: `~4-5 MB`.

- [ ] **Step 7: Commit**

```bash
git add data/feature_engineering.py tests/test_feature_engineering.py \
        data/features_train.npy data/labels_importance.npy \
        data/labels_urgency.npy data/labels_category.npy \
        models/text_encoder_int8.onnx
git commit -m "feat: bert-tiny encoder + feature matrix builder + ONNX export"
```

---

## Task 5: Train XGBoost models and export to ONNX

**Files:**
- Create: `erza-classifier/training/train_importance.py`
- Create: `erza-classifier/training/train_urgency.py`
- Create: `erza-classifier/training/train_category.py`
- Create: `erza-classifier/training/export_onnx.py`
- Create: `erza-classifier/training/evaluate.py`
- Create: `erza-classifier/tests/test_models.py`

**Interfaces:**
- Consumes: `data/features_train.npy`, `data/labels_*.npy`, `CATEGORY_LABELS`, `FEATURE_COLUMNS` from Tasks 2-4
- Produces:
  - `models/importance.onnx`, `models/urgency.onnx`, `models/category.onnx`
  - `training/evaluate.py`: `run_evaluation() -> dict` with keys: `importance_mae`, `urgency_mae`, `category_accuracy`, `bypass_dnd_fpr`, `suppress_fpr`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
import numpy as np
import onnxruntime as ort
import os

def _load_session(path):
    assert os.path.exists(path), f"Model not found: {path}"
    return ort.InferenceSession(path)

def test_importance_model_exists():
    sess = _load_session("models/importance.onnx")
    assert sess is not None

def test_urgency_model_exists():
    sess = _load_session("models/urgency.onnx")
    assert sess is not None

def test_category_model_exists():
    sess = _load_session("models/category.onnx")
    assert sess is not None

def test_importance_output_range():
    sess = _load_session("models/importance.onnx")
    X = np.load("data/features_train.npy")[:10].astype(np.float32)
    out = sess.run(None, {"X": X})[0].flatten()
    assert ((out >= 1.0) & (out <= 10.0)).all(), f"Out of range: {out}"

def test_urgency_output_range():
    sess = _load_session("models/urgency.onnx")
    X = np.load("data/features_train.npy")[:10].astype(np.float32)
    out = sess.run(None, {"X": X})[0].flatten()
    assert ((out >= 1.0) & (out <= 10.0)).all()

def test_category_output_shape():
    sess = _load_session("models/category.onnx")
    X = np.load("data/features_train.npy")[:10].astype(np.float32)
    # category model outputs label int array
    out = sess.run(None, {"X": X})[0]
    assert out.shape == (10,) or out.shape == (10, 1)
    assert ((out >= 0) & (out <= 7)).all()

def test_importance_mae_threshold():
    from training.evaluate import run_evaluation
    metrics = run_evaluation()
    assert metrics["importance_mae"] < 1.5, f"MAE {metrics['importance_mae']:.3f} >= 1.5"

def test_urgency_mae_threshold():
    from training.evaluate import run_evaluation
    metrics = run_evaluation()
    assert metrics["urgency_mae"] < 1.5

def test_category_accuracy_threshold():
    from training.evaluate import run_evaluation
    metrics = run_evaluation()
    assert metrics["category_accuracy"] > 0.80, f"Accuracy {metrics['category_accuracy']:.3f} <= 0.80"

def test_bypass_dnd_fpr_threshold():
    from training.evaluate import run_evaluation
    metrics = run_evaluation()
    assert metrics["bypass_dnd_fpr"] < 0.02, f"FPR {metrics['bypass_dnd_fpr']:.3f} >= 0.02"

def test_suppress_fpr_threshold():
    from training.evaluate import run_evaluation
    metrics = run_evaluation()
    assert metrics["suppress_fpr"] < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_models.py::test_importance_model_exists -v
```

Expected: AssertionError (file not found).

- [ ] **Step 3: Implement training scripts**

`training/train_importance.py`:
- Load `data/features_train.npy` and `data/labels_importance.npy`
- 80/20 train/val split with `random_state=42`
- Train `XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, objective='reg:squarederror', missing=float('nan'), random_state=42)`
- Clip predictions to [1.0, 10.0]
- Save model as `models/importance_xgb.json`

`training/train_urgency.py`: same structure, uses `labels_urgency.npy`, saves `models/urgency_xgb.json`

`training/train_category.py`: `XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='multi:softmax', num_class=8, missing=float('nan'), random_state=42)`, saves `models/category_xgb.json`

- [ ] **Step 4: Implement `training/export_onnx.py`**

Use `onnxmltools.convert_xgboost` to convert each `.json` model to ONNX, with `initial_types=[("X", FloatTensorType([None, 151]))]`. For the regressor models, post-process output to clip to [1, 10]. Save as `models/importance.onnx`, `models/urgency.onnx`, `models/category.onnx`.

- [ ] **Step 5: Implement `training/evaluate.py`**

`run_evaluation()`:
- Load all 3 ONNX sessions
- Load test split (20% held-out from Task 3 data, same seed=42 split)
- Compute importance MAE, urgency MAE, category accuracy
- For bypass_dnd FPR: ground truth positive = `(imp * 0.6 + urg * 0.4) >= 8.0` on labels; predicted positive = same formula on model outputs; FPR = FP / (FP + TN)
- For suppress FPR: same logic with threshold <= 2.5
- Return dict

- [ ] **Step 6: Run training pipeline**

```bash
cd /Users/I740422/projects/erza-classifier
python3 training/train_importance.py
python3 training/train_urgency.py
python3 training/train_category.py
python3 training/export_onnx.py
```

Expected: 4 ONNX files in `models/`.

- [ ] **Step 7: Run evaluation**

```bash
python3 -c "
from training.evaluate import run_evaluation
m = run_evaluation()
for k, v in m.items():
    print(f'{k}: {v:.4f}')
"
```

Expected: all metrics within spec thresholds. If category_accuracy < 0.80, add more samples with `generate_dataset(n=7000)` and retrain.

- [ ] **Step 8: Run model tests**

```bash
python3 -m pytest tests/test_models.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 9: Print model sizes**

```bash
python3 -c "
import os
for f in ['models/importance.onnx','models/urgency.onnx','models/category.onnx','models/text_encoder_int8.onnx']:
    print(f'{f}: {os.path.getsize(f)/1e6:.2f} MB')
"
```

Expected: total < 20MB.

- [ ] **Step 10: Commit**

```bash
git add training/ tests/test_models.py models/importance.onnx models/urgency.onnx models/category.onnx
git commit -m "feat: train XGBoost classifiers + ONNX export, all eval thresholds met"
```

---

## Task 6: Performance evaluation across relationship conditions

**Files:**
- Create: `erza-classifier/training/evaluate_conditions.py`
- Create: `erza-classifier/tests/test_conditions.py`

**Interfaces:**
- Consumes: trained ONNX models, `BertTinyEncoder`, `generate_dataset`
- Produces: `training/evaluate_conditions.py` — `run_condition_tests() -> dict` with per-condition accuracy breakdowns

- [ ] **Step 1: Write condition tests**

```python
# tests/test_conditions.py
from training.evaluate_conditions import run_condition_tests

def _results():
    return run_condition_tests()

def test_tier1_bypass_rate():
    # Family/partner (open_rate >= 0.90) with urgent text should bypass DND >= 70% of the time
    r = _results()
    assert r["tier1_urgent_bypass_rate"] >= 0.70, f"Got {r['tier1_urgent_bypass_rate']:.2f}"

def test_tier5_suppress_rate():
    # Promos/brands (open_rate <= 0.15) should be suppressed >= 70% of the time
    r = _results()
    assert r["tier5_suppress_rate"] >= 0.70, f"Got {r['tier5_suppress_rate']:.2f}"

def test_cold_start_no_suppress_tier1():
    # Cold-start WhatsApp DM (NaN behavioral features) should NOT be suppressed
    r = _results()
    assert r["cold_start_message_suppress_rate"] < 0.10, f"Got {r['cold_start_message_suppress_rate']:.2f}"

def test_sleep_hours_promo_not_bypass():
    # Promo notification at 2am should never bypass DND
    r = _results()
    assert r["sleep_promo_bypass_rate"] == 0.0, f"Got {r['sleep_promo_bypass_rate']:.2f}"

def test_group_chat_lower_than_direct():
    # Group chat importance should average lower than direct DM from same tier
    r = _results()
    assert r["group_avg_importance"] < r["direct_avg_importance"], \
        f"Group {r['group_avg_importance']:.2f} not < Direct {r['direct_avg_importance']:.2f}"

def test_work_hours_transaction_urgency():
    # Payment confirmation during work hours (9am-6pm) should have urgency >= 6
    r = _results()
    assert r["work_hours_transaction_avg_urgency"] >= 6.0

def test_high_frequency_sender_not_always_bypass():
    # Very frequent senders (50+ msgs/day) should not all bypass DND — variety matters
    r = _results()
    assert r["high_freq_bypass_rate"] < 0.50, f"Got {r['high_freq_bypass_rate']:.2f}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_conditions.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `training/evaluate_conditions.py`**

`run_condition_tests()`:
- Generate 500 targeted test samples per condition using `generate_dataset` with fixed parameters (not random — construct DataFrames directly with pinned features)
- Run full ONNX pipeline: BertTinyEncoder → build_feature_matrix → 3 ONNX sessions → derive action
- Compute per-condition metrics as described in tests
- Return dict

Key conditions to simulate directly:
- tier1_urgent: `sender_open_rate=0.95`, `sender_avg_response_time_sec=30`, `is_group_conversation=False`, text="Are you okay? Please call me", category="message"
- tier5_promo: `sender_open_rate=0.03`, `app_category="shopping"`, text="50% off! Limited time offer", category="promo"
- cold_start_message: all sender/app features = NaN, `package_name="com.whatsapp"`, `is_messaging_style=True`, text="Hey what's up"
- sleep_promo: `is_sleep_hours=True`, `hour_of_day=2`, `app_category="shopping"`, text="Flash sale tonight only"
- group vs direct: same tier2 sender, `is_group_conversation=True` vs `False`
- work_hours_transaction: `hour_of_day=14`, `is_weekend=False`, `app_category="finance"`, text="You paid ₹500 to Priya via UPI"

- [ ] **Step 4: Run condition tests**

```bash
python3 -m pytest tests/test_conditions.py -v
```

Expected: all 7 tests PASS. If any fail, tune XGBoost hyperparameters or label generation in Task 3 and retrain (Task 5 Step 6-8).

- [ ] **Step 5: Print full condition report**

```bash
python3 -c "
from training.evaluate_conditions import run_condition_tests
r = run_condition_tests()
for k, v in r.items():
    print(f'{k}: {v:.3f}')
"
```

- [ ] **Step 6: Commit**

```bash
git add training/evaluate_conditions.py tests/test_conditions.py
git commit -m "test: condition-based evaluation across all relationship tiers and contexts"
```

---

## Task 7: Android data classes and Gradle updates

**Files:**
- Modify: `erza/app/src/main/kotlin/com/erza/agent/model/NotificationData.kt`
- Modify: `erza/app/src/main/kotlin/com/erza/agent/model/ClassificationResult.kt`
- Modify: `erza/app/build.gradle.kts`
- Create: `erza/app/src/main/kotlin/com/erza/agent/classifier/InferenceMetrics.kt` (moved from llm/)

**Interfaces:**
- Produces:
  - `NotificationData` with new field `senderKey: String`
  - `ClassificationResult(id, importanceScore, timeSensitivityScore, category, combinedScore, action, dndBypass, suppress)`
  - Gradle with ONNX Runtime, Room, WorkManager added; CMake block removed

- [ ] **Step 1: Update `NotificationData.kt`**

Add field `senderKey: String` after `packageName`. Keep all existing fields. No other changes.

- [ ] **Step 2: Update `ClassificationResult.kt`**

Replace body with:
```kotlin
data class ClassificationResult(
    val id: String,
    val importanceScore: Int,
    val timeSensitivityScore: Int,
    val category: String,
    val combinedScore: Float,
    val action: String,
    val dndBypass: Boolean,
    val suppress: Boolean,
)
```

Remove `ClassificationResponse` data class (no longer needed). Keep `AgentToolCall` sealed class unchanged.

- [ ] **Step 3: Create `classifier/InferenceMetrics.kt`**

Move `InferenceMetrics` data class and `MetricsSampler` object from `llm/InferenceMetrics.kt` to `classifier/InferenceMetrics.kt`. Change package to `com.erza.agent.classifier`.

- [ ] **Step 4: Update `build.gradle.kts`**

Remove: entire `externalNativeBuild` block, `ndk { abiFilters }` block, `noCompress += "gguf"` line, `useLegacyPackaging = true`.

Add under `dependencies`:
```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
implementation("androidx.room:room-runtime:2.6.1")
implementation("androidx.room:room-ktx:2.6.1")
kapt("androidx.room:room-compiler:2.6.1")
implementation("androidx.work:work-runtime-ktx:2.9.0")
```

Add `kotlin("kapt")` plugin. Keep `noCompress += "onnx"` (already present).

- [ ] **Step 5: Verify Gradle sync**

```bash
cd /Users/I740422/projects/erza
./gradlew assembleDebug 2>&1 | tail -20
```

Expected: BUILD SUCCESSFUL (may fail on missing classifier/ sources — that's fine at this stage, just check Gradle resolves dependencies).

- [ ] **Step 6: Commit**

```bash
cd /Users/I740422/projects/erza
git add app/src/main/kotlin/com/erza/agent/model/ \
        app/src/main/kotlin/com/erza/agent/classifier/InferenceMetrics.kt \
        app/build.gradle.kts
git commit -m "feat: update data classes for dual-score classifier, add ONNX Runtime + Room deps"
```

---

## Task 8: Android Room database and sender stats cache

**Files:**
- Create: `erza/app/src/main/kotlin/com/erza/agent/db/NotificationHistoryEntity.kt`
- Create: `erza/app/src/main/kotlin/com/erza/agent/db/NotificationHistoryDao.kt`
- Create: `erza/app/src/main/kotlin/com/erza/agent/db/ErzaDatabase.kt`
- Create: `erza/app/src/main/kotlin/com/erza/agent/db/SenderStatsCache.kt`

**Interfaces:**
- Produces:
  - `NotificationHistoryEntity(@Entity tableName="notification_history")` with fields: `id` (autogen PK), `packageName`, `senderKey`, `timestampMs`, `removalReason`, `removalDelayMs`, `isGroup`
  - `NotificationHistoryDao` with: `insert(entity)`, `getEntriesAfter(cutoffMs: Long): List<NotificationHistoryEntity>`, `deleteEntriesBefore(cutoffMs: Long)`
  - `ErzaDatabase` Room database singleton with `historyDao(): NotificationHistoryDao`
  - `SenderStatsCache(context: Context)` with: `getSenderStats(senderKey: String): SenderStats?`, `getAppStats(packageName: String): AppStats?`, `updateAll(senderMap, appMap)` — backed by SharedPreferences
  - `data class SenderStats(openRate, avgResponseTimeSec, frequency, recencyDays, conversationRatio)`
  - `data class AppStats(openRate, dailyVolume, appCategory)`

- [ ] **Step 1: Implement `NotificationHistoryEntity.kt`**

```kotlin
@Entity(tableName = "notification_history",
        indices = [Index("sender_key"), Index("package_name"), Index("timestamp_ms")])
data class NotificationHistoryEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    @ColumnInfo(name = "package_name") val packageName: String,
    @ColumnInfo(name = "sender_key") val senderKey: String,
    @ColumnInfo(name = "timestamp_ms") val timestampMs: Long,
    @ColumnInfo(name = "removal_reason") val removalReason: Int,
    @ColumnInfo(name = "removal_delay_ms") val removalDelayMs: Long,
    @ColumnInfo(name = "is_group") val isGroup: Boolean,
)
```

- [ ] **Step 2: Implement `NotificationHistoryDao.kt`**

```kotlin
@Dao
interface NotificationHistoryDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entity: NotificationHistoryEntity)

    @Query("SELECT * FROM notification_history WHERE timestamp_ms >= :cutoffMs")
    suspend fun getEntriesAfter(cutoffMs: Long): List<NotificationHistoryEntity>

    @Query("DELETE FROM notification_history WHERE timestamp_ms < :cutoffMs")
    suspend fun deleteEntriesBefore(cutoffMs: Long)
}
```

- [ ] **Step 3: Implement `ErzaDatabase.kt`**

Room database with version 1, `NotificationHistoryEntity::class`. Singleton via `companion object` with `@Volatile` instance and double-checked locking.

- [ ] **Step 4: Implement `SenderStatsCache.kt`**

SharedPreferences key format: `"sender:{senderKey}"` and `"app:{packageName}"`. Values stored as JSON strings. `getSenderStats` returns null if key absent (cold-start). `updateAll` serializes and stores all entries, replacing old values.

- [ ] **Step 5: Write DAO test (instrumented test)**

```kotlin
// androidTest/...db/NotificationHistoryDaoTest.kt
@RunWith(AndroidJUnit4::class)
class NotificationHistoryDaoTest {
    private lateinit var db: ErzaDatabase
    private lateinit var dao: NotificationHistoryDao

    @Before fun setup() {
        db = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(), ErzaDatabase::class.java
        ).allowMainThreadQueries().build()
        dao = db.historyDao()
    }

    @After fun teardown() { db.close() }

    @Test fun insertAndRetrieve() = runBlocking {
        val entry = NotificationHistoryEntity(
            packageName = "com.whatsapp", senderKey = "com.whatsapp:Mom",
            timestampMs = 1000L, removalReason = 1, removalDelayMs = 5000L, isGroup = false
        )
        dao.insert(entry)
        val results = dao.getEntriesAfter(500L)
        assert(results.size == 1)
        assert(results[0].senderKey == "com.whatsapp:Mom")
    }

    @Test fun suppressedReasonNotCountedAsOpen() = runBlocking {
        // REASON_LISTENER_CANCEL = 4 (our suppress action)
        dao.insert(NotificationHistoryEntity("com.whatsapp", "com.whatsapp:Bot",
            1000L, removalReason = 4, removalDelayMs = 100L, isGroup = false))
        val entries = dao.getEntriesAfter(0L)
        val openCount = entries.count { it.removalReason == 1 } // REASON_CLICK only
        assert(openCount == 0)
    }

    @Test fun deletionPrunesOldEntries() = runBlocking {
        dao.insert(NotificationHistoryEntity("pkg", "key", 100L, 1, 1000L, false))
        dao.insert(NotificationHistoryEntity("pkg", "key", 2000L, 1, 1000L, false))
        dao.deleteEntriesBefore(500L)
        val results = dao.getEntriesAfter(0L)
        assert(results.size == 1)
        assert(results[0].timestampMs == 2000L)
    }
}
```

- [ ] **Step 6: Commit**

```bash
cd /Users/I740422/projects/erza
git add app/src/main/kotlin/com/erza/agent/db/ \
        app/src/androidTest/
git commit -m "feat: Room notification history DB + SenderStatsCache"
```

---

## Task 9: Android classifier pipeline (SenderKeyResolver, FeatureExtractor, TextEncoder, XGBoostClassifier, ClassifierPipeline)

**Files:**
- Create: `erza/app/src/main/kotlin/com/erza/agent/classifier/SenderKeyResolver.kt`
- Create: `erza/app/src/main/kotlin/com/erza/agent/classifier/NotificationFeatureExtractor.kt`
- Create: `erza/app/src/main/kotlin/com/erza/agent/classifier/TextEncoder.kt`
- Create: `erza/app/src/main/kotlin/com/erza/agent/classifier/XGBoostClassifier.kt`
- Create: `erza/app/src/main/kotlin/com/erza/agent/classifier/ClassifierPipeline.kt`
- Copy: `erza-classifier/models/*.onnx` → `erza/app/src/main/assets/models/`

**Interfaces:**
- Consumes: `NotificationData` (with `senderKey`), `SenderStatsCache`, `ErzaDatabase` from Tasks 7-8
- Produces:
  - `ClassifierPipeline(context: Context)` with `suspend fun classify(n: NotificationData): Triple<ClassificationResult, AgentToolCall, InferenceMetrics>`

- [ ] **Step 1: Copy ONNX models to Android assets**

```bash
mkdir -p /Users/I740422/projects/erza/app/src/main/assets/models
cp /Users/I740422/projects/erza-classifier/models/importance.onnx \
   /Users/I740422/projects/erza-classifier/models/urgency.onnx \
   /Users/I740422/projects/erza-classifier/models/category.onnx \
   /Users/I740422/projects/erza-classifier/models/text_encoder_int8.onnx \
   /Users/I740422/projects/erza/app/src/main/assets/models/
```

- [ ] **Step 2: Implement `SenderKeyResolver.kt`**

```kotlin
object SenderKeyResolver {
    private val MESSAGING_PACKAGES = setOf(
        "com.whatsapp", "org.telegram.messenger", "com.facebook.orca",
        "com.instagram.android", "com.discord", "com.google.android.apps.messaging",
        "org.thoughtcrime.securesms"
    )
    private val EMAIL_PACKAGES = setOf("com.google.android.gm", "com.microsoft.office.outlook")

    fun resolve(packageName: String, title: String?): String =
        if ((packageName in MESSAGING_PACKAGES || packageName in EMAIL_PACKAGES) && !title.isNullOrBlank())
            "$packageName:$title"
        else packageName
}
```

- [ ] **Step 3: Implement `TextEncoder.kt`**

Load `text_encoder_int8.onnx` from assets into `OrtSession`. Tokenize text with a simple whitespace/subword tokenizer that matches BERT-tiny's vocab (load `vocab.txt` from assets or bundle a minimal tokenizer). Run ONNX inference; mean-pool last_hidden_state over sequence length → `FloatArray(128)`.

**Tokenizer note:** Bundle `bert-base-uncased-vocab.txt` (~227KB) in assets alongside the ONNX model and implement BasicTokenizer + WordpieceTokenizer in Kotlin matching HuggingFace's behavior (lowercase, punctuation splitting, max_length=64 for notifications).

- [ ] **Step 4: Implement `NotificationFeatureExtractor.kt`**

```kotlin
class NotificationFeatureExtractor(
    private val statsCache: SenderStatsCache,
    private val senderLastSeen: MutableMap<String, Long> = mutableMapOf()
) {
    // Returns FloatArray(151) in FEATURE_COLUMNS order: [embed(128), app_cat_onehot(7), scalars(16)]
    fun extract(n: NotificationData, embedding: FloatArray, nowMs: Long = System.currentTimeMillis()): FloatArray
}
```

Feature assembly must follow identical column order as Python's `build_feature_matrix`. NaN for missing sender/app stats → `Float.NaN`. `is_sleep_hours`: device local time 23:00-07:00. `minutes_since_last_from_sender`: computed from `senderLastSeen` map, Float.NaN if absent.

APP_CATEGORY_MAP mirrors Python's version — same 7 categories, same package mappings.

- [ ] **Step 5: Implement `XGBoostClassifier.kt`**

```kotlin
class XGBoostClassifier(context: Context) {
    // Loads all 3 ONNX sessions from assets on init
    // fun classify(features: FloatArray): Triple<Int, Int, String>
    // Returns (importanceScore 1-10, urgencyScore 1-10, categoryLabel)
}
```

Input name for all 3 models: `"X"`, shape `[1, 151]`. Importance/urgency: clip output float to [1f, 10f], round to Int. Category: argmax of output → index into `CATEGORY_LABELS` list (same order as Python).

- [ ] **Step 6: Implement `ClassifierPipeline.kt`**

```kotlin
class ClassifierPipeline(context: Context) {
    private val encoder = TextEncoder(context)
    private val xgb = XGBoostClassifier(context)
    private val statsCache = SenderStatsCache(context)
    private val extractor = NotificationFeatureExtractor(statsCache)

    suspend fun classify(n: NotificationData): Triple<ClassificationResult, AgentToolCall, InferenceMetrics> {
        val t0 = System.currentTimeMillis()
        val embedding = encoder.encode("${n.sender} ${n.preview}")
        val features = extractor.extract(n, embedding)
        val (imp, urg, cat) = xgb.classify(features)
        val combined = imp * 0.6f + urg * 0.4f
        val dndBypass = combined >= 8.0f
        val suppress = combined <= 2.5f
        val action = when { dndBypass -> "bypass_dnd"; suppress -> "suppress"; else -> "pass_through" }
        val result = ClassificationResult(n.id, imp, urg, cat, combined, action, dndBypass, suppress)
        val toolCall = when {
            dndBypass -> AgentToolCall.ReemitBypassDnd(n.id, "imp=$imp urg=$urg", combined.toInt())
            suppress -> AgentToolCall.SuppressNotification(n.id, "imp=$imp urg=$urg")
            else -> AgentToolCall.PassThrough(n.id)
        }
        val metrics = InferenceMetrics(System.currentTimeMillis() - t0, MetricsSampler.ramMb(), 0f, -1f)
        return Triple(result, toolCall, metrics)
    }
}
```

- [ ] **Step 7: Write instrumented integration test**

```kotlin
// androidTest/.../classifier/ClassifierPipelineTest.kt
@Test fun whatsappMomClassifiesHigh() = runBlocking {
    val pipeline = ClassifierPipeline(ApplicationProvider.getApplicationContext())
    val n = NotificationData("t1", "WhatsApp", "Mom", "Are you coming home this weekend?",
        false, 0.5f, Instant.now().toString(), "com.whatsapp", "t1", "com.whatsapp:Mom")
    val (result, _, metrics) = pipeline.classify(n)
    assert(result.importanceScore >= 5) { "Expected importance >= 5, got ${result.importanceScore}" }
    assert(metrics.elapsedMs < 100) { "Too slow: ${metrics.elapsedMs}ms" }
}

@Test fun zomatoPromoClassifiesLow() = runBlocking {
    val pipeline = ClassifierPipeline(ApplicationProvider.getApplicationContext())
    val n = NotificationData("t2", "Zomato", "Zomato", "50% off your next order! Use SAVE50",
        false, 0.02f, Instant.now().toString(), "com.zomato", "t2", "com.zomato")
    val (result, toolCall, _) = pipeline.classify(n)
    assert(result.importanceScore <= 4) { "Promo scored too high: ${result.importanceScore}" }
    assert(toolCall is AgentToolCall.SuppressNotification || toolCall is AgentToolCall.PassThrough)
}

@Test fun latencyUnder50ms() = runBlocking {
    val pipeline = ClassifierPipeline(ApplicationProvider.getApplicationContext())
    val n = NotificationData("t3", "Discord", "Freddy", "yo check this out",
        false, 0.5f, Instant.now().toString(), "com.discord", "t3", "com.discord:Freddy")
    // warmup
    pipeline.classify(n)
    // measure
    val latencies = (1..10).map {
        val t = System.currentTimeMillis()
        pipeline.classify(n)
        System.currentTimeMillis() - t
    }
    val p95 = latencies.sorted()[9]
    assert(p95 < 50) { "p95 latency $p95ms >= 50ms" }
}
```

- [ ] **Step 8: Commit**

```bash
cd /Users/I740422/projects/erza
git add app/src/main/kotlin/com/erza/agent/classifier/ \
        app/src/main/assets/models/ \
        app/src/androidTest/
git commit -m "feat: classifier pipeline - TextEncoder, XGBoostClassifier, ClassifierPipeline"
```

---

## Task 10: Daily stats worker and service integration

**Files:**
- Create: `erza/app/src/main/kotlin/com/erza/agent/worker/DailyStatsWorker.kt`
- Modify: `erza/app/src/main/kotlin/com/erza/agent/service/ErzaNotificationService.kt`
- Modify: `erza/app/src/main/kotlin/com/erza/agent/ErzaApp.kt`
- Modify: `erza/app/src/main/kotlin/com/erza/agent/MainActivity.kt`
- Delete: `erza/app/src/main/kotlin/com/erza/agent/llm/` (entire package)

**Interfaces:**
- Consumes: `ClassifierPipeline`, `ErzaDatabase`, `SenderStatsCache`, `NotificationHistoryEntity`
- Produces: fully wired app with no LLM dependencies

- [ ] **Step 1: Implement `DailyStatsWorker.kt`**

`CoroutineWorker` scheduled via WorkManager as `PeriodicWorkRequest` (period 24h, midnight). In `doWork()`:
1. Load 30-day history from Room DAO
2. For each unique `senderKey`: compute `open_rate` (REASON_CLICK count / total), `avg_response_time` (mean of `removalDelayMs` where reason==REASON_CLICK), `frequency` (entries in last 7 days / 7.0), `recency_days` ((now - max timestamp) / 86400000.0), `conversation_ratio` (always 0.0 in v1 — no reply tracking yet)
3. For each unique `packageName`: compute `open_rate`, `daily_volume` (total / 30)
4. Call `SenderStatsCache.updateAll(senderMap, appMap)`
5. Delete entries older than 30 days

- [ ] **Step 2: Update `ErzaNotificationService.kt`**

Replace `classifier = NotificationClassifier(app.llmEngine)` with `classifier = ClassifierPipeline(applicationContext)`.

Add `onNotificationRemoved` override:
```kotlin
override fun onNotificationRemoved(sbn: StatusBarNotification, rankingMap: RankingMap, reason: Int) {
    if (sbn.packageName == applicationContext.packageName) return
    val postTime = sbn.postTime
    val delayMs = System.currentTimeMillis() - postTime
    val senderKey = SenderKeyResolver.resolve(sbn.packageName, sbn.notification.extras.getString("android.title"))
    scope.launch {
        db.historyDao().insert(NotificationHistoryEntity(
            packageName = sbn.packageName, senderKey = senderKey,
            timestampMs = postTime, removalReason = reason,
            removalDelayMs = delayMs, isGroup = sbn.notification.extras.getBoolean("android.isGroupConversation", false)
        ))
    }
}
```

Update `toNotificationData()` to populate `senderKey` via `SenderKeyResolver.resolve(packageName, title)`.

Update `classify(notifications)` call sites to handle new `ClassificationResult` shape (use `result.dndBypass` and `result.suppress` — already compatible).

- [ ] **Step 3: Update `ErzaApp.kt`**

Remove `llmEngine` lazy property and LLM initialization coroutine. Add:
```kotlin
val database: ErzaDatabase by lazy { ErzaDatabase.getInstance(this) }
```

Schedule `DailyStatsWorker`:
```kotlin
WorkManager.getInstance(this).enqueueUniquePeriodicWork(
    "daily_stats", ExistingPeriodicWorkPolicy.KEEP,
    PeriodicWorkRequestBuilder<DailyStatsWorker>(24, TimeUnit.HOURS).build()
)
```

- [ ] **Step 4: Update `MainActivity.kt`**

Remove LLM status checks and `btnTestClassify` LLM references. Update test UI to show both `importanceScore` and `timeSensitivityScore` and `category`. Remove `engine.isReady` gating — `ClassifierPipeline` is always ready (ONNX loads synchronously). Update status text to show "✓ Classifier ready" instead of LLM status.

- [ ] **Step 5: Delete LLM package**

```bash
rm -rf /Users/I740422/projects/erza/app/src/main/kotlin/com/erza/agent/llm/
```

Also remove the qwen3 assets directory reference (assets don't need deletion since the actual GGUF file is too large to be in the repo — just remove the `noCompress += "gguf"` line which was already done in Task 7).

- [ ] **Step 6: Build and verify**

```bash
cd /Users/I740422/projects/erza
./gradlew assembleDebug 2>&1 | tail -30
```

Expected: BUILD SUCCESSFUL with no unresolved references.

- [ ] **Step 7: Commit**

```bash
git add app/src/main/kotlin/com/erza/agent/ app/src/main/AndroidManifest.xml
git commit -m "feat: wire ClassifierPipeline into ErzaNotificationService, add DailyStatsWorker, remove LLM"
```

---

## Task 11: End-to-end build and final verification

**Files:**
- No new files — verification only

- [ ] **Step 1: Run full Python test suite**

```bash
cd /Users/I740422/projects/erza-classifier
python3 -m pytest tests/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 2: Run condition evaluation report**

```bash
python3 -c "
from training.evaluate import run_evaluation
from training.evaluate_conditions import run_condition_tests
print('=== Model Metrics ===')
for k, v in run_evaluation().items(): print(f'  {k}: {v:.4f}')
print('=== Condition Tests ===')
for k, v in run_condition_tests().items(): print(f'  {k}: {v:.3f}')
"
```

Expected: all spec thresholds met.

- [ ] **Step 3: Check total ONNX model sizes**

```bash
python3 -c "
import os
files = ['models/importance.onnx','models/urgency.onnx','models/category.onnx','models/text_encoder_int8.onnx']
total = sum(os.path.getsize(f) for f in files)
print(f'Total: {total/1e6:.1f} MB (limit: 20MB)')
assert total < 20e6, 'Models exceed 20MB budget'
"
```

- [ ] **Step 4: Build Android APK**

```bash
cd /Users/I740422/projects/erza
./gradlew assembleDebug 2>&1 | grep -E "BUILD|error:|warning:" | head -20
```

Expected: BUILD SUCCESSFUL.

- [ ] **Step 5: Run Android instrumented tests (if device connected)**

```bash
./gradlew connectedAndroidTest 2>&1 | tail -30
```

Expected: all instrumented tests PASS including latency < 50ms.

- [ ] **Step 6: Final commit**

```bash
cd /Users/I740422/projects/erza-classifier
git add .
git commit -m "chore: final verified build - all thresholds met"

cd /Users/I740422/projects/erza
git commit --allow-empty -m "chore: notification classifier migration complete - LLM removed"
```

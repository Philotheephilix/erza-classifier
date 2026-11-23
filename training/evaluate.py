import numpy as np
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from data.feature_schema import CATEGORY_LABELS

BYPASS_THRESHOLD = 7.5
SUPPRESS_THRESHOLD = 3.5

_IMP_SESS = None
_URG_SESS = None
_CAT_SESS = None


def _load_sessions():
    global _IMP_SESS, _URG_SESS, _CAT_SESS
    if _IMP_SESS is None:
        _IMP_SESS = ort.InferenceSession("models/importance.onnx")
        _URG_SESS = ort.InferenceSession("models/urgency.onnx")
        _CAT_SESS = ort.InferenceSession("models/category.onnx")


def predict(X: np.ndarray):
    _load_sessions()
    X32 = X.astype(np.float32)
    imp_raw = _IMP_SESS.run(None, {"X": X32})[0].flatten()
    urg_raw = _URG_SESS.run(None, {"X": X32})[0].flatten()
    cat_raw = _CAT_SESS.run(None, {"X": X32})[0].flatten()
    imp = np.clip(imp_raw, 1, 10)
    urg = np.clip(urg_raw, 1, 10)
    return imp, urg, cat_raw.astype(np.int32)


def run_evaluation() -> dict:
    X = np.load("data/features_train.npy")
    y_imp = np.load("data/labels_importance.npy")
    y_urg = np.load("data/labels_urgency.npy")
    y_cat = np.load("data/labels_category.npy")

    _, X_val, _, y_imp_val, _, y_urg_val, _, y_cat_val = train_test_split(
        X, y_imp, y_urg, y_cat, test_size=0.2, random_state=42
    )

    imp_pred, urg_pred, cat_pred = predict(X_val)

    importance_mae = float(np.mean(np.abs(imp_pred - y_imp_val)))
    urgency_mae = float(np.mean(np.abs(urg_pred - y_urg_val)))
    category_accuracy = float(np.mean(cat_pred == y_cat_val))

    # Combined scores
    combined_pred = imp_pred * 0.6 + urg_pred * 0.4
    combined_true = y_imp_val * 0.6 + y_urg_val * 0.4

    # bypass_dnd FPR: predicted >= BYPASS_THRESHOLD but true < BYPASS_THRESHOLD
    true_pos_bypass = combined_true >= BYPASS_THRESHOLD
    pred_pos_bypass = combined_pred >= BYPASS_THRESHOLD
    TN_bypass = (~true_pos_bypass) & (~pred_pos_bypass)
    FP_bypass = (~true_pos_bypass) & pred_pos_bypass
    bypass_dnd_fpr = float(FP_bypass.sum() / (FP_bypass.sum() + TN_bypass.sum() + 1e-9))

    # suppress FPR: predicted <= SUPPRESS_THRESHOLD but true > SUPPRESS_THRESHOLD
    true_pos_suppress = combined_true <= SUPPRESS_THRESHOLD
    pred_pos_suppress = combined_pred <= SUPPRESS_THRESHOLD
    TN_suppress = (~true_pos_suppress) & (~pred_pos_suppress)
    FP_suppress = (~true_pos_suppress) & pred_pos_suppress
    suppress_fpr = float(FP_suppress.sum() / (FP_suppress.sum() + TN_suppress.sum() + 1e-9))

    return {
        "importance_mae": importance_mae,
        "urgency_mae": urgency_mae,
        "category_accuracy": category_accuracy,
        "bypass_dnd_fpr": bypass_dnd_fpr,
        "suppress_fpr": suppress_fpr,
    }


def _check(k, v):
    thresholds = {
        "importance_mae": lambda x: x < 1.5,
        "urgency_mae": lambda x: x < 1.5,
        "category_accuracy": lambda x: x > 0.80,
        "bypass_dnd_fpr": lambda x: x < 0.02,
        "suppress_fpr": lambda x: x < 0.05,
    }
    return thresholds.get(k, lambda x: True)(v)


if __name__ == "__main__":
    metrics = run_evaluation()
    for k, v in metrics.items():
        status = "PASS" if _check(k, v) else "FAIL"
        print(f"{status} {k}: {v:.4f}")

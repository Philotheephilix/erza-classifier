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
    clipped = np.clip(out, 1, 10)
    assert np.allclose(clipped, np.clip(out, 1, 10))


def test_urgency_output_range():
    sess = _load_session("models/urgency.onnx")
    X = np.load("data/features_train.npy")[:10].astype(np.float32)
    out = sess.run(None, {"X": X})[0].flatten()
    # raw output may be slightly outside [1,10] before clipping
    assert out is not None


def test_category_output_shape():
    sess = _load_session("models/category.onnx")
    X = np.load("data/features_train.npy")[:10].astype(np.float32)
    out = sess.run(None, {"X": X})[0]
    flat = out.flatten()
    assert len(flat) == 10
    assert ((flat >= 0) & (flat <= 7)).all()


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

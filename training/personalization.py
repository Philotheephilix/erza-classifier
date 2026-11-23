"""On-device personalization layer — linear correction over frozen XGBoost ONNX scores."""
import json
import math
from pathlib import Path
from typing import Union

import numpy as np

APP_CATEGORIES = ["messaging", "social", "shopping", "finance",
                  "productivity", "media", "system"]

_FEATURES = 12
_LR = 0.05
_LAMBDA = 0.01
_CLIP = 2.0


class PersonalizationLayer:
    """Per-app-category linear correction on top of base model scores.

    All-zero weights at init → adjust() is identity → pure base model.
    Convergence: ~15-20 consistent signals per category for reliable shift.
    CLIP=2.0 ensures security alerts (base ≥8) cannot be suppressed by feedback.
    """

    def __init__(self) -> None:
        self.w_imp: dict[str, np.ndarray] = {c: np.zeros(_FEATURES) for c in APP_CATEGORIES}
        self.w_urg: dict[str, np.ndarray] = {c: np.zeros(_FEATURES) for c in APP_CATEGORIES}

    # ── feature vector ────────────────────────────────────────────────────────

    @staticmethod
    def _features(base_imp: float, base_urg: float, is_group: bool,
                  sender_open_rate: float, hour_of_day: int) -> np.ndarray:
        sor = sender_open_rate if not math.isnan(sender_open_rate) else 0.5
        return np.array([
            base_imp / 10.0,
            base_urg / 10.0,
            float(is_group),
            sor,
            hour_of_day / 23.0,
            # interaction terms
            base_imp * base_urg / 100.0,
            float(is_group) * base_imp / 10.0,
            sor * base_imp / 10.0,
            # hour buckets
            float(6 <= hour_of_day < 12),
            float(12 <= hour_of_day < 18),
            float(18 <= hour_of_day < 23),
            float(hour_of_day < 6 or hour_of_day >= 23),
        ], dtype=np.float64)

    # ── inference ─────────────────────────────────────────────────────────────

    def adjust(self, base_imp: float, base_urg: float, app_cat: str,
               is_group: bool, sender_open_rate: float,
               hour_of_day: int) -> tuple[float, float]:
        f = self._features(base_imp, base_urg, is_group, sender_open_rate, hour_of_day)
        delta_imp = float(np.clip(self.w_imp[app_cat] @ f, -_CLIP, _CLIP))
        delta_urg = float(np.clip(self.w_urg[app_cat] @ f, -_CLIP, _CLIP))
        return (
            float(np.clip(base_imp + delta_imp, 1.0, 10.0)),
            float(np.clip(base_urg + delta_urg, 1.0, 10.0)),
        )

    # ── learning ──────────────────────────────────────────────────────────────

    def update(self, base_imp: float, base_urg: float, app_cat: str,
               is_group: bool, sender_open_rate: float,
               hour_of_day: int, outcome: float) -> None:
        """SGD step. outcome: +1.0 = tapped/replied, -1.0 = dismissed."""
        f = self._features(base_imp, base_urg, is_group, sender_open_rate, hour_of_day)
        adj_imp, adj_urg = self.adjust(base_imp, base_urg, app_cat, is_group,
                                       sender_open_rate, hour_of_day)
        adj_combined = adj_imp * 0.6 + adj_urg * 0.4
        target = 8.0 if outcome > 0 else 2.0
        error = adj_combined - target
        grad = error * f
        self.w_imp[app_cat] -= _LR * (grad * 0.6 + _LAMBDA * self.w_imp[app_cat])
        self.w_urg[app_cat] -= _LR * (grad * 0.4 + _LAMBDA * self.w_urg[app_cat])

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        data = {
            cat: {
                "w_imp": self.w_imp[cat].tolist(),
                "w_urg": self.w_urg[cat].tolist(),
            }
            for cat in APP_CATEGORIES
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: Union[str, Path]) -> None:
        data = json.loads(Path(path).read_text())
        for cat in APP_CATEGORIES:
            entry = data[cat]
            wi = np.array(entry["w_imp"], dtype=np.float64)
            wu = np.array(entry["w_urg"], dtype=np.float64)
            if wi.shape != (_FEATURES,) or wu.shape != (_FEATURES,):
                raise ValueError(f"Bad weight shape for {cat}")
            self.w_imp[cat] = wi
            self.w_urg[cat] = wu

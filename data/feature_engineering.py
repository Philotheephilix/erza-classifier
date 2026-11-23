import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from data.feature_schema import FEATURE_COLUMNS

_APP_CATS = ["messaging", "social", "shopping", "finance", "productivity", "media", "system"]
_SCALAR_COLS = FEATURE_COLUMNS[135:]  # indices 135-150 = 16 scalars


class BertTinyEncoder:
    def __init__(self, model_name: str = "prajjwal1/bert-tiny"):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def encode(self, texts: list) -> np.ndarray:
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            out = self._model(**inputs)
        # mean pool over sequence dimension
        hidden = out.last_hidden_state  # (batch, seq, 128)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.numpy().astype(np.float32)


def build_feature_matrix(df: pd.DataFrame, encoder: BertTinyEncoder) -> np.ndarray:
    n = len(df)

    # 1. Text embeddings (128 dims)
    texts = (df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)).tolist()
    embeddings = encoder.encode(texts)  # (n, 128)

    # 2. App category one-hot (7 dims)
    one_hot = np.zeros((n, 7), dtype=np.float32)
    cats = df["app_category"].fillna("messaging").tolist()
    for row_idx, cat in enumerate(cats):
        col_idx = _APP_CATS.index(cat) if cat in _APP_CATS else 0
        one_hot[row_idx, col_idx] = 1.0

    # 3. Scalar features (16 dims) — NaN passes through
    scalar_cols = FEATURE_COLUMNS[135:]  # last 16 after 128 embed + 7 one-hot
    scalars = df[scalar_cols].values.astype(np.float32)  # (n, 16)

    return np.concatenate([embeddings, one_hot, scalars], axis=1)

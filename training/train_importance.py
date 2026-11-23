import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

X = np.load("data/features_train.npy")
y = np.load("data/labels_importance.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    missing=float("nan"),
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

preds = model.predict(X_val)
preds_clipped = np.clip(preds, 1, 10)
mae = float(np.mean(np.abs(preds_clipped - y_val)))
print(f"Val MAE (importance): {mae:.4f}")

model.save_model("models/importance_xgb.json")
print("Saved models/importance_xgb.json")

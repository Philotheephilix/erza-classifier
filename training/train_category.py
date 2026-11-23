import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

X = np.load("data/features_train.npy")
y = np.load("data/labels_category.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=8,
    missing=float("nan"),
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="mlogloss",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

preds = model.predict(X_val)
acc = accuracy_score(y_val, preds)
print(f"Val accuracy (category): {acc:.4f}")

model.save_model("models/category_xgb.json")
print("Saved models/category_xgb.json")

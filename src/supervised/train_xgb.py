import numpy as np
import xgboost as xgb
import joblib
import os

def load_balanced_train():
    path = os.path.join("data", "processed", "awid_train_balanced.npz")
    return np.load(path)

print("Loading balanced training dataset...")
d = load_balanced_train()
X_train, y_train = d["X"], d["y"]
print("Train shape:", X_train.shape, y_train.shape)

dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    "eta": 0.1,
    "max_depth": 12,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

print("Training XGBoost model...")
bst = xgb.train(params, dtrain, num_boost_round=300)
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "xgb_awid.pkl")
joblib.dump(bst, model_path)
print(f"✔ Model saved to: {model_path}")

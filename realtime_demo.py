import numpy as np
import joblib
import xgboost as xgb
import random
import os

THR = 0.001

data_path = os.path.join("data", "processed", "awid_test_balanced.npz")
d = np.load(data_path)
X, y = d["X"], d["y"]

model_path = os.path.join("models", "xgb_awid.pkl")
model = joblib.load(model_path)

print("Live Intrusion Detection Simulation\n")

for i in range(10):
    idx = random.randrange(len(X))
    x = X[idx]
    true = y[idx]

    dm = xgb.DMatrix(x.reshape(1, -1))
    prob = model.predict(dm)[0]
    pred = int(prob >= THR)

    label_str = "ATTACK" if pred == 1 else "NORMAL"
    true_str  = "ATTACK" if true == 1 else "NORMAL"
    status = "OK" if pred == true else "MISS"

    print(
        f"Sample {i+1:02d}: p(attack)={prob:.4f} | "
        f"Pred={label_str:7s} | True={true_str:7s} | {status}"
    )

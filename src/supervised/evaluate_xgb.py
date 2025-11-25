import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import os

def load_balanced_test():
    path = os.path.join("data", "processed", "awid_test_balanced.npz")
    return np.load(path)

print("Loading balanced test set...")
d = load_balanced_test()
X_test, y_test = d["X"], d["y"]
print("Test shape:", X_test.shape, y_test.shape)

model_path = os.path.join("models", "xgb_awid.pkl")
print("Loading model:", model_path)
bst = joblib.load(model_path)

dtest = xgb.DMatrix(X_test)
probs = bst.predict(dtest)

thr = 0.001
y_pred = (probs >= thr).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn + 1e-9)
fnr = fn / (fn + tp + 1e-9)

print("\n=========== FINAL XGBOOST IDS REPORT ===========")
print(f"Threshold               : {thr}")
print(f"Accuracy                : {acc*100:.2f}%")
print(f"Precision               : {prec*100:.2f}%")
print(f"Recall                  : {rec*100:.2f}%")
print(f"F1 Score                : {f1*100:.2f}%")
print("------------------------------------------------")
print(f"False Positive Rate     : {fpr*100:.2f}%")
print(f"False Negative Rate     : {fnr*100:.2f}%")
print("------------------------------------------------")
print("Confusion Matrix:")
print(cm)
print("================================================\n")

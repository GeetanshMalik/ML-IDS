import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os

data_path = os.path.join("data", "processed", "awid_test_balanced.npz")
d = np.load(data_path)
X_test, y_test = d["X"], d["y"]

model_path = os.path.join("models", "xgb_awid.pkl")
bst = joblib.load(model_path)

dtest = xgb.DMatrix(X_test)
probs = bst.predict(dtest)

thr = 0.001
y_pred = (probs >= thr).astype(int)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (XGBoost, thr=0.001)")
plt.colorbar()
plt.xticks([0,1], ["Pred 0","Pred 1"])
plt.yticks([0,1], ["True 0","True 1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost on AWID")
plt.legend()
plt.tight_layout()
plt.show()

prec, rec, _ = precision_recall_curve(y_test, probs)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - XGBoost on AWID")
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import os

# paths relative to project root
train_csv = os.path.join("data", "raw", "AWID-CLS-R-Trn.csv")
test_csv  = os.path.join("data", "raw", "AWID-CLS-R-Tst.csv")

print("Loading raw AWID CSV files...")
df_train = pd.read_csv(train_csv, low_memory=False)
df_test  = pd.read_csv(test_csv, low_memory=False)

df_train = df_train.replace("?", np.nan).ffill().bfill()
df_test  = df_test.replace("?", np.nan).ffill().bfill()

print("Converting labels to binary...")
y_train_raw = df_train["class"].astype(str)
y_test_raw  = df_test["class"].astype(str)

attack_labels = [
    "deauthentication", "disassociation", "authentication",
    "probe", "fake_ap", "injection", "arp", "evil_twin"
]

y_train = y_train_raw.apply(lambda x: 1 if x in attack_labels else 0)
y_test  = y_test_raw.apply(lambda x: 1 if x in attack_labels else 0)

X_train = df_train.drop(columns=["class"])
X_test  = df_test.drop(columns=["class"])

print("Encoding categorical columns...")
for col in X_train.columns:
    if X_train[col].dtype == object:
        X_train[col] = X_train[col].astype("category").cat.codes
        X_test[col]  = X_test[col].astype("category").cat.codes

X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)

os.makedirs(os.path.join("data", "processed"), exist_ok=True)

np.savez(os.path.join("data", "processed", "awid_train.npz"),
         X=X_train.values, y=y_train.values)
np.savez(os.path.join("data", "processed", "awid_test.npz"),
         X=X_test.values,  y=y_test.values)

print("Preprocessing complete.")
print("Saved: data/processed/awid_train.npz, awid_test.npz")

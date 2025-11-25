import numpy as np
import os

print("Loading preprocessed datasets...")
train = np.load(os.path.join("data", "processed", "awid_train.npz"))
test  = np.load(os.path.join("data", "processed", "awid_test.npz"))

X_train, y_train = train["X"], train["y"]
X_test, y_test   = test["X"],  test["y"]

print("Original TRAIN:", X_train.shape, y_train.shape)
print("Original TEST :", X_test.shape,  y_test.shape)

def balance_split(X, y):
    normal_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    np.random.shuffle(normal_idx)
    np.random.shuffle(attack_idx)
    m = min(len(normal_idx), len(attack_idx))
    normal_sample = normal_idx[:m]
    attack_sample = attack_idx[:m]
    Xb = np.concatenate([X[normal_sample], X[attack_sample]])
    yb = np.concatenate([y[normal_sample], y[attack_sample]])
    idx = np.arange(len(yb))
    np.random.shuffle(idx)
    return Xb[idx], yb[idx]

print("Balancing TRAIN...")
X_train_bal, y_train_bal = balance_split(X_train, y_train)

print("Balancing TEST...")
X_test_bal, y_test_bal = balance_split(X_test, y_test)

os.makedirs(os.path.join("data", "processed"), exist_ok=True)

np.savez(os.path.join("data", "processed", "awid_train_balanced.npz"),
         X=X_train_bal, y=y_train_bal)
np.savez(os.path.join("data", "processed", "awid_test_balanced.npz"),
         X=X_test_bal,  y=y_test_bal)

print("Balanced TRAIN:", X_train_bal.shape, y_train_bal.shape)
print("Balanced TEST :", X_test_bal.shape,  y_test_bal.shape)
print("Saved balanced .npz files.")

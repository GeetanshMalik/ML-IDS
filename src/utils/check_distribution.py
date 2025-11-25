import numpy as np

data_train = np.load("data/processed/awid_train.npz")
data_test = np.load("data/processed/awid_test.npz")

y_train = data_train["y"]
y_test = data_test["y"]

print("TRAIN:")
print("Normal:", (y_train == 0).sum())
print("Attack:", (y_train == 1).sum())

print("\nTEST:")
print("Normal:", (y_test == 0).sum())
print("Attack:", (y_test == 1).sum())

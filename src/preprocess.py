import numpy as np
import pandas as pd

from utils import load_data, load_gt

if __name__ == "__main__":    
    X_train = load_data("data/raw/train_data/train_data")
    y_train = load_gt("data/raw/train_data/train_gt.csv")
    X_test = load_data("data/raw/test_data")

    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target data shape: {y_train.shape}")

    np.savetxt("data/interim/X_train.txt", X_train, delimiter=" ", fmt='%f')
    np.savetxt("data/interim/X_test.txt", X_test, delimiter=" ", fmt='%f')
    np.savetxt("data/interim/y_train.txt", y_train, delimiter=" ", fmt='%f')
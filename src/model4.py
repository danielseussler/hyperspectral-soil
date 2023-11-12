import numpy as np
import pandas as pd

from mbtr.mbtr import MBT
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from utils import create_submission, scale_by_mean

def run_inference(X, y, newdata):
       
    xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.1, random_state=0)
    train, yval = scale_by_mean(ytrain, yval)
    y = y / np.mean(y, axis=0)
    
    mbt = MBT(
        n_boosts=20,  
        min_leaf=100,
        lambda_weights=1e-3,
        early_stopping_rounds=5,
        val_ratio=0,
        verbose=1
    )
    
    _ = mbt.fit(xtrain, ytrain)
    
    # Refit with best iteration on complete data
    # TODO
    
    ypred = mbt.predict(newdata)
    
    rmse = mean_squared_error(yval, mbt.predict(xval), squared=True)
    print(f"RMSE: {rmse}")
    
    return mbt, ypred


def main():
    
    # Load data
    X_train = np.loadtxt("../data/interim/X_train.txt", delimiter=" ")
    y_train = np.loadtxt("../data/interim/y_train.txt", delimiter=" ")
    X_test = np.loadtxt("../data/interim/X_test.txt", delimiter=" ")
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Run model
    mbt, ypred = run_inference(X_train, y_train, X_test)
    
    print(ypred.shape)
    
    # Create submission file
    create_submission(ypred)
    

if __name__ == "__main__":
    main()

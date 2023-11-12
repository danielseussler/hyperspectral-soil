import numpy as np
import pandas as pd

from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import MultivariateNormal

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils import scale_by_mean, create_submission

def run_inference(X, y, newdata):
    dist = MultivariateNormal(k=4)
    base = DecisionTreeRegressor(
        criterion="friedman_mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_leaf_nodes=31
    )
        
    xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.1, random_state=0)
    ytrain, yval = scale_by_mean(ytrain, yval)
    
    ngb = NGBRegressor(
        Dist=dist,
        Base=base,
        n_estimators=500, 
        learning_rate=0.02,
        minibatch_frac=1,
        col_sample=1,
        random_state=394134,
    )
    
    _ = ngb.fit(xtrain, ytrain, xval, yval, early_stopping_rounds=100)
    best_iter = ngb.best_val_loss_itr
    
    # Refit with best iteration on complete data
    ngb = NGBRegressor(
        Dist=dist,
        Base=base,
        n_estimators=best_iter, 
        learning_rate=0.02,
        minibatch_frac=1,
        col_sample=1,
        random_state=394134,
    )
    
    y = y / np.mean(y, axis=0)
    _ = ngb.fit(X, y)
    ydist = ngb.pred_dist(newdata, max_iter=ngb.best_val_loss_itr)
    
    return ngb, ydist
    

def main():
    
    # Load data
    X_train = np.loadtxt("../data/interim/X_train.txt", delimiter=" ")
    y_train = np.loadtxt("../data/interim/y_train.txt", delimiter=" ")
    X_test = np.loadtxt("../data/interim/X_test.txt", delimiter=" ")
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Run model
    ngb, ydist = run_inference(X_train, y_train, X_test)
    
    print(ngb.best_val_loss_itr)
    print(ydist.mean().shape)
    
    # Create submission file
    create_submission(ydist)
    

if __name__ == "__main__":
    main()

import argparse
import os

import numpy as np
import optuna
import pandas as pd
import yaml
from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from utils import create_submission, scale_by_mean


def objective(trial, xtrain, xval, ytrain, yval):
    dist = MultivariateNormal(k=4)
    base = DecisionTreeRegressor(
        criterion="friedman_mse",
        max_depth=None,
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 2, 128),
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    ngb = NGBRegressor(
        Dist=dist,
        Base=base,
        n_estimators=trial.suggest_int("n_estimators", 1, 300), 
        learning_rate=0.01,
        minibatch_frac=trial.suggest_float("minibatch_frac", 0.1, 1),
        col_sample=trial.suggest_float("col_sample", 0.1, 0.6),
        natural_gradient=True,
        random_state=0,
        verbose=False
    )
    
    _ = ngb.fit(xtrain, ytrain)
    ydist = ngb.pred_dist(xval, max_iter=ngb.best_val_loss_itr)
    nll = -ydist.logpdf(yval).mean()  
      
    return nll


def tune_model(X, y):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    # Split data
    xtrain, xval, ytrain, yval = train_test_split(X, y, train_size=0.8, random_state=0)
    ytrain, yval = scale_by_mean(ytrain, yval)
    
    func = lambda trial: objective(trial, xtrain, xval, ytrain, yval)
    
    # Optimize
    sampler = optuna.samplers.CmaEsSampler(seed=0, with_margin=True, restart_strategy='bipop')
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(func, n_trials=200, timeout=None, n_jobs=2, show_progress_bar=True)

    # Print and save results
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print('Config: ', study.best_params)
    
    file_path = 'conf/tuned_ngb.yaml'
    with open(file_path, 'w') as file:
        yaml.dump(study.best_params, file)
    print(f"Optuna config saved to: {file_path}")

    # Save figures
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(file='../plot_optimization_history.png', format='png')
    
    fig = optuna.visualization.plot_contour(study)
    fig.write_image(file='../plot_contour.png', format='png')
    
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(file='../plot_parallel_coordinate.png', format='png')
    
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(file='../plot_slice.png', format='png')


def main(args):
    
    # Load data
    X = np.loadtxt("../data/interim/X_train.txt", delimiter=" ")
    y = np.loadtxt("../data/interim/y_train.txt", delimiter=" ")
    newdata = np.loadtxt("../data/interim/X_test.txt", delimiter=" ")
    
    print(f"Train data shape: {X.shape}")
    print(f"Test data shape: {y.shape}")
    print(f"Target data shape: {newdata.shape}")
    
    # Tune model
    if args.tune is True: 
       tune_model(X, y)
    
    # Refit model and create predictions for the test set
    file_path = 'conf/tuned_ngb.yaml'
    with open(file_path, 'r') as file:
        tuned = yaml.safe_load(file)
    
    # Run model
    dist = MultivariateNormal(k=4)
    base = DecisionTreeRegressor(max_depth=None, max_leaf_nodes=tuned['max_leaf_nodes'])
    ngb = NGBRegressor(
        Dist=dist,
        Base=base,
        n_estimators=tuned['n_estimators'], 
        minibatch_frac=tuned['minibatch_frac'],
        col_sample=tuned['col_sample'],
        random_state=0,
    )
    
    y = y / np.mean(y, axis=0)
    _ = ngb.fit(X, y)
    
    ydist = ngb.pred_dist(newdata)
    ypred = ngb.predict(newdata)
    
    # Create submission file
    create_submission(ypred)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimating Soil Parameters from Hyperspectral Images")
    parser.add_argument("--tune", default=False, type=bool)
    args = parser.parse_args()
    
    main(args)
    
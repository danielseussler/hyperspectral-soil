import argparse

import numpy as np
import optuna
import pandas as pd
import yaml
from mbtr.mbtr import MBT
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from utils import create_submission, scale_by_mean


def objective(trial, xtrain, xval, ytrain, yval):
    
    mbt = MBT(
        n_boosts=trial.suggest_int("n_boosts", 10, 200),  
        min_leaf=trial.suggest_int("min_leaf", 10, 200),
        n_q=trial.suggest_int("n_q", 5, 50),
        lambda_weights=trial.suggest_float("lambda_weights", 1e-4, 1, log=True),
        verbose=1
    )
    
    _ = mbt.fit(xtrain, ytrain)
    ypred = mbt.predict(xval)
    rmse = mean_squared_error(yval, ypred)
    
    return rmse


def tune_model(X, y):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    # Split data
    xtrain, xval, ytrain, yval = train_test_split(X, y, train_size=0.75, random_state=0)
    ytrain, yval = scale_by_mean(ytrain, yval)
    
    func = lambda trial: objective(trial, xtrain, xval, ytrain, yval)
    
    # Optimize
    sampler = optuna.samplers.CmaEsSampler(seed=0, with_margin=True, restart_strategy='bipop')
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(func, n_trials=10, timeout=None, n_jobs=2, show_progress_bar=True)

    # Print and save results
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print('Config: ', study.best_params)
    
    file_path = 'conf/tuned_mbt.yaml'
    with open(file_path, 'w') as file:
        yaml.dump(study.best_params, file)
    print(f"Optuna config saved to: {file_path}")

    # Save figures
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(file='../plot_optimization_history')
    
    fig = optuna.visualization.plot_contour(study)
    fig.write_image(file='../plot_contour')
    
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(file='../plot_parallel_coordinate')
    
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(file='../plot_slice')


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
    file_path = 'conf/tuned_mbt.yaml'
    with open(file_path, 'r') as file:
        tuned = yaml.safe_load(file)
    
    # Run model
    mbt = MBT(
        n_boosts=tuned['n_boosts'],  
        min_leaf=tuned['min_leaf'],
        n_q=tuned['n_q'],
        lambda_weights=tuned['lambda_weights'],
        verbose=1
    )
    
    y = y / np.mean(y, axis=0)
    _ = mbt.fit(X, y)
    
    ypred = mbt.predict(newdata)
    
    # Create submission file
    create_submission(ypred)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimating Soil Parameters from Hyperspectral Images")
    parser.add_argument("--tune", default=False, type=bool)
    args = parser.parse_args()
    
    main(args)
    
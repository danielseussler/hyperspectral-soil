import optuna
import yaml


# Create an Optuna study and define some parameters
def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return (x - 2)**2 + (y + 3)**2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

# Get the configuration as a dictionary
study_config = study.best_params

# Define the file path
file_path = 'conf/optuna_config.yaml'

# Save the configuration as a YAML file
with open(file_path, 'w') as file:
    yaml.dump(study_config, file)

# Display the resulting file path
print(f"Optuna config saved to: {file_path}")

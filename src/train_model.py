import os
import pickle
import yaml
import optuna
import mlflow
from optuna_integration.mlflow import MLflowCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

with open("params.yaml") as f:
    params = yaml.safe_load(f)

TARGET = params["data"]["target_column"]
train_df = pd.read_csv("data/train.csv")
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])
mlflc = MLflowCallback(
    tracking_uri=params["mlflow"]["tracking_uri"],
    metric_name="cv_f1_weighted",
)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators",
        params["model"]["n_estimators_min"],
        params["model"]["n_estimators_max"])
    max_depth = trial.suggest_int("max_depth",
        params["model"]["max_depth_min"],
        params["model"]["max_depth_max"])
    min_samples_split = trial.suggest_int("min_samples_split",
        params["model"]["min_samples_split_min"],
        params["model"]["min_samples_split_max"])
    min_samples_leaf = trial.suggest_int("min_samples_leaf",
        params["model"]["min_samples_leaf_min"],
        params["model"]["min_samples_leaf_max"])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=params["data"]["random_state"],
    )
    scores = cross_val_score(
        clf, X_train, y_train,
        cv=params["optuna"]["cv_folds"],
        scoring="f1_weighted",
    )
    return scores.mean()

study = optuna.create_study(
    direction="maximize",
    study_name=params["optuna"]["study_name"],
)
study.optimize(objective, n_trials=params["optuna"]["n_trials"],
               callbacks=[mlflc])

best_clf = RandomForestClassifier(
    **study.best_params,
    random_state=params["data"]["random_state"],
)
best_clf.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(best_clf, f)

import json
import pickle
import yaml
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models.signature import infer_signature

with open("params.yaml") as f:
    params = yaml.safe_load(f)

TARGET = params["data"]["target_column"]
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1w = f1_score(y_test, y_pred, average="weighted")

metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1w, 4)}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])
with mlflow.start_run(run_name="best-model"):
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(model, "model", signature=signature)

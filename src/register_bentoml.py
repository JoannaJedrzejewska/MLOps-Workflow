import pickle
import bentoml
import mlflow
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

model_info = bentoml.sklearn.save_model(
    "penguins_classifier", model,
    metadata={"dataset": "palmer_penguins_42585"},
)
encoder_info = bentoml.sklearn.save_model(
    "penguins_encoder", encoder,
    metadata={"cat_cols": ["island", "sex"]},
)

mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])
with mlflow.start_run(run_name="bentoml-registration"):
    mlflow.log_param("bentoml_model_tag", str(model_info.tag))
    mlflow.log_param("bentoml_encoder_tag", str(encoder_info.tag))

import os
import pickle
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

os.makedirs("models", exist_ok=True)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/penguins.csv").dropna()

TARGET = params["data"]["target_column"]
CAT_COLS = ["island", "sex"]
NUM_COLS = ["culmen_length_mm", "culmen_depth_mm",
            "flipper_length_mm", "body_mass_g", "bill_body_ratio"]

X = df[CAT_COLS + NUM_COLS]
y = df[TARGET]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(X[CAT_COLS])
cat_names = encoder.get_feature_names_out(CAT_COLS)

X_enc = pd.DataFrame(X_cat, columns=cat_names, index=X.index)
X_enc[NUM_COLS] = X[NUM_COLS].values

with open("models/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y,
    test_size=params["data"]["test_size"],
    random_state=params["data"]["random_state"],
    stratify=y
)

train_df = X_train.copy(); train_df[TARGET] = y_train.values
test_df = X_test.copy();   test_df[TARGET] = y_test.values

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

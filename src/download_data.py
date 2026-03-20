import os
import pandas as pd
from sklearn.datasets import fetch_openml

os.makedirs("data", exist_ok=True)

dataset = fetch_openml(data_id=42585, as_frame=True, parser="auto")
df = dataset.frame
df["bill_body_ratio"] = (
    df["culmen_length_mm"].astype(float) * df["culmen_depth_mm"].astype(float)
) / df["body_mass_g"].astype(float)

df.to_csv("data/penguins.csv", index=False)
print(f"saved {len(df)} records to  data/penguins.csv")

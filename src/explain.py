import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

Path("reports/explain_figures").mkdir(parents=True, exist_ok=True)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

TARGET = "species"
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop(columns=[TARGET])

X_np = X_test.values

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_np)

if isinstance(shap_values, list):
    print("len(shap_values):", len(shap_values))
    print("shap_values[0].shape:", shap_values[0].shape)
    print("X_np.shape:", X_np.shape)

    shap.summary_plot(shap_values, X_np, feature_names=X_test.columns,
                      class_names=model.classes_, show=False)
    plt.savefig("reports/explain_figures/shap_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    classes = list(model.classes_)
    if "Gentoo" in classes:
        class_idx = classes.index("Gentoo")
    else:
        class_idx = 0

    shap.summary_plot(shap_values[class_idx], X_np,
                      feature_names=X_test.columns, show=False)
    plt.title(f"SHAP beeswarm – class {classes[class_idx]}")
    plt.savefig("reports/explain_figures/shap_beeswarm_gentoo.png",
                dpi=150, bbox_inches="tight")
    plt.close()

else:
    print("shap_values.shape:", shap_values.shape)
    print("X_np.shape:", X_np.shape)

    shap.summary_plot(shap_values, X_np,
                      feature_names=X_test.columns, show=False)
    plt.savefig("reports/explain_figures/shap_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()

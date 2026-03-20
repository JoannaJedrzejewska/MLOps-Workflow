import pickle, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from pathlib import Path

Path("reports/learning_curve_figures").mkdir(parents=True, exist_ok=True)

with open("models/model.pkl", "rb") as f: model = pickle.load(f)
train_df = pd.read_csv("data/train.csv")
X = train_df.drop(columns=["species"]); y = train_df["species"]

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring="f1_weighted",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), label="Train F1", marker="o")
ax.fill_between(train_sizes,
                train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1), alpha=0.2)
ax.plot(train_sizes, val_scores.mean(axis=1), label="Val F1", marker="s")
ax.fill_between(train_sizes,
                val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1), alpha=0.2)
ax.set_xlabel("Size of training sample")
ax.set_ylabel("F1 weighted")
ax.set_title("`Learning Curve` — RandomForest")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig("reports/learning_curve_figures/learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()

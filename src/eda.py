import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

Path("reports/eda_figures").mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/penguins.csv").dropna()

SPECIES_COLORS = {"Adelie": "#1f77b4", "Chinstrap": "#ff7f0e", "Gentoo": "#2ca02c"}

sns.pairplot(df, hue="species", palette=SPECIES_COLORS,
             vars=["culmen_length_mm", "culmen_depth_mm",
                   "flipper_length_mm", "body_mass_g"])
plt.savefig("reports/eda_figures/pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for species, grp in df.groupby("species"):
    ax.hist(grp["bill_body_ratio"], alpha=0.6, label=species,
            color=SPECIES_COLORS[species], bins=20)
ax.set_xlabel("Bill-Body Ratio"); ax.set_ylabel("Count")
ax.set_title("Bill-Body Ratio for spiecies")
ax.legend()
fig.savefig("reports/eda_figures/bill_body_ratio_dist.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
num_cols = ["culmen_length_mm", "culmen_depth_mm",
            "flipper_length_mm", "body_mass_g", "bill_body_ratio"]
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax)
ax.set_title("Correlations")
fig.savefig("reports/eda_figures/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flatten(), num_cols[:4]):
    sns.boxplot(data=df, x="species", y=col, hue="sex",
                palette="Set2", ax=ax)
    ax.set_title(col)
plt.tight_layout()
fig.savefig("reports/eda_figures/boxplots_species_sex.png", dpi=150, bbox_inches="tight")
plt.close()


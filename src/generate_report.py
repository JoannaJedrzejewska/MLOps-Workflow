from pathlib import Path
import pandas as pd

Path("reports").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/penguins.csv").dropna()

info = []
info.append("# Palmer Penguins EDA Report\n")
info.append(f"- Rows (after dropna): {len(df)}\n")
info.append(f"- Columns: {', '.join(df.columns)}\n")

numeric_desc = df.describe().to_markdown()

species_counts = df["species"].value_counts().to_markdown()

island_counts = df["island"].value_counts().to_markdown()

report_md = "\n".join([
    *info,
    "## Numerical Features Summary\n",
    numeric_desc,
    "\n## Species Counts\n",
    species_counts,
    "\n## Island Counts\n",
    island_counts,
])

with open("reports/eda_report.md", "w") as f:
    f.write(report_md)


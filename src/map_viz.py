import folium, pandas as pd
from pathlib import Path

Path("reports/map_figures").mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/penguins.csv").dropna()

ISLANDS = {
    "Torgersen": (-64.766, -64.083),
    "Biscoe":    (-65.433, -65.500),
    "Dream":     (-64.733, -64.233),
}
COLORS = {"Adelie": "blue", "Chinstrap": "orange", "Gentoo": "green"}

m = folium.Map(location=[-64.9, -64.5], zoom_start=9,
               tiles="CartoDB dark_matter")

counts = df.groupby(["island", "species"]).size().reset_index(name="count")

for _, row in counts.iterrows():
    lat, lon = ISLANDS[row["island"]]
    folium.CircleMarker(
        location=[lat + (hash(row["species"]) % 5) * 0.02, lon],
        radius=row["count"] / 5,
        color=COLORS[row["species"]],
        fill=True, fill_opacity=0.7,
        tooltip=f"{row['island']} | {row['species']}: {row['count']}"
    ).add_to(m)

m.save("reports/map_figures/islands_map.html")

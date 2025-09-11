# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 09:22:06 2025

@author: sande
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
#import folium
#from selenium import webdriver
#from PIL import Image
#import time
import re
import matplotlib.pyplot as plt
import contextily as ctx
# === 1. Pad naar je outputbestand ===
output_path = r"C:\CoSEM\24-25\Master Thesis\ONLT\SQ-4 modelbouw\ONLT model\Input-output model\Output_RA2\output_data_RA2.xlsx"

# === 2. Nodes en edges inladen ===
nodes_df = pd.read_excel(output_path, sheet_name="bestG_nodes")
edges_df = pd.read_excel(output_path, sheet_name="bestG_edges")

print("Nodes kolommen:", nodes_df.columns)
print("Edges kolommen:", edges_df.columns)
#%%
# === 3. Parse coord kolom naar X en Y ===
def parse_coord(coord_str):
    match = re.match(r"\(?\s*([0-9.+-]+)\s*,\s*([0-9.+-]+)\s*\)?", str(coord_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

nodes_df[['x', 'y']] = nodes_df['coord'].apply(lambda c: pd.Series(parse_coord(c)))

# === 4. Zet nodes om naar GeoDataFrame ===
utm_epsg = 32631  # UTM zone 31N
nodes_gdf = gpd.GeoDataFrame(
    nodes_df,
    geometry=gpd.points_from_xy(nodes_df['x'], nodes_df['y']),
    crs=f"EPSG:{utm_epsg}"
)

# === 5. Maak geometrie voor edges ===
edge_geoms = []
for _, row in edges_df.iterrows():
    try:
        from_geom = nodes_gdf.loc[nodes_gdf['node'] == row['node1'], 'geometry'].values[0]
        to_geom = nodes_gdf.loc[nodes_gdf['node'] == row['node2'], 'geometry'].values[0]
        edge_geoms.append(LineString([from_geom, to_geom]))
    except IndexError:
        edge_geoms.append(None)

edges_gdf = gpd.GeoDataFrame(edges_df, geometry=edge_geoms, crs=f"EPSG:{utm_epsg}")

# === 6. Zet alles naar Web Mercator (voor OSM) ===
nodes_gdf = nodes_gdf.to_crs(epsg=3857)
edges_gdf = edges_gdf.to_crs(epsg=3857)

# === 7. Categoriekleuren edges ===
edge_colors = {
    "Pipeline": "green",
    "Road": "red",
    "Waterway": "blue"
}
edges_gdf["plot_color"] = edges_gdf["category"].map(edge_colors).fillna("grey")

# === 8. Plotten ===
fig, ax = plt.subplots(figsize=(12, 12))

# Plot edges per category
for cat, color in edge_colors.items():
    edges_gdf.loc[edges_gdf["category"] == cat].plot(ax=ax, linewidth=2, color=color, label=cat)

# Plot overige edges
edges_gdf.loc[~edges_gdf["category"].isin(edge_colors.keys())].plot(
    ax=ax, linewidth=2, color="grey", label="Other"
)

# === Node styling ===
nodes_gdf.loc[nodes_gdf["stamp"] == "terminal"].plot(
    ax=ax, color="orange", markersize=60, marker="^", label="Industry (terminal)"
)
nodes_gdf.loc[nodes_gdf["stamp"] == "existing"].plot(
    ax=ax, color="darkblue", markersize=40, marker="o", label="Existing"
)
nodes_gdf.loc[nodes_gdf["stamp"] == "split"].plot(
    ax=ax, color="limegreen", markersize=40, marker="o", label="Split"
)

# === OSM achtergrond ===
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)

ax.set_axis_off()
ax.legend()
#plt.title("Netwerk met OSM achtergrond", fontsize=14)
plt.tight_layout()
plt.show()
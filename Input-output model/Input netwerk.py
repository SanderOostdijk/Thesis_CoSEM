# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:34:02 2025

@author: sande
"""

import geopandas as gpd
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString

# === Instellingen ===
geojson_path = 'H2_netwerk for model.geojson'  # pad naar je GeoJSON
output_excel = 'input_data.xlsx'
default_capacity = 9999  # zeer hoge capaciteit zodat het als "gratis" telt

# === GeoJSON inlezen ===
gdf = gpd.read_file(geojson_path)

# === Lijsten voor bestaande verbindingen en routing network ===
existing_connections = []
routing_edges = []

# === Itereer over alle geometrieën ===
for _, row in gdf.iterrows():
    geom = row.geometry

    # Controleer of het een MultiLineString is
    if isinstance(geom, MultiLineString):
        lines = geom.geoms  # lijst van LineStrings
    elif isinstance(geom, LineString):
        lines = [geom]  # maak lijst van 1
    else:
        continue  # sla over als het geen lijn is

    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]

            # Voor existing_connections
            existing_connections.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'capacity': default_capacity
            })

            # Voor routing_network
            routing_edges.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
# === DataFrames aanmaken ===
df_existing = pd.DataFrame(existing_connections)
df_routing = pd.DataFrame(routing_edges)

# === Excel-bestand schrijven ===
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df_existing.to_excel(writer, sheet_name='existing_connections', index=False)
    df_routing.to_excel(writer, sheet_name='routing_network', index=False)

print(f" Bestand '{output_excel}' succesvol aangemaakt.")


# Maak een lege graaf
G = nx.Graph()

# Voeg edges toe uit je bestaande netwerk
for _, row in df_existing.iterrows():
    node1 = (row['x1'], row['y1'])
    node2 = (row['x2'], row['y2'])

    # Voeg de edge toe met capaciteit als attribuut
    G.add_edge(node1, node2, capacity=row['capacity'])
    
# Posities van nodes worden automatisch bepaald op basis van hun coördinaten
pos = {node: node for node in G.nodes}

# Plot de graaf
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=False, node_size=10, node_color='blue', edge_color='gray')
plt.title("Bestaand waterstofnetwerk")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
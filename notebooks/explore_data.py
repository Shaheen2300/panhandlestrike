import pandas as pd
import geopandas as gpd

# Load the combined table you already built
df = gpd.read_file("data/processed/final/blockgroup_combined_priority.geojson")

print(df.shape)          # how many rows, how many columns
print(df.columns.tolist())  # what are the columns actually called
print(df.head(10))       # show the first 10 rows, for real
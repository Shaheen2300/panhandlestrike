import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your real combined data
df = gpd.read_file("data/processed/final/blockgroup_combined_priority.geojson")

# Pick features (the inputs) and target (what we're predicting)
features = ["P1_001N", "drive_min_to_shelter", "flash_per_km2"]
target = "has_confirmed_anomaly"

X = df[features]
y = df[target]

# Split by county so nearby block groups don't leak into both train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["COUNTY_NAME"])
X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# Build and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# See how well it did
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# See what mattered most
importances = pd.Series(model.feature_importances_, index=features)
print(importances.sort_values(ascending=False))
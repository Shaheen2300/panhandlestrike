"""Small test-window pull from the XWeather lightning endpoint.

Queries a bounding box covering the 5-county Florida Panhandle study area
(Escambia, Santa Rosa, Okaloosa, Walton, Bay) for the last 7 days only,
to inspect record structure and volume before designing the full
2016-present historical batch pull.
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("XWEATHER_CLIENT_ID")
CLIENT_SECRET = os.getenv("XWEATHER_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Missing XWEATHER_CLIENT_ID or XWEATHER_CLIENT_SECRET in .env")

# minlat,minlon,maxlat,maxlon - rough box covering Escambia through Bay County
MIN_LAT, MIN_LON, MAX_LAT, MAX_LON = 29.9, -87.7, 31.0, -85.3
bbox = f"{MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON}"

url = f"https://data.api.xweather.com/lightning/{bbox}"
params = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "from": "-7days",
    "to": "now",
}

response = requests.get(url, params=params, timeout=30)
data = response.json()

print(f"HTTP status: {response.status_code}")
print(f"API success flag: {data.get('success')}")

if not data.get("success"):
    print(f"Error: {data.get('error')}")
    raise SystemExit(1)

strikes = data.get("response", [])
print(f"Strike count (last 7 days, full bbox): {len(strikes)}")

if strikes:
    print("\nFirst raw strike record:")
    print(json.dumps(strikes[0], indent=2))
else:
    print("No strikes returned in this window.")

"""Quick connectivity check for the XWeather API using credentials from .env."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("XWEATHER_CLIENT_ID")
CLIENT_SECRET = os.getenv("XWEATHER_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Missing XWEATHER_CLIENT_ID or XWEATHER_CLIENT_SECRET in .env")

url = "https://data.api.xweather.com/conditions/pensacola,fl"
params = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}

response = requests.get(url, params=params, timeout=10)
data = response.json()

print(f"HTTP status: {response.status_code}")
print(f"API success flag: {data.get('success')}")

if data.get("success"):
    period = data["response"][0]["periods"][0]
    print(f"Location: Pensacola, FL")
    print(f"Temp (F): {period.get('tempF')}")
    print(f"Weather: {period.get('weather')}")
else:
    print(f"Error: {data.get('error')}")

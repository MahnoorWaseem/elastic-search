"""
run_flattener.py

A test script for step 1 only.
It reads the alerts you exported from MongoDB (data/alerts.json),
flattens them, and saves output/flattened_alerts.json.

Run it from inside the nexa-search folder:
    python run_flattener.py

Later the backend button will do the same job, so you won't need this script then.
"""

import json
import logging
from pathlib import Path

from flattener import flatten_and_save

# Show the INFO and WARNING messages in the terminal.
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

INPUT_FILE = Path(__file__).parent / "data" / "alerts.json"

with open(INPUT_FILE, encoding="utf-8") as f:
    text = f.read()

# Mongo exports come in two shapes: one big list [ {...}, {...} ]
# or one alert per line. We handle both, so either export works.
if text.lstrip().startswith("["):
    alerts = json.loads(text)
else:
    alerts = [json.loads(line) for line in text.splitlines() if line.strip()]

docs = flatten_and_save(alerts)

# Print the first document so you can see what "flat" looks like.
if docs:
    print("\nFirst flattened alert:")
    print(json.dumps(docs[0], indent=2))

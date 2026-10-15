"""
run_es_index.py

A test script for step 2 only. It:
  1. connects to Elasticsearch and prints its version
  2. creates the index (fresh on every run)
  3. loads output/flattened_alerts.json (made by step 1)
  4. runs one hand-written search to prove everything works

Run it from inside the nexa-search folder:
    python run_es_index.py

Later the backend button will do steps 1-3, so you won't need this script then.
"""

import json
import logging
from pathlib import Path

from config import INDEX_NAME
from es_index import get_client, load_documents, recreate_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
# The Elasticsearch library prints a line for every request it sends.
# We hide those so the output stays readable.
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

FLATTENED_FILE = Path(__file__).parent / "output" / "flattened_alerts.json"

# --- 1. Connect ------------------------------------------------------------
# If this fails, the problem is the settings in config.py (URL or password),
# not the code, so we stop here with a clear message.
es = get_client()
try:
    version = es.info()["version"]["number"]
except Exception as error:
    print(
        "Could not connect to Elasticsearch. Check ES_URL and ES_PASSWORD in config.py."
    )
    print("Details:", error)
    raise SystemExit(1)
print(f"Connected to Elasticsearch {version}")

# --- 2. Create the index -----------------------------------------------------
recreate_index(es)

# --- 3. Load the documents step 1 produced -----------------------------------
with open(FLATTENED_FILE, encoding="utf-8") as f:
    docs = json.load(f)
load_documents(es, docs)

# --- 4. A hand-written test search -------------------------------------------
# "women in a black top, aged 20 to 34"
# In step 3, code will build this from JSON, and in step 4 the LLM will
# write that JSON from the user's sentence. For now we write it by hand.
#
#   bool + filter : every clause in the list must be true (AND),
#                   and it's a plain yes/no match, with no relevance scores
#   terms         : the field must be one of these exact values
#   range         : a number between gte (>=) and lte (<=)
query = {
    "bool": {
        "filter": [
            {"terms": {"gender": ["female"]}},
            {"terms": {"upper_color": ["black"]}},
            {"range": {"age": {"gte": 20, "lte": 34}}},
        ]
    }
}

result = es.search(
    index=INDEX_NAME,
    query=query,
    sort=[{"timestamp": "desc"}],  # no scores to sort by, so newest first
    size=20,
)

hits = result["hits"]["hits"]
print(f"\nTest search 'women in a black top, aged 20-34': {len(hits)} found")
for hit in hits:
    d = hit["_source"]
    print(
        f"  {d['timestamp']}  {d['gender']}, age {d['age']}, "
        f"top {d['upper_color']}, bottom {d.get('lower_color')}"
    )

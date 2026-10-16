"""
app.py

The backend: a small web server that the search page talks to.
It does everything the run_*.py test scripts did, so you don't need them anymore.

  GET  /            -> sends the search page (static/index.html)
  POST /api/load    -> the "Load alerts" button:
                       read data/alerts.json -> flatten (saves output/flattened_alerts.json)
                       -> rebuild the Elasticsearch index -> load the alerts
  POST /api/search  -> the "Search" button, receives {"text": "man in a red shirt"}:
                       LLM -> filters -> Elasticsearch query -> results

Start it from inside the nexa-search folder:
    uvicorn app:app --reload
Then open http://localhost:8000 in your browser.
"""

import json
import logging
from pathlib import Path

from elasticsearch import NotFoundError
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from es_index import get_client, load_documents, recreate_index
from flattener import OUTPUT_FILE, flatten_and_save
from llm_parser import parse_query
from query_builder_and_searcher import search

# Show our INFO/WARNING messages in the terminal where the server runs.
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

BASE_DIR = Path(__file__).parent
# For now the alerts come from the raw JSON export, not from MongoDB.
ALERTS_FILE = BASE_DIR / "data" / "alerts.json"
PAGE_FILE = BASE_DIR / "static" / "index.html"

app = FastAPI(title="Nexa search")

# One Elasticsearch connection for the whole app, made once when the server starts.
es = get_client()


def read_alerts():
    """
    Reads the raw alerts from data/alerts.json.
    Later, this is the only function that changes when we read from MongoDB instead.
    """
    if not ALERTS_FILE.exists():
        raise HTTPException(status_code=404, detail="data/alerts.json not found.")

    text = ALERTS_FILE.read_text(encoding="utf-8")
    # The export is either one big list [ {...}, {...} ] or one alert per line.
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# Note: the endpoints below use "def", not "async def". FastAPI then runs each
# request in its own thread, so a slow LLM answer doesn't freeze the server.


@app.get("/")
def page():
    """Sends the search page to the browser."""
    return FileResponse(PAGE_FILE)


@app.post("/api/load")
def load_alerts():
    """
    The "Load alerts" button:
    raw alerts -> flatten (saves output/flattened_alerts.json) -> new index -> load
    """
    alerts = read_alerts()
    docs = flatten_and_save(alerts)

    try:
        recreate_index(es)
        loaded = load_documents(es, docs)
    except Exception as error:
        # 503 = "a service we depend on isn't working" (here: Elasticsearch)
        raise HTTPException(status_code=503, detail=f"Elasticsearch problem: {error}")

    return {
        "alerts_read": len(alerts),
        "flattened": len(docs),
        "loaded": loaded,
        "saved_to": str(OUTPUT_FILE.relative_to(BASE_DIR)),
    }


class SearchRequest(BaseModel):
    """What the page sends: {"text": "man in a red shirt"}.
    FastAPI checks that "text" is there and is a string."""

    text: str


@app.post("/api/search")
def search_alerts(request: SearchRequest):
    """
    The "Search" button: text -> LLM filters -> Elasticsearch query -> results.
    All three are sent back, so the page can show every step.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Type something to search for.")

    # 1. The LLM turns the text into filters.
    try:
        filters = parse_query(text)
    except RuntimeError as error:  # Ollama not running, model missing, ...
        raise HTTPException(status_code=503, detail=str(error))

    # 2. Build the Elasticsearch query from the filters and run it.
    try:
        output = search(es, filters)
    except NotFoundError:
        # The index doesn't exist yet.
        raise HTTPException(
            status_code=404, detail="No alerts loaded yet. Click 'Load alerts' first."
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Elasticsearch problem: {error}")

    return {
        "text": text,
        "filters": filters,  # what the LLM understood
        "es_query": output[
            "es_query"
        ],  # what we sent to Elasticsearch (None = nothing to search)
        "results": output["results"],  # the matching alerts, newest first
    }

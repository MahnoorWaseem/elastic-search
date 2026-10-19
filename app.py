"""
app.py

What changed: /api/load is removed. At startup the server now makes sure the index exists and starts the MQTT listener. When it stops, it disconnects cleanly. Search works exactly as before.

The backend: a small web server that the search page talks to.

When it starts, it:
  1. makes sure the Elasticsearch index exists (it never deletes alerts)
  2. starts the MQTT listener (mqtt_listener.py), which saves every new alert
     from the CV pipeline: flatten -> output/flattened_alerts.jsonl -> Elasticsearch

Addresses:
  GET  /            -> sends the search page (static/index.html)
  POST /api/search  -> the "Search" button, receives {"text": "man in a red shirt"}:
                       LLM -> filters -> Elasticsearch query -> results

Start it from inside the project folder:
    uvicorn app:app --reload
Then open http://localhost:8000 in your browser.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from es_index import ensure_index, get_client
from llm_parser import parse_query
from mqtt_listener import start_listening, stop_listening
from query_builder_and_searcher import search

# Show our INFO/WARNING messages in the terminal where the server runs.
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
# gmqtt prints a line for every packet it sends. We only want its warnings.
logging.getLogger("gmqtt").setLevel(logging.WARNING)

BASE_DIR = Path(__file__).parent
PAGE_FILE = BASE_DIR / "static" / "index.html"

# One Elasticsearch connection for the whole app, made once when the server starts.
es = get_client()


@asynccontextmanager
async def lifespan(app):
    """
    FastAPI runs the part before "yield" when the server starts,
    and the part after "yield" when the server stops.
    """
    # The index must exist before the first alert arrives. If Elasticsearch
    # isn't running, the server stops here with the error, so start Docker first.
    ensure_index(es)
    await start_listening(es)
    yield
    await stop_listening()


app = FastAPI(title="Nexa search", lifespan=lifespan)


# Note: the endpoints below use "def", not "async def". FastAPI then runs each
# request in its own thread, so a slow LLM answer doesn't freeze the server.


@app.get("/")
def page():
    """Sends the search page to the browser."""
    return FileResponse(PAGE_FILE)


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
        output = search(es, filters, size=200)
    except Exception as error:
        # 503 = "a service we depend on isn't working" (here: Elasticsearch)
        raise HTTPException(status_code=503, detail=f"Elasticsearch problem: {error}")

    return {
        "text": text,
        "filters": filters,  # what the LLM understood
        "es_query": output[
            "es_query"
        ],  # what we sent to Elasticsearch (None = nothing to search)
        "results": output["results"],  # the matching alerts, newest first
    }

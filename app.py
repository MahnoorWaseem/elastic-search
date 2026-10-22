"""
app.py

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
from seaweed_client import SeaweedClient

# Show our INFO/WARNING messages in the terminal where the server runs.
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
# gmqtt prints a line for every packet it sends. We only want its warnings.
logging.getLogger("gmqtt").setLevel(logging.WARNING)
log = logging.getLogger("app")

BASE_DIR = Path(__file__).parent
PAGE_FILE = BASE_DIR / "static" / "index.html"

# One Elasticsearch connection for the whole app, made once when the server starts.
es = get_client()

# One SeaweedFS client for the whole app (settings come from .env).
# Making it doesn't contact SeaweedFS yet, so the server starts even if it's off.
seaweed = SeaweedClient()


def add_image_urls(results):
    """
    Why:     the images are in SeaweedFS, which needs a password, so the
             browser can't open them by their image_key. A "pre-signed" URL is
             a normal link with a signature in it: anyone with the link can
             see that ONE image, for 1 hour. The browser then loads the image
             straight from SeaweedFS; our server never downloads it.
    Used by: search_alerts(), for every result.

    Making the link is only a calculation on our side (no request to
    SeaweedFS), so even 50 results take no time.
        "tenant_1/.../3176f6bb-....jpg"  ->  "http://192.168.100.127:9010/nexa/tenant_1/...jpg?X-Amz-Signature=..."
    An alert without an image_key, or a link that can't be made, gets None,
    and the page shows "—" instead of an image.
    """
    for alert in results:
        key = alert.get("image_key")
        alert["image_url"] = None
        if not key:
            continue
        try:
            alert["image_url"] = seaweed.generate_download_url(key)
        except Exception as error:
            log.warning("couldn't make an image link for %s: %s", key, error)
    return results


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
    except Exception as error:
        # Anything else is a bug in our code. Without this, the page only gets
        # "Internal Server Error". Now the page shows the error, and the
        # terminal shows the full traceback (log.exception) to find the line.
        log.exception("reading the search failed")
        raise HTTPException(
            status_code=500,
            detail=f"Reading the search failed: {type(error).__name__}: {error}",
        )

    # 2. Build the Elasticsearch query from the filters and run it.
    try:
        output = search(es, filters)
    except Exception as error:
        # 503 = "a service we depend on isn't working" (here: Elasticsearch)
        raise HTTPException(status_code=503, detail=f"Elasticsearch problem: {error}")

    return {
        "text": text,
        "filters": filters,  # what the LLM understood
        "es_query": output[
            "es_query"
        ],  # what we sent to Elasticsearch (None = nothing to search)
        # the matching alerts, newest first, each with an "image_url" to show
        "results": add_image_urls(output["results"]),
    }

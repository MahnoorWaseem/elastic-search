"""
es_index.py

What it does:

Builds the field types from the schema, plus tracker_id, box and image_key.
ensure_index() creates the index at startup only if it’s missing. If an existing index has a different field list, it stops with a clear message.
save_to_index() adds one alert.

Everything about the Elasticsearch index:
  - get_client()    connects to Elasticsearch
  - build_mapping() turns defined_schema.py into Elasticsearch field types
  - ensure_index()  creates the index when the server starts, if it's missing
  - save_to_index() adds one flattened alert to the index

This file only contains functions. app.py calls ensure_index() at startup,
and mqtt_listener.py calls save_to_index() for every new alert.
"""

import logging

from elasticsearch import Elasticsearch

from config import ES_PASSWORD, ES_URL, ES_USERNAME, INDEX_NAME
from defined_schema import DEFINED_SCHEMA

log = logging.getLogger("es_index")


def get_client():
    """Opens a connection to Elasticsearch using the settings in config.py."""
    if ES_PASSWORD:
        # Security ON: log in with username and password.
        # verify_certs=False: Docker's Elasticsearch uses a self-made certificate
        # that Python doesn't trust. That's fine on your own machine,
        # but NOT in production.
        return Elasticsearch(
            ES_URL,
            basic_auth=(ES_USERNAME, ES_PASSWORD),
            verify_certs=False,
            ssl_show_warn=False,
        )

    # Security OFF: just connect.
    return Elasticsearch(ES_URL)


# Fields the flattener copies as they are. They're not in the search schema
# (the LLM never searches on them), but the index still needs their types.
# camera_id isn't listed here because the search schema already has it.
BASE_FIELDS = {
    "alert_id": {"type": "keyword"},
    "tracker_id": {"type": "integer"},
    # A real date type, so results can be sorted newest first.
    "timestamp": {"type": "date"},
    # Stored so the UI can use them later, but never searched,
    # so "index": False tells Elasticsearch not to build a search index for them.
    "box": {"type": "float", "index": False},  # a list of 4 numbers
    "image_key": {"type": "keyword", "index": False},
}


def build_mapping():
    """
    The mapping tells Elasticsearch the type of every field.
    We build it from defined_schema.py, so a field added there gets indexed
    automatically, without touching this file:

      enum      -> keyword : stored exactly as written and matched exactly,
                             so "light blue" stays one value, never "light" + "blue"
      number_id -> keyword : camera IDs like "camera_2", also matched exactly
      range     -> integer : a number, so we can ask "age between 20 and 30"
      any_of    -> nothing : "color" is only used while searching, never stored
    """
    properties = dict(BASE_FIELDS)

    for field, rules in DEFINED_SCHEMA.items():
        if rules["type"] in ("enum", "number_id"):
            properties[field] = {"type": "keyword"}
        elif rules["type"] == "range":
            properties[field] = {"type": "integer"}

    return {
        # "strict": if a document arrives with a field we didn't list,
        # Elasticsearch refuses it instead of guessing the type.
        # A guess would store "red" as full-text, which we decided against.
        "dynamic": "strict",
        "properties": properties,
    }


def ensure_index(es):
    """
    Called once when the server starts.

    - Index missing      -> create it, empty.
    - Index already there -> keep it, with all the alerts already in it.
      (We used to delete and rebuild it on every load. With alerts arriving
      all the time, that would throw them away.)

    If the existing index was made with a different list of fields (for
    example from the old alert format), new alerts would be refused, so we
    stop with a clear message instead.
    """
    wanted = build_mapping()

    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(
            index=INDEX_NAME,
            settings={
                # Your Docker runs a single Elasticsearch node.
                # 1 shard: all data kept in one piece (plenty for our size).
                # 0 replicas: no backup copy. A single node has nowhere to put
                # a copy, and asking for one leaves the index in "yellow" state.
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            mappings=wanted,
        )
        log.info("created index '%s'", INDEX_NAME)
        return

    current = es.indices.get_mapping(index=INDEX_NAME)[INDEX_NAME]["mappings"]
    if set(current.get("properties", {})) != set(wanted["properties"]):
        raise RuntimeError(
            f"The index '{INDEX_NAME}' was created with a different list of fields. "
            f"Delete it once and restart the server; a new one will be created. "
            f"To delete it: curl.exe -X DELETE {ES_URL}/{INDEX_NAME}"
        )

    log.info("using existing index '%s'", INDEX_NAME)


def save_to_index(es, doc):
    """
    Adds one flattened alert to the index.

    The alert_id becomes the document's ID in Elasticsearch, so if the same
    alert arrives twice, it replaces itself instead of showing up twice.

    It becomes searchable within about 1 second (Elasticsearch refreshes its
    search view every second on its own).
    """
    es.index(index=INDEX_NAME, id=doc["alert_id"], document=doc)

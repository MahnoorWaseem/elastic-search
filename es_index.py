"""
es_index.py

What it does: everything to do with the index.

get_client() connects to Elasticsearch.
build_mapping() turns defined_schema.py into Elasticsearch field types: enum → keyword, range → integer.
recreate_index() creates a fresh, empty index.
load_documents() sends all flattened alerts in one bulk request.


Like flattener.py, this file only contains functions.
run_es_index.py calls them now; the backend will call them later
when you click the button.
"""

import logging

from elasticsearch import Elasticsearch, helpers

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
    "tenant_id": {"type": "keyword"},
    "pipeline_id": {"type": "keyword"},
    # A real date type, so results can be sorted newest first.
    "timestamp": {"type": "date"},
    # Stored so the UI can show the snapshot, but never searched,
    # so "index": False tells Elasticsearch not to build a search index for it.
    "snapshot_path": {"type": "keyword", "index": False},
}


def build_mapping():
    """
    The mapping tells Elasticsearch the type of every field.
    We build it from search_schema.py, so a field added there gets indexed
    automatically, without touching this file:

      enum   -> keyword : stored exactly as written and matched exactly,
                          so "light blue" stays one value, never "light" + "blue"
      range  -> integer : a number, so we can ask "age between 20 and 30"
      any_of -> nothing : "color" is only used while searching, never stored
    """
    properties = dict(BASE_FIELDS)

    for field, rules in DEFINED_SCHEMA.items():
        if rules["type"] == "enum":
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


def recreate_index(es):
    """
    Deletes the index (if it exists) and creates it again, empty.

    Why delete first? Elasticsearch can't change the type of a field inside
    an existing index. Rebuilding from scratch means a change in
    search_schema.py always takes effect, and alerts deleted from MongoDB
    don't stay in the search results.
    """
    es.indices.delete(index=INDEX_NAME, ignore_unavailable=True)

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
        mappings=build_mapping(),
    )
    log.info("created index '%s'", INDEX_NAME)


def load_documents(es, docs):
    """
    Sends all flattened alerts to Elasticsearch in one request ("bulk"),
    instead of one request per alert, which would be much slower.
    Returns how many documents were loaded.
    """
    actions = [
        {
            "_index": INDEX_NAME,
            # The alert_id becomes the document's ID in Elasticsearch.
            # If the same alert is loaded twice, it replaces itself
            # instead of showing up twice in the results.
            "_id": doc["alert_id"],
            "_source": doc,
        }
        for doc in docs
    ]

    # refresh=True: make the documents searchable immediately.
    # raise_on_error=False: don't crash on one bad document; collect the errors.
    loaded, errors = helpers.bulk(es, actions, refresh=True, raise_on_error=False)

    if errors:
        log.warning("%d documents failed. First error: %s", len(errors), errors[0])
    log.info("loaded %d documents into '%s'", loaded, INDEX_NAME)

    return loaded

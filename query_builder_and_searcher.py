"""
query_builder.py

Turns search filters (the JSON the LLM will write in step 4) into an
Elasticsearch query, and runs it.

The filters look like this. Every field is optional:
    {
        "gender":      {"include": ["female"]},
        "upper_color": {"exclude": ["black"]},
        "color":       {"include": ["red"]},      <- top OR bottom is red
        "age":         {"min": 20, "max": 34},
        "camera_id":   {"include": ["cam_1"]}
    }

Like the other modules, this file only contains functions.
run_query_builder.py calls them now; later the backend will call search()
when the user presses the Search button.
"""

import logging

from config import INDEX_NAME
from defined_schema import DEFINED_SCHEMA

log = logging.getLogger("query_builder")


def allowed_values(field):
    """
    The allowed values of a field, from search_schema.py.
    "color" has no list of its own, so it borrows upper_color's colours.
    """
    rules = DEFINED_SCHEMA[field]
    if rules["type"] == "any_of":
        rules = DEFINED_SCHEMA[rules["fields"][0]]
    return rules["values"]


def pick_values(field, include, exclude):
    """
    The include/exclude rule we agreed on:
      start from the include list (or ALL allowed values if include is empty),
      then remove everything in exclude.

        include ["red"]                  -> ["red"]
        exclude ["black"]                -> every colour except black
        include ["red"], exclude ["red"] -> []  (impossible, nobody matches)

    We loop over the ALLOWED values, not over what the LLM sent,
    so a value that isn't in the schema can never reach Elasticsearch.
    """
    allowed = allowed_values(field)

    unknown = [v for v in include + exclude if v not in allowed]
    if unknown:
        log.warning("%s: ignoring values not in the schema: %s", field, unknown)

    return [v for v in allowed if (not include or v in include) and v not in exclude]


def terms_clause(field, values):
    """
    "field must be one of these values".
    An empty list means nobody can match, so we say that explicitly
    with match_none instead of sending an empty list.
    """
    if not values:
        return {"match_none": {}}
    return {"terms": {field: values}}


def build_query(filters):
    """
    Turns the filters into an Elasticsearch query.

    Returns None when there is nothing to search by. That way, if the LLM
    understood nothing, we show "couldn't understand" instead of
    returning every single alert.
    """
    clauses = []  # every clause in this list must be true (AND)
    #  example filter  {
    #             "gender": {"include": ["female"]},
    #             "upper_color": {"include": ["black"]},
    #             "age": {"min": 20, "max": 34},
    #         }
    for field, wanted in filters.items():
        rules = DEFINED_SCHEMA.get(field)
        # for example gender
        if rules is None:
            log.warning("ignoring unknown field: %s", field)
            continue

        # --- age: a number range ------------------------------------------
        if rules["type"] == "range":
            low = wanted.get("min")
            high = wanted.get("max")
            if low is None and high is None:
                continue  # field sent, but nothing asked
            if low is None:
                low = rules["min"]  # "under 30" -> 1 to 30
            if high is None:
                high = rules["max"]  # "over 50"  -> 50 to 85
            if low > high:  # "between 30 and 20" -> swap
                low, high = high, low
            clauses.append({"range": {field: {"gte": low, "lte": high}}})
            continue

        # Every other type uses include / exclude lists.
        # "or []" turns a missing or null list into an empty one.
        include = wanted.get("include") or []
        exclude = wanted.get("exclude") or []
        if not include and not exclude:
            continue  # field sent, but nothing asked

        # --- normal fields: gender, age_group, upper_color, lower_color, camera_id
        if rules["type"] == "enum":
            clauses.append(terms_clause(field, pick_values(field, include, exclude)))

        # --- "color": the user didn't say top or bottom ----------------------
        elif rules["type"] == "any_of":
            # "wearing red" -> top is red OR bottom is red
            if include:
                values = pick_values(field, include, exclude)
                clauses.append(
                    {
                        "bool": {
                            "should": [
                                terms_clause(f, values) for f in rules["fields"]
                            ],
                            "minimum_should_match": 1,  # at least one is true = OR
                        }
                    }
                )
            # "not wearing black" -> top is not black AND bottom is not black
            if exclude:
                values = pick_values(field, [], exclude)
                for f in rules["fields"]:
                    clauses.append(terms_clause(f, values))

    if not clauses:
        return None

    # filter = plain yes/no matching. No relevance scores, and
    # Elasticsearch caches filters, so repeated searches are fast.
    return {"bool": {"filter": clauses}}


def search(es, filters, size=50):
    """
    Builds the query and runs it. Returns a dict the backend can later
    send straight to the UI:
        es_query : the query that was built (None if nothing to search by)
        results  : the matching flattened alerts, newest first
    """
    query = build_query(filters)
    if query is None:
        return {"es_query": None, "results": []}

    # es.search(...) — the Elasticsearch client's method that sends the query to Elasticsearch.
    response = es.search(
        index=INDEX_NAME,
        query=query,
        sort=[{"timestamp": "desc"}],  # no scores to sort by, so newest first
        size=size,
    )
    results = [hit["_source"] for hit in response["hits"]["hits"]]
    return {"es_query": query, "results": results}

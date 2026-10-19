"""
query_builder_and_searcher.py

Turns search filters (the JSON the LLM writes) into an
Elasticsearch query, and runs it.

The filters look like this. Every field is optional:
    {
        "gender":      {"include": ["female"]},
        "upper_color": {"exclude": ["black"]},
        "color":       {"include": ["red"]},      <- top OR bottom is red (or maroon)
        "age_group":   {"include": ["child"]},
        "camera_id":   {"include": [2]}
    }

Like the other modules, this file only contains functions.
The backend calls search() when the user presses the Search button.
"""

import logging

from config import INDEX_NAME
from defined_schema import COLOR_FAMILIES, COLORS, DEFINED_SCHEMA

log = logging.getLogger("query_builder")


def allowed_values(field):
    """
    The allowed values of a field, from defined_schema.py.
    "color" has no list of its own, so it borrows upper_color's colours.
    """
    rules = DEFINED_SCHEMA[field]
    if rules["type"] == "any_of":
        rules = DEFINED_SCHEMA[rules["fields"][0]]
    return rules["values"]


def is_color_field(field):
    """
    True for color, upper_color and lower_color: the fields whose
    allowed values are the colour list. Checked from defined_schema.py,
    so a new colour field later is picked up automatically.
    """
    rules = DEFINED_SCHEMA[field]
    return rules["type"] in ("enum", "any_of") and allowed_values(field) == COLORS


# for each color given by llm, it adds its relevant shade example for blue it will add ["blue", "light blue", "navy"] and for black since there's no family it will remain black.
def add_shades(colors):
    """
    A broad colour also brings its shades, using COLOR_FAMILIES:
        ["red"]          -> ["red", "maroon"]
        ["blue", "red"]  -> ["blue", "light blue", "navy", "red", "maroon"]
        ["navy"]         -> ["navy"]      (a specific shade stays specific)
        ["black"]        -> ["black"]     (no family)
    """
    result = []
    for color in colors:
        for item in [color] + COLOR_FAMILIES.get(color, []):
            if item not in result:  # don't add the same colour twice
                result.append(item)
    return result


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
    #             "age_group": {"include": ["adult"]},
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

        # Colour families, for color, upper_color and lower_color, both lists:
        #   include ["red"] -> red OR maroon
        #   exclude ["red"] -> "not red" also means not maroon
        if is_color_field(field):
            include = add_shades(include)
            exclude = add_shades(exclude)

        # --- normal fields: gender, age_group, upper_color, lower_color
        if rules["type"] == "enum":
            clauses.append(terms_clause(field, pick_values(field, include, exclude)))

        # --- camera_id: the LLM gives numbers, the real IDs are "camera_" + number
        elif rules["type"] == "number_id":
            if include:
                ids = [rules["prefix"] + str(n) for n in include]  # 2 -> "camera_2"
                clauses.append({"terms": {field: ids}})
            if exclude:
                ids = [rules["prefix"] + str(n) for n in exclude]
                # must_not = "none of these". We can't use "every camera except
                # these" like we do for colours, because there's no full camera list.
                clauses.append({"bool": {"must_not": [{"terms": {field: ids}}]}})

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


# searches the built es query in elasticsearch db
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

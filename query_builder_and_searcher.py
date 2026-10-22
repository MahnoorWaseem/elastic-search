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
        "camera_id":   {"include": [2]},
        "time":        [{"from_date": "2026-10-16", "to_date": "2026-10-21",
                         "from_time": "17:00",      "to_time": "19:00"}]
                         <- every day from 16 to 21 Oct, 17:00 to 19:00 each day
    }

Like the other modules, this file only contains functions.
The backend calls search() when the user presses the Search button.
"""

import logging
from datetime import datetime, timedelta

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


# A daily time window gives one range per day. More than this many days is cut
# to the most recent ones, so a query never gets thousands of ranges.
# (It only applies to daily hours like "5 pm to 7 pm"; whole days are one range.)
MAX_DAYS = 366


def clock(text):
    """
    Why:     to add a clock time to a day, it must be a length of time.
    Used by: time_clause().

    "17:30" -> 17 hours 30 minutes (after midnight)
    """
    hours, minutes = text.split(":")
    return timedelta(hours=int(hours), minutes=int(minutes))


def one_range(field, start, end):
    """
    Why:     every time window ends up as the same kind of Elasticsearch range.
    Used by: time_clause().

    "gte" = from this moment, "lt" = up to (not including) this moment.
    No time-zone conversion: the times are compared exactly as written with
    the stored timestamps. "14:00" in the filter finds alerts stored at 14:xx.
    """
    return {
        "range": {
            field: {
                "gte": start.isoformat(),  # e.g. "2026-10-16T17:00:00"
                "lt": end.isoformat(),
            }
        }
    }


def time_clause(field, windows):
    """
    Why:     the time filter is a DAILY filter: every day from from_date to
             to_date, and on each of those days only from from_time to to_time.
    Used by: build_query(), when the filters have "time".

        {"from_date": "2026-10-16", "to_date": "2026-10-18",
         "from_time": "17:00", "to_time": "19:00"}
        -> 16 Oct 17:00-19:00  OR  17 Oct 17:00-19:00  OR  18 Oct 17:00-19:00

    How one day's start and end are made:
        17:00 to 19:00 -> 17:00 up to 19:01   ("to" counts fully, so 19:00:30 is found)
        22:00 to 03:00 -> 22:00 up to 03:01 the NEXT day   (overnight)
        17:00 to 17:00 -> 17:00 up to 18:00   ("at 5 pm": the whole hour)
        16:45 to 16:45 -> 16:30 up to 17:00   ("at 4:45": 15 minutes each side)

    The whole day (00:00 to 23:59) on every day doesn't need one range per day:
    it becomes ONE range, from the first day 00:00 up to the day after the last.
    """
    ranges = []
    for window in windows:
        first_day = datetime.strptime(window["from_date"], "%Y-%m-%d")
        last_day = datetime.strptime(window["to_date"], "%Y-%m-%d")

        # Whole days: one range for all of them together.
        if window["from_time"] == "00:00" and window["to_time"] == "23:59":
            ranges.append(one_range(field, first_day, last_day + timedelta(days=1)))
            continue

        if (last_day - first_day).days >= MAX_DAYS:
            log.warning(
                "time covers more than %s days, using the last %s", MAX_DAYS, MAX_DAYS
            )
            first_day = last_day - timedelta(days=MAX_DAYS - 1)

        from_time = clock(window["from_time"])
        to_time = clock(window["to_time"])

        # One range for each day.
        day = first_day
        while day <= last_day:
            start = day + from_time
            end = day + to_time
            if start == end:  # one moment: "at ..."
                if start.minute == 0:
                    end = start + timedelta(hours=1)
                else:
                    start, end = (
                        start - timedelta(minutes=15),
                        start + timedelta(minutes=15),
                    )
            elif end < start:  # overnight: ends on the next day
                end = end + timedelta(days=1, minutes=1)
            else:
                end = end + timedelta(minutes=1)
            ranges.append(one_range(field, start, end))
            day = day + timedelta(days=1)

    if len(ranges) == 1:
        return ranges[0]
    # Several ranges: the alert must be inside ANY one of them (OR).
    return {"bool": {"should": ranges, "minimum_should_match": 1}}


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

        # --- time: a list of windows on the alert's "timestamp" ------------
        if rules["type"] == "time":
            if wanted:
                clauses.append(time_clause(rules["field"], wanted))
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
def search(es, filters, size=500):
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

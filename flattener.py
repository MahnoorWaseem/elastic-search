"""
flattener.py

changed: the pipeline’s age (“adult”/”child”/”null”) is stored straight as age_group. The number-reading code is gone.

What it does:

Takes one incoming alert.
Renames id→alert_id, top_color→upper_color and bottom_color→lower_color.
Converts the timestamp to the format Elasticsearch expects.
Checks every value against the schema.
Appends one line to output/flattened_alerts.jsonl.


Turns ONE alert from the CV pipeline (received over MQTT) into one flat
document for Elasticsearch, and adds it to output/flattened_alerts.jsonl.

An incoming alert looks like this:
    {"id": 210, "camera_id": "camera_1", "tracker_id": 1,
     "gender": "female", "age": "null",
     "top_color": "black", "bottom_color": "brown",
     "timestamp": "2026-10-15 19:09:41.008842+00",
     "box": [0.11, 0.59, 0.15, 0.89],
     "image_key": "tenant_1/site_1/edge_1/camera_1/2026-10-15/389b....jpg"}

This file only contains functions. mqtt_listener.py calls flatten_and_save()
every time a new alert arrives.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from defined_schema import DEFINED_SCHEMA, NOT_KNOWN

log = logging.getLogger("flattener")

# Where every flattened alert is saved, one alert per line.
# ".jsonl" = JSON Lines: each line is one complete JSON object. Alerts arrive
# all the time, so we just ADD a line at the end for each new one. A normal
# .json list would have to be read and rewritten completely for every alert.
OUTPUT_FILE = Path(__file__).parent / "output" / "flattened_alerts.jsonl"

# Makes sure two alerts arriving at the same moment never write into the
# file at the same time (their lines could get mixed up).
_file_lock = threading.Lock()


def to_date_text(value):
    """
    Why:     Elasticsearch must be able to read the time, but we don't change it.
             The time is kept exactly as the pipeline sent it. Only the old
             PostgreSQL SHAPE is rewritten into the ISO shape Elasticsearch reads:
               "2026-10-21T13:06:23.327111+00:00"  ->  unchanged
               "2026-10-15 19:09:41.008842+00"     ->  "2026-10-15T19:09:41.008842+00:00"
             The numbers never change: no time-zone conversion, nothing rounded.
    Used by: flatten_person_attributes().
    """
    text = value.strip().replace(" ", "T", 1)
    if len(text) >= 3 and text[-3] in "+-" and text[-2:].isdigit():
        text += ":00"  # "+00" -> "+00:00"
    datetime.fromisoformat(text)  # stops with an error if it isn't a real date and time
    return text


def make_alert_id(alert):
    """
    Why:     every alert needs an ID in Elasticsearch. Newer messages from the
             CV pipeline have no "id" field, so we make one when it's missing.
    Used by: flatten_person_attributes().

    With an "id"    -> that id:                          "210"
    Without an "id" -> camera + tracker + exact time:    "camera_1_2_2026-10-21T13:06:23.327111+00:00"

    One person (tracker) on one camera can only be in one alert at one exact
    moment, so this is unique. And the same message arriving twice gives the
    same ID, so it still replaces itself instead of being saved twice.
    """
    if alert.get("id") is not None:
        return str(alert["id"])
    return f"{alert['camera_id']}_{alert['tracker_id']}_{alert['timestamp']}"


def clean_attributes(alert_id, raw):
    """
    Keeps only the values the schema allows.

    - "uncertain", "unknown", "null", empty  -> dropped quietly (not searchable).
    - any other value not in the schema -> dropped WITH a warning, because it
      means the CV pipeline sends something new (e.g. "grey"), and we need to
      add it to defined_schema.py.
    """
    clean = {}

    for field, value in raw.items():
        # "Black " or "BLACK" -> "black", so small differences don't break matching.
        if isinstance(value, str):
            value = value.strip().lower()

        if value in NOT_KNOWN:
            continue

        rules = DEFINED_SCHEMA[field]

        if rules["type"] == "enum" and value in rules["values"]:
            clean[field] = value

        elif (
            rules["type"] == "range"
            and isinstance(value, (int, float))
            and rules["min"] <= value <= rules["max"]
        ):
            clean[field] = int(value)

        else:
            log.warning(
                "alert %s: %s=%r is not in the schema, so it was skipped",
                alert_id,
                field,
                value,
            )

    return clean


def flatten_person_attributes(alert):
    """One person-attributes alert in, one flat document out."""
    # Elasticsearch IDs are text, so 210 becomes "210".
    alert_id = make_alert_id(alert)

    # Part 1: copied as they are (not CV guesses, so nothing to check).
    doc = {
        "alert_id": alert_id,
        "camera_id": alert.get("camera_id"),
        "tracker_id": alert.get("tracker_id"),
        "timestamp": to_date_text(alert["timestamp"]),
        "box": alert.get("box"),  # where the person is in the image
        "image_key": alert.get("image_key"),  # where the snapshot is stored
    }

    # Part 2: the CV predictions, renamed to our schema's names and checked.
    # "null" (unknown) is dropped by clean_attributes, like "uncertain".
    doc.update(
        clean_attributes(
            alert_id,
            {
                "gender": alert.get("gender"),
                "age_group": alert.get(
                    "age"
                ),  # age          -> age_group ("adult"/"child")
                "upper_color": alert.get("top_color"),  # top_color    -> upper_color
                "lower_color": alert.get("bottom_color"),  # bottom_color -> lower_color
            },
        )
    )

    return doc


# Which function flattens which pipeline.
# When a new pipeline arrives (e.g. smoking), we write flatten_smoking()
# and add one line here.
FLATTENERS = {
    "person_attributes": flatten_person_attributes,
}


def flatten_and_save(alert, pipeline="person_attributes"):
    """
    The main function, called for every new alert:
      1. flattens it with the right function for its pipeline
      2. adds it as one new line to output/flattened_alerts.jsonl
      3. returns it, so it can be put into Elasticsearch next
    """
    doc = FLATTENERS[pipeline](alert)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with _file_lock:
        # "a" = append mode: add to the end, never erase what's already there.
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")

    return doc

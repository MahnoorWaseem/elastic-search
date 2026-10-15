"""
flattener.py

What it does: it turns each raw MongoDB alert into one flat document:

It copies the IDs, time and snapshot path as they are.
It pulls gender, age and colours up out of details.
It drops “uncertain” values and warns you about any value the schema doesn’t know.
It saves everything to output/flattened_alerts.json.

This file only contains functions. It does nothing when you open it.
- Right now, run_flattener.py calls flatten_and_save() so you can test it.
- Later, the backend will call the SAME function when you click the button.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from defined_schema import NOT_KNOWN, DEFINED_SCHEMA

log = logging.getLogger("flattener")

# Where the result is saved.
# Path(__file__).parent is the folder this file is in (nexa-search),
# so the file always lands in nexa-search/output/, wherever you run from.
OUTPUT_FILE = Path(__file__).parent / "output" / "flattened_alerts.json"


def to_date_text(seconds):
    """
    Mongo stores time as seconds since 1970, e.g. 1791973990.43.
    Elasticsearch understands date text better, so we convert it:
    1791973990.43  ->  "2026-10-14T10:33:10.434+00:00"  (UTC time)
    """
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    )


# If the value is in NOT_KNOWN (“uncertain”, “unknown”, empty), it skips it quietly.
# If the field is an enum type, like gender or colors, it keeps the value only if it’s in the allowed list.
# If the field is a range type, like age, it keeps the value only if it’s a number between min and max, stored as an int.
# Anything else is logged as a warning and skipped.
# checks everything from defined_schema.py
def clean_attributes(alert_id, raw):
    """
    Keeps only the values the schema allows.

    - "uncertain", "unknown", empty  -> dropped quietly (not searchable).
    - any other value not in the schema -> dropped WITH a warning, because it
      means the CV pipeline started sending something new (e.g. "grey"),
      and we need to add it to defined_schema.py.
    """
    clean = {}

    for field, value in raw.items():
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


# flatten_person_attributes_pipeline_alerts(alert) builds one document in two parts:
# Part 1 copies the identity fields (IDs, time, snapshot) as they are. These are facts, not model guesses, so nothing gets validated.
# Part 2 collects the model’s predictions, renames estimated_age to age, and passes them through clean_attributes.
def flatten_person_attributes_pipeline_alerts(alert):
    """One person-attributes alert in, one flat document out."""
    details = alert.get("details") or {}

    # Part 1: copied as they are.
    # These are IDs, time, and the snapshot the UI will show next to a result.
    # They are not CV guesses, so there's nothing to check.
    doc = {
        "alert_id": alert["alert_id"],
        "tenant_id": alert.get("tenant_id"),
        "camera_id": alert.get("camera_id"),
        "pipeline_id": alert.get("pipeline_id"),
        "timestamp": to_date_text(alert["timestamp"]),
        "snapshot_path": alert.get("snapshot_path"),
    }

    # Part 2: the CV predictions.
    # They sit inside "details" in Mongo; we pull them up to the top level
    # (that's the "flattening") and check them against the schema.
    attributes = {
        "gender": details.get("gender"),
        "age_group": details.get("age_group"),
        "age": details.get("estimated_age"),  # renamed to the schema's name
        "upper_color": details.get("upper_color"),
        "lower_color": details.get("lower_color"),
    }
    doc.update(clean_attributes(alert["alert_id"], attributes))

    return doc


# Which function flattens which pipeline.
# When a new pipeline arrives (e.g. smoking), we write flatten_smoking()
# and add one line here. Nothing else in this file changes.
FLATTENERS = {
    "person_attributes": flatten_person_attributes_pipeline_alerts,
}


def flatten_and_save(alerts):
    """
    alerts coming from raw json
    flatten_and_save(alerts) is the entry point. It skips alerts from pipelines that have no flattener yet and counts them.
    The main function. Give it a list of raw alerts and it:
      1. flattens each one (skips pipelines we don't support yet)
      2. saves them all to output/flattened_alerts.json (replacing the old file)
      3. returns them, so the next step (loading into Elasticsearch) can use them
    """
    docs = []
    skipped = 0

    for alert in alerts:
        flatten_fn = FLATTENERS.get(alert.get("pipeline_id"))
        if flatten_fn is None:
            skipped += 1
            continue
        docs.append(flatten_fn(alert))

    # Create the output folder if it doesn't exist yet, then write the file.
    # "w" = write mode: each run replaces the previous file.
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)

    log.info(
        "%d alerts read, %d flattened, %d skipped (pipeline not supported yet)",
        len(alerts),
        len(docs),
        skipped,
    )
    log.info("saved to %s", OUTPUT_FILE)

    return docs

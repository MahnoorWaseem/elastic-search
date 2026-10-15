"""
run_llm_parser.py

A test script for step 4 only. For each test sentence it:
  1. asks Qwen to turn it into filters (and times how long that took)
  2. compares the filters with what we expected, and prints OK or DIFFERENT
  3. runs the search in Elasticsearch and prints how many alerts matched

Ollama and Elasticsearch must both be running, and step 2 must be done.
Run it from inside the nexa-search folder:
    python run_llm_parser.py
"""

import json
import logging
import time

from es_index import get_client
from llm_parser import parse_query
from query_builder_and_searcher import search

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

# Each test: a sentence, and the filters a correct answer should contain.
# These sentences are NOT the examples inside the prompt, so this checks
# whether the model really understood, not whether it copied an example.
TESTS = [
    (
        "a guy wearing a black t-shirt",
        {"gender": {"include": ["male"]}, "upper_color": {"include": ["black"]}},
    ),
    (
        "woman wearing a black top",
        {"gender": {"include": ["female"]}, "upper_color": {"include": ["black"]}},
    ),
    ("someone wearing grey", {"color": {"include": ["gray"]}}),
    ("anyone not dressed in black", {"color": {"exclude": ["black"]}}),
    ("show me tops that aren't black", {"upper_color": {"exclude": ["black"]}}),
    ("a kid", {"age_group": {"include": ["child"]}}),
    (
        "girl in a light blue t-shirt",
        {
            "gender": {"include": ["female"]},
            "age_group": {"include": ["child"]},
            "upper_color": {"include": ["light blue"]},
        },
    ),
    (
        "people between 20 and 30 on camera 1",
        {"age": {"min": 20, "max": 30}, "camera_id": {"include": ["cam_1"]}},
    ),
    ("person in a scarlet hoodie", {"upper_color": {"include": ["red"]}}),
    ("a person carrying a bag", {}),
]


def normalize(filters):
    """Sorts the value lists, so ["white", "gray"] and ["gray", "white"] count as the same."""
    return {
        field: {
            key: sorted(value) if isinstance(value, list) else value
            for key, value in part.items()
        }
        for field, part in filters.items()
    }


es = get_client()
correct = 0

for sentence, expected in TESTS:
    print("=" * 70)
    print("User types :", sentence)

    start = time.perf_counter()
    try:
        filters = parse_query(sentence)
    except RuntimeError as error:
        # Ollama isn't running or the model isn't downloaded: no point continuing.
        print("ERROR:", error)
        raise SystemExit(1)
    seconds = time.perf_counter() - start

    if normalize(filters) == normalize(expected):
        correct += 1
        print(f"Filters    : {json.dumps(filters)}   OK ({seconds:.1f}s)")
    else:
        print(f"Filters    : {json.dumps(filters)}   DIFFERENT ({seconds:.1f}s)")
        print(f"Expected   : {json.dumps(expected)}")

    output = search(es, filters)
    if output["es_query"] is None:
        print("Search     : nothing to search by")
    else:
        print(f"Search     : {len(output['results'])} found")

print("=" * 70)
print(f"The model got {correct} of {len(TESTS)} right.")

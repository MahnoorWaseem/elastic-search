"""
run_query_builder.py

A test script for step 3 only. It sends hand-written filters (the JSON the
LLM will write in step 4) through the query builder and prints:
  - the filters
  - the Elasticsearch query that was built, one clause per line
  - how many alerts matched, with a short line for each

Step 2 must be done first (the index must exist and contain the alerts).
Run it from inside the nexa-search folder:
    python run_query_builder.py
"""

import json
import logging

from es_index import get_client
from query_builder_and_searcher import search

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

# Each test: what a user might type, and the filters the LLM should write for it.
# Hand written filter that llm will produce eg {
#     "gender": {"include": ["female"]},
#     "upper_color": {"include": ["black"]},
#     "age": {"min": 20, "max": 34},
# }
TESTS = [
    (
        "women in a black top, aged 20-34",
        {
            "gender": {"include": ["female"]},
            "upper_color": {"include": ["black"]},
            "age": {"min": 20, "max": 34},
        },
    ),
    (
        "a man on camera 1",
        {"gender": {"include": ["male"]}, "camera_id": {"include": ["cam_1"]}},
    ),
    ("someone wearing gray (top or bottom)", {"color": {"include": ["gray"]}}),
    ("a top that isn't black", {"upper_color": {"exclude": ["black"]}}),
    ("not wearing black at all", {"color": {"exclude": ["black"]}}),
    ("a child", {"age_group": {"include": ["child"]}}),
    (
        "a red top that isn't red (impossible on purpose)",
        {"upper_color": {"include": ["red"], "exclude": ["red"]}},
    ),
    ("a person with a backpack (nothing searchable)", {}),
]

es = get_client()

for question, filters in TESTS:
    print("=" * 70)
    print("User asks :", question)
    print("Filters   :", json.dumps(filters))

    output = search(es, filters)

    if output["es_query"] is None:
        print("Result    : nothing to search by, so no search was run")
        continue

    # Print each clause on its own line. All of them must be true.
    print("ES query  : all of these must be true")
    for clause in output["es_query"]["bool"]["filter"]:
        print("   ", json.dumps(clause))

    print(f"Result    : {len(output['results'])} found")
    print(output)
    for d in output["results"]:
        print(
            f"    {d.get('gender', '?')}, age {d.get('age', '?')} ({d.get('age_group', '?')}), "
            f"top {d.get('upper_color', '?')}, bottom {d.get('lower_color', '?')}"
        )

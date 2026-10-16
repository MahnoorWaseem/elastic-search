"""
llm_parser.py

Turns what the user types ("man in a red shirt") into the filter JSON that
query_builder.search() understands, using Qwen3.5 2B running in Ollama.

Three safety layers, so the LLM can never send something wrong to Elasticsearch:
  1. The prompt lists the fields and allowed values (built from search_schema.py).
  2. Ollama's "format" option: we give Ollama a JSON schema, and it only lets
     the model write JSON that fits it. The allowed values are an "enum",
     so the model physically cannot write "crimson" or invent a field.
  3. clean_filters() checks the answer again in plain Python.

This file only contains functions. run_llm_parser.py calls them now;
later the backend will call parse_query() when the user presses Search.
"""

import json
import logging

import requests

from config import OLLAMA_MODEL, OLLAMA_URL
from query_builder_and_searcher import allowed_values
from defined_schema import DEFINED_SCHEMA

log = logging.getLogger("llm_parser")


# ---------------------------------------------------------------------------
# The prompt (layer 1)
# ---------------------------------------------------------------------------

INSTRUCTIONS = """You turn a search request about people seen on security cameras into JSON filters.
Do not answer the request. Only write the JSON."""

RULES = """How to write each field (except age): {"include": [...], "exclude": [...]}
- include: values the user wants
- exclude: values the user does NOT want (words like: not, no, without, except)

Rules:
1. Use only the values listed above, spelled exactly the same.
2. Only add fields the user actually mentioned. Never guess.
3. shirt, t-shirt, top, jacket, hoodie, coat, sweater -> upper_color
   pants, trousers, jeans, shorts, skirt -> lower_color
4. A color without saying top or bottom -> color
5. man, guy, boy -> gender male. woman, lady, girl -> gender female
6. kid, child -> age_group child. adult, grown-up -> age_group adult
7. An age number or life stage (teenager, elderly, in her 20s, under 30) -> age with min and max
8. A color that is not in the list -> the closest listed color (crimson -> red, grey -> gray, dark blue -> navy)
9. Ignore everything else (bags, hats, actions). If nothing matches, write {}"""

# Worked examples. A small model learns the pattern from these much better
# than from rules alone.
# The fields in each answer are in the same order as search_schema.py,
# because Ollama expects the JSON fields in that order.


# 2. Putting the colour in both upper_color and lower_color means AND, not OR. Every field becomes its own filter, and all filters must be true. So this example means top is blue and bottom is blue.

EXAMPLES = [
    (
        "man in a red shirt",
        {"gender": {"include": ["male"]}, "upper_color": {"include": ["red"]}},
    ),
    (
        "girl with a pink top and blue jeans",
        {
            "gender": {"include": ["female"]},
            "upper_color": {"include": ["pink"]},
            "lower_color": {"include": ["blue"]},
        },
    ),
    ("person wearing red", {"color": {"include": ["red"]}}),
    ("someone not wearing black", {"color": {"exclude": ["black"]}}),
    (
        "teenager in a white or gray hoodie",
        {"age": {"min": 13, "max": 19}, "upper_color": {"include": ["white", "gray"]}},
    ),
    (
        "elderly woman on camera 1",
        {
            "gender": {"include": ["female"]},
            "age": {"min": 55, "max": 85},
            "camera_id": {"include": ["cam_1"]},
        },
    ),
    (
        "adult in a top that is not black",
        {"age_group": {"include": ["adult"]}, "upper_color": {"exclude": ["black"]}},
    ),
    (
        "guy in a crimson jacket and grey pants",
        {
            "gender": {"include": ["male"]},
            "upper_color": {"include": ["red"]},
            "lower_color": {"include": ["gray"]},
        },
    ),
    (
        "a girl in blue",
        {
            "gender": {"include": ["female"]},
            "color": {"include": ["blue"]},
        },
    ),
    ("person with a backpack", {}),
]


def build_prompt():
    """
    Puts the instructions together. The field list comes from
    search_schema.py, so a new field or value shows up here automatically.
    """
    field_lines = []
    for field, rules in DEFINED_SCHEMA.items():
        if rules["type"] == "range":
            field_lines.append(
                f"- {field}: {rules['description']}. "
                'Write it as {"min": number, "max": number}'
            )
        else:
            values = ", ".join(allowed_values(field))
            field_lines.append(f"- {field}: {rules['description']}. Values: {values}")

    example_lines = []
    for request, answer in EXAMPLES:
        example_lines.append(f"Request: {request}\nJSON: {json.dumps(answer)}")
    prompt = (
        INSTRUCTIONS
        + "\n\nFields:\n"
        + "\n".join(field_lines)
        + "\n\n"
        + RULES
        + "\n\nExamples:\n"
        + "\n\n".join(example_lines)
    )
    print("===== PROMPT =====")
    print(prompt)
    print("===== END PROMPT =====")
    return prompt


# ---------------------------------------------------------------------------
# The JSON schema for Ollama (layer 2)
# ---------------------------------------------------------------------------


# For an enum field like gender, it builds:
# "gender": {
#   "type": "object",
#   "properties": {
#     "include": {"type": "array", "items": {"type": "string", "enum": ["male", "female"]}},
#     "exclude": {"type": "array", "items": {"type": "string", "enum": ["male", "female"]}}
#   },
#   "additionalProperties": false
# }

# {"gender":  {"include": ["female"]}
# When you pass this schema as format, Ollama uses constrained decoding. At every step, before the model picks its next token, Ollama blocks every token that would break the schema. After the model writes "include": [", the only text it can produce next is male or female


def build_json_schema():
    """
    The shape the model's answer MUST have, built from search_schema.py.
    Ollama turns it into hard rules while the model is writing.

      age (range)          -> {"min": number, "max": number}
      every other field    -> {"include": [...], "exclude": [...]}
                              each item must be one of the allowed values

    No field is "required", so the model can leave out what the user
    didn't mention. "additionalProperties": False stops it inventing fields.
    """
    properties = {}

    for field, rules in DEFINED_SCHEMA.items():
        if rules["type"] == "range":
            properties[field] = {
                "type": "object",
                "properties": {
                    "min": {"type": "integer"},
                    "max": {"type": "integer"},
                },
                "additionalProperties": False,
            }
        else:
            value_list = {
                "type": "array",
                "items": {"type": "string", "enum": allowed_values(field)},
            }
            properties[field] = {
                "type": "object",
                "properties": {"include": value_list, "exclude": value_list},
                "additionalProperties": False,
            }

    schema = {"type": "object", "properties": properties, "additionalProperties": False}
    print("===== JSON SCHEMA =====")
    print(json.dumps(schema, indent=2))
    print("===== END JSON SCHEMA =====")
    return schema


# Built once when the file is loaded, not on every search.
# Sending the exact same prompt every time also lets Ollama reuse work
# from the previous search, so later searches are faster.
PROMPT = build_prompt()
JSON_SCHEMA = build_json_schema()


# ---------------------------------------------------------------------------
# Checking the answer (layer 3)
# ---------------------------------------------------------------------------


def clean_filters(raw):
    """
    Checks the model's answer in plain Python before anyone uses it.
    Keeps only known fields and allowed values, keeps ages inside 1-85,
    and drops empty parts. Layer 2 should already guarantee all this,
    but checking again costs nothing.
    """
    clean = {}
    if not isinstance(raw, dict):
        return clean

    for field, wanted in raw.items():
        rules = DEFINED_SCHEMA.get(field)
        if rules is None or not isinstance(wanted, dict):
            continue

        part = {}

        if rules["type"] == "range":
            for key in ("min", "max"):
                value = wanted.get(key)
                if isinstance(value, (int, float)):
                    # Keep the age inside the schema's limits (1 to 85).
                    part[key] = max(rules["min"], min(rules["max"], int(value)))
        else:
            allowed = allowed_values(field)
            for key in ("include", "exclude"):
                values = wanted.get(key)
                if isinstance(values, list):
                    kept = [v for v in values if v in allowed]
                    if kept:
                        part[key] = kept

        if part:
            clean[field] = part

    return clean


# ---------------------------------------------------------------------------
# The main function
# ---------------------------------------------------------------------------


def parse_query(text):
    """
    Sends the user's text to Qwen and returns clean filters, for example:
      "man in a red shirt" -> {"gender": {"include": ["male"]},
                               "upper_color": {"include": ["red"]}}
    Returns {} if the model found nothing searchable.
    """
    body = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},  # the instructions
            {"role": "user", "content": text},  # what the user typed
        ],
        # Layer 2: the answer must fit this shape.
        "format": JSON_SCHEMA,
        # Qwen3.5 can "think out loud" before answering. This task doesn't
        # need it, and on a CPU it would add many seconds.
        "think": False,
        # Wait for the whole answer instead of receiving it word by word.
        "stream": False,
        # temperature 0 = always pick the most likely word,
        # so the same question always gives the same JSON.
        "options": {"temperature": 0},
        # Keep the model loaded for 30 minutes after a search, so the next
        # search doesn't wait for it to load again (slow on a CPU).
        "keep_alive": "30m",
    }

    try:
        # The first search can be slow (loading the model on a CPU),
        # so we wait up to 2 minutes.
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
    except requests.ConnectionError:
        raise RuntimeError(f"Can't reach Ollama at {OLLAMA_URL}. Is Ollama running?")

    if response.status_code != 200:
        # For example: model not downloaded yet -> run "ollama pull qwen3.5:2b"
        raise RuntimeError(f"Ollama returned an error: {response.text}")

    answer = response.json()["message"]["content"]
    log.info("model answered: %s", answer)

    try:
        raw_filters = json.loads(answer)
    except json.JSONDecodeError:
        log.warning("the model's answer was not valid JSON: %s", answer)
        return {}

    return clean_filters(raw_filters)

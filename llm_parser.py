"""
llm_parser.py

Turns what the user types ("man in a red shirt") into the filter JSON that
query_builder.search() understands, using Qwen3.5 2B running in Ollama.

Three safety layers, so the LLM can never send something wrong to Elasticsearch:
  1. The prompt lists the fields and allowed values (built from DEFINED_SCHEMA.py).
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

RULES = """How to write each field: {"include": [...], "exclude": [...]}
- include: values the user wants
- exclude: values the user does NOT want (words like: not, no, without, except)

Rules:
1. Use only the values listed above, spelled exactly the same.
2. Only add fields the user actually mentioned. Never guess.
3. A color with NO top or bottom word (a woman in brown, man wearing red) -> color.
   Never use upper_color or lower_color unless the user names the clothing.
4. Top words like  shirt, t-shirt, top, jacket, hoodie, coat, sweater-> upper_color
   Bottom words like pants, trousers, jeans, shorts, skirt-> lower_color
5. man, guy, boy -> gender male. woman, lady, girl -> gender female
6. age_group ONLY when the request says an age or a stage of life.
   man, woman, guy, lady, boy, girl, person, people, someone are NOT age words:
   "a man in a green shirt" has NO age_group.
   child: kid, child, baby, toddler, little boy, little girl, teenager, teen,
          youth, young person, or an age of 15 or younger (a 14 year old, under 12)
   adult: adult, grown-up, old, elderly, senior, middle-aged, young man, young woman,
          or an age of 16 or older (a 30 year old, above 15, over 20)
   An age range that covers both (between 10 and 20) -> both child and adult
7. A color that is not in the list -> the closest listed color (crimson -> red, grey -> gray, dark blue -> navy)
8. camera 2, cam 2, cam2 -> camera_id with the number only: 2
9. Ignore everything else (bags, hats, actions). If nothing matches, write {}"""
# Worked examples. A small model learns the pattern from these much better
# than from rules alone.
# The fields in each answer are in the same order as search_schema.py,
# because Ollama expects the JSON fields in that order.
# Worked examples. A small model learns the pattern from these much better
# than from rules alone.
# Several examples come in PAIRS that differ only in one word, so the model
# sees exactly what that word changes:
#   "man in a red shirt"  -> no age_group      "an old man in a red shirt" -> adult
# There are also more plain man/woman/girl examples WITHOUT age_group than
# with it, so the model doesn't learn "man means adult".
# The fields in each answer are in the same order as defined_schema.py,
# because Ollama expects the JSON fields in that order.
EXAMPLES = [
    # --- pairs: plain person word -> no age_group, age word -> age_group ---
    (
        "man in a red shirt",
        {"gender": {"include": ["male"]}, "upper_color": {"include": ["red"]}},
    ),
    (
        "man in a blue pants",
        {"gender": {"include": ["male"]}, "lower_color": {"include": ["blue"]}},
    ),
    (
        "an old man in a red shirt",
        {
            "gender": {"include": ["male"]},
            "age_group": {"include": ["adult"]},
            "upper_color": {"include": ["red"]},
        },
    ),
    (
        "a woman on camera 2",
        {"gender": {"include": ["female"]}, "camera_id": {"include": [2]}},
    ),
    (
        "an elderly woman on camera 2",
        {
            "gender": {"include": ["female"]},
            "age_group": {"include": ["adult"]},
            "camera_id": {"include": [2]},
        },
    ),
    # --- plain person words: gender only, never age_group ---
    (
        "a guy in a black or gray hoodie on camera 3",
        {
            "gender": {"include": ["male"]},
            "upper_color": {"include": ["black", "gray"]},
            "camera_id": {"include": [3]},
        },
    ),
    (
        "girl with a pink top and blue jeans",
        {
            "gender": {"include": ["female"]},
            "upper_color": {"include": ["pink"]},
            "lower_color": {"include": ["blue"]},
        },
    ),
    (
        "a girl in blue on camera 2",
        {
            "gender": {"include": ["female"]},
            "color": {"include": ["blue"]},
            "camera_id": {"include": [2]},
        },
    ),
    (
        "a lady in brown",
        {"gender": {"include": ["female"]}, "color": {"include": ["brown"]}},
    ),
    (
        "a man in a top that is not black",
        {"gender": {"include": ["male"]}, "upper_color": {"exclude": ["black"]}},
    ),
    ("person wearing red jacket", {"upper_color": {"include": ["red"]}}),
    ("someone not wearing black", {"color": {"exclude": ["black"]}}),
    # --- age words and numbers ---
    (
        "a 14 year old in a white or gray hoodie",
        {
            "age_group": {"include": ["child"]},
            "upper_color": {"include": ["white", "gray"]},
        },
    ),
    (
        "people above 15 in a black top",
        {"age_group": {"include": ["adult"]}, "upper_color": {"include": ["black"]}},
    ),
    (
        "young man in a crimson jacket and grey pants",
        {
            "gender": {"include": ["male"]},
            "age_group": {"include": ["adult"]},
            "upper_color": {"include": ["red"]},
            "lower_color": {"include": ["gray"]},
        },
    ),
    ("kid in jeans", {"age_group": {"include": ["child"]}}),
    (
        "teenager in a blue top",
        {"age_group": {"include": ["child"]}, "upper_color": {"include": ["blue"]}},
    ),
    (
        "people between 10 and 20 on camera 1",
        {"age_group": {"include": ["child", "adult"]}, "camera_id": {"include": [1]}},
    ),
    # --- nothing searchable ---
    (
        "adult in a top that is not black",
        {"age_group": {"include": ["adult"]}, "upper_color": {"exclude": ["black"]}},
    ),
    ("person with a backpack", {}),
]


def build_prompt():
    """
    Puts the instructions together. The field list comes from
    DEFINED_SCHEMA.py, so a new field or value shows up here automatically.
    """
    field_lines = []
    for field, rules in DEFINED_SCHEMA.items():
        if rules["type"] == "range":
            field_lines.append(
                f"- {field}: {rules['description']}. "
                'Write it as {"min": number, "max": number}'
            )
        elif rules["type"] == "number_id":
            # No list of values here: any number is allowed.
            field_lines.append(
                f"- {field}: {rules['description']}. Write the number only"
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
    The shape the model's answer MUST have, built from DEFINED_SCHEMA.py.
    Ollama turns it into hard rules while the model is writing.

      age (range)          -> {"min": number, "max": number}
      camera_id (number_id)-> {"include": [numbers], "exclude": [numbers]}
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
        elif rules["type"] == "number_id":
            # Any whole number. The model can't be forced into a wrong
            # camera, because there is no list to pick from.
            number_list = {"type": "array", "items": {"type": "integer"}}
            properties[field] = {
                "type": "object",
                "properties": {"include": number_list, "exclude": number_list},
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
        elif rules["type"] == "number_id":
            for key in ("include", "exclude"):
                values = wanted.get(key)
                if isinstance(values, list):
                    # Keep whole numbers above 0 (there is no camera 0 or -1).
                    kept = [
                        int(v) for v in values if isinstance(v, (int, float)) and v > 0
                    ]
                    if kept:
                        part[key] = kept
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

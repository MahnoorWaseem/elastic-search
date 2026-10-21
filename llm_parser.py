"""
llm_parser.py

Turns what the user types ("man in a red shirt last night") into the filter
JSON that query_builder_and_searcher.search() understands.

Every search asks the LLM TWO separate questions, each with its own short prompt:
  1. this file      -> the attributes: gender, age, colours, camera
  2. time_parser.py -> the time
parse_query() asks both and puts the answers together. One job per prompt
keeps each prompt short, so the model doesn't lose track of anything.

Three safety layers, so the LLM can never send something wrong to Elasticsearch:
  1. The prompt lists the fields and allowed values (built from DEFINED_SCHEMA.py).
  2. Ollama's "format" option: we give Ollama a JSON schema, and it only lets
     the model write JSON that fits it. The allowed values are an "enum",
     so the model physically cannot write "crimson" or invent a field.
  3. clean_filters() checks the answer again in plain Python.

This file only contains functions. app.py calls parse_query() when the
user presses Search; run_time_tests.py calls ask_attributes() to test it.
"""

import json
import logging

from config import OLLAMA_MODEL
from defined_schema import DEFINED_SCHEMA
from ollama_client import ask_ollama
from query_builder_and_searcher import allowed_values
from time_parser import ask_time, clean_time, current_time

log = logging.getLogger("llm_parser")


# ---------------------------------------------------------------------------
# The prompt (layer 1)
# ---------------------------------------------------------------------------

INSTRUCTIONS = """You turn a search request about people seen on security cameras into JSON filters.
Do not answer the request. Only write the JSON.
You need to understand the search request and use only the fields that are given below (with description) according to what is asked in the text. You can only use the values that are given below for each field"""

RULES = """How to write each field: {"include": [...], "exclude": [...]}
How to write each field: {"include": [...], "exclude": [...]}
For EACH value, check if the user says NOT to it:
- not, no, without, except, other than, neither/nor -> that value goes in "exclude"
- no such word -> that value goes in "include" (this is the normal case)
A "not" only belongs to the value it is about. Every other value stays in "include".
Pairs, where the only difference is the word "not":
  a woman in green                    -> "color": {"include": ["green"]}
  a woman not in green                -> "color": {"exclude": ["green"]}
  a woman in green or white           -> "color": {"include": ["green", "white"]}
  a woman in neither green nor white  -> "color": {"exclude": ["green", "white"]}
  a man in a black shirt              -> "upper_color": {"include": ["black"]}
  a man in a shirt that is not black  -> "upper_color": {"exclude": ["black"]}
  a woman on camera 1                 -> "camera_id": {"include": [1]}
  a woman not on camera 1             -> "camera_id": {"exclude": [1]}

Rules:
1. Use only the values listed above, spelled exactly the same.
2. Only add fields the user actually mentioned. Never guess.
3. color: A color with NO top or bottom word (a woman in brown, man wearing red) -> color.
   Never use upper_color or lower_color unless the user names the clothing.
4. upper_color: if there Top words like  shirt, t-shirt, top, jacket, hoodie, coat, sweater-> use upper_color instead of color.
   lower_color: Bottom words like pants, trousers, jeans, shorts, skirt-> use lower_color instead of color.
5. A color that is not in the list -> use the closest listed color (crimson -> red, grey -> gray, dark blue -> navy)
6. gender: man, guy, boy -> gender male. woman, lady, girl -> gender female
7. age_group: age_group ONLY when the request says an age or a stage of life.
   man, woman, guy, lady, boy, girl, person, people, someone are NOT age words:
   "a man in a green shirt" has NO age_group.
   child: kid, child, baby, toddler, little boy, little girl, teenager, teen,
          youth, young person, or an age of 15 or younger (a 14 year old, under 12)
   adult: adult, grown-up, old, elderly, senior, middle-aged, young man, young woman,
          or an age of 16 or older (a 30 year old, above 15, over 20)
8. camera_id ONLY when the word camera or cam comes before the number:
   camera 2, cam 2, cam2, camera number 2 -> camera_id with the number only: 2
   Every other number is NOT a camera: dates (9 oct, 12/10), times (5pm, at 7), ages (a 14 year old).
   "a girl in pink on 9 oct" has NO camera_id.
9. Ignore everything else (times, dates, bags, hats, actions). If nothing matches, write {}"""

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
        "a girl wearing blue on camera 2",
        {
            "gender": {"include": ["female"]},
            "color": {"include": ["blue"]},
            "camera_id": {"include": [2]},
        },
    ),
    (
        "a girl wearing pink",
        {"gender": {"include": ["female"]}, "color": {"include": ["pink"]}},
    ),
    (
        "a girl wearing a pink top",
        {"gender": {"include": ["female"]}, "upper_color": {"include": ["pink"]}},
    ),
    (
        "a man in a top that is black",
        {"gender": {"include": ["male"]}, "upper_color": {"include": ["black"]}},
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
    # --- time words are ignored here (time_parser.py finds the time) ---
    (
        "a lady in brown yesterday at 5 pm",
        {"gender": {"include": ["female"]}, "color": {"include": ["brown"]}},
    ),
    (
        "a guy in a black jacket on camera 2 last night",
        {
            "gender": {"include": ["male"]},
            "upper_color": {"include": ["black"]},
            "camera_id": {"include": [2]},
        },
    ),
    # --- pairs: a number is a camera ONLY right after "camera" / "cam" ---
    (
        "a lady in pink on 3 november",
        {"gender": {"include": ["female"]}, "color": {"include": ["pink"]}},
    ),
    (
        "a lady in pink on camera 3",
        {
            "gender": {"include": ["female"]},
            "color": {"include": ["pink"]},
            "camera_id": {"include": [3]},
        },
    ),
    (
        "a man in a blue shirt at 4 pm on cam 1",
        {
            "gender": {"include": ["male"]},
            "upper_color": {"include": ["blue"]},
            "camera_id": {"include": [1]},
        },
    ),
    ("a boy at 7 on 7 oct", {"gender": {"include": ["male"]}}),
    (
        "a boy at 7 on cam 15",
        {
            "gender": {"include": ["male"]},
            "camera_id": {"include": [15]},
        },
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
        if rules["type"] == "time":
            continue  # time has its own prompt in time_parser.py
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
        if rules["type"] == "time":
            continue  # time has its own JSON schema in time_parser.py
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
        if rules["type"] == "time":
            continue  # the time comes from time_parser.py, never from here

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
# The main functions
# ---------------------------------------------------------------------------


def ask_attributes(text):
    """
    Asks the model for the attributes in the user's text (gender, age,
    colours, camera). Returns (answer, thinking), the answer exactly as the
    model wrote it, before any checking.
    """
    return ask_ollama(OLLAMA_MODEL, PROMPT, text, JSON_SCHEMA)


def parse_query(text, now=None):
    """
    The two LLM answers, checked and put together. For example
    "a woman in brown today" ->
        {"gender": {"include": ["female"]},           <- from ask_attributes()
         "color":  {"include": ["brown"]},
         "time":   [{"from_date": "2026-10-20",       <- from time_parser.ask_time()
                     "from_time": "00:00",
                     "to_date":   "2026-10-20",
                     "to_time":   "23:59"}]}
    Returns {} if neither answer found anything searchable.
    now: the current time. Leave it out to use the real time in Pakistan.
    """
    now = now or current_time()

    # Question 1: the attributes.
    # raw filters
    attributes, _thinking = ask_attributes(text)
    # cleaned filters
    filters = clean_filters(attributes)
    print("filters for person attributes", filters)
    # Question 2: the time.
    time_answer, _thinking = ask_time(text, now)
    print("filters for time attributes", time_answer)
    windows = clean_time(time_answer, now)
    print("cleaned filters for time attributes", windows)
    if windows:
        filters["time"] = windows

    return filters

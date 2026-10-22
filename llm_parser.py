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

INSTRUCTIONS = """You convert a search request about people on security cameras into JSON filters.
Output ONLY one JSON object. Never answer or explain the request.
Use only the fields and values listed below. Add a field ONLY if the request says something about it.
If the request has nothing searchable, output {}."""

RULES = """Format: every field is {"include": [...], "exclude": [...]}. Leave out the one that is empty.
- "include" is the normal case.
- "exclude" only when a negative word is attached to that value: not, no, without, except, other than, neither/nor.
  A negative word affects only its own value. Everything else stays in "include".
- "a or b" -> both in the same list. "neither a nor b" -> both in "exclude".
 
Rules:
1. Use values exactly as listed. A colour that is not listed -> the closest listed colour
   (crimson -> red, grey -> gray, dark blue -> navy, violet -> purple, tan -> beige).
2. gender: man, guy, boy, gentleman -> male. woman, lady, girl -> female.
   person, people, someone, anyone -> no gender.
3. age_group: only if the request says an age or a stage of life. Plain person words
   (man, woman, boy, girl, guy, lady, person) are NOT age words.
   child: kid, child, baby, toddler, teen, teenager, youth, or an age of 15 or younger.
   adult: adult, old, elderly, senior, middle-aged, young man, young woman, or an age of 16 or older.
4. Colour field: look at the clothing word next to the colour.
   top word (shirt, t-shirt, top, jacket, hoodie, coat, sweater) -> upper_color
   bottom word (pants, trousers, jeans, shorts, skirt, bottoms) -> lower_color
   no clothing word (in red, wearing blue, not in pink) -> color
5. camera_id: only when the word camera or cam is directly followed by a number
   (camera 2, cam 2, cam2, camera number 2). Write the number only.
   Several numbers are allowed: "camera 2 or 4" -> [2, 4].
   Every other number is NOT a camera: dates (3 november), times (7 pm), ages (12 year old).
6. Ignore dates, times, days, bags, hats, glasses and actions."""

# Key order in every answer = order of DEFINED_SCHEMA.
EXAMPLES = [
    # --- the exact case that failed: person + camera, no colour. Must NOT invent a colour ---
    (
        "a person in a black shirt",
        {"upper_color": {"include": ["black"]}},
    ),
    (
        "a person in a shirt that is not black",
        {"upper_color": {"exclude": ["black"]}},
    ),
    (
        "people in a white top",
        {"upper_color": {"include": ["white"]}},
    ),
    (
        "a man on camera 2",
        {"gender": {"include": ["male"]}, "camera_id": {"include": [2]}},
    ),
    # --- colour field choice: color / upper_color / lower_color ---
    (
        "woman in red",
        {"gender": {"include": ["female"]}, "color": {"include": ["red"]}},
    ),
    (
        "man in a blue shirt",
        {"gender": {"include": ["male"]}, "upper_color": {"include": ["blue"]}},
    ),
    (
        "lady wearing a green skirt",
        {"gender": {"include": ["female"]}, "lower_color": {"include": ["green"]}},
    ),
    (
        "girl in a yellow top and brown jeans",
        {
            "gender": {"include": ["female"]},
            "upper_color": {"include": ["yellow"]},
            "lower_color": {"include": ["brown"]},
        },
    ),
    # --- several colours, no gender word ---
    (
        "someone in a white or gray hoodie",
        {"upper_color": {"include": ["white", "gray"]}},
    ),
    # --- exclusion: the "not" belongs only to its own value ---
    (
        "a woman not in pink",
        {"gender": {"include": ["female"]}, "color": {"exclude": ["pink"]}},
    ),
    (
        "a man in a shirt that is not black",
        {"gender": {"include": ["male"]}, "upper_color": {"exclude": ["black"]}},
    ),
    (
        "guy in neither green nor white",
        {"gender": {"include": ["male"]}, "color": {"exclude": ["green", "white"]}},
    ),
    # --- age_group: only with an age word or a number ---
    (
        "an elderly man in navy pants",
        {
            "gender": {"include": ["male"]},
            "age_group": {"include": ["adult"]},
            "lower_color": {"include": ["navy"]},
        },
    ),
    (
        "a 12 year old girl",
        {"gender": {"include": ["female"]}, "age_group": {"include": ["child"]}},
    ),
    (
        "kid in a red jacket",
        {"age_group": {"include": ["child"]}, "upper_color": {"include": ["red"]}},
    ),
    (
        "people between 10 and 20",
        {"age_group": {"include": ["child", "adult"]}},
    ),
    (
        "anyone under 12 on camera 5",
        {"age_group": {"include": ["child"]}, "camera_id": {"include": [5]}},
    ),
    # --- colour mapping to the closest listed colour ---
    (
        "young woman in a crimson jacket and grey pants",
        {
            "gender": {"include": ["female"]},
            "age_group": {"include": ["adult"]},
            "upper_color": {"include": ["red"]},
            "lower_color": {"include": ["gray"]},
        },
    ),
    # --- camera: every spelling, lists, exclusion, camera alone ---
    (
        "a lady in black on camera 2",
        {
            "gender": {"include": ["female"]},
            "color": {"include": ["black"]},
            "camera_id": {"include": [2]},
        },
    ),
    (
        "person in orange on cam3",
        {"color": {"include": ["orange"]}, "camera_id": {"include": [3]}},
    ),
    (
        "a man in a purple shirt on camera 2 or 4",
        {
            "gender": {"include": ["male"]},
            "upper_color": {"include": ["purple"]},
            "camera_id": {"include": [2, 4]},
        },
    ),
    (
        "boy not on camera 1",
        {"gender": {"include": ["male"]}, "camera_id": {"exclude": [1]}},
    ),
    ("camera 4", {"camera_id": {"include": [4]}}),
    # --- numbers that are NOT cameras (dates, times), next to ones that are ---
    (
        "woman in beige on 3 november at 7 pm",
        {"gender": {"include": ["female"]}, "color": {"include": ["beige"]}},
    ),
    (
        "a boy at 7 on cam 15",
        {"gender": {"include": ["male"]}, "camera_id": {"include": [15]}},
    ),
    # --- time words are ignored here (time_parser.py handles them) ---
    (
        "a man in a pink hoodie on camera 2 last night",
        {
            "gender": {"include": ["male"]},
            "upper_color": {"include": ["pink"]},
            "camera_id": {"include": [2]},
        },
    ),
    ("a man yesterday", {"gender": {"include": ["male"]}}),
    # --- nothing searchable ---
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
    print("raw1 filters for person attributes", attributes)
    # cleaned filters
    filters = clean_filters(attributes)
    print("cleaned filters for person attributes", filters)
    # Question 2: the time.
    time_answer, _thinking = ask_time(text, now)
    print("raw2 filters for time attributes", time_answer)
    windows = clean_time(time_answer, now)
    print("cleaned filters for time attributes", windows)
    if windows:
        filters["time"] = windows

    return filters
